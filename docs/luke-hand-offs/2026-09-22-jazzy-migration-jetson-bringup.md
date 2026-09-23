# Hand-off: Jazzy / Ubuntu 24.04 migration and Jetson v4 bring-up (2026-09-22)

## Goal

Move the project from ROS 2 Humble to **Jazzy**, and get the **v4 detector
running as a ROS node on the Orin Nano**. Both halves moved: the repo's ROS
config is migrated (uncommitted, unbuilt), and the board was successfully
reflashed to JetPack 7.2. The session stopped mid-bring-up, at the point where
`torch` needs installing on the board.

## State

- Branch **`ModelTraining`**, HEAD **`e52d06c`**, remote `origin` =
  `github.com/Elijahtab/SkyPilot.git`. This hand-off is the only commit.
- **The whole Jazzy migration is uncommitted.** Modified:
  `.devcontainer/devcontainer.json`, `docker/Dockerfile.dev`,
  `docker/Dockerfile.jetson`, `docker/compose.yaml`,
  `docker/requirements-ros.txt`, `ros2_ws/README.md`,
  `ros2_ws/src/skypilot_vision/setup.py`. Untracked:
  `ros2_ws/src/skypilot_vision/skypilot_vision/v4_detector_node.py`.
- Still untracked on purpose (unchanged from 2026-09-15):
  `Vehicle_type_detection/runs/Vehicle_type_detection_v8/`,
  `Vehicle_type_detection/runs_cls/`, `YoloTraining/`, `yolo11n.pt`,
  `weights/pretrained/yolo11s-cls.pt`, and the three 0-byte placeholders.

### Verified vs pending

| Thing | Status |
|---|---|
| Board flashed to JetPack 7.2 / Ubuntu 24.04 | ✅ confirmed — apt resolved `dists/noble` |
| ROS 2 Jazzy + `cv_bridge` importing on the board | ✅ confirmed by Luke |
| `torch` on the board | ❌ **not installed — this is where we stopped** |
| `v4_detector` node ever run | ❌ never executed. Syntax-checked only |
| `Dockerfile.dev` / `Dockerfile.jetson` build | ❌ **never built.** No Docker on the Windows workstation |
| `numpy<2` / `cv_bridge` interaction on real hardware | ❌ predicted from upstream issue, not observed |

**Nothing in the Docker half of this migration has been executed.** Treat every
Dockerfile change as a reasoned edit, not a working image.

## Key findings

### The Humble decision in `e52d06c` was based on a wrong premise

That commit argued for Humble because "the Orin Nano is on JetPack 5" and
JetPack 7.2 would be a two-hop reflash. **The board was actually already on
R36 (JetPack 6).** Firmware was therefore already ≥ 36.0, the JetPack 5.1.3 →
`nvidia-l4t-jetson-orin-nano-qspi-updater` bridge hop was never needed, and the
reflash was a single hop. `ros2_ws/README.md` has been rewritten accordingly.

### JetPack 7.2 install path for the Orin Nano (this cost the most to establish)

- JetPack **7.2 is the first 7.x that supports Orin at all** — 7.0 and 7.1 are
  Thor-only. Ships Ubuntu 24.04, Python 3.12, CUDA 13.2, L4T r39.2.
- There is **no downloadable SD card image** as of 7.2. But microSD is still a
  valid *install target* via the new **Jetson ISO installer**: write the ISO to
  a 16 GB+ USB stick with Balena Etcher, boot the Jetson from it (`Esc` →
  Boot Manager → USB), GRUB → "Install Jetson ISO r39.2.1", pick the microSD.
- Consequences, both of which reverse earlier advice in this session: **no NVMe
  purchase required**, and **no Ubuntu x86_64 host / SDK Manager required** —
  the host PC only writes a USB stick, so Windows is fine.
- A firmware capsule update prompt appears with a ~30 second window during
  boot; press `Y`. That is the one genuinely risky moment.

### The Jetson Docker image had to invert its layering

Under Humble, `docker/Dockerfile.jetson` started `FROM
ultralytics/ultralytics:latest-jetson-jetpack6` and added ROS, because that
image carried a *validated* aarch64 CUDA torch. **No equivalent exists for
JetPack 7 on an Orin** — Ultralytics' only JetPack 7 tag is Thor-only and
PyTorch-only, with no TensorRT. So the file now goes:

```
nvcr.io/nvidia/l4t-jetpack:r39.2.1   (Ubuntu 24.04)
  → apt ros-jazzy-*
  → pip --pre torch torchvision --extra-index-url .../cu132
  → pip --no-deps -r requirements-ros.txt
```

It now mirrors `Dockerfile.dev` instead of contradicting it. The cost: torch is
a **prerelease wheel**, not a vendor-validated container. `TORCH_CUDA` is a
build arg (default `cu132`) because JetPack 7.1 would need `cu130`.

### numpy had to go DOWN, 2.1.2 → 1.26.4

Jazzy's apt `cv_bridge` on noble is compiled against the **numpy 1.x C ABI**.
numpy 2.x makes it fail on import with *"module compiled using NumPy 1.x cannot
be run in NumPy 2.0.0"* — at run time, inside the detector node, not at build
time. Upstream: https://github.com/ros-perception/vision_opencv/issues/535

1.26.4 is exactly what noble ships as `python3-numpy`. Checked before pinning:
the runtime path uses only `np.asarray`, `np.ascontiguousarray` and `np.zeros`,
all identical across numpy 1 and 2, so this costs nothing.

Note the knock-on: under Humble, "don't reuse the root `requirements.txt`" was
self-enforcing because Python 3.10 physically could not install it. Jazzy is
3.12, the same as the training workstation, so **it will now install happily
and quietly pull numpy 2.4.** That gotcha is now a live trap, and is documented
as such in `ros2_ws/README.md`.

### `type_cls_v1` weights are not in git — a clone yields a broken node

`git ls-files` confirms `Vehicle_type_detection/runs/Vehicle_type_detection_v4/weights/best.pt`
**is** tracked, but `Vehicle_type_detection/runs_cls/type_cls_v1/weights/best.pt`
**is not** (`runs_cls/` is untracked). `scripts/pipeline/two_stage.py:44-45`
loads both and the constructor raises `ValueError` on a class-name mismatch, so
**a fresh clone produces a `detector_node` that dies at startup.** This is a
real deployment gap for anyone but Luke.

### Deployment payload is 50 MB, not 1.18 GiB

`ros2_ws/src` is **55 KB across 13 files**; v4's `best.pt` is **50 MB**. The
repo's tracked history is 1.18 GiB. There is no reason to clone it onto the
board — `scp` of those two things is sufficient.

## Gotchas

- **Humble and Jazzy cannot share a ROS graph** (Jazzy added rosidl/DDS type
  hashes). This is never a per-machine migration; every machine moves together
  or they silently fail to discover each other.
- **`torch` must be installed before `ultralytics`**, or pip resolves its own
  generic wheel that cannot see the GPU. `--pre` is equally load-bearing:
  without it pip silently takes the stable **CPU-only** wheel, which imports
  fine and reports `torch.cuda.is_available() == False`.
- **`--break-system-packages` is mandatory on noble** (PEP 668). Keep installs
  in the system environment, *not* a venv — `rclpy` and `cv_bridge` live there.
- **`scp` from Windows needs a leading `.\`**. `scp S:\GitHub\...` makes scp
  read `S:` as a hostname.
- **Sourcing is per-shell.** Both `/opt/ros/jazzy/setup.bash` and
  `~/ros2_ws/install/setup.bash`. Most "module not found" reports this session
  traced back to this.
- **Do not `pip install cv_bridge`.** There is an unrelated PyPI package by
  that name. `cv_bridge` comes from `apt install ros-jazzy-cv-bridge`.
- **Orin Nano devkits have no RTC battery.** After a flash the clock can be far
  enough off to fail TLS certificate validation — which presents as an apt
  "certificate is NOT trusted" error against `packages.ros.org`. Check `date`
  first. Never "fix" it with `Acquire::https::Verify-Peer false`.
- **Any TensorRT `.engine` built under JetPack 6 is now dead.** Engines are
  tied to the JetPack/CUDA/GPU they were built on.

## Next steps

1. **Install torch on the board.** Confirm the channel first with
   `nvcc --version` (13.2 → `cu132`, 13.0 → `cu130`):
   ```bash
   pip install --break-system-packages --pre torch torchvision \
       --extra-index-url https://download.pytorch.org/whl/cu132
   pip install --break-system-packages "numpy<2"
   pip install --break-system-packages ultralytics
   ```
2. **Verify before launching anything:**
   ```bash
   python3 -c "import torch, numpy, cv_bridge; print(numpy.__version__, torch.cuda.is_available())"
   ```
   Want `1.26.x` and `True`. If numpy reads 2.x, ultralytics bumped it — rerun
   step 1's numpy line. If torch's version string carries `+cpu`, the `--pre`
   or index URL was wrong.
3. **Deploy and run v4** (from `S:\GitHub\SkyPilot`, PowerShell):
   ```powershell
   scp -r .\ros2_ws\src\skypilot_msgs .\ros2_ws\src\skypilot_vision user@jetson:~/ros2_ws/src/
   scp .\Vehicle_type_detection\runs\Vehicle_type_detection_v4\weights\best.pt user@jetson:~/v4.pt
   ```
   then on the board:
   ```bash
   cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
   ros2 run skypilot_vision v4_detector --ros-args -p weights:=$HOME/v4.pt
   ```
4. **Nothing publishes `/camera/image_raw` yet.** The node will start, log
   `v4 detector ready`, and sit silent — that is not a failure. `tello_driver`
   is unwritten. Needs a bag or a small image publisher to prove end-to-end.
   An `image_publisher` node was offered and not yet written.
5. **Decide what to commit.** The Jazzy migration is 7 modified files plus the
   new node, none of it build-verified. Committing it is reasonable — it is
   reasoned work and the board really is on 7.2 now — but the commit message
   should say the Docker images are unbuilt.
6. **Fix the `type_cls_v1` distribution gap** (see Key findings): either track
   the 11 MB weights, or add a fetch script beside
   `scripts/labeling/download_kaggle.py`.
7. **Open question, deferred:** whether to drop torch on the board in favour of
   a TensorRT `.engine` (ships with JetPack, ~2–3× faster, zero install) at the
   cost of hand-writing letterbox/NMS/rescale pre- and post-processing, and
   diverging from the ultralytics-based `two_stage.py` and `jetson_bench.py`.
   Recommendation given: finish with torch first, optimise only with numbers.

## References

- `ros2_ws/README.md` — rewritten "Why Jazzy, and what it cost" section holds
  the current decision record; supersedes the rationale in commit `e52d06c`.
- `docker/Dockerfile.jetson` — the inverted-layering explanation is in the
  header comment, including why `--pre` and `--no-deps` are load-bearing.
- `docker/requirements-ros.txt` — the numpy pin and its reasoning.
- `ros2_ws/src/skypilot_vision/skypilot_vision/v4_detector_node.py` — the
  standalone node; its docstring states exactly what it trades away against
  `detector_node.py`.
- https://github.com/ros-perception/vision_opencv/issues/535 — the cv_bridge /
  numpy 2 ABI break.
- NVIDIA Orin Nano devkit user guide — Jetson ISO installer procedure and the
  JetPack 6.x firmware update path.
- Prior hand-off: `docs/luke-hand-offs/2026-09-15-kaggle-train-split-assessment.md`
