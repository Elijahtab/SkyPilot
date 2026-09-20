# SkyPilot ROS 2 workspace

**ROS 2 Humble** (Ubuntu 22.04, Python 3.10). Everything runs in Docker, on
every machine, so a teammate's laptop and the Jetson run identical versions.

## Why Humble and not Jazzy

Jazzy is the longer-lived LTS (May 2029 against Humble's **May 2027**), and on
paper it is what NVIDIA recommends for JetPack 7.2. We are on Humble anyway,
for the board:

- The Orin Nano is on JetPack 5 today, and JetPack 7.2 **requires JetPack 6.x
  UEFI firmware first** — it is a two-hop reflash, with reported UEFI capsule
  failures on the way.
- JetPack 6 gives us `ultralytics/ultralytics:latest-jetson-jetpack6`, a
  *validated* aarch64 CUDA torch build. JetPack 7.2 needs undocumented
  prerelease `--pre cu132` wheels, and Ultralytics has not validated it.
- Humble is native on JetPack 6's Ubuntu 22.04, so no version straddling.

**This is a dated decision.** Revisit before May 2027.

## Layout

| Package | What it is |
|---|---|
| `skypilot_msgs` | `Vehicle.msg`, `VehicleArray.msg` — the wire contract |
| `skypilot_vision` | `vehicle_detector` node wrapping the two-stage pipeline |

## The topic contract

```
/camera/image_raw  (sensor_msgs/Image)
        │
        ▼
  vehicle_detector          ← wraps scripts/pipeline/two_stage.py
        │
        ▼
  vehicles  (skypilot_msgs/VehicleArray)
```

The node does **not** reimplement detection. It imports
`scripts/pipeline/two_stage.py` from the repo, so the offline tools
(`index_frames.py`, `search_vehicles.py`) and the live robot run the same code.
If they ever disagree, that is a bug in the node.

Two semantics worth knowing before you consume `vehicles`:

- **`type` of `Vehicle` is the umbrella class** — true but unspecific. It is
  what you get when the box was too small to type (`type_status: too_small`)
  or the classifier was not confident (`unsure`). It is not a junk bucket.
  Branch on `type_status`, and use `type_guess` if you want to apply your own
  threshold.
- **Absent values are NaN and empty string**, because ROS messages cannot carry
  `None`.

## Running it

### On a laptop

Open the repo in VS Code and *Reopen in Container* — `.devcontainer/` builds
`docker/Dockerfile.dev` (CPU torch) and runs `colcon build` for you. Then:

```bash
source install/setup.bash
ros2 launch skypilot_vision vision.launch.py image_topic:=/camera/image_raw
```

No camera and no GPU needed to develop against this: replay a bag, or publish
into `/camera/image_raw` from any source.

### On the Jetson

```bash
docker compose -f docker/compose.yaml up vision
```

Verify CUDA is actually attached before trusting any timing:

```bash
docker compose -f docker/compose.yaml run --rm vision \
    python3 -c "import torch; print(torch.cuda.is_available())"
```

## Gotchas

- **`ROS_DOMAIN_ID` must match** across every machine on one graph. Set it in
  the environment; compose passes it through. Give each person a different one
  when testing on a shared network, or you will discover each other's nodes.
- **`network_mode: host` and `ipc: host` are required.** DDS discovery needs
  multicast and shared memory; bridged networking silently breaks both and
  nodes just never see each other.
- **Do not reuse the root `requirements.txt` here.** It targets the Python 3.12
  training workstation and pins `numpy 2.4` / `scipy 1.16`, both of which need
  Python ≥ 3.11 and will not install on 3.10. Use
  `docker/requirements-ros.txt`, which drops scipy entirely (nothing in the
  runtime path imports it — only `scripts/archive/` does).
- **Frames are dropped on purpose.** Inference is ~69 ms on the Orin Nano,
  slower than the frame interval, so the subscription is best-effort depth-1.
  A tracker wants the freshest frame, not a growing backlog.

## Not built yet

`tello_driver`, `tracker`, `voice` and `llm_agent` are mapped but not written.
The existing implementations live in `Drone+OpenAI/` and still run standalone.
