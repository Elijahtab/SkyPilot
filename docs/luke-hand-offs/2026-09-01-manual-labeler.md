# Hand-off: Manual vehicle-type labeler (2026-09-01)

## Goal

Move onto the reorganized `scripts/` tree and build a fast keyboard labeler for
assigning the 7-class vehicle schema to the Kaggle traffic-camera set. New tool
`scripts/labeling/manual_label.py`; the Kaggle dataset is now downloaded locally.
Work continues on the **desktop** next — see "Desktop pickup" below.

## State

- Repo: branch **`ModelTraining`**, HEAD **`48a3ca5`** ("Fix two broken eval
  scripts, add resolution sweep") + the commit this hand-off ships with.
- Session started on local `79abd6c` (5 behind); fast-forwarded to `48a3ca5`.
- macOS checkout: interpreter is **`.venv/bin/python`**, not the Windows
  `myenv\Scripts\python.exe` the docs assume. YOLO runs on **`mps`**.

### This session's commit

`scripts/labeling/manual_label.py` (new), `scripts/README.md` (+"Hand-label"
flow), this doc. Nothing else staged.

### Deliberately uncommitted / untracked

| Path | Why |
|---|---|
| `Labeling/kaggle_dataset/` (194 MB) | dataset media, gitignored (`.gitignore:50`). Re-fetch with `scripts/labeling/download_kaggle.py`. |
| `Labeling/kaggle_dataset/train/labels_gpt/*.txt` (16 files) | **the labels made so far** — gitignored with the rest of `train/`. Not in the commit. See "Labels don't travel" below. |
| `Labeling/kaggle_dataset/preview/` | 191 tracked previews deleted + 16 modified by `download_kaggle.py`'s `rmtree` / labeler re-render. Cleanup deferred — restore (`git checkout --`) or gitignore `preview/`. |
| `Labeling/new_recognition_dataset/`, `Labeling/auto_label_new_dataset.py`, `YoloTraining/`, `integration_snippet.py`, `releases/`, root `yolov8m.pt` | Prior "Thread 1" work (new Kaggle vehicle-type-recognition set, 362 imgs). Untouched this session. Root `yolov8m.pt` is redundant with `weights/pretrained/yolov8m.pt`. |
| `git stash@{0}` | "WIP pre-reorg: MAX_IMAGES=500 in auto_label_yolo" — one-line tweak to a now-archived script. Drop it unless you want that number. |

### Verified

- `manual_label.py` compiles; `--help` clean; headless smoke tests pass for:
  image collection + dedupe (5248 → 2015 unique source frames), resume (skips
  frames with an existing `labels_gpt/*.txt`), `read_yolo_txt` seed + round-trip,
  all three `--boxes` modes, empty-negative write, preview write.
- yolov8m loads and detects on `mps`.
- Kaggle download: **5248 train / 582 valid / 291 test**, `train/labels/` are
  human-drawn, single class `0`, ~10 boxes/frame, in-bounds.
- `--boxes auto` seeds from `train/labels/` for **100%** of the 2015 deduped
  frames (all have a sibling label), so the detector never runs on this set.

### NOT verified

- The interactive cv2 loop (window, mouse hover, `waitKeyEx`) was never run in a
  GUI — only its pieces. First real drive was by the user this session and it
  "works well enough", but there may be rough edges (hover precision on 2 px
  boxes, key-code quirks) once you use it in volume on the desktop.
- No training has been done. No `diagnose_labels.py` run on the new labels.

## What was built — `scripts/labeling/manual_label.py`

Single-window OpenCV labeler. Per image: draw candidate boxes, hover one, press a
key to assign its type; auto-writes the YOLO `.txt` and loads the next unlabeled
image the moment every box is resolved.

- **Keys**: `1`–`7` = Bus / Vehicle / Motorcycle / SUV / Standard Car / Truck /
  Van (schema order — writes canonical id `key-1`). `d`/Del/⌫ delete hovered box,
  left-drag draw a new box, `u` undo, `n` next (only when resolved), `b` back
  (reopens for edit), `s` skip, `k` confirm empty, `q` quit.
- **`--boxes {auto,existing,detect}`** (default `auto`):
  - `auto` — sibling `labels/` boxes if the frame has them, else yolov8m
  - `existing` — always the human boxes; their class is discarded, you assign it.
    Model is never loaded.
  - `detect` — always yolov8m (`weights/pretrained/yolov8m.pt`, conf 0.25,
    imgsz 960, COCO classes 2/3/5/7). For imagery with no boxes yet.
- **`--out`** defaults to `Labeling/kaggle_dataset/train/labels_gpt/` — same dir
  `auto_label_kaggle.py` writes, so the batch flows into `extract_good_kaggle.py`
  → `diagnose_labels.py` unchanged, and both labelers resume off the same files.
- **`--dedupe`** (default on) — one frame per unique source name (before
  `_jpg.rf.`), 5248 → 2015. `--no-dedupe` to label every export.
- Other: `--source`, `--labels-dir`, `--conf`, `--imgsz`, `--view` (on-screen px,
  upscales small frames), `--limit N`, `--no-preview`, `--redo`.
- Resume is automatic. Model is lazy-loaded (not imported in `existing` mode).
  Decoded images are held in a ±2 window to cap RAM.

## Key finding — don't detect on the Kaggle set, seed from its human boxes

The tool started (per the original spec) running yolov8m to propose boxes. On
this data that misses ~80% of vehicles. Not mainly image *quality* — it's scale +
domain + augmentation:

- Frames are **416×416** Roboflow exports, ~10 vehicles each (23 on busy ones),
  **median box 23 px**, some 2 px. Many frames are rotation/mosaic augmentations
  (sideways cars). Matches the `2026-08-06` hand-off's findings.
- Measured on `Aptakisic-at-Bond-IP-East-0` (**23** human boxes):
  - stock **yolov8m**: **0** boxes at conf 0.25, 1–3 even at 0.05
  - **v4** (the fine-tuned in-domain model): 2 at 0.25, 5 at 0.10 — i.e. the
    ~0.219 recall the last hand-off already measured for this imagery
- MPS also throws `NMS time limit exceeded`, which can zero out results.

The Kaggle set already ships complete human boxes. So `--boxes auto` uses those
and you only classify; the detector is kept for box-less imagery (the Thread-1
photo set).

## Labeling progress — 16 frames, 7 need redo

`labels_gpt/` has 16 `.txt` (frames `Bond-IP-East-0..8`, `Bond-IP-North-0..9`
deduped). The **first 7** (`East-0,1,2,3,4,5,7`, done 15:22–15:30 before the
`--boxes` change) are **detector-seeded and undercount** the vehicles
(East-0 has 5 vs 23). The rest (`East-6,8`, `North-*`, done 15:46+) were
human-box-seeded and match, give or take 1–2 boxes deliberately deleted (too
small/ambiguous to type).

Redo the undercounted ones — delete any `labels_gpt` file with fewer boxes than
its `labels/` twin, then re-run:

```bash
cd Labeling/kaggle_dataset/train
for f in labels_gpt/*.txt; do
  g=$(grep -c . "$f"); h=$(grep -c . "labels/$(basename "$f")")
  [ "$g" -lt "$h" ] && echo "rm $f  (gpt=$g human=$h)" && rm "$f"
done
```

(PowerShell equivalent on the desktop, or just delete `East-0,1,2,3,4,5,7`.)

## Labels don't travel with git

`labels_gpt/` is under `.gitignore` (`Labeling/kaggle_dataset/train/`), so the 16
labels are **not in the commit**. To have them on the desktop, pick one:

1. **Force-add** them in a follow-up commit — `git add -f
   Labeling/kaggle_dataset/train/labels_gpt/` — deliberate exception to the
   ignore, but they're small and hand-made (the only copy). Recommended if the
   desktop is the primary machine from here.
2. Copy the dir across manually (rsync / USB / cloud). Matches the repo's
   existing "local only, back up by copy" pattern for `labels_gpt`.
3. Redo them — it's ~15 min for 16 frames once the dataset is down.

Whichever: also copy or don't-lose them before running `download_kaggle.py`
again — it `rmtree`s the whole `kaggle_dataset/` tree and only backs up
`labels_gpt/` when you pass `--force`.

## Desktop pickup

1. `git pull` on `ModelTraining` (after this commit is pushed).
2. Kaggle auth: put the API token at `%USERPROFILE%\.kaggle\access_token`
   (or `kaggle.json` with `{"username","key"}`; kagglehub reads either).
3. `python scripts\labeling\download_kaggle.py` — pulls ~1 GB to the kagglehub
   cache + copies splits into `Labeling\kaggle_dataset\`.
4. Bring the labels over (see above) if you want to keep this session's 13 good
   frames.
5. `python scripts\labeling\manual_label.py` — resumes at the first unlabeled
   deduped frame.
6. Confirm `.venv` / `myenv` has `opencv-python` (full, **not** headless — needs
   the GUI build), `ultralytics`, `torch`.

## Gotchas

1. **`download_kaggle.py` is destructive.** `rmtree(kaggle_dataset/)` — wipes
   `labels_gpt/` (refuses without `--force`, backs up to
   `Labeling/kaggle_dataset_labels_gpt_backup`) and also deleted 191 tracked
   `preview/*.jpg` from the working tree this session.
2. **`labels_gpt/` and everything under `kaggle_dataset/train/` is gitignored.**
   Nothing you label lands in git unless force-added.
3. **MPS `NMS time limit exceeded`** on yolov8m — degrades/zeros detections.
   Irrelevant while `--boxes auto` is used on the Kaggle set; matters if you
   `--boxes detect` on the Mac.
4. **v4's `best.pt` still reports class 1 as `CAR`**, not `Vehicle` (last
   hand-off gotcha #2). The labeler doesn't touch v4, but keep it in mind if you
   wire v4 in as a detector.
5. **Taxonomy is still 7-class incl. generic `Vehicle`.** The labeler writes it;
   `diagnose_labels.py` expects it. If you later collapse SUV/Standard Car into
   Vehicle (last hand-off's open question), the already-made labels need a remap.
6. **`requirements.txt` pins CUDA torch + older versions** — don't `pip install
   -r` against the Mac `.venv`. The desktop (CUDA) is what that file targets.

## Next steps

1. Redo the 7 undercounted frames (script above).
2. Decide how the labels travel (force-add vs copy).
3. Label a useful batch — even 100–200 frames of real subtypes is more than the
   pool has ever had that wasn't GPT-contradicted.
4. `extract_good_kaggle.py` → **`diagnose_labels.py`** (healthy ref: recall 0.804
   / precision 0.813 / agreement 0.934). Do not train until agreement is well off
   the 0.19 floor the GPT batches hit.
5. Only then retrain and compare against v4 (still champion, 0.430).
6. Housekeeping: resolve the `preview/` deletions; drop `git stash@{0}`; delete
   root `yolov8m.pt`.

## References

- `scripts/labeling/manual_label.py` — module docstring has the full key list
- `docs/luke-hand-offs/2026-08-06-vision-pipeline-reorg.md` — the regression
  diagnosis, class schema, the label gate, the augmentation problem
- `scripts/README.md` — "Hand-label the vehicle types" flow
- `scripts/evaluation/diagnose_labels.py` — the gate; healthy numbers in its docstring
- Dataset: `ryankraus/traffic-camera-object-detection` (Kaggle, Roboflow export)
