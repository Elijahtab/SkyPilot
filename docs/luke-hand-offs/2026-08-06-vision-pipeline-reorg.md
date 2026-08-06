# Hand-off: Vision pipeline diagnosis + reorganization (2026-08-06)

## Goal

Research pass over the labeling / prediction / Kaggle / training code, then act on
what it found. The driving question: **why has every training run that added GPT
auto-labeled data regressed against `v4`?** The answer turned out to be two
independent data defects, not hyper-parameters. Fixed both, renamed the generic
class `CAR` → `Vehicle`, and restructured all vision code into a single tree.

## State

- Repo: `S:\GitHub\SkyPilot` — branch **`ModelTraining`**, HEAD **`8ec058a`**
- Session start was `c0e56ff`. **One commit made this session:** `8ec058a`
  "Reorganize vision pipeline, rename CAR->Vehicle, drop poisoned label batch"
- Working tree **clean**. Remote `origin` = `github.com/Elijahtab/SkyPilot.git`;
  **commit is local, NOT pushed** (not requested).
- `Drone+OpenAI/` untouched, as instructed.

### Deliberately uncommitted / untracked

| Path | Why |
|---|---|
| `Vehicle_type_detection/runs/*/weights/*_vehicle.pt` | 596MB of retagged copies, regenerable in seconds via `scripts/tools/retag_checkpoint.py --all` |
| `Labeling/quarantine_branchB_yolov8n/` | 17MB of dropped label batch, kept on disk for recovery |
| `Labeling/kaggle_dataset/{train,valid,test}/` | 199MB dataset media; matches the intent of `c0e56ff` "Removed old dataset images" |
| `preds/`, `Labeling/preview/` | regenerable output; untracked this session (216 + 91 files) |

### Verified

- **v4 control reproduces its recorded run**: val mAP50-95 `0.4396` vs `0.4303`
  logged in `results.csv`. The measurement harness is sound.
- All diagnostic numbers below, measured directly.
- Retag correctness: names change, **475 weight tensors bit-identical**.
- All scripts compile (`compileall`); all four training scripts' `sanity_check()`
  resolve paths and report correct counts (967 / 216 / 108 base, 207 pool).
- Pool is exactly 207 images / 207 labels == branch A stem set.

### Pending — NOT verified

1. **No model has been retrained since the cleanup.** `v4` is still champion at
   0.430. The fixes are not yet proven to *improve* anything.
2. **A post-cleanup re-run of `diagnose_labels.py` on the 207-only pool was still
   running when this hand-off was written.** Expect precision to improve (the
   under-annotated batch is gone) and agreement to stay ~0.19–0.21 (the taxonomy
   fix only affects *future* labeling — see Gotcha 1).
3. The 207 labels in the pool **still use the old vocabulary**.

## Key findings

### The regression is a data problem, and it is two problems

Run history (best epoch, val):

| run | init | GPT data | mAP50-95 | best epoch |
|---|---|---|---|---|
| **v4** | v1 best | none | **0.430** | 50 / 55 |
| v5 | v4 best | 1x | 0.407 | **5** / 25 |
| v7 | v4 best | 2x | 0.363 | **1** / 21 |
| v6 | yolov8m | 1x | 0.291 | 39 / 41 |
| v6_oversampled | yolov8m | 5x | 0.215 | 37 / 40 |

v5 and v7 peaking at epoch 5 and **1** is the tell: starting from v4's weights,
every subsequent epoch on the mixed pool made validation worse. Monotonic decline
with more GPT data; worst at 5x oversampling.

### The diagnostic that separates the causes

`scripts/evaluation/diagnose_labels.py` splits class-agnostic detection from class
agreement over *matched* boxes, so detection failures cannot contaminate the
taxonomy number. v4 as reference, conf>=0.25, IoU>=0.5:

| | base val | GPT pool (306) | **A**: kaggle GT boxes (207) | **B**: yolov8n boxes (99) |
|---|---|---|---|---|
| class-agnostic recall | 0.804 | 0.371 | **0.219** | 0.910 |
| class-agnostic precision | 0.813 | 0.331 | 0.505 | **0.255** |
| class agreement (matched only) | **0.934** | 0.189 | 0.212 | 0.169 |

The two branches fail in **opposite** directions:

- **Branch B (99 imgs) — under-annotated.** v4 finds 91% of the labeled boxes then
  finds 3.6x more real vehicles that aren't labeled. `1017_png.rf.103b4230…jpg`
  has **1** GT box; v4 correctly detects **16**. ~72% of vehicles unlabeled →
  trained as background → suppresses correct detections. **Dropped.**
- **Branch A (207 imgs) — v4 can't see the imagery.** Labels are complete human
  Kaggle boxes; v4 only reaches 0.219 recall on them.

### Why branch A's imagery defeats v4

Not a small-object problem — that hypothesis is **refuted**. Kaggle objects are
*larger* (median 54px vs 28px at 640 input) and *sparser* (10/img vs 24/img), and
matched-vs-missed box scale is identical (0.088 vs 0.082) where the base val split
shows the normal small-objects-are-harder signature (0.049 vs 0.022).

The real cause, from rendering the overlays: **every image in the Kaggle train
split is a 416x416 Roboflow export, and many are rotation/mosaic augmentations** —
four rotated tiles packed into one frame. Measured:

- 100% of the split is 416x416
- **2,015 unique source frames → 5,248 images (62% augmentation duplicates)**
- splits are clean: **zero** source-frame leakage between train/valid/test

So `auto_label_kaggle.py` has been sending GPT-4o crops of **sideways vehicles**.

### The taxonomy collision

On the same physical object, where GPT says X, v4 says the generic class:

| GPT label | v4 says generic (A) | (B) |
|---|---|---|
| SUV | 89.9% | 98.4% |
| Standard Car | 89.4% | 98.5% |
| Van | 71.4% | 62.8% |

Cause: the old prompt said *"CAR (generic, but you MUST try to classify into a
subclass below first!)"* while the base dataset resolves 78% of boxes to generic.
Renamed `CAR` → **`Vehicle`** and reversed the prompt bias.

**The base labels are independently weak here too**: in the control, GT `SUV` →
`CAR` 41.5%, GT `Standard Car` → `CAR` 70%. `Standard Car` scores AP50 **0.240**,
recall **0.111** — v4's worst class, on 10 matched boxes.

## Gotchas

1. **The 207 labels in the pool predate the prompt fix.** Renaming the class did
   not relabel anything. Training on the pool today still hits the taxonomy
   conflict. Re-run the labeler before trusting it.
2. **Ultralytics reads class names from the `.pt`, not the data yaml.** Editing
   `configs/` does not change what an existing model reports. `*_vehicle.pt`
   copies exist alongside every checkpoint; originals still say `CAR`.
   `eval_model.py` warns on mismatch.
3. **`download_kaggle.py` used to `rmtree` the GPT labels** (each one an API call)
   with no warning. Now refuses without `--force` and auto-backs-up. If you see
   labels_gpt empty, check `Labeling/kaggle_dataset_labels_gpt_backup`.
4. **Stale `.cache` files silently reuse old box counts.** Delete
   `labels/kaggle_gpt.cache` after changing the pool. `extract_good_kaggle.py`
   now does this automatically.
5. **The `test` split has 0 `Standard Car` and 3 `SUV` boxes**, whose per-class AP
   is noise averaged into the headline mAP. `compare_models.py` now prints GT
   support per class — read it before trusting a small delta.
6. **Don't run multiple GPU jobs concurrently on the 2070 SUPER (8GB)** — it OOMs
   and the failure surfaces as a confusing `GET was unable to find an engine`.
7. **The active `python` on PATH is another project's venv** (`FractureFinder`).
   Always use `.\myenv\Scripts\python.exe`.
8. `find_best_mAP50-95.py` (now `promote_best_model.py`) picked by mtime, which
   would have promoted **v7 (0.363) over v4 (0.430)**. Now picks by recorded mAP.

## Next steps

1. **Confirm the post-cleanup diagnostic** (was mid-run at hand-off):
   ```powershell
   .\myenv\Scripts\python.exe scripts\evaluation\diagnose_labels.py
   .\myenv\Scripts\python.exe scripts\evaluation\diagnose_labels.py --control
   ```
2. **Decide the taxonomy policy before any retraining.** Either (a) collapse
   `SUV`/`Standard Car` into `Vehicle` in the pool labels — cheap, and defensible
   given `Standard Car` has 18 val / 0 test boxes — or (b) relabel val/test to the
   fine-grained scheme. Option (a) gives a 5-class schema.
3. **Re-label branch A against un-augmented source frames.** Dedupe to the 2,015
   unique frames first (`_jpg.rf.` prefix split) — that alone is a ~62% API cost
   saving, and it removes the sideways-vehicle problem.
4. **Reconstruct `train_vehicle_v4.py` while the weights still exist.** No script
   reproduces the best model or the init point for v5/v7. Recorded args:
   `lr0 0.01, optimizer auto, mosaic 1.0, freeze null, epochs 50`, initialised
   from a `runs/Vehicle_type_detection/weights/best.pt` that is gone.
5. **Only then retrain**, and gate on `diagnose_labels.py` before merging.
6. Decide whether to push `8ec058a` to `origin/ModelTraining` — currently local.
7. Optional: branch B's 99 images are recoverable if re-proposed with v4 as the
   detector (0.910 recall on that imagery) instead of yolov8n.

## References

- `scripts/README.md` — pipeline, run history, class schema, the label gate
- `Labeling/quarantine_branchB_yolov8n/README.md` — evidence for the drop, recovery
- `configs/vehicle_7class.yaml` — source of truth for class names
- `scripts/_paths.py` — all repo paths; replaces ~20 hardcoded `S:\GitHub\SkyPilot`
- `scripts/evaluation/diagnose_labels.py` — the gate; healthy reference numbers in
  its docstring
- Commit `8ec058a` — full change description
- Dataset: `ryankraus/traffic-camera-object-detection` (Kaggle, Roboflow export)
