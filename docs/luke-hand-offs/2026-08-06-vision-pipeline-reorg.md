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
- **Post-cleanup `diagnose_labels.py` on the 207-only pool completed and matches
  the pre-cleanup branch-A measurement exactly** — recall `0.219`, precision
  `0.505`, agreement `0.212`. Two things confirmed at once: the reorganized
  script is correct, and the pool is cleanly branch A with no branch B residue.

  Versus the old mixed 306-image pool: precision **0.331 → 0.505** (the
  under-annotated batch is gone, as predicted) and agreement **0.189 → 0.212**
  (unchanged, as predicted). Recall **fell, 0.371 → 0.219** — not a regression,
  just arithmetic: branch B had 0.910 recall and was inflating the average. The
  remaining 0.219 is the branch A augmented-imagery problem in isolation.

### Pending — NOT verified

1. **No model has been retrained since the cleanup.** `v4` is still champion at
   0.430. The fixes are not yet proven to *improve* anything.
2. The 207 labels in the pool **still use the old vocabulary**.

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

### Resolution sweep (added after the reorg)

`scripts/evaluation/sweep_imgsz.py`, v4 on val. **Inference-only** — v4 was
trained at 640, so this is evidence about retraining, not a substitute for it.

| imgsz | class-agnostic recall | mAP50-95 | precision | ms/img |
|---|---|---|---|---|
| 640 | 0.811 | **0.4396** | 0.727 | 13.8 |
| 800 | 0.844 | 0.4227 | 0.629 | 21.4 |
| **960** | **0.855** | 0.4154 | 0.666 | 32.8 |
| 1280 | 0.841 | 0.3366 | 0.565 | 50.4 |

**Recall rises, mAP falls, and both are real.** At 960 v4 finds 85.5% of vehicles
vs 81.1% at 640 — a **23% reduction in vehicles missed entirely**. mAP falls
because precision drops 0.727 → 0.666: the classic scale-prior mismatch of
running a 640-trained model at 1.5x scale. Retraining at 960 is what removes it.

Mean AP50 split by class support (threshold 200 GT boxes) — the headline mAP is
actively misleading here:

| imgsz | all 7 | Bus/Vehicle/Motorcycle/Truck | SUV/Standard Car/Van |
|---|---|---|---|
| 640 | 0.6306 | 0.6926 | 0.5480 |
| **800** | 0.6165 | **0.7204 (+0.028)** | 0.4781 (-0.070) |
| 960 | 0.6113 | 0.7149 (+0.022) | 0.4730 |
| 1280 | 0.4944 | 0.6834 | 0.2424 |

**SUV + Standard Car + Van hold 85 boxes out of 5,105 (1.7%) yet invert the
headline**, because mAP weights all classes equally. Until the taxonomy is fixed,
every experiment on this dataset will be distorted the same way.

Box sizes explain the ceiling: 21.6% of val boxes are <16px at 640 (below YOLO's
8px stride, effectively invisible). 960 lifts 11.8% of all boxes over 16px, 1280
lifts 17.4%, leaving only 4.2% still tiny.

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
9. **Ultralytics only populates `confusion_matrix` when `plots=True`.** With
   `plots=False` the matrix stays all zeros and any metric derived from it
   (class-agnostic recall) silently reads 0.000 rather than erroring. Bit me
   once in `sweep_imgsz.py`; the comment there now says so.
10. **Never trust the headline mAP on this dataset without checking per-class
   support.** Three classes with 85 boxes between them reversed the entire
   conclusion of the resolution sweep.

## Next steps

1. **Decide the taxonomy policy before any retraining.** Either (a) collapse
   `SUV`/`Standard Car` into `Vehicle` in the pool labels — cheap, and defensible
   given `Standard Car` has 18 val / 0 test boxes — or (b) relabel val/test to the
   fine-grained scheme. Option (a) gives a 5-class schema.
2. **Retrain at imgsz 960.** Strongest measured lever after the taxonomy fix:
   inference-only already gives +4.4pts class-agnostic recall (23% fewer misses),
   and the accompanying precision drop is scale-prior mismatch that retraining
   removes. Cost ~2.4x train time; 32.8 ms/img at inference (~30fps, still fine
   for the drone tracker). Do this AFTER step 1 or the thin classes will mask it.
3. **Re-label branch A against un-augmented source frames.** Dedupe to the 2,015
   unique frames first (`_jpg.rf.` prefix split) — that alone is a ~62% API cost
   saving, and it removes the sideways-vehicle problem. This is also the only
   thing that will move the 0.219 recall on that pool.
4. **Reconstruct `train_vehicle_v4.py` while the weights still exist.** No script
   reproduces the best model or the init point for v5/v7. Recorded args:
   `lr0 0.01, optimizer auto, mosaic 1.0, freeze null, epochs 50`, initialised
   from a `runs/Vehicle_type_detection/weights/best.pt` that is gone.
5. **Gate any new auto-labeled batch on `diagnose_labels.py`** before merging it
   into the pool.
6. Decide whether to push the local commits to `origin/ModelTraining` — nothing
   has been pushed.
7. Optional: branch B's 99 images are recoverable if re-proposed with v4 as the
   detector (0.910 recall on that imagery) instead of yolov8n.
8. Minor script refinement: `diagnose_labels.py` prints
   "UNDER-ANNOTATED: model finds real objects the labels omit" whenever
   precision < 0.6. On branch A that explanation is wrong — precision is 0.505
   because v4 emits only 1,063 boxes on the augmented imagery and half miss,
   not because labels are absent. The number is right, the canned diagnosis
   isn't; consider gating that message on `recall > 0.6` as well.

## References

- `scripts/README.md` — pipeline, run history, class schema, the label gate
- `Labeling/quarantine_branchB_yolov8n/README.md` — evidence for the drop, recovery
- `configs/vehicle_7class.yaml` — source of truth for class names
- `scripts/_paths.py` — all repo paths; replaces ~20 hardcoded `S:\GitHub\SkyPilot`
- `scripts/evaluation/diagnose_labels.py` — the gate; healthy reference numbers in
  its docstring
- Commit `8ec058a` — full change description
- Dataset: `ryankraus/traffic-camera-object-detection` (Kaggle, Roboflow export)
