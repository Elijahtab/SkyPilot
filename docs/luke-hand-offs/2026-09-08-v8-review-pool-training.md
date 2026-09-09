# Hand-off: v8 — training on the hand-reviewed >=48px pool (2026-09-08)

## Goal

Build a human-in-the-loop labeling tool, use it to hand-label the legible subset
of the clean Kaggle frames, then merge that batch into v4 and measure what
happens. Luke labeled 1,201 crops over roughly two hours. The merge (run `v8`)
**regressed** — 0.3959 vs v4's 0.4322 — for the reason predicted before it ran:
the new labels use a different labeling policy from the base val/test splits.
The run is worth keeping because it separates two hypotheses that four previous
runs had conflated.

## State

- Repo `S:\GitHub\SkyPilot`, branch **`ModelTraining`**, HEAD **`48a3ca5`**
  (unchanged this session apart from this hand-off's own commit).
- Remote `origin` = `github.com/Elijahtab/SkyPilot.git`. **Nothing pushed.**
  Prior sessions' commits are also still local.
- `v4` remains champion at **0.4322** val mAP50-95. `promote_best_model.py` picks
  by recorded mAP, so v8 will not displace it.

### Uncommitted — deliberate, and this hand-off is the only thing committed

Per the wrap-up instruction only the hand-off was staged. **Everything below is
real work sitting dirty in the tree** and should be committed deliberately next
session:

| Path | What it is |
|---|---|
| `scripts/labeling/build_crops.py` | builds the size-gated crop corpus + manifest |
| `scripts/labeling/label_app.py` / `.html` | the review app (stdlib http.server + sqlite3) |
| `scripts/labeling/export_labels.py` | decisions -> YOLO labels + bias report |
| `scripts/tools/find_duplicate_boxes.py` | umbrella/subtype double-label diagnostic |
| `scripts/training/train_vehicle_v8.py` | the run described here |
| `scripts/evaluation/diagnose_labels.py` | **bug fix** (OOM, see Gotchas) |
| `scripts/evaluation/compare_models.py` | **two bug fixes** (see Gotchas) |
| `scripts/_paths.py` | adds `REVIEW_IMG` / `REVIEW_LBL` |
| `configs/vehicle_7class.yaml`, `scripts/README.md` | umbrella semantics + v8 in run history |
| `docs/v4-integration-plan.md` | the plan + the v8 result |

Not tracked by design: `images/kaggle_review`, `labels/kaggle_review` (under the
existing `/images/` `/labels/` ignores), `Labeling/review/crops/` (regenerable),
`Vehicle_type_detection/runs/.../v8/` (52MB weights x2).

**`Labeling/review/decisions.sqlite` is two hours of irreplaceable human work.**
It is gitignored, so a dated copy exists at
`Labeling/review/decisions.backup-2026-09-08.sqlite` and `.gitignore` was given a
negation rule so that backup **is** trackable. Commit it.

### Verified

- v8 trained to completion: 37 epochs, best **epoch 22**, val mAP50-95 **0.3959**.
- v4 vs v8 measured on the same base val split, twice (mAP via
  `compare_models.py`, class-agnostic via `diagnose_labels.py --control --run`).
- Label gate run on the exported pool before training.
- The 5->7 class remap of Luke's first 53 decisions verified against a backup:
  **zero name mismatches** across all 53 rows.
- All new scripts compile; the app's endpoints smoke-tested including path
  traversal (404) and decide/undo round-trip.

### NOT verified

- Nothing has been re-trained since v8. The classifier route (Path A) is
  untouched — no code exists for it yet.
- The duplicate-box defect is measured but **not fixed**; `--fix-dir` has never
  been run.
- The 112 undecided crops were exported as generic `Vehicle`, never reviewed.

## Key findings

### v8 result: regressed, but informatively

| metric, base val | v4 | v8 | delta |
|---|---|---|---|
| mAP50-95 | **0.4322** | 0.3959 | **-0.036** |
| mAP50 | 0.6195 | 0.5785 | -0.041 |
| class-agnostic recall | **0.802** | 0.770 | **-0.032** |
| class-agnostic precision | 0.806 | **0.821** | +0.015 |
| class agreement | 0.934 | 0.931 | -0.003 |

Per-class AP50 fell for **every class**: Bus -0.031, Vehicle -0.009, Motorcycle
-0.059, SUV -0.041, Standard Car -0.107, Truck -0.019, Van -0.021.

Updated run history:

| run | init | added data | mAP50-95 | best epoch |
|---|---|---|---|---|
| **v4** | v1 best | none | **0.430** | 50 / 55 |
| v5 | v4 best | GPT 1x | 0.407 | 5 / 25 |
| **v8** | v4 best | **human review 1x** | **0.396** | **22 / 37** |
| v7 | v4 best | GPT 2x | 0.363 | 1 / 21 |
| v6 | yolov8m | GPT 1x | 0.291 | 39 / 41 |

**The epoch number is the finding.** v5 and v7 peaked at epoch 5 and 1 — training
made validation worse immediately, because the GPT pool's *geometry* was broken.
v8 peaked at **22**. The boxes are genuinely good, and the decline is the
taxonomy conflict accumulating gradually. Two failure modes that looked identical
in the mAP column are now separated.

**The detection-side hope did not survive.** The reason to run v8 was that 521
real traffic-cam frames with good boxes might improve *finding* vehicles even
while *naming* them conflicts. Class-agnostic recall fell 0.802 -> 0.770.
Precision rose only because v8 emits fewer boxes (4,787 vs 5,081) — more
conservative, not more accurate. There is no win hiding under the mAP drop.

### Why it was always going to regress

Measured by `diagnose_labels.py` on `images/kaggle_review` before training:

```
class-agnostic recall    0.723   (GPT pool was 0.371)   <- boxes GOOD
class-agnostic precision 0.783   (GPT pool was 0.331)
of boxes a human called SUV,          92.9% are 'Vehicle' to v4
of boxes a human called Standard Car, 99.2% are 'Vehicle' to v4
```

At the same size range (>=48px), base **val** labels 1,848/2,375 boxes (78%)
`Vehicle`; this batch labels 1/1,201 (0.08%). A correct SUV therefore scores as
a false positive on `SUV` **and** a false negative on `Vehicle`.

### `Vehicle` is the UMBRELLA class — corrected this session

Luke corrected a wrong assumption that had been baked into the docs: `Vehicle` is
not a "generic bucket" sibling of the body styles — **it is the umbrella covering
all of them** (Bus, Truck, Motorcycle, Van, Standard Car, SUV). A 5-class schema
that collapsed SUV/Standard Car into `Vehicle` was built and then **deleted**;
`configs/vehicle_5class.yaml` no longer exists. Do not re-propose that collapse.

### The umbrella relation is not expressible in YOLO — and the data proves it

Because `Vehicle` is an umbrella, some frames label each object **twice**, once
`Vehicle` and once its subtype, as two near-identical boxes. Frame
`0003_jpg.rf.e5ed04ae...` is 24 boxes that are really 12 objects.

`scripts/tools/find_duplicate_boxes.py` (IoU>=0.85):

| split | frames affected | boxes in a pair |
|---|---|---|
| train | 37 of 948 | 362 (1.8%) |
| val | 6 of 215 | 92 (1.8%) |
| test | **0** of 108 | 0 |

1.8% overall is misleading. Per class: **31% of all SUV labels (115/370)** and
**20% of Standard Car (81/403)** are half of a contradictory pair. That is a
strong candidate explanation for v4 predicting `Vehicle` on GT `SUV` 41.5% of the
time, and it means **train and test disagree about what `Vehicle` means.**

### The labeling pass itself

1,201 of 1,313 crops at the >=48px gate, 521 frames, ~2 hours.
Result: SUV 508, Standard Car 400, Truck 222, Van 45, Bus 25, Vehicle 1, Moto 0.

- **Agreement with the v4 prior: 0.028** — no rubber-stamping whatsoever.
  871 of 1,201 decisions convert `Vehicle` into a body style.
- 178 crops were decided more than once and 170 changed, **79 of them SUV <->
  Standard Car swaps**. These are in-session corrections, not blind re-tests, so
  it is *not* a self-consistency score — but the churn sits exactly on the
  boundary the research predicted would be unrecoverable at traffic-cam scale.
- 5,293 of the exported pool's 6,494 boxes are sub-gate and were auto-labeled
  `Vehicle` without review. **The pool therefore also teaches "large car = SUV,
  small car = Vehicle" — size, not appearance.** If v8's regression has a second
  cause, this is it, and this run cannot separate the two.

### Model structure, for reference

v4/v8 are **YOLOv8m**: 25,860,373 params, 295 modules, `depth 0.67 / width 0.75`.
Backbone layers 0-9 (Conv/C2f + SPPF), PANet neck 10-21, `Detect` head at layer
22 reading layers 15/18/21 at strides 8/16/32, `reg_max 16` (DFL). Stride 8 is
why sub-16px boxes are effectively invisible at imgsz 640.

## Gotchas

1. **`v5`, `v6`, `v7` are taken.** The new run is `v8`. Do not name anything v5.
2. **`diagnose_labels.py` OOM'd on 521 images — FIXED.** Ultralytics sets the
   predictor batch size to `len(source)` when `source` is a **list**, so it tried
   a 9.54 GiB allocation on the 8GB 2070 SUPER. It silently worked at 207 images
   and died at 521. `stream=True` does **not** cap it and `batch=1` is **silently
   ignored for list sources**. Pass the directory — that gives `bs=1`.
3. **`compare_models.py` had two bugs — both FIXED.**
   (a) per-class AP was keyed by `model.names`, and v4's `best.pt` still says
   `CAR`, so v4's largest class printed as `--`. Now indexes `CLASS_NAMES`.
   (b) `--split val` left `val` behind as a positional and it was evaluated as a
   run tag.
4. **scipy in `myenv` is BROKEN**: `cannot import name '_ccallback_c'`.
   Training completes and saves weights, then ultralytics calls scipy to draw
   `results.png` and the process **exits code 1 despite succeeding**. Any
   automation checking exit status will report a false failure. Fix:
   `.\myenv\Scripts\python.exe -m pip install --force-reinstall scipy`.
5. **A schema change under a running app silently corrupts labels.** The review
   app loads the manifest **once at startup**. Rebuilding the manifest from
   5-class to 7-class mid-session left the browser writing 5-class IDs into a
   7-class store, where `3` flipped from Truck to SUV. Caught and remapped, but
   **restart the app after any `build_crops.py` run.**
6. `.pt` files carry their own class names. v4's `best.pt` says `CAR`;
   `best_vehicle.pt` says `Vehicle`; weights identical.
7. Don't run concurrent GPU jobs on the 2070 SUPER (8GB).
8. The active `python` on PATH is another project's venv — always
   `.\myenv\Scripts\python.exe`.
9. `yolo11n.pt` (5.35MB) was auto-downloaded to the repo root by ultralytics' AMP
   check. Untracked junk; safe to delete.
10. `scripts/check_labels.py`, `scripts/label_distribution.py` and
    `Labeling/auto_label_yolo.py` are **0-byte placeholders**, not work in
    progress. The last one collides by name with the retired
    `scripts/archive/auto_label_yolo.py`.

## Next steps

1. **Commit the session's work.** It is all still dirty — see the State table.
   Include `Labeling/review/decisions.backup-2026-09-08.sqlite`; the `.gitignore`
   negation rule already permits it.
2. **Fix scipy** so training runs stop exiting 1:
   `.\myenv\Scripts\python.exe -m pip install --force-reinstall scipy`
3. **Build the subtype classifier (Path A)** — the recommendation v8 did not
   displace. v4 stays the detector untouched; a small classifier types the crops,
   so `Vehicle` and `SUV` never compete for one slot. Training set combining this
   batch with the base dataset's own >=48px subtype crops: Truck 951, SUV 818,
   Bus 789, Standard Car 771, Motorcycle 234, Van 186. Hold out 20% of *this
   batch* (not the base close-ups) as validation. **Gate: if SUV vs Standard Car
   lands near chance, merge them into one `Car` class** — that is a finding, not
   a failure. Details in `docs/v4-integration-plan.md` §3.
4. **Decide the duplicate-box question.** Run
   `find_duplicate_boxes.py --fix-dir <scratch>` and re-measure v4 against the
   deduped labels. Until then, any conclusion about SUV or Standard Car — the
   classifier's included — is unreliable, because the defect is concentrated in
   exactly those classes.
5. **Optional: finish the last 112 crops.** They exported as `Vehicle`.
   `.\myenv\Scripts\python.exe scripts\labeling\label_app.py` resumes at the
   first undecided page.
6. Do **not** retry the merge with different hyper-parameters. The conflict is in
   the labels; oversampling ratio and LR were not the binding constraint. If a
   single-stage model is still wanted, `docs/v4-integration-plan.md` §4 prices it:
   re-reviewing base val+test at the same gate is **3,258 crops**, ~2.5x the pass
   just completed.
7. Decide whether to push `ModelTraining` to `origin` — nothing has ever been
   pushed.

## References

- `docs/v4-integration-plan.md` — the plan, the §2 conflict analysis, §5b the v8 result
- `docs/labeling-tool-plan.md` — what to label and why most boxes stay `Vehicle`
- `docs/labeling-web-app-research.md` — the size distribution behind the 48px gate
- `docs/luke-hand-offs/2026-08-06-vision-pipeline-reorg.md` — the v5/v6/v7 diagnosis
- `scripts/README.md` — pipeline, run history (now incl. v8), class schema
- `configs/vehicle_7class.yaml` — schema + the umbrella semantics and its caveat
- `Vehicle_type_detection/runs/Vehicle_type_detection_v8/` — weights, results.csv,
  confusion matrices (`results.png` missing: the scipy break)
- `Labeling/review/manifest.json` — exactly which 1,313 crops were in scope
