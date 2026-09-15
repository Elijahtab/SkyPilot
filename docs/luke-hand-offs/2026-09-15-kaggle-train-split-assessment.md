# Hand-off: should the Kaggle train split go into v8? (2026-09-15)

## Goal

Luke proposed adding about 2,000 images from the Kaggle traffic-camera dataset
(`ryankraus/traffic-camera-object-detection`) to the v8 training set to raise
accuracy, and then hand-typing those frames in the review app. This session
measured whether either idea holds up. **No code was written and nothing was
trained.** The discussion was paused at Luke's request before a direction was
chosen.

## State

- Branch **`ModelTraining`**, HEAD before this hand-off **`6169f5b`**, remote
  `origin` = `github.com/Elijahtab/SkyPilot.git`. This hand-off is the only
  commit made this session.
- ⚠ **The whole 2026-09-14 session is still uncommitted and has no hand-off of
  its own.** Its only record is the **untracked** `docs/two-stage-pipeline.md`.
  At risk: `scripts/_crops.py`, `scripts/_color.py`,
  `scripts/training/build_type_crops.py`, `scripts/training/train_type_classifier.py`,
  `scripts/evaluation/eval_type_classifier.py`, `scripts/evaluation/explore_color.py`,
  `scripts/pipeline/{two_stage,index_frames,search_vehicles}.py`, plus edits to
  `scripts/_paths.py` (`TYPE_CLS`, `TYPE_RUNS`) and `scripts/README.md`. This is
  the same state that a `git reset --hard` destroyed on 2026-09-10.
  **Commit it before anything else.** Luke was asked twice and has not answered
  yet, so it was not swept into this commit.
- Still untracked on purpose: `Vehicle_type_detection/runs/Vehicle_type_detection_v8/`,
  `Vehicle_type_detection/runs_cls/type_cls_v1/`, `YoloTraining/`,
  `yolo11n.pt`, `weights/pretrained/yolo11s-cls.pt`, and the three 0-byte
  placeholders (`scripts/check_labels.py`, `scripts/label_distribution.py`,
  `Labeling/auto_label_yolo.py`).

### Recap of 2026-09-14 (full record: `docs/two-stage-pipeline.md`)

Path A was built: v4 detects (class output ignored) → `type_cls_v1`
(yolo11s-cls) types boxes of 48px or more → colour rules → SQLite index + text
search.

- Classifier top-1 accuracy is **0.807** on the Kaggle holdout (20 of 93
  intersections) and 0.806 on base val.
- SUV vs Standard Car gate **passed**: 0.865 against a 0.560 baseline.
- Colour rules score about 0.62, measured against Claude's eye calls, not human
  labels.
- The biggest defect: v4 detects traffic signals and the classifier types them
  (every "motorcycle" search hit was wrong).
- scipy was fixed (cp311 wheels had been installed in the Python 3.12 venv).

### Verified this session

- `diagnose_labels.py` detection numbers in the table below (conf 0.25,
  IoU 0.5), for v4 and v8.
- The Kaggle dataset's structure (counts, copies, camera overlap, mosaics).

### NOT verified

- Anything to do with training on the Kaggle train split: no run was made.
- The 12,935-crop estimate is an upper bound. Zoom inflates box size, and the
  number of repeated vehicles was not measured.
- Whether un-augmented originals exist on Roboflow Universe: not checked.

## Key findings

### 1. This is not a new dataset

It is the same Kaggle set as `Labeling/kaggle_dataset/` (fetched by
`scripts/labeling/download_kaggle.py`). It is the source of the GPT pool
(v5–v7), of the 1,201 hand-reviewed crops, and of v8.

### 2. Correction: v8 improved detection on traffic-cam frames

The 09-08 hand-off said v8 had "no win hiding under the mAP drop". That was
measured on base val only. Class-agnostic results:

| model | base val recall | base val precision | Kaggle valid recall | Kaggle valid precision |
|---|---|---|---|---|
| v4 | **0.802** | 0.806 | 0.689 | 0.751 |
| v8 | 0.770 | 0.821 | **0.795** | **0.804** |

v8 is a trade: -0.032 recall on base val, **+0.106** on Kaggle valid, from only
521 Kaggle frames.

Caveat: **all 258 Kaggle-valid cameras also appear in Kaggle train** (295
cameras), so the Kaggle-valid figures measure performance on cameras the model
has seen. They are optimistic.

Reproduce:

```
.\myenv\Scripts\python.exe scripts\evaluation\diagnose_labels.py Labeling\kaggle_dataset\valid\images Labeling\kaggle_dataset\valid\labels --run Vehicle_type_detection_v8
```

(Drop `--run` for v4. Ignore section (B): Kaggle class 0 is `car`, which lines
up with index 0 `Bus` in the 7-class schema.)

### 3. The Kaggle labels have one class

`data.yaml` has `nc: 1`, `names: ['car']`, with 74,586 train boxes. The dataset
carries no type information. Adding it to the 7-class detector means labeling
every box `Vehicle`: about 24k generic boxes, more than base train's roughly
20k boxes. That reproduces the taxonomy conflict that regressed v5–v8
(`docs/v4-integration-plan.md` §2).

### 4. The train split has no clean frames

- 5,248 images come from **2,015 source frames**. Copies per frame: 1 copy for
  161 frames, 2 for 475, 3 for 1,379.
- **Every sampled training image is a 2×2 mosaic** of rotated, sheared or zoomed
  tiles, often mixing cameras (24 of 24 in a contact sheet). The filename
  (`<camera>-<n>_jpg.rf.<hash>`) names only one of the four sources.
- The kagglehub download contains only `traffic/{train,valid,test}`. There is no
  un-augmented folder and no Roboflow README.
- Box sizes (long side in px at 416): Kaggle valid median **16**, 12% are 48px
  or more. Base val median 43, 47% are 48px or more.
- Taking the best copy of each train frame gives **12,935** boxes of 48px or
  more (31% of boxes). That share is inflated by zoom, so many look legible but
  carry about 24px of real detail. Repeated vehicles probably cut the distinct
  count to about a third. At Luke's 09-08 pace (1,201 crops in about 2 hours),
  labeling all of them is roughly 20 hours of mostly repeats.
- The review app has already covered all 1,313 crops of 48px or more in the
  clean valid+test frames (1,201 decided).

### 5. Where type labels can go

Type labels only help the **stage-2 classifier**. They cannot help any detector
that is scored on base val, which labels 78% of its boxes of 48px or more as
`Vehicle`. The classifier's thin spots:

- Van: 174 training crops
- Motorcycle: 231, with 0 in its validation split
- Bus: 5 in its validation split
- pickups being called SUV, its largest error

### Options on the table (none chosen)

**A. One-class detector (no labeling).** Collapse base train to a single
`vehicle` class, add about 2,000 Kaggle train frames, and exclude the 20
classifier-holdout intersections. Score class-agnostic against v4 and v8 on:

- base val, minus the 24 near-duplicate frames
- a Kaggle set with whole intersections held out

If it wins, it replaces v4 as stage 1; the classifier is unchanged. About an
hour on the 2070 SUPER.

**B. A targeted typing pass on train mosaic crops (Luke labels).**

- Remove repeats of the same vehicle.
- Let `type_cls_v1` pre-sort the crops so Luke sees predicted Van, Bus,
  Motorcycle and Truck, low-confidence crops, and pickup candidates.
- Target 1,000–1,500 crops, about 2 hours.
- Use the labels to retrain the classifier.

**C. Get un-augmented originals.** Watermarks show Lake County, IL traffic
cameras ("IL 83 / Arlington Heights", "Lake County"). Look for the original
Roboflow Universe project or capture fresh frames. This is a larger project.

**Rules for B, agreed in discussion:**

1. **A separate manifest and database.** Use something like
   `Labeling/review_train/{manifest.json,decisions.sqlite,crops/}`. Decisions
   reference crops by ID through the manifest, so re-running `build_crops.py`
   into `Labeling/review/` would orphan the 1,201 existing decisions. The same
   mechanism corrupted 53 decisions on 09-08.
2. **Mosaic crops are used for training only.** Classifier validation stays on
   the clean-frame crops, split by intersection.
3. **Holdout intersections are removed from the mosaic training crops too.**
   Otherwise the same cars leak into training and inflate the score.

## Gotchas

1. `build_crops.py` hardcodes `SPLITS = ("valid", "test")` and writes to the
   fixed `Labeling/review/manifest.json`. Pointing it at `train` needs a new
   output directory, not just a split change. `label_app.py` also hardcodes
   `MANIFEST` and `DB` (lines 55–56).
2. `diagnose_labels.py` taxonomy output is meaningless on Kaggle labels (one
   class, index 0). Read only section (A).
3. The Kaggle-valid numbers are same-camera figures (finding 2). Do not quote
   them as generalisation.
4. **Which imagery is the target is still undecided.** The README describes a
   DJI Tello drone, and base val is drone and street footage. The Kaggle set and
   the 09-14 search demo are fixed pole cameras. This decides how to weigh
   v8-style trades (a base-val loss against a traffic-cam gain). Luke was asked
   and has not answered.
5. The Bash tool's heredocs failed on this long markdown document (unmatched
   quote error). Write documents with the file tool instead.
6. Everything in the 09-10 and 09-08 hand-offs still applies. In particular:
   restart `label_app.py` after any manifest rebuild.

## Next steps

1. **Commit the 2026-09-14 work** (list under State), and commit
   `docs/two-stage-pipeline.md` with it.
2. Luke decides the **target imagery** (drone vs traffic cam) and which of
   options A, B or C to pursue. They are not exclusive: A needs no labeling and
   can run while B is being labeled.
3. If A: write `scripts/training/train_detector_1class.py`. Build the two
   leak-controlled evaluation sets first.
4. If B: extend `build_crops.py` / `label_app.py` with an output-directory
   argument, then build the deduplicated, classifier-sorted mosaic crop set.
5. Still open from 09-14: add a "not a vehicle" class to stage 2, colour
   labels, re-split base val/test by video sequence.

## References

- `docs/two-stage-pipeline.md` (**untracked**): the 09-14 pipeline, classifier
  and colour record
- `docs/v4-integration-plan.md`: §2 the taxonomy conflict, §3 Path A, §5b v8
- `docs/luke-hand-offs/2026-09-10-stopping-point.md`,
  `2026-09-08-v8-review-pool-training.md`: v8 and the review tool
- `docs/luke-hand-offs/2026-08-06-vision-pipeline-reorg.md`: first discovery
  of the Kaggle augmentation (2,015 sources, 62% duplicates)
- `scripts/labeling/build_crops.py`, `scripts/labeling/label_app.py`,
  `scripts/evaluation/diagnose_labels.py`
- Dataset: https://www.kaggle.com/datasets/ryankraus/traffic-camera-object-detection
