# Two-stage pipeline: detect, type, colour, search (2026-09-14)

Path A from `docs/v4-integration-plan.md` §3, built and measured, plus a first
exploration of colour so the product can answer "red SUV" or "white van".

```
frame ──> v4 detector ──> boxes (class output ignored, agnostic NMS)
                            │
                            ├─ long side >= 48px ─> type classifier ─> Bus / Motorcycle / SUV /
                            │                                         Standard Car / Truck / Van
                            │                                         (or 'Vehicle' if unsure)
                            ├─ long side <  48px ─> 'Vehicle'  (too small to type)
                            └─ long side >= 24px ─> colour rules ─> white / gray / black / red / ...
                                                      │
                                   SQLite index <─────┘ ──> search_vehicles.py "red suv"
```

The detector never has to choose between `Vehicle` and `SUV`, so the taxonomy
conflict that regressed v5, v6, v7 and v8 cannot occur. v4 is untouched.

## Why v4 and not stock yolov8m for stage 1

Measured on base val, same matching as `diagnose_labels.py` (conf 0.25, IoU 0.5),
stock COCO yolov8m restricted to car/motorcycle/bus/truck:

| model | recall | precision | recall <16px | 16-32px | 32-64px | >=64px |
|---|---|---|---|---|---|---|
| **v4** | **0.788** | 0.792 | **0.56** | **0.86** | **0.92** | **0.82** |
| stock yolov8m @640 | 0.403 | 0.848 | 0.09 | 0.33 | 0.65 | 0.68 |
| stock yolov8m @960 | 0.461 | 0.850 | 0.17 | 0.42 | 0.70 | 0.66 |

Stock yolov8m misses more than half the vehicles; 58% of val boxes are under
32px, and that is what v4's fine-tuning bought.

## Stage 2: the type classifier (`type_cls_v1`)

`yolo11s-cls`, ImageNet init, 224px, 31 epochs (early-stopped, best epoch 16),
about 2 minutes on the 2070 SUPER.

### Data — `build_type_crops.py`

| class | train | val (Kaggle holdout) | test (base val) |
|---|---|---|---|
| Bus | 778 | 5 | 152 |
| Motorcycle | 231 | 0 | 68 |
| SUV | 696 | 108 | 46 |
| Standard Car | 665 | 85 | 13 |
| Truck | 899 | 46 | 202 |
| Van | 174 | 10 | 9 |
| **total** | **3,443** | **254** | **490** |

- train = base train subtype boxes >=48px + 80% of the 1,200 typed review decisions
- **val = the other 20%, split by intersection** (20 of 93), so no camera scene
  is in both. Of 500 seeded splits, the one whose class mix best matches the
  batch is kept.
- square crops with 15% context; half the base crops over 96px are downsampled
  to 40-110px so the high-res base vehicles look like traffic-cam vehicles
- the duplicate-box defect barely touches this: it is almost entirely
  `Vehicle`+subtype pairs, and `Vehicle` boxes are not used. Only 1 contradictory
  subtype pair exists in base train.

### Results — `eval_type_classifier.py`

| metric | val: Kaggle holdout | test: base val |
|---|---|---|
| top-1 accuracy | **0.807** [0.754, 0.851] | 0.806 [0.769, 0.839] |
| majority baseline | 0.425 | 0.412 |
| balanced accuracy | 0.790 | 0.767 |
| v4's own class output on the same crops | 0.031 (84% `Vehicle`) | — |

Per class on the Kaggle holdout: SUV recall 0.852, Standard Car 0.824, Truck
0.674, Van 0.800 (n=10), Bus 0.800 (n=5).

**The gate passes.** SUV vs Standard Car head-to-head on the Kaggle holdout:
**0.865** [0.810, 0.906] against a 0.560 baseline, 193 crops — separable at
traffic-cam scale. Keep both classes. (On base val the same test is
INCONCLUSIVE: 59 crops, 46 of them SUV.)

Confidence → coverage on the Kaggle holdout. Below the threshold the pipeline
answers `Vehicle`:

| conf >= | coverage | accuracy |
|---|---|---|
| 0.5 (pipeline default) | 0.937 | 0.824 |
| 0.7 | 0.764 | 0.840 |
| 0.9 | 0.461 | 0.897 |

Caveats:

- **val also picked the best epoch**, so 0.807 is slightly optimistic; the last
  epoch scored 0.780. Base val (test) was never used for selection and agrees.
- On the 40 crops Luke changed his mind about, accuracy is 0.775 vs 0.813 on the
  rest — the model struggles where the human did.
- Error sheet (`runs_cls/type_cls_v1/eval_val_errors.jpg`): the recurring
  mistakes are **pickups called SUV**, night frames, vehicles cut off at the
  frame edge, and one school bus called Truck. Several "errors" look like label
  problems (step vans labeled Truck). Base labels have visible noise too — a
  top-down sedan and a taxi labeled `Van`, a minivan labeled `Truck`.

## Colour — `_color.py`, `explore_color.py`

No colour labels exist, so the first attempt is rules, not learning: a weighted
pixel vote inside a central ellipse, with the road around the box used to
down-weight road-coloured pixels, grey-world white balance, and reduced votes
for glass and glare.

**Accuracy against 112 crops colour-called by eye: 0.607 (first rules) → 0.616
(after white balance and road-relative white).**

⚠ Those 112 calls are Claude's visual judgement from contact sheets, not human
ground truth. Night frames and ambiguous crops (silver vs white) were skipped.
They are good enough to show the ceiling, not to report as a product number.

| colour | n | recall |
|---|---|---|
| red | 7 | 1.000 |
| black | 25 | 0.680 |
| white | 45 | 0.667 |
| gray | 15 | 0.667 |
| blue | 8 | 0.375 |
| yellow | 10 | 0.200 |

The fixes **moved errors between colours rather than removing them** (white
0.53 → 0.67, blue 0.75 → 0.38), which is the signature of a rule-based ceiling.
Tuning stopped there on purpose rather than fitting thresholds to 112 eye calls.

Failure modes, all visible in `preds/color_explore/sheet_*.jpg`:

1. white paint in shade reads gray or blue
2. yellow school buses and taxis read white (roof) or brown
3. black vs dark gray is decided by exposure
4. **night: tail lights make a dark car red** — no brightness threshold separates
   night from dark daytime frames (night median V reaches 0.43, day goes down to 0.19)

## The product shape: index + search

```powershell
.\myenv\Scripts\python.exe scripts\pipeline\index_frames.py <frames dir>      # -> preds\vehicle_index.sqlite
.\myenv\Scripts\python.exe scripts\pipeline\search_vehicles.py "red suv"      # -> preds\search\red_suv.jpg
```

Demo on the **191 frames from the 20 holdout intersections** (never trained on by
either model): 2,025 vehicles, **37 ms/frame** end to end. 16% typed, 1% unsure,
83% too small to type — the same ~12% legibility the review pass measured.

Spot check of the result sheets:

| query | results | colour right | type right |
|---|---|---|---|
| red suv | 12 | 11 (1 night tail-light) | ~6-8 (pickup, dump truck, sedans) |
| white van | 9 | 9 | 6 (pickup, small SUV, semi trailer) |
| motorcycle | 8 | — | **0** |

**Every "motorcycle" was a traffic signal or a night-time car.** v4 detects
signal heads as vehicles, and the classifier, which has no "not a vehicle"
answer, confidently types them. The holdout contains no motorcycles at all.
This is the biggest defect in the pipeline as built.

## Data findings made along the way

1. **The base `test` split is 108 byte-identical copies of `train` images.**
   Every image, same filename, same bytes. Any v4 test-set number measures
   memorisation. Nothing in this work uses it.
2. **Base `val` leaks consecutive video frames from `train`** — `DJI_0005-0175`
   in val, `-0174` in train; `MunichStreet02-MOS76/77`. 24 of 216 val frames are
   within 6/255 mean pixel difference of a train frame. v4's 0.43 val mAP is
   somewhat optimistic for the same reason. `build_type_crops.py` excludes
   those frames from its test split.
3. **The scipy break was a Python-version mismatch, not corruption.** `myenv` is
   Python 3.12 but scipy (and pydantic_core, jiter, PyYAML, MarkupSafe,
   charset_normalizer) had `cp311` binaries. Reinstalled at the same versions
   with `--no-deps`; training now exits 0 and `results.png` is drawn again.
   pydantic_core being broken meant `openai` could not import either.

## Next steps, in order of value

1. **Add a "not a vehicle" class to stage 2.** Mine v4 detections >=48px that
   overlap no ground-truth box (IoU < 0.1) on the Kaggle frames, have a human
   confirm them in the review app (some will be unlabeled real vehicles — the
   branch-B lesson), and train them as a seventh class. This removes the
   traffic-signal motorcycles and raises detection precision for free.
2. **Colour needs learning to pass ~60%.** Two routes:
   - **label colours** on the existing 1,313 review crops (one keypress each,
     roughly 20-30 min), then train a colour classifier with HSV augmentation
     OFF — and get the first real colour accuracy number;
   - **CLIP zero-shot** (`open_clip_torch`, ~600MB of weights): no labels, and
     it would allow free-text search ("white box truck") — but it still needs
     the colour labels above to be measured.
3. **Pickups.** The largest type error. Decide whether a pickup is `Truck`, and
   label more of them — base `Truck` is mostly box trucks and semis.
4. **Re-split base val/test** by video sequence before quoting any new detector
   number (finding 1 and 2).
5. Video input for `index_frames.py` (sample every Nth frame), for flight logs.

## Files

| path | what |
|---|---|
| `scripts/_crops.py` | the one square-crop implementation, shared by training and inference |
| `scripts/_color.py` | colour rules + white balance |
| `scripts/training/build_type_crops.py` | crop dataset, leak-controlled splits |
| `scripts/training/train_type_classifier.py` | yolo11s-cls training |
| `scripts/evaluation/eval_type_classifier.py` | per-class, gate, coverage, error sheet |
| `scripts/evaluation/explore_color.py` | colour sheets + scoring against a labels CSV |
| `scripts/pipeline/two_stage.py` | detect → type → colour for one frame |
| `scripts/pipeline/index_frames.py` | frames → SQLite index |
| `scripts/pipeline/search_vehicles.py` | query → matches + contact sheet |
| `Vehicle_type_detection/runs_cls/type_cls_v1/` | classifier run (untracked) |
