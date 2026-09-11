# Plan: integrating the 2026-09-08 review batch into v4

Covers the 1,201 crops labeled by hand on 2026-09-08 from the clean Kaggle frames
at the >=48px gate, and what can and cannot be done with them.

**Headline: this batch is not "more training data". It is a taxonomy change, and
merging it into v4 training as-is regresses the model — the same mechanism that
sank v5, v6 and v7.** The labels are good; the conflict is with what the base
dataset means by `Vehicle`. This was predicted in §2 and then confirmed by
running it (§5b).

---

## 1. What the pass produced

1,201 of 1,313 crops decided (112 left undecided), across 521 frames.

| class | crops | median px |
|---|---|---|
| SUV | 508 | 66 |
| Standard Car | 400 | 66 |
| Truck | 222 | 74 |
| Van | 45 | 74 |
| Bus | 25 | 92 |
| Vehicle | 1 | 80 |
| Motorcycle | 0 | — |

Mechanically the session is sound: no schema corruption after the 5->7 class fix,
no rubber-stamping, an append-only log with every decision timestamped.

### Agreement with the v4 prior: 0.028

Near-total disagreement. That rules out pre-fill bias — nobody was accepting the
model's answer. But read the correction table and the reason becomes clear:

| correction | n |
|---|---|
| Vehicle -> SUV | 474 |
| Vehicle -> Standard Car | 397 |
| Vehicle -> Truck | 126 |
| Bus -> Truck | 62 |
| Truck -> SUV | 34 |

871 of 1,201 decisions convert `Vehicle` into a body style. That is not a
correction of model error. It is a different labeling policy.

---

## 2. The blocking problem

At the **same size range**, the two label sets disagree almost completely:

| corpus, boxes >=48px | labeled `Vehicle` | labeled a body style |
|---|---|---|
| base **val** split | 1,848 / 2,375 (**78%**) | 527 (22%) |
| base **test** split | 659 / 883 (**75%**) | 224 (25%) |
| **this batch** | 1 / 1,201 (**0.08%**) | 1,200 (99.9%) |

Train a model on this batch and evaluate it on the base val split and every
correctly-recognised SUV is scored **twice wrong** — a false positive on `SUV`
and a false negative on `Vehicle`. mAP collapses. This is arithmetic, not a risk
estimate, and it is precisely the v5/v6/v7 failure: class agreement on the GPT
pool measured 0.189 against 0.934 between the base splits.

**There is currently no validation set in this taxonomy, so there is also no way
to measure whether the new labels help.**

### The second defect: the labels are size-conditioned

Export writes 6,494 boxes for the 521 touched frames: 1,201 reviewed, and
**5,293 sub-gate boxes labeled `Vehicle` because nobody looked at them.**

So within one dataset, the same physical object type is labeled:

- `SUV` / `Standard Car` when its box is >=48px
- `Vehicle` when its box is <48px

A detector trained on this learns *size*, not appearance — "large car = SUV,
small car = Vehicle". That is unlearnable as a visual class and it will degrade
the detector's existing behaviour. It is a worse defect than the taxonomy
conflict because it is internally contradictory rather than merely different.

### Third: SUV vs Standard Car is unstable even for one person

178 crops were decided more than once; 170 of those changed class. **79 of the
changes are SUV <-> Standard Car swaps** (58 one way, 21 the other), plus 46
Truck <-> Standard Car.

Caveat, stated plainly: these are in-session corrections, not blind re-tests, so
0.045 is **not** a self-consistency score — the tool never re-showed a crop with
the answer hidden. But the churn is concentrated on exactly the boundary the
earlier research predicted would be unrecoverable at traffic-cam scale, and that
is worth taking as a warning rather than an artifact.

---

## 3. Recommended: Path A — two-stage, don't retrain v4's classes

**Keep v4 exactly as it is** and add a separate crop classifier.

- v4's job is *finding* vehicles, and it is good at it (0.802 class-agnostic
  recall on its own val, 0.723 on this pool). The taxonomy dispute does not touch
  that job at all.
- The batch you produced is, literally, a crop-classification dataset: 1,201
  fixed rectangles each with one human body-style label.
- In two stages the detector never has to choose between `Vehicle` and `SUV`, so
  **the collision disappears instead of being fought.** Sub-gate boxes are not
  mislabeled either — they simply never reach the classifier, and the system
  returns `Vehicle` as an explicit "too small to type".

### Training set for the classifier

Combining this batch with the base dataset's own >=48px subtype crops:

| class | this batch | base train >=48px | total |
|---|---|---|---|
| Truck | 222 | 729 | 951 |
| SUV | 508 | 310 | 818 |
| Bus | 25 | 764 | 789 |
| Standard Car | 400 | 371 | 771 |
| Motorcycle | 0 | 234 | 234 |
| Van | 45 | 141 | 186 |

Six classes, four of them near or above the ~300-instance threshold where a
classifier generalises rather than memorises. **This is the first time the
project has had a workable SUV/Standard Car sample** — the starvation the
2026-08-06 hand-off called unfixable from traffic-cam imagery.

**Watch the scale gap:** base subtype crops have a median long side of 282px
(SUV) and 210px (Standard Car); this batch's are 66px. Two very different
domains. Normalise crops to a fixed input, and hold out a Kaggle-only validation
split so the number reported is the one that matters.

### Steps

1. Export: `export_labels.py --partial` (see §5 first).
2. New `scripts/training/train_subtype_classifier.py` — crops from both sources,
   fixed input size, stratified split, class-balanced loss for Van/Motorcycle.
3. Hold out 20% of *this batch* as the validation set. Report per-class accuracy
   and the SUV/Standard Car confusion specifically.
4. Inference wrapper: v4 detects, crops >=48px go to the classifier, crops below
   the gate return `Vehicle` explicitly as "too small to type".
5. **Gate:** if SUV-vs-Standard Car accuracy is at or near chance, drop those two
   to a single `Car` class and ship the rest. That result would be a finding, not
   a failure — it settles a question three training runs have stumbled over.

**Effort ~1 day. No re-labeling. v4 stays the champion and is never put at risk.**

---

## 4. Path B — single-stage, if you want one model

Only viable if the evaluation sets are moved to the same taxonomy first.

1. Review the base **val** and **test** splits at the same >=48px gate:
   **2,375 + 883 = 3,258 crops**, about 2.5x the pass just completed, so roughly
   1.5–2 hours. `build_crops.py` needs a flag to point at the base dataset instead
   of Kaggle — about 20 lines.
2. Decide and *document* what `Vehicle` means for sub-gate boxes. The honest
   options are "too small to type" (making it partly a size class, which must
   then be consistent across train, val and test) or dropping sub-gate boxes
   entirely — which reintroduces the under-annotation that poisoned branch B.
   Neither is clean. This is the cost of one-stage.
3. Retrain from v4 at imgsz 960 (the resolution sweep's +4.4pt class-agnostic
   recall), gate with `diagnose_labels.py`, compare on the *new* val split.
4. The current 0.430 mAP50-95 is **not** a comparable baseline afterwards.
   Re-measure v4 against the new val split before claiming any delta.

Do not do step 3 before steps 1 and 2. That ordering is the whole lesson of
v5/v6/v7 — and of v8 below.

---

## 5. Do these regardless of path

1. **The 112 undecided crops.** On export they silently become `Vehicle` inside
   their frames. If that matches the intent ("inconclusive stays Vehicle"),
   make it explicit; only 1 crop was actually keyed to `Vehicle`, so the record
   currently shows skips, not decisions.
2. **Back up `Labeling/review/decisions.sqlite`.** It is ~2 hours of
   irreplaceable human work in a gitignored file.
3. **Run the gate before anything trains:**
   `diagnose_labels.py images\kaggle_review`. Expect low class agreement — that
   is this batch's known property, not a surprise, and the gate is there to stop
   it reaching training silently.
4. **Fix the duplicate-box defect first** (`tools/find_duplicate_boxes.py`).
   31% of the base dataset's SUV labels and 20% of Standard Car are half of a
   contradictory `Vehicle`+subtype pair on the same object. Any conclusion about
   those two classes — including the classifier's — is unreliable until this is
   resolved, because it is concentrated in exactly the classes under study.

---

## 5b. RESULT — v8 was run anyway (2026-09-08)

Path B's merge was tested directly rather than argued about. `train_vehicle_v8.py`,
fine-tuned from v4 on base train + the reviewed pool at 1x, imgsz 640, v7's recipe.

| metric, base val | v4 | v8 | delta |
|---|---|---|---|
| mAP50-95 | **0.4322** | 0.3959 | **-0.036** |
| mAP50 | 0.6195 | 0.5785 | -0.041 |
| class-agnostic recall | **0.802** | 0.770 | **-0.032** |
| class-agnostic precision | 0.806 | **0.821** | +0.015 |
| class agreement | 0.934 | 0.931 | -0.003 |

Per-class AP50 fell for **every single class**: Bus -0.031, Vehicle -0.009,
Motorcycle -0.059, SUV -0.041, Standard Car -0.107, Truck -0.019, Van -0.021.

Two things worth keeping from this:

1. **It is the best of the four merge attempts** (0.396 vs v5 0.407 — within
   noise of it — and clearly above v7's 0.363), and it peaked at **epoch 22**
   rather than epoch 1 or 5. The boxes really are good; the decline is the
   taxonomy conflict accumulating gradually, not geometry collapsing instantly.
2. **The detection-side hope did not survive contact.** Class-agnostic recall
   *fell* 0.802 -> 0.770. Precision rose slightly because v8 predicts fewer boxes
   (4,787 vs 5,081) — it became more conservative, not more accurate. So there is
   no salvageable win hiding under the mAP drop, and no hyper-parameter change
   would expose one.

Conclusion: merging this pool into the detector does not work, for the reason
predicted in §2, and the reviewed labels should go to a classifier (§3) rather
than into detector training.

> The v8 run directory was lost to a `git reset --hard` + clean on 2026-09-10,
> then **rebuilt by re-running `train_vehicle_v8.py` the same day**. The rerun
> reproduced the original **exactly** — same 37 epochs, same best epoch 22, same
> 0.3959, same per-class AP50, same 4,787 predicted boxes — because ultralytics
> defaults to `seed=0, deterministic=True`. So the numbers above rest on two
> independent runs, not one, and the run is reproducible on demand from a
> committed script. The run directory itself is untracked (104MB of weights).

---

## 6. What not to do

- **Do not merge `images/kaggle_review` into v4 training and retrain.** Tested;
  see §5b. The expected outcome is a worse model.
- **Do not collapse SUV and Standard Car into `Vehicle` to make the conflict go
  away.** `Vehicle` is the umbrella covering every type; merging deletes the only
  body-style labels the project has, including the 908 produced by hand.
- **Do not compare any new number to 0.430** until it is measured on a val split
  in the same taxonomy.

## References

- `docs/labeling-tool-plan.md` — why the 48px gate, why most boxes stay `Vehicle`
- `docs/labeling-web-app-research.md` — the size distribution behind the gate
- `docs/luke-hand-offs/2026-09-08-v8-review-pool-training.md` — the session record
- `docs/luke-hand-offs/2026-08-06-vision-pipeline-reorg.md` — the v5/v6/v7 diagnosis
- `scripts/tools/find_duplicate_boxes.py` — the umbrella/subtype double-label defect
- `scripts/evaluation/diagnose_labels.py` — the merge gate
- `configs/vehicle_7class.yaml` — schema and the umbrella semantics
