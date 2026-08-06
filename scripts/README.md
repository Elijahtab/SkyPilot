# Vision pipeline — scripts

All Python for the detection work lives here. Every script resolves paths through
[`_paths.py`](_paths.py), so there are no hardcoded `S:\GitHub\SkyPilot` paths and the
repo works from any checkout location.

Run everything from the repo root with the project venv:

```powershell
.\myenv\Scripts\python.exe scripts\<area>\<script>.py
```

## Layout

| Path | What's in it |
|---|---|
| [`_paths.py`](_paths.py) | Repo paths + the class schema. Single source of truth. |
| [`labeling/`](labeling/) | Fetch the Kaggle dataset, GPT auto-labeling, merge into the pool |
| [`training/`](training/) | `train_vehicle_v1/v5/v6/v7.py` |
| [`evaluation/`](evaluation/) | Eval, model comparison, inference, **label-quality gate** |
| [`tools/`](tools/) | Val splits, promoting checkpoints, retagging class names |
| [`archive/`](archive/) | Dead or superseded scripts. Each carries a banner saying why. Don't run these. |

Configs live in [`../configs/`](../configs/), weights in [`../weights/`](../weights/)
(`pretrained/` backbones, `released/` promoted checkpoints).

## The class schema

[`../configs/vehicle_7class.yaml`](../configs/vehicle_7class.yaml) is authoritative;
`_paths.py` reads `names` from it.

```
0 Bus   1 Vehicle   2 Motorcycle   3 SUV   4 Standard Car   5 Truck   6 Van
```

Class 1 `Vehicle` (renamed from `CAR`) is the **generic** bucket — used when a vehicle is
present but its subtype isn't confidently determinable. It deliberately overlaps
SUV / Standard Car / Van, and it's ~78% of the boxes in the base dataset.

> Existing checkpoints have their class names baked in. `*_vehicle.pt` copies next to each
> `best.pt`/`last.pt` say `Vehicle`; the originals still say `CAR`. Weights are identical —
> only the `names` dict differs. Use `tools/retag_checkpoint.py` to regenerate.

## Typical flows

**Add auto-labeled data** — never skip step 3.

```powershell
.\myenv\Scripts\python.exe scripts\labeling\download_kaggle.py        # once
.\myenv\Scripts\python.exe scripts\labeling\auto_label_kaggle.py      # GPT-4o, costs money
.\myenv\Scripts\python.exe scripts\labeling\extract_good_kaggle.py    # -> images/labels/kaggle_gpt
.\myenv\Scripts\python.exe scripts\evaluation\diagnose_labels.py      # GATE
```

**Train / evaluate**

```powershell
.\myenv\Scripts\python.exe scripts\training\train_vehicle_v7.py
.\myenv\Scripts\python.exe scripts\evaluation\eval_model.py
.\myenv\Scripts\python.exe scripts\evaluation\compare_models.py v4 v7 --split val
.\myenv\Scripts\python.exe scripts\tools\promote_best_model.py
```

## Read this before training on auto-labeled data

`v4` is still the best model (val mAP50-95 **0.430**). Every run that added GPT auto-labels
regressed, monotonically with the amount added:

| run | init | GPT data | val mAP50-95 | best epoch |
|---|---|---|---|---|
| **v4** | v1 best | none | **0.430** | 50 / 55 |
| v5 | v4 best | 1× | 0.407 | 5 / 25 |
| v7 | v4 best | 2× | 0.363 | **1** / 21 |
| v6 | yolov8m | 1× | 0.291 | 39 / 41 |
| v6_oversampled | yolov8m | 5× | 0.215 | 37 / 40 |

v5 and v7 peaking at epoch 5 and 1 means training on the mixed pool made validation worse
from the very first epoch. Measured with `diagnose_labels.py`:

|  | base val (healthy) | GPT pool |
|---|---|---|
| class-agnostic recall | 0.804 | 0.371 |
| class-agnostic precision | 0.813 | 0.331 |
| class agreement (matched boxes only) | **0.934** | **0.189** |

Two independent causes, both now addressed:

1. **Under-annotation** — the 99 images from `auto_label_yolo.py` had ~72% of their vehicles
   unlabeled (yolov8n proposed the boxes). Quarantined to
   `Labeling/quarantine_branchB_yolov8n/`; the pool is now the 207 human-boxed images only.
2. **Taxonomy collision** — GPT called vehicles `SUV`/`Standard Car` where the val set says
   `Vehicle` (93–95% of the time on the same object). The prompt in `auto_label_kaggle.py`
   now biases toward the generic class to match. **Existing labels in the pool predate this
   fix and still use the old vocabulary — re-run the labeler before trusting them.**

Still open: the Kaggle imagery is 416×416 Roboflow exports, 62% augmentation duplicates,
many of them rotated/mosaicked — so GPT is classifying sideways vehicles. Dedupe to the
2,015 unique source frames before the next labeling pass.

Also open: **no script reproduces v4**, the best model and the init point for v5/v7. Its
recorded args differ from `train_vehicle_v1.py` — see that file's header.
