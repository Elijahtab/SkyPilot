"""
Train Vehicle Type Detection v8 — fine-tune v4 with the 2026-09-08 human-reviewed
Kaggle pool (images/labels/kaggle_review).

Named v8, not v5: v5/v6/v7 are taken by the GPT-pool runs.

WHAT THIS RUN IS FOR
--------------------
This is a deliberate test, run with the conflict known in advance, not a bid for
a better checkpoint. The reviewed labels use a different labeling policy from the
base val/test splits, measured by diagnose_labels.py on this exact pool:

    class-agnostic recall    0.723   <- boxes are GOOD (GPT pool was 0.371)
    class-agnostic precision 0.783   <- (GPT pool was 0.331)
    class agreement          0.723   <- dominated by the sub-gate Vehicle rows

    of the boxes a human called SUV,          92.9% are 'Vehicle' to v4
    of the boxes a human called Standard Car, 99.2% are 'Vehicle' to v4

So headline mAP on the base val split is EXPECTED TO FALL, because base val calls
those same objects 'Vehicle' and scores a correct SUV as two errors at once.

RESULT (2026-09-08): regressed to val mAP50-95 **0.3959**, best epoch **22** of
37, vs v4's 0.4322. Every per-class AP50 fell. Class-agnostic recall on base val
fell 0.802 -> 0.770, so the hoped-for detection-side win did not appear either —
precision rose only because v8 emits fewer boxes (4,787 vs 5,081).

Worth keeping: it peaked at epoch 22 rather than epoch 1 or 5 like v5/v7. The
boxes really are good; the decline is the taxonomy conflict accumulating, not
geometry collapsing. That separates two failure modes four runs had conflated.

Second confound, unresolved: 5,293 of the pool's 6,494 boxes are sub-gate (<48px)
and were auto-labeled 'Vehicle' without review, so the pool also teaches "large
car = SUV, small car = Vehicle" — size, not appearance. This run cannot separate
that from the taxonomy conflict.

Full write-up: docs/v4-integration-plan.md

    python scripts/training/train_vehicle_v8.py
    python scripts/training/train_vehicle_v8.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO

from _paths import (CLASS_NAMES, REVIEW_IMG, VTD_RUNS, VTD_TEST, VTD_TRAIN,
                    VTD_VAL, best_weights, build_data_yaml, labels_for, require)

V4_WEIGHTS = best_weights("Vehicle_type_detection_v4")
RUN_NAME   = "Vehicle_type_detection_v8"

EPOCHS     = 40
BATCH_SIZE = 16
IMAGE_SIZE = 640           # match v4 so mAP stays comparable
LR0        = 0.0008        # v7's fine-tune LR
LRF        = 0.01
PATIENCE   = 15
FREEZE     = 10            # preserve v4 backbone features

OVERSAMPLE_FACTOR = 1      # no oversampling: v6_oversampled at 5x was the worst
                           # run on record. Let the pool sit at its natural size.

DEVICE = 0 if torch.cuda.is_available() else "cpu"

TRAIN_SOURCES = [VTD_TRAIN] + [REVIEW_IMG] * OVERSAMPLE_FACTOR


def sanity_check():
    require(V4_WEIGHTS)
    for d in set(TRAIN_SOURCES) | {VTD_VAL, VTD_TEST}:
        require(d, labels_for(d))

    total = 0
    for d in TRAIN_SOURCES:
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        lbl = len(list(labels_for(d).glob("*.txt")))
        total += n
        print(f"[INFO] train source {d.name:16s} {n:5d} images / {lbl:5d} labels")
    print(f"[INFO] total training images: {total}")
    print(f"[INFO] val: {VTD_VAL.name} "
          f"({len(list(VTD_VAL.glob('*.jpg'))) + len(list(VTD_VAL.glob('*.png')))} images)")

    # Stale .cache files silently reuse old box counts after a pool changes.
    for d in TRAIN_SOURCES:
        c = labels_for(d).with_suffix(".cache")
        if c.exists():
            c.unlink()
            print(f"[INFO] removed stale cache {c.name}")


def train():
    print(f"[INFO] device {DEVICE} | init {V4_WEIGHTS.name} | "
          f"lr0 {LR0} freeze {FREEZE} imgsz {IMAGE_SIZE} epochs {EPOCHS}")

    data = build_data_yaml("v8", train=TRAIN_SOURCES, val=VTD_VAL, test=VTD_TEST)

    YOLO(str(V4_WEIGHTS)).train(
        data=str(data),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LR0,
        lrf=LRF,
        freeze=FREEZE,
        cache="disk",
        workers=4,
        optimizer="AdamW",
        half=True,
        plots=True,             # REQUIRED: confusion_matrix stays all zeros
                                # without it, and class-agnostic metrics derived
                                # from it silently read 0.000 instead of erroring
        patience=PATIENCE,
        cos_lr=True,
        project=str(VTD_RUNS),
        name=RUN_NAME,
        exist_ok=False,
        device=DEVICE,
        mosaic=0.5,
        mixup=0.0,
    )

    print(f"\n[OK] v8 done. Logs: {VTD_RUNS / RUN_NAME}")
    print("Compare against v4 — read class-agnostic recall, not just mAP:")
    print("  .\\myenv\\Scripts\\python.exe scripts\\evaluation\\compare_models.py "
          "v4 v8 --split val")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="sanity check and exit without training")
    args = ap.parse_args()

    print(f"[INFO] classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    sanity_check()
    if args.dry_run:
        print("[INFO] --dry-run: not training.")
    else:
        train()
