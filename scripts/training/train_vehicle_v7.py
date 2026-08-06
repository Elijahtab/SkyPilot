"""
Train Vehicle Type Detection v7 — fine-tune v4 best weights with the Kaggle
GPT auto-labels.

Key settings:
  • Starts from v4 best.pt (strongest checkpoint)
  • Lower lr0 (0.0008), freeze first 10 backbone layers
  • Oversamples the GPT pool by OVERSAMPLE_FACTOR

⚠ RESULT: this run REGRESSED (val mAP50-95 0.363 vs v4's 0.430), peaking at
  epoch 1 — i.e. every epoch trained on the mixed pool made val worse. The cause
  is a labeling-policy conflict, not a hyper-parameter problem: the GPT labels
  call vehicles 'SUV'/'Standard Car' where the val set calls them 'Vehicle'
  (class agreement 0.19, vs 0.93 on the base val set). Run
  scripts/evaluation/diagnose_labels.py before re-running this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO
from _paths import (CLASS_NAMES, POOL_IMG, VTD_RUNS, VTD_TRAIN, VTD_VAL,
                    VTD_TEST, best_weights, build_data_yaml, labels_for, require)

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
V4_WEIGHTS = best_weights("Vehicle_type_detection_v4")
RUN_NAME   = "Vehicle_type_detection_v7"

EPOCHS     = 40
BATCH_SIZE = 16
IMAGE_SIZE = 640
LR0        = 0.0008        # lower LR for fine-tuning
LRF        = 0.01          # final LR fraction
PATIENCE   = 20
FREEZE     = 10            # freeze backbone features

OVERSAMPLE_FACTOR = 2      # duplicate the GPT pool to give it more weight

DEVICE = 0 if torch.cuda.is_available() else "cpu"

TRAIN_SOURCES = [VTD_TRAIN] + [POOL_IMG] * OVERSAMPLE_FACTOR


def sanity_check():
    """Verify every image dir has a matching label dir, then print counts."""
    require(V4_WEIGHTS)
    for d in set(TRAIN_SOURCES) | {VTD_VAL, VTD_TEST}:
        require(d, labels_for(d))

    total = 0
    for d in TRAIN_SOURCES:
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        total += n
        print(f"[INFO] Train source: {d}  →  {n} images")
    print(f"[INFO] Total training images (with oversampling): {total}")


def train():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Fine-tuning from: {V4_WEIGHTS}")
    print(f"[INFO] LR: {LR0}  |  Epochs: {EPOCHS}  |  Freeze: {FREEZE} layers")

    data = build_data_yaml("v7", train=TRAIN_SOURCES, val=VTD_VAL, test=VTD_TEST)

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
        plots=True,
        patience=PATIENCE,
        cos_lr=True,
        project=str(VTD_RUNS),
        name=RUN_NAME,
        exist_ok=False,
        device=DEVICE,
        mosaic=0.5,        # reduced mosaic for stability
        mixup=0.0,         # disabled mixup to reduce noise
    )

    print(f"\n✅ v7 Training complete! Logs at {VTD_RUNS / RUN_NAME}")


if __name__ == "__main__":
    print(f"[INFO] Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    sanity_check()
    train()
