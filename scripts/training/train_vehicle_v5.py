"""
Train Vehicle Type Detection v5 — fine-tune v4 best weights with the Kaggle
auto-labeled pool at a lower learning rate.

Key settings vs v4:
  • Starts from v4 best.pt (not from scratch)
  • Lower lr0 (0.0005), cosine LR schedule
  • Kaggle pool kept in its own folder (multi-dir train config, no oversampling)
  • Freezes first 10 backbone layers to preserve learned features

FIXED: this script previously pointed at Labeling/kaggle_dataset/labeled/images,
a directory that no longer exists (download_kaggle.py restructured the dataset to
train|valid|test), so sanity_check() exited before training. It now uses the
actual merged pool, _paths.POOL_IMG.

⚠ RESULT: regressed (val mAP50-95 0.407 vs v4's 0.430), peaking at epoch 5 of 25.
  Same labeling-policy conflict as v6/v7 — see the v7 header and
  scripts/evaluation/diagnose_labels.py.
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
RUN_NAME   = "Vehicle_type_detection_v5"

EPOCHS     = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LR0        = 0.0005        # conservative LR for fine-tuning
LRF        = 0.01
PATIENCE   = 20
FREEZE     = 10            # preserve v4 backbone features

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# absolute paths in a list so each dataset stays in its own folder
TRAIN_SOURCES = [VTD_TRAIN, POOL_IMG]


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
    print(f"[INFO] Total training images: {total}")


def train():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Fine-tuning from: {V4_WEIGHTS}")
    print(f"[INFO] LR: {LR0}  |  Epochs: {EPOCHS}  |  Freeze: {FREEZE} layers")

    data = build_data_yaml("v5", train=TRAIN_SOURCES, val=VTD_VAL, test=VTD_TEST)

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
        exist_ok=True,
        device=DEVICE,
        mosaic=0.5,        # reduced mosaic for stability
        mixup=0.0,         # disabled mixup to reduce noise
    )

    print(f"\n✅ v5 Training complete! Logs at {VTD_RUNS / RUN_NAME}")


if __name__ == "__main__":
    print(f"[INFO] Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    sanity_check()
    train()
