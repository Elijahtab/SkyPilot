"""
Train Vehicle Type Detection v6 — train from base COCO yolov8m with the Kaggle
GPT auto-labels oversampled.

Key settings:
  • Fresh start from yolov8m.pt (not from v4), all layers unfrozen
  • Higher lr0 (0.015) for faster convergence from base
  • Auto-resumes from last.pt if the run directory already exists

⚠ RESULT: this is the WORST run in the project. At OVERSAMPLE_FACTOR=5 val
  mAP50-95 reached only 0.215 (vs v4's 0.430); at factor 1 it reached 0.291.
  Performance degrades monotonically with more GPT data, which is the signature
  of a labeling conflict rather than a training-config problem. See
  scripts/evaluation/diagnose_labels.py and the v7 header before re-running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO
from _paths import (CLASS_NAMES, POOL_IMG, PRETRAINED, VTD_RUNS, VTD_TRAIN,
                    VTD_VAL, VTD_TEST, build_data_yaml, labels_for, require)

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
BASE_WEIGHTS = PRETRAINED / "yolov8m.pt"
RUN_NAME     = "Vehicle_type_detection_v6_oversampled"

EPOCHS     = 40
BATCH_SIZE = 16
IMAGE_SIZE = 640
LR0        = 0.015         # higher LR for convergence from base COCO
LRF        = 0.01
PATIENCE   = 20
FREEZE     = 0             # train all layers

OVERSAMPLE_FACTOR = 5

DEVICE = 0 if torch.cuda.is_available() else "cpu"

TRAIN_SOURCES = [VTD_TRAIN] + [POOL_IMG] * OVERSAMPLE_FACTOR


def sanity_check():
    """Verify every image dir has a matching label dir, then print counts."""
    require(BASE_WEIGHTS)
    for d in set(TRAIN_SOURCES) | {VTD_VAL, VTD_TEST}:
        require(d, labels_for(d))

    total = 0
    for d in TRAIN_SOURCES:
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        total += n
        print(f"[INFO] Train source: {d}  →  {n} images")
    print(f"[INFO] Total training images (with oversampling): {total}")


def train():
    ckpt = VTD_RUNS / RUN_NAME / "weights" / "last.pt"

    print(f"[INFO] Using device: {DEVICE}")
    if ckpt.exists():
        print(f"\n[INFO] Auto-Resume Triggered! Found {ckpt}")
        YOLO(str(ckpt)).train(resume=True)
        print(f"\n✅ v6 Training complete! Logs at {VTD_RUNS / RUN_NAME}")
        return

    print(f"[INFO] Fresh training from: {BASE_WEIGHTS}")
    print(f"[INFO] LR: {LR0}  |  Epochs: {EPOCHS}  |  Freeze: {FREEZE} layers")

    data = build_data_yaml("v6", train=TRAIN_SOURCES, val=VTD_VAL, test=VTD_TEST)

    YOLO(str(BASE_WEIGHTS)).train(
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

    print(f"\n✅ v6 Training complete! Logs at {VTD_RUNS / RUN_NAME}")


if __name__ == "__main__":
    print(f"[INFO] Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    sanity_check()
    train()
