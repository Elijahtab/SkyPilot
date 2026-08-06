"""
Train Vehicle Type Detection v1 — baseline, from COCO yolov8m on the base
vehicle-type dataset only (no auto-labeled data).

Was YoloTraining/test_yolo.py. Renamed because it is a training script, not a
test: RUN_NAME/PRETRAINED_WEIGHTS pointed at the v1 run and its recorded args
(lr0 0.001, patience 10) match runs/Vehicle_type_detection_v1.

⚠ NOTE: no script in this repo reproduces **v4**, which is still the best model
  (val mAP50-95 0.430) and the init point for v5 and v7. v4's recorded args
  (Vehicle_type_detection/runs/Vehicle_type_detection_v4/args.yaml) are
  lr0 0.01, optimizer auto, mosaic 1.0, freeze null, epochs 50, initialised from
  a runs/Vehicle_type_detection/weights/best.pt that no longer exists. Those
  differ from this file's settings — do not assume this reproduces v4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO
from _paths import (CLASS_NAMES, PRETRAINED, VTD_RUNS, VTD_TRAIN, VTD_VAL,
                    VTD_TEST, build_data_yaml, labels_for, require)

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
BASE_WEIGHTS = PRETRAINED / "yolov8m.pt"
RUN_NAME     = "Vehicle_type_detection_v1"

EPOCHS     = 75
BATCH_SIZE = 16
IMAGE_SIZE = 640
LR0        = 0.001
PATIENCE   = 10

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def sanity_check():
    require(BASE_WEIGHTS)
    for d in (VTD_TRAIN, VTD_VAL, VTD_TEST):
        require(d, labels_for(d))
    for d in (VTD_TRAIN, VTD_VAL, VTD_TEST):
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        print(f"[INFO] {d.name:14s} → {n} images")


def train():
    print(f"[INFO] Using device: {DEVICE}")
    data = build_data_yaml("v1", train=VTD_TRAIN, val=VTD_VAL, test=VTD_TEST)

    YOLO(str(BASE_WEIGHTS)).train(
        data=str(data),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LR0,
        cache=True,
        workers=12,
        half=True,
        plots=False,
        patience=PATIENCE,
        project=str(VTD_RUNS),
        name=RUN_NAME,
        exist_ok=True,
        device=DEVICE,
    )

    print(f"\n✅ Training complete! Logs at {VTD_RUNS / RUN_NAME}")


if __name__ == "__main__":
    print(f"[INFO] Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    sanity_check()
    train()
