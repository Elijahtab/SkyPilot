"""
Train the Stage-2 vehicle TYPE classifier on crops from build_type_crops.py.

Stage 1 is v4, untouched, used only to find vehicles. This model names the type
of each box >= 48px: Bus, Motorcycle, SUV, Standard Car, Truck, Van. Below the
gate, or when this model is unsure, the pipeline answers 'Vehicle' -- the
umbrella class -- which is true-but-unspecific rather than wrong.

Validation is the Kaggle intersection holdout (images/type_cls/val), not the base
crops: traffic-cam scale is where the classifier has to work, and the only split
where its SUV vs Standard Car number means anything. Per-class metrics, the
SUV/Standard Car gate and the confidence->coverage table come from
scripts/evaluation/eval_type_classifier.py -- top-1 alone hides all three.

Usage:
    python scripts/training/build_type_crops.py          # once
    python scripts/training/train_type_classifier.py
    python scripts/training/train_type_classifier.py --model yolo11m-cls.pt --name type_cls_v2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO

from _crops import TYPE_CLASSES
from _paths import PRETRAINED, TYPE_CLS, TYPE_RUNS, require

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
BASE_MODEL = "yolo11s-cls.pt"         # ImageNet-pretrained, auto-downloaded once
RUN_NAME   = "type_cls_v1"

EPOCHS     = 60
PATIENCE   = 15
IMAGE_SIZE = 224                      # ImageNet resolution; crops are upsampled to it
BATCH_SIZE = 64

# RandomResizedCrop keeps (1 - SCALE, 1.0) of the crop area. The default 0.5 can
# cut half the vehicle away; the crops are already tight (15% context per side),
# so keep at least 70% of the area.
SCALE = 0.3

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def sanity_check():
    require(TYPE_CLS / "train", TYPE_CLS / "val")
    for split in ("train", "val", "test"):
        dirs = sorted(p.name for p in (TYPE_CLS / split).iterdir() if p.is_dir())
        if dirs != sorted(TYPE_CLASSES):
            sys.exit(f"[ERR] {split}/ has class folders {dirs}, expected {sorted(TYPE_CLASSES)}.\n"
                     f"      ImageFolder numbers classes by folder, so a mismatch mislabels "
                     f"every crop. Re-run build_type_crops.py.")
        counts = {d: len(list((TYPE_CLS / split / d).glob('*.jpg'))) for d in dirs}
        print(f"[INFO] {split:5s} {sum(counts.values()):5d}  {counts}")


def train(model_name, run_name):
    weights = PRETRAINED / model_name     # downloads here on first use
    print(f"[INFO] Using device: {DEVICE}   base: {weights.name}")

    YOLO(str(weights)).train(
        data=str(TYPE_CLS),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        scale=SCALE,
        seed=0,
        deterministic=True,
        workers=8,
        plots=True,
        project=str(TYPE_RUNS),
        name=run_name,
        exist_ok=True,
        device=DEVICE,
    )
    print(f"\n✅ Training complete! Logs at {TYPE_RUNS / run_name}")
    print(f"Next:  .\\myenv\\Scripts\\python.exe scripts\\evaluation\\eval_type_classifier.py {run_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--name", default=RUN_NAME)
    args = ap.parse_args()
    sanity_check()
    train(args.model, args.name)
