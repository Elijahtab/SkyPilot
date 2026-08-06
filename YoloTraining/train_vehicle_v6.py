"""
Train Vehicle Type Detection v6 — fine-tune v4 best weights with additional
Kaggle labeled data (including the new GPT auto-labels) at a lower learning rate.

Key changes from v5:
  • Pulls in the newly extracted `kaggle_gpt` dataset (306 strictly filtered GPT-labeled images)
  • Maintains v5 settings (cosine LR, lower lr0, freeze 10)
"""

import sys, tempfile, yaml, torch
from pathlib import Path
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
YOLOV8M_WEIGHTS = Path(r"S:\GitHub\SkyPilot\yolov8m.pt")
DATASET_ROOT  = Path(r"S:\GitHub\SkyPilot\Vehicle_type_detection")
PROJECT_DIR   = DATASET_ROOT / "runs"
RUN_NAME      = "Vehicle_type_detection_v6_oversampled"

KAGGLE_LABELED_IMG = Path(r"S:\GitHub\SkyPilot\Labeling\kaggle_dataset\labeled\images")
KAGGLE_GPT_IMG     = Path(r"S:\GitHub\SkyPilot\images\kaggle_gpt")

TRAIN_IMG_DIR = DATASET_ROOT / "images" / "images_train"

# ────────────────────────────────────────────────────────────
# Hyper-parameters  (tuned down from v4 for fine-tuning)
# ────────────────────────────────────────────────────────────
EPOCHS     = 40
BATCH_SIZE = 16
IMAGE_SIZE = 640
LR0        = 0.015         # slightly higher LR for faster convergence from base
LRF        = 0.01          # final LR fraction
PATIENCE   = 20            # moderate patience
FREEZE     = 0             # train all layers from base COCO

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Bus", "CAR", "Motorcycle", "SUV", "Standard Car", "Truck", "Van"]

OVERSAMPLE_FACTOR = 5  # Duplicate Kaggle GPT images to give them more weight

# Use absolute paths in a list so each dataset stays in its own folder
DATA_CFG = {
    "train": [
        str(DATASET_ROOT / "images" / "images_train"),
    ] + [str(KAGGLE_GPT_IMG)] * OVERSAMPLE_FACTOR,
    "val":   str(DATASET_ROOT / "images" / "images_val"),
    "test":  str(DATASET_ROOT / "images" / "images_test"),
    "nc":    len(CLASS_NAMES),
    "names": CLASS_NAMES,
}

# ────────────────────────────────────────────────────────────
# Sanity-check directories
# ────────────────────────────────────────────────────────────
def sanity_check():
    """Verify all image/label directories exist and print counts."""
    missing = []

    # Check train dirs (list of absolute paths)
    for train_dir in DATA_CFG["train"]:
        img_dir = Path(train_dir)
        if img_dir.name == "images":                       # kaggle: images/ → labels/
            lbl_dir = img_dir.parent / "labels"
        elif img_dir.name == "kaggle_gpt":                 # Auto-label output: images/kaggle_gpt -> labels/kaggle_gpt
            lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
        else:
            lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
            
        if not img_dir.exists():
            missing.append(img_dir)
        if not lbl_dir.exists():
            missing.append(lbl_dir)

    # Check val / test
    for split_key in ("val", "test"):
        img_dir = Path(DATA_CFG[split_key])
        lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
        
        if not img_dir.exists():
            missing.append(img_dir)
        if not lbl_dir.exists():
            missing.append(lbl_dir)

    if missing:
        print("[ERR] Missing folders:")
        for p in missing:
            print(" •", p)
        sys.exit(1)

    # Print dataset summary
    total_train = 0
    for train_dir in DATA_CFG["train"]:
        count = len(list(Path(train_dir).glob("*.jpg"))) + len(list(Path(train_dir).glob("*.png")))
        total_train += count
        print(f"[INFO] Train source: {train_dir}  →  {count} images")
    print(f"[INFO] Total training images so far (jpg+png): {total_train}")

# ────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────
def train():
    ckpt_path = PROJECT_DIR / RUN_NAME / "weights" / "last.pt"
    
    print(f"[INFO] Using device: {DEVICE}")
    if ckpt_path.exists():
        print(f"\n[INFO] Auto-Resume Triggered! Found {ckpt_path}")
        model = YOLO(str(ckpt_path))
        model.train(resume=True)
        print(f"\n✅ v6 Training complete! Logs at {PROJECT_DIR / RUN_NAME}")
        return
        
    print(f"[INFO] Fresh training from: {YOLOV8M_WEIGHTS}")
    print(f"[INFO] LR: {LR0}  |  Epochs: {EPOCHS}  |  Freeze: {FREEZE} layers")

    tmp_yaml = Path(tempfile.gettempdir()) / "vehicle_type_v6.yaml"
    with tmp_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DATA_CFG, f, sort_keys=False)

    model = YOLO(str(YOLOV8M_WEIGHTS))

    model.train(
        data=str(tmp_yaml),
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
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        device=DEVICE,
        mosaic=0.5,        # reduced mosaic for stability
        mixup=0.0,         # disabled mixup to reduce noise
    )

    print(f"\n✅ v6 Training complete! Logs at {PROJECT_DIR / RUN_NAME}")

# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sanity_check()
    train()
