"""
Train Vehicle Type Detection v5 — fine-tune v4 best weights with additional
Kaggle labeled data at a lower learning rate.

Key changes from v4:
  • Starts from v4 best.pt (not from scratch)
  • Lower lr0 (0.0005 → half of v4's 0.001)
  • Kaggle data kept in its own folder (not merged) — uses multi-dir train config
  • Cosine LR scheduler for smoother convergence
  • Freeze first 10 backbone layers to preserve learned features
"""

import sys, tempfile, yaml, torch
from pathlib import Path
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
V4_WEIGHTS    = Path(r"S:\GitHub\SkyPilot\Vehicle_type_detection\runs\Vehicle_type_detection_v4\weights\best.pt")
DATASET_ROOT  = Path(r"S:\GitHub\SkyPilot\Vehicle_type_detection")
PROJECT_DIR   = DATASET_ROOT / "runs"
RUN_NAME      = "Vehicle_type_detection_v5"

KAGGLE_LABELED_IMG = Path(r"S:\GitHub\SkyPilot\Labeling\kaggle_dataset\labeled\images")
KAGGLE_LABELED_LBL = Path(r"S:\GitHub\SkyPilot\Labeling\kaggle_dataset\labeled\labels")

TRAIN_IMG_DIR = DATASET_ROOT / "images" / "images_train"

# ────────────────────────────────────────────────────────────
# Hyper-parameters  (tuned down from v4 for fine-tuning)
# ────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LR0        = 0.0005        # restore conservative LR
LRF        = 0.01          # final LR fraction
PATIENCE   = 20            # moderate patience
FREEZE     = 10            # restore backbone freeze to preserve v4 features

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Bus", "CAR", "Motorcycle", "SUV", "Standard Car", "Truck", "Van"]

# Use absolute paths in a list so each dataset stays in its own folder
DATA_CFG = {
    "train": [
        str(DATASET_ROOT / "images" / "images_train"),
        str(KAGGLE_LABELED_IMG),
    ],
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
        count = len(list(Path(train_dir).glob("*.jpg")))
        total_train += count
        print(f"[INFO] Train source: {train_dir}  →  {count} images")
    print(f"[INFO] Total training images: {total_train}")

# ────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────
def train():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Fine-tuning from: {V4_WEIGHTS}")
    print(f"[INFO] LR: {LR0}  |  Epochs: {EPOCHS}  |  Freeze: {FREEZE} layers")

    tmp_yaml = Path(tempfile.gettempdir()) / "vehicle_type_v5.yaml"
    with tmp_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DATA_CFG, f, sort_keys=False)

    model = YOLO(str(V4_WEIGHTS))

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

    print(f"\n✅ v5 Training complete! Logs at {PROJECT_DIR / RUN_NAME}")

# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sanity_check()
    train()
