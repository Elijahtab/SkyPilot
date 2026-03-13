from pathlib import Path
import sys, tempfile, yaml, torch
from xml.parsers.expat import model
from ultralytics import YOLO

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
PRETRAINED_WEIGHTS = r"S:/GitHub/SkyPilot/Vehicle_type_detection/runs/Vehicle_type_detection_v1/weights/best.pt"
EPOCHS             = 75
BATCH_SIZE         = 16
IMAGE_SIZE         = 640
LR0                = 0.001
RUN_NAME           = "Vehicle_type_detection_v1"
PROJECT_DIR        = Path(r"S:/GitHub/SkyPilot/Vehicle_type_detection/runs")

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# Dataset mapping
DATASET_ROOT = Path(r"S:/GitHub/SkyPilot/Vehicle_type_detection")
TRAIN_SPLIT  = "images/images_train"
VAL_SPLIT    = "images/images_val"
TEST_SPLIT   = "images/images_test"

CLASS_NAMES = ["Bus", "CAR", "Motorcycle", "SUV", "Standard Car", "Truck", "Van"]
DATA_CFG = {
    "path": str(DATASET_ROOT),
    "train": TRAIN_SPLIT,
    "val": VAL_SPLIT,
    "test": TEST_SPLIT,
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def sanity_check_paths():
    missing = []
    for split_key in ("train", "val", "test"):
        split_rel = DATA_CFG[split_key]
        if not split_rel:
            continue
        img_dir = DATASET_ROOT / split_rel
        lbl_dir = DATASET_ROOT / split_rel.replace("images", "labels", 1)
        if not img_dir.exists():
            missing.append(img_dir)
        if not lbl_dir.exists():
            missing.append(lbl_dir)
    if missing:
        print("\n[ERR] Missing folders:")
        for p in missing:
            print(" •", p)
        sys.exit(1)


def main():
    sanity_check_paths()

    tmp_yaml = Path(tempfile.gettempdir()) / "vehicle_type_v4.yaml"
    with tmp_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DATA_CFG, f, sort_keys=False)

    model = YOLO("yolov8m.pt")

    model.train(
        data=str(tmp_yaml),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LR0,
        cache=True,
        workers=12,
        half=True,
        plots=False,
        patience=10,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        device=DEVICE,
    )

    print("\n✅ Training extended! Logs at", PROJECT_DIR / RUN_NAME)


if __name__ == "__main__":
    main()
