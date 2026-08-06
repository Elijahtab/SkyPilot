"""
Download the full Kaggle traffic-camera-object-detection dataset
and copy it into kaggle_dataset/ preserving the train/valid/test splits
with both images and labels.
"""

import kagglehub
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
KAGGLE_DEST = Path(r"S:\GitHub\SkyPilot\Labeling\kaggle_dataset")

# ── 1. Download dataset ────────────────────────────────
print("[1/3] Downloading dataset from Kaggle...")
downloaded_path = Path(kagglehub.dataset_download("ryankraus/traffic-camera-object-detection"))
traffic_root = downloaded_path / "traffic"
print(f"      Downloaded to: {traffic_root}")

# ── 2. Show source structure ───────────────────────────
print(f"\n[2/3] Source dataset structure:")
for split in ["train", "valid", "test"]:
    split_dir = traffic_root / split
    if split_dir.exists():
        imgs = len(list((split_dir / "images").glob("*")))
        lbls = len(list((split_dir / "labels").glob("*")))
        print(f"      {split:6s} → {imgs} images, {lbls} labels")

# Show data.yaml info
data_yaml = traffic_root / "data.yaml"
if data_yaml.exists():
    print(f"\n      data.yaml contents:")
    print(f"      {data_yaml.read_text().strip()}")

# ── 3. Copy splits to kaggle_dataset ──────────────────
print(f"\n[3/3] Copying splits to {KAGGLE_DEST}...")

# Clear and recreate destination
if KAGGLE_DEST.exists():
    shutil.rmtree(KAGGLE_DEST)
KAGGLE_DEST.mkdir(parents=True, exist_ok=True)

# Copy each split folder
for split in ["train", "valid", "test"]:
    src = traffic_root / split
    dst = KAGGLE_DEST / split
    if src.exists():
        shutil.copytree(src, dst)
        imgs = len(list((dst / "images").glob("*")))
        lbls = len(list((dst / "labels").glob("*")))
        print(f"      ✅ {split:6s} → {imgs} images, {lbls} labels")
    else:
        print(f"      ⚠️  {split} not found in source")

# Copy data.yaml
if data_yaml.exists():
    shutil.copy2(data_yaml, KAGGLE_DEST / "data.yaml")
    print(f"      ✅ data.yaml copied")

print(f"\n✅ Done! Dataset copied to {KAGGLE_DEST}")
print(f"   Structure: kaggle_dataset/train|valid|test/images + labels")
