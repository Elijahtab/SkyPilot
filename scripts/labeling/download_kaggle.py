"""
Download the Kaggle traffic-camera-object-detection dataset into
Labeling/kaggle_dataset/, preserving the train/valid/test splits.

⚠ DESTRUCTIVE. This wipes Labeling/kaggle_dataset/ before copying, which
  includes train/labels_gpt/ — the GPT auto-labels, each of which cost an API
  call to produce. The script now refuses to proceed if labels_gpt/ is present
  unless you pass --force, and always offers to back it up first.

Note on the data: 62% of the train split is Roboflow augmentation duplicates
(2015 unique source frames → 5248 images), and every image is a 416x416 export,
many of them rotation/mosaic augmentations. Splits are clean — no source frame
leaks between train/valid/test.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kagglehub

from _paths import KAGGLE, KAGGLE_GPT

DATASET = "ryankraus/traffic-camera-object-detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="proceed even though existing GPT labels will be destroyed")
    ap.add_argument("--no-backup", action="store_true", help="skip the labels_gpt backup")
    args = ap.parse_args()

    # ── guard the expensive artifacts ──────────────────────
    existing = sorted(KAGGLE_GPT.glob("*.txt")) if KAGGLE_GPT.exists() else []
    if existing:
        print(f"⚠ {len(existing)} GPT auto-labels exist at {KAGGLE_GPT}")
        print("  Re-downloading deletes them (the whole kaggle_dataset/ tree is replaced).")
        if not args.force:
            sys.exit("  Refusing to continue. Re-run with --force if that is intended.")
        if not args.no_backup:
            backup = KAGGLE.parent / f"{KAGGLE.name}_labels_gpt_backup"
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(KAGGLE_GPT, backup)
            print(f"  ✓ Backed up {len(existing)} labels → {backup}")

    # ── download ───────────────────────────────────────────
    print("[1/3] Downloading dataset from Kaggle...")
    traffic_root = Path(kagglehub.dataset_download(DATASET)) / "traffic"
    print(f"      Downloaded to: {traffic_root}")

    print("\n[2/3] Source dataset structure:")
    for split in ("train", "valid", "test"):
        d = traffic_root / split
        if d.exists():
            print(f"      {split:6s} → {len(list((d / 'images').glob('*')))} images, "
                  f"{len(list((d / 'labels').glob('*')))} labels")

    data_yaml = traffic_root / "data.yaml"
    if data_yaml.exists():
        print(f"\n      data.yaml contents:\n      {data_yaml.read_text().strip()}")

    # ── copy ───────────────────────────────────────────────
    print(f"\n[3/3] Copying splits to {KAGGLE}...")
    if KAGGLE.exists():
        shutil.rmtree(KAGGLE)
    KAGGLE.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src = traffic_root / split
        if not src.exists():
            print(f"      ⚠  {split} not found in source")
            continue
        dst = KAGGLE / split
        shutil.copytree(src, dst)
        print(f"      ✅ {split:6s} → {len(list((dst / 'images').glob('*')))} images, "
              f"{len(list((dst / 'labels').glob('*')))} labels")

    if data_yaml.exists():
        shutil.copy2(data_yaml, KAGGLE / "data.yaml")
        print("      ✅ data.yaml copied")

    print(f"\n✅ Done! Dataset at {KAGGLE}")
    print("   Structure: kaggle_dataset/train|valid|test/{images,labels}")


if __name__ == "__main__":
    main()
