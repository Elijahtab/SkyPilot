"""
Copy GPT-labeled Kaggle images + labels into the training pool
(images/kaggle_gpt + labels/kaggle_gpt).

This is the ONLY supported path into the pool. The other historical merger,
scripts/archive/merge_yolo_labels.py, fed in yolov8n-proposed boxes that were
~72% under-annotated; that batch has been quarantined (see
Labeling/quarantine_branchB_yolov8n/README.md).

    python scripts/labeling/extract_good_kaggle.py
    python scripts/labeling/extract_good_kaggle.py --check   # verify only
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _paths import KAGGLE_GPT, KAGGLE_IMG, POOL_IMG, POOL_LBL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, copy nothing")
    args = ap.parse_args()

    print(f"Reading labels from: {KAGGLE_GPT}")
    POOL_IMG.mkdir(parents=True, exist_ok=True)
    POOL_LBL.mkdir(parents=True, exist_ok=True)

    copied = missing = 0
    for txt_path in KAGGLE_GPT.glob("*.txt"):
        img_path = KAGGLE_IMG / f"{txt_path.stem}.jpg"
        if not img_path.exists():
            img_path = KAGGLE_IMG / f"{txt_path.stem}.png"

        if not img_path.exists():
            missing += 1
            print(f"Warning: label with no matching image: {txt_path.stem}")
            continue

        if not args.check:
            shutil.copy2(img_path, POOL_IMG / img_path.name)
            shutil.copy2(txt_path, POOL_LBL / txt_path.name)
        copied += 1

    verb = "Would copy" if args.check else "Copied"
    print(f"\nDone! {verb} {copied} image/label pairs.")
    if missing:
        print(f"Warning: {missing} labels had no matching image.")
    print(f"Images → {POOL_IMG}")
    print(f"Labels → {POOL_LBL}")

    # a stale ultralytics cache will silently reuse the old box counts
    for cache in POOL_LBL.parent.glob(f"{POOL_LBL.name}.cache"):
        if not args.check:
            cache.unlink()
            print(f"Removed stale label cache: {cache}")

    print("\n⚠ Verify before training:")
    print("    python scripts/evaluation/diagnose_labels.py")


if __name__ == "__main__":
    main()
