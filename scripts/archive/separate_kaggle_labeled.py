"""
ARCHIVED - DEAD PATHS. Targets kaggle_dataset/{images,labels}/ and writes a
labeled/ + unlabeled/ split that no longer exists after the train|valid|test
restructure. Kept only as a record of the original layout.
"""

"""
Separate kaggle_dataset images into labeled vs unlabeled folders.

- kaggle_dataset/labeled/images/   → images that have a matching .txt in labels/
- kaggle_dataset/labeled/labels/   → corresponding label files
- kaggle_dataset/unlabeled/        → images with NO label file
"""

import shutil
from pathlib import Path

KAGGLE_ROOT = Path(r"S:\GitHub\SkyPilot\Labeling\kaggle_dataset")
IMG_DIR     = KAGGLE_ROOT / "images"
LBL_DIR     = KAGGLE_ROOT / "labels"

# Output directories
LABELED_IMG   = KAGGLE_ROOT / "labeled" / "images"
LABELED_LBL   = KAGGLE_ROOT / "labeled" / "labels"
UNLABELED_DIR = KAGGLE_ROOT / "unlabeled"

def main():
    # Create output dirs
    LABELED_IMG.mkdir(parents=True, exist_ok=True)
    LABELED_LBL.mkdir(parents=True, exist_ok=True)
    UNLABELED_DIR.mkdir(parents=True, exist_ok=True)

    labeled_count = 0
    unlabeled_count = 0

    for img_path in sorted(IMG_DIR.glob("*.jpg")):
        label_path = LBL_DIR / (img_path.stem + ".txt")

        if label_path.exists():
            shutil.copy2(img_path, LABELED_IMG / img_path.name)
            shutil.copy2(label_path, LABELED_LBL / label_path.name)
            labeled_count += 1
        else:
            shutil.copy2(img_path, UNLABELED_DIR / img_path.name)
            unlabeled_count += 1

    print(f"\n✅ Done!")
    print(f"   Labeled:   {labeled_count} images + labels → {LABELED_IMG}")
    print(f"   Unlabeled: {unlabeled_count} images        → {UNLABELED_DIR}")

if __name__ == "__main__":
    main()
