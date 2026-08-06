import os
import shutil
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent # SkyPilot
KAGGLE_IMG_DIR = ROOT_DIR / "Labeling" / "kaggle_dataset" / "train" / "images"
KAGGLE_LBL_DIR = ROOT_DIR / "Labeling" / "kaggle_dataset" / "train" / "labels_gpt"

OUT_IMG_DIR = ROOT_DIR / "images" / "kaggle_gpt"
OUT_LBL_DIR = ROOT_DIR / "labels" / "kaggle_gpt"

def main():
    print(f"Reading labels from: {KAGGLE_LBL_DIR}")
    
    # Create the output directories if they don't exist
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    
    count = 0
    missing_images = 0
    
    # Scan labels_gpt for generated .txt files
    for txt_path in KAGGLE_LBL_DIR.glob("*.txt"):
        stem = txt_path.stem
        
        # Check if the corresponding image exists
        img_path = KAGGLE_IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            img_path = KAGGLE_IMG_DIR / f"{stem}.png" # fallback just in case
            
        if img_path.exists():
            # Copy both image and label to their final destinations
            shutil.copy2(img_path, OUT_IMG_DIR / img_path.name)
            shutil.copy2(txt_path, OUT_LBL_DIR / txt_path.name)
            count += 1
        else:
            missing_images += 1
            print(f"Warning: Found label but missing image for {stem}")
            
    print(f"\nDone! Copied {count} images and labels.")
    if missing_images > 0:
        print(f"Warning: {missing_images} labels had no matching image.")
    print(f"Images output to: {OUT_IMG_DIR}")
    print(f"Labels output to: {OUT_LBL_DIR}")

if __name__ == "__main__":
    main()
