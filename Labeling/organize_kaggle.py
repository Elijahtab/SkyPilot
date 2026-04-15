import os, pathlib, shutil

ROOT_DIR = pathlib.Path(__file__).resolve().parent
IMG_DIR  = ROOT_DIR / "images_to_label"
LBL_DIR  = ROOT_DIR / "labels"
PREV_DIR = ROOT_DIR / "preview"

KAGGLE_DIR = ROOT_DIR / "kaggle_dataset"
KAGGLE_IMG = KAGGLE_DIR / "images"
KAGGLE_LBL = KAGGLE_DIR / "labels"
KAGGLE_PRV = KAGGLE_DIR / "preview"

KAGGLE_LBL_CACHE = pathlib.Path(os.path.expanduser("~/.cache/kagglehub/datasets/ryankraus/traffic-camera-object-detection/versions/1/traffic/train/labels"))

def setup():
    KAGGLE_IMG.mkdir(parents=True, exist_ok=True)
    KAGGLE_LBL.mkdir(parents=True, exist_ok=True)
    KAGGLE_PRV.mkdir(parents=True, exist_ok=True)

def run():
    kaggle_stems = set(p.stem for p in KAGGLE_LBL_CACHE.glob("*.txt"))
    
    # Move raw images
    moved_imgs = 0
    for img_path in IMG_DIR.glob("*.[jp][pn]g"):
        if img_path.stem in kaggle_stems:
            shutil.move(str(img_path), str(KAGGLE_IMG / img_path.name))
            moved_imgs += 1
            
    # Move labels
    moved_lbls = 0
    for txt_path in LBL_DIR.glob("*.txt"):
        if txt_path.stem in kaggle_stems:
            shutil.move(str(txt_path), str(KAGGLE_LBL / txt_path.name))
            moved_lbls += 1
            
    # Move previews
    for prv_path in PREV_DIR.glob("*_preview.jpg"):
        original_stem = prv_path.name.replace("_preview.jpg", "")
        if original_stem in kaggle_stems:
            shutil.move(str(prv_path), str(KAGGLE_PRV / prv_path.name))

    print(f"Isolated {moved_imgs} images and {moved_lbls} labels into {KAGGLE_DIR}")

if __name__ == '__main__':
    setup()
    run()
