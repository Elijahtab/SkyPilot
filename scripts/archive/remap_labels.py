"""
ARCHIVED - LEGACY (Stanford Cars 11-class schema). Remaps the 196 Stanford class
ids into 11 coarse classes and emits class_remap_11.csv. Note it also writes a
duplicate generic box for every specific-class box; the current 7-class data does
NOT use that convention.
"""

import os
import re
import shutil
import yaml
from pathlib import Path
from collections import Counter
import scipy.io

# ---------- CONFIG ----------
ROOT = Path("S:/GitHub/SkyPilot")
META_FILE = ROOT / "devkit" / "cars_meta.mat"

IMG_TRAIN = ROOT / "images/train"
LBL_TRAIN = ROOT / "labels/train"
IMG_VAL = ROOT / "images/val"
LBL_VAL   = ROOT / "labels/val"

NEW_CLASSES = [
    "CAR", "Convertible", "Coupe", "ExoticSports", "Hatchback",
    "Minivan", "PickupTruck", "SUV", "Sedan", "Van", "Wagon"
]

# Matching rules: first match wins
RULES = [
    ("ExoticSports", r"(ferrari|lamborghini|bugatti|mclaren|spyker|aston martin|bentley|maybach|rolls-royce|koenigsegg|pagani|gtr|z06|superleggera|reventon|murcielago|gallardo|viper)"),
    ("PickupTruck", r"(pickup|crew cab|quad cab|regular cab|extended cab|sut|silverado|f-150|f-250|f-350|ram pickup)"),
    ("Minivan", r"\bminivan\b"),
    ("Van", r"\bvan\b"),
    ("SUV", r"\bsuv\b"),
    ("Wagon", r"\bwagon\b"),
    ("Convertible", r"(convertible|roadster|drophead|cabrio)"),
    ("Hatchback", r"\bhatchback\b"),
    ("Coupe", r"\bcoupe\b"),
    ("Sedan", r"\bsedan\b"),
    ("CAR", r".*")  # fallback default
]

DEFAULT_CLASS = "CAR"
BACKUP_DIR = ROOT / "labels_backup"
# --------------------------------------------

def pick_new_class(name: str) -> str:
    ln = name.lower()
    for new_cls, pattern in RULES:
        if re.search(pattern, ln):
            return new_cls
    return DEFAULT_CLASS

def main():
    # Load original class names
    mat = scipy.io.loadmat(META_FILE)
    orig_names = [str(n[0]) for n in mat["class_names"][0]]

    # Build mapping old_id -> new_id
    name_to_id = {c: i for i, c in enumerate(NEW_CLASSES)}
    old_to_new = {}
    for old_id, full_name in enumerate(orig_names):
        new_cls = pick_new_class(full_name)
        old_to_new[old_id] = name_to_id[new_cls]

    # Save for inspection
    with open(ROOT / "class_remap_11.csv", "w", encoding="utf-8") as f:
        f.write("old_id,old_name,new_class,new_id\n")
        for i, n in enumerate(orig_names):
            f.write(f"{i},{n},{NEW_CLASSES[old_to_new[i]]},{old_to_new[i]}\n")

    print("📄 Wrote class_remap_11.csv")

    # Backup original labels once
    if not BACKUP_DIR.exists():
        shutil.copytree(ROOT / "labels", BACKUP_DIR)
        print("✅ Backed up labels/ -> labels_backup/")

    def remap_folder(lbl_dir: Path, split_name: str):
        counts = Counter()
        for txt in lbl_dir.glob("*.txt"):
            new_lines = []
            with open(txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    coords = parts[1:]
                    new_id = old_to_new.get(old_id, None)
                    if new_id is not None:
                        # Add specific class
                        new_lines.append(f"{new_id} {' '.join(coords)}")
                        counts[NEW_CLASSES[new_id]] += 1

                        # Add general CAR class if not already CAR (id 0)
                        if new_id != 0:
                            new_lines.append(f"0 {' '.join(coords)}")
                            counts["CAR"] += 1
            with open(txt, "w") as f:
                f.write("\n".join(new_lines))
        print(f"[{split_name}] ✅ Processed {len(list(lbl_dir.glob('*.txt')))} files.")
        for cls in NEW_CLASSES:
            print(f"{cls:14}: {counts[cls]} boxes")

    remap_folder(LBL_TRAIN, "TRAIN")
    remap_folder(LBL_VAL, "VAL")

    # Update original data.yaml
    data_yaml = ROOT / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, "r") as f:
            y = yaml.safe_load(f)
        y["names"] = NEW_CLASSES
        y["nc"] = len(NEW_CLASSES)
        with open(data_yaml, "w") as f:
            yaml.safe_dump(y, f, sort_keys=False)
        print("✅ Updated data.yaml with 11-class schema")

    # Remove YOLO caches
    for cache in ROOT.rglob("*.cache"):
        cache.unlink()
        print(f"🗑️ Deleted cache {cache}")

    print("\n🎉 Done. You can now train both pools with data2.yaml!")

if __name__ == "__main__":
    main()