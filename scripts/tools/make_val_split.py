"""
Carve a validation split out of a training split by MOVING files.

    python scripts/tools/make_val_split.py <train_images_dir> [--frac 0.2] [--seed 0]
    python scripts/tools/make_val_split.py --dry-run <dir>

Label paths are derived with the /images/ → /labels/ convention. The split is
seeded so it is reproducible, and it refuses to run if the destination already
holds files (re-running used to silently move a second 20% out of train).
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _paths import labels_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_images", help="training image directory")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_img = Path(args.train_images)
    train_lbl = labels_for(train_img)
    val_img = train_img.parent / train_img.name.replace("train", "val")
    val_lbl = labels_for(val_img)

    if val_img == train_img:
        sys.exit(f"[ERR] cannot derive a val dir name from {train_img}")
    for d in (train_img, train_lbl):
        if not d.exists():
            sys.exit(f"[ERR] missing {d}")

    existing = list(val_img.glob("*.jpg")) if val_img.exists() else []
    if existing and not args.dry_run:
        sys.exit(f"[ERR] {val_img} already holds {len(existing)} images — "
                 f"refusing to move more out of train. Delete it first if intended.")

    images = sorted(p for p in train_img.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    n = int(len(images) * args.frac)
    random.Random(args.seed).shuffle(images)
    picked = images[:n]

    print(f"train images : {train_img}  ({len(images)})")
    print(f"val images   : {val_img}")
    print(f"moving       : {n} ({args.frac:.0%}, seed {args.seed})")

    if args.dry_run:
        for p in picked[:10]:
            print("   ", p.name)
        print(f"    ... ({n} total)  — dry run, nothing moved")
        return

    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    moved = skipped = 0
    for img in picked:
        lbl = train_lbl / f"{img.stem}.txt"
        if not lbl.exists():
            skipped += 1                   # background image, leave it in train
            continue
        shutil.move(str(img), str(val_img / img.name))
        shutil.move(str(lbl), str(val_lbl / lbl.name))
        moved += 1

    print(f"\nMoved {moved} image/label pairs to the validation set.")
    if skipped:
        print(f"Skipped {skipped} images with no label file (left in train).")


if __name__ == "__main__":
    main()
