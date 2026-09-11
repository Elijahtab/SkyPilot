"""
Build the review corpus for the human labeling pass: crop tiles + a manifest.

Complements scripts/labeling/manual_label.py. That tool draws and assigns boxes
one image at a time on the augmented `train` split; this one classifies boxes
that ALREADY EXIST on the clean `valid`/`test` splits, in a 60-per-page grid.

Scope is set by a SIZE GATE, and the gate is the whole point. Measured over the
873 clean Kaggle frames (valid+test), 10,655 human-drawn boxes, all 416x416:

    >= 16px  5,451 (51.2%)     >= 48px  1,313 (12.3%)
    >= 24px  3,597 (33.8%)     >= 64px    727 ( 6.8%)
    >= 32px  2,491 (23.4%)     >= 96px    207 ( 1.9%)

Median box long side is 16.5px, p10 is 6.5px. A human cannot assign a vehicle
subtype to a 16px crop; blown up to a tile it is an 8x magnification of about
250 pixels of information, and the answer is a guess. Guesses are indistinguish-
able from real labels downstream -- they are what regressed v5, v6 and v7. So
the default gate is 48px: 1,313 crops, roughly 25-40 minutes of clicking for one
person, and every one of them is actually legible.

Output (all under Labeling/review/):
    crops/<file>.jpg      padded, upscaled tiles
    manifest.json         one record per crop, ordered into review pages

Boxes are NEVER modified. This pass assigns a class to boxes that already exist,
using the full 7-class schema. 'Vehicle' is the UMBRELLA term -- it covers Bus,
Truck, Motorcycle, Van, Standard Car and SUV alike -- so leaving a crop as
Vehicle is a true-but-unspecific answer, never a wrong one. SUV and Standard Car
are not merged away: they are the only body-style labels this project has.

Usage:
    python scripts/labeling/build_crops.py                 # >=48px, v4 priors
    python scripts/labeling/build_crops.py --min-size 32   # widen the gate
    python scripts/labeling/build_crops.py --no-prior      # no model pre-fill
    python scripts/labeling/build_crops.py --stats         # measure only
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from _paths import CLASS_NAMES, KAGGLE, REPO, VEHICLE_ID, best_weights

REVIEW   = REPO / "Labeling" / "review"
CROPS    = REVIEW / "crops"
MANIFEST = REVIEW / "manifest.json"

SPLITS        = ("valid", "test")      # the CLEAN frames; train/ is augmented
REFERENCE_RUN = "Vehicle_type_detection_v4"
CONF          = 0.25
IOU_MATCH     = 0.5
TILE          = 160                    # px, longest side of the rendered tile
PAD           = 0.20                   # fraction of box size added on each side
PAGE_SIZE     = 60

# Full 7-class schema, straight from configs/vehicle_7class.yaml. SUV and
# Standard Car are NOT collapsed: 'Vehicle' is the umbrella term covering every
# type, not a sibling bucket to be merged into. Collapsing them would delete the
# only labels that distinguish body styles anywhere in this project.
CLASSES = CLASS_NAMES
VEHICLE = VEHICLE_ID


def frames():
    """Yield (split, image_path, label_path) for every clean frame."""
    for split in SPLITS:
        img_dir = KAGGLE / split / "images"
        lbl_dir = KAGGLE / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            for ext in (".jpg", ".jpeg", ".png"):
                img = img_dir / (lbl.stem + ext)
                if img.exists():
                    yield split, img, lbl
                    break


def read_boxes(lbl, w, h):
    """Return [(line_idx, x1, y1, x2, y2)] in pixels. Handles bbox + polygon rows."""
    out = []
    for i, line in enumerate(lbl.read_text().splitlines()):
        p = line.split()
        if not p:
            continue
        v = np.array(p[1:], float)
        if len(p) == 5:
            xc, yc, bw, bh = v
            x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
        else:                                  # segmentation polygon -> its bbox
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        out.append((i, x1 * w, y1 * h, x2 * w, y2 * h))
    return out


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def report_sizes(all_sizes):
    a = np.array(all_sizes)
    print(f"\n  {len(a)} boxes | median long side {np.median(a):.1f}px | "
          f"p10 {np.percentile(a, 10):.1f}px")
    for t in (16, 24, 32, 48, 64, 96):
        print(f"    >= {t:3d}px : {(a >= t).sum():6d}  ({(a >= t).mean() * 100:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=48,
                    help="gate on box long side in px (default 48)")
    ap.add_argument("--no-prior", action="store_true",
                    help="skip v4 inference; every crop starts as generic Vehicle")
    ap.add_argument("--stats", action="store_true", help="measure only, write nothing")
    ap.add_argument("--clean", action="store_true", help="delete existing crops first")
    args = ap.parse_args()

    todo, all_sizes, n_frames = [], [], 0
    print(f"Scanning {KAGGLE} splits {SPLITS} ...")
    for split, img, lbl in frames():
        n_frames += 1
        w, h = Image.open(img).size
        keep = []
        for idx, x1, y1, x2, y2 in read_boxes(lbl, w, h):
            long_side = max(x2 - x1, y2 - y1)
            all_sizes.append(long_side)
            if long_side >= args.min_size:
                keep.append((idx, x1, y1, x2, y2, long_side))
        if keep:
            todo.append((split, img, lbl, w, h, keep))

    n_crops = sum(len(t[5]) for t in todo)
    print(f"  {n_frames} frames scanned")
    report_sizes(all_sizes)
    print(f"\n  GATE >= {args.min_size}px -> {n_crops} crops in {len(todo)} frames")

    if args.stats:
        return
    if n_crops == 0:
        sys.exit("[ERR] gate excluded every box; lower --min-size")

    # -- model priors ------------------------------------------------
    # Grouping a page by predicted class turns each page into "spot the odd one
    # out", which is the single biggest throughput win. The prior is recorded
    # SEPARATELY from the human decision so pre-fill bias stays measurable
    # rather than invisible -- export_labels.py reports the agreement rate.
    priors = {}
    if not args.no_prior:
        weights = best_weights(REFERENCE_RUN)
        # Ultralytics reads class names from the .pt, not the data yaml. The
        # *_vehicle.pt copies carry the renamed schema; originals still say CAR.
        retagged = weights.with_name("best_vehicle.pt")
        if retagged.exists():
            weights = retagged
        if not weights.exists():
            sys.exit(f"[ERR] no weights at {weights}; re-run with --no-prior")

        from ultralytics import YOLO
        print(f"\nRunning {REFERENCE_RUN} over {len(todo)} frames for priors ...")
        model = YOLO(str(weights))
        for n, (split, img, lbl, w, h, keep) in enumerate(todo, 1):
            r = model.predict(str(img), conf=CONF, verbose=False)[0]
            has = len(r.boxes) > 0
            pb = r.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4))
            pc = r.boxes.cls.cpu().numpy().astype(int) if has else np.zeros(0, int)
            gt = [[b[1], b[2], b[3], b[4]] for b in keep]
            m = iou_matrix(gt, pb)
            for row, box in enumerate(keep):
                key = f"{split}/{lbl.stem}#{box[0]}"
                if m.shape[1] and m[row].max() >= IOU_MATCH:
                    priors[key] = int(pc[m[row].argmax()])
                else:
                    priors[key] = VEHICLE          # v4 saw nothing -> umbrella
            if n % 100 == 0:
                print(f"    {n}/{len(todo)} frames")

    # -- render tiles ------------------------------------------------
    if args.clean and CROPS.exists():
        shutil.rmtree(CROPS)
    CROPS.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"\nRendering tiles -> {CROPS}")
    for split, img, lbl, w, h, keep in todo:
        im = Image.open(img).convert("RGB")
        for idx, x1, y1, x2, y2, long_side in keep:
            key = f"{split}/{lbl.stem}#{idx}"
            fname = f"{split}_{lbl.stem}_{idx}.jpg"

            # pad so the vehicle is not cut at the tile edge -- roof line and
            # ground contact are exactly the cues separating Truck from Van
            px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
            box = (max(0, x1 - px), max(0, y1 - py),
                   min(w, x2 + px), min(h, y2 + py))
            tile = im.crop(tuple(int(round(v)) for v in box))
            scale = TILE / max(tile.size)
            if scale > 1:
                tile = tile.resize((max(1, int(tile.width * scale)),
                                    max(1, int(tile.height * scale))),
                                   Image.LANCZOS)
            tile.save(CROPS / fname, quality=88)

            records.append({
                "id": key,
                "file": fname,
                "split": split,
                "stem": lbl.stem,
                "image": img.name,
                "box_index": idx,
                "size_px": round(long_side, 1),
                "box_norm": [round(x1 / w, 6), round(y1 / h, 6),
                             round(x2 / w, 6), round(y2 / h, 6)],
                "prior": priors.get(key, VEHICLE),
            })

    # Order pages by prior class, largest crops first inside each class. All
    # crops on a page then share one prior, so scanning beats judging.
    records.sort(key=lambda r: (r["prior"], -r["size_px"]))
    for i, r in enumerate(records):
        r["page"] = i // PAGE_SIZE

    REVIEW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "classes": CLASSES,
        "generic_id": VEHICLE,
        "min_size_px": args.min_size,
        "page_size": PAGE_SIZE,
        "has_prior": not args.no_prior,
        "prior_model": None if args.no_prior else REFERENCE_RUN,
        "crops": records,
    }, indent=1), encoding="utf-8")

    dist = Counter(CLASSES[r["prior"]] for r in records)
    print(f"\n  {len(records)} tiles, {records[-1]['page'] + 1} pages of {PAGE_SIZE}")
    print(f"  prior distribution: {dict(dist)}")
    print(f"  manifest -> {MANIFEST}")
    print("\nNext:  .\\myenv\\Scripts\\python.exe scripts\\labeling\\label_app.py")
    print("NOTE: the app loads the manifest ONCE at startup — restart it after "
          "any build_crops.py run, or it will write stale class ids.")


if __name__ == "__main__":
    main()
