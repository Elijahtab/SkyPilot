"""
Find objects that carry TWO labels: the umbrella class 'Vehicle' and a body-style
subtype, on the same box.

Why this exists. 'Vehicle' is the umbrella term covering every type, so an
annotator working a hierarchy can legitimately think "this is a Vehicle AND it is
an SUV" and draw both. YOLO cannot express that -- each box gets exactly one
class -- so the pair lands as two near-identical boxes with contradictory labels
over the same pixels. The detector is then trained to call those pixels 'Vehicle'
and 'SUV' at once, and NMS keeps only one of them at inference.

Measured on the base dataset (IoU >= 0.85):

    split          frames affected     boxes in a pair
    images_train   37 of 948           362 (1.8%)
    images_val      6 of 215            92 (1.8%)
    images_test     0 of 108             0

The split-level share looks negligible; the per-class share does not. The pairs
land almost entirely on the two rarest body styles:

    31% of all SUV labels          (115 of 370)
    20% of all Standard Car labels  (81 of 403)
     1% of Truck labels

which is a strong candidate explanation for why v4 confuses SUV and Standard Car
with Vehicle so heavily (GT SUV -> predicted Vehicle 41.5%, GT Standard Car ->
Vehicle 70%), and why those two are its worst classes.

Note that train and val carry the defect and test does NOT, so the splits
disagree about what 'Vehicle' means.

Usage:
    python scripts/tools/find_duplicate_boxes.py                 # report
    python scripts/tools/find_duplicate_boxes.py --list          # name the frames
    python scripts/tools/find_duplicate_boxes.py --fix-dir OUT   # write deduped labels
    python scripts/tools/find_duplicate_boxes.py --iou 0.7       # looser matching

--fix-dir writes a corrected copy of every label file to OUT/<split>/ and never
touches the originals. The fix drops the umbrella box and keeps the subtype,
which is the information-preserving direction: 'SUV' implies 'Vehicle', not the
other way round.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _paths import CLASS_NAMES, VEHICLE_ID, VTD_LABELS

SPLITS = ("images_train", "images_val", "images_test")


def to_xyxy(rows):
    v = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in rows])
    return np.stack([v[:, 0] - v[:, 2] / 2, v[:, 1] - v[:, 3] / 2,
                     v[:, 0] + v[:, 2] / 2, v[:, 1] + v[:, 3] / 2], 1)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def scan_frame(rows, thr):
    """Return (indices of umbrella rows to drop, Counter of partner subtypes)."""
    cls = np.array([int(r[0]) for r in rows])
    box = to_xyxy(rows)
    g = np.flatnonzero(cls == VEHICLE_ID)
    o = np.flatnonzero(cls != VEHICLE_ID)
    if len(g) == 0 or len(o) == 0:
        return [], Counter()
    m = iou(box[g], box[o])
    drop, partners = [], Counter()
    for i in range(len(g)):
        j = int(m[i].argmax())
        if m[i, j] >= thr:
            drop.append(int(g[i]))
            partners[CLASS_NAMES[cls[o[j]]]] += 1
    return drop, partners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.85)
    ap.add_argument("--list", action="store_true", help="print affected frame stems")
    ap.add_argument("--fix-dir", type=Path, default=None,
                    help="write deduped labels here (originals untouched)")
    args = ap.parse_args()

    class_totals = Counter()
    for split in SPLITS:
        for f in (VTD_LABELS / split).glob("*.txt"):
            for line in f.read_text().splitlines():
                p = line.split()
                if p:
                    class_totals[CLASS_NAMES[int(p[0])]] += 1

    grand = Counter()
    for split in SPLITS:
        d = VTD_LABELS / split
        n_frames = n_boxes = affected = dropped = 0
        partners = Counter()
        names = []
        for f in sorted(d.glob("*.txt")):
            rows = [l.split() for l in f.read_text().splitlines() if l.split()]
            if not rows:
                continue
            n_frames += 1
            n_boxes += len(rows)
            drop, part = scan_frame(rows, args.iou)
            if drop:
                affected += 1
                dropped += len(drop)
                partners += part
                names.append(f.stem)
            if args.fix_dir:
                out = args.fix_dir / split
                out.mkdir(parents=True, exist_ok=True)
                keep = [" ".join(r) for i, r in enumerate(rows) if i not in set(drop)]
                (out / f.name).write_text("\n".join(keep) + "\n", encoding="utf-8")

        grand += partners
        print(f"--- {split} ---  {n_frames} frames, {n_boxes} boxes")
        print(f"  frames with a duplicate pair : {affected}")
        print(f"  boxes involved               : {dropped * 2} "
              f"({dropped * 2 / max(n_boxes, 1) * 100:.1f}%)")
        if partners:
            print(f"  subtype in the pair          : {dict(partners)}")
        if args.list and names:
            for n in names:
                print(f"    {n}")

    if grand:
        print("\nShare of each subtype's labels that are half of a duplicate pair:")
        for name, n in grand.most_common():
            tot = class_totals[name]
            print(f"  {name:14s} {n:4d} / {tot:5d}  ({n / max(tot, 1) * 100:.0f}%)")

    if args.fix_dir:
        print(f"\nDeduped labels -> {args.fix_dir}")
        print("Originals untouched. Re-measure before trusting this:")
        print("  .\\myenv\\Scripts\\python.exe scripts\\evaluation\\eval_model.py")


if __name__ == "__main__":
    main()
