"""
Build the crop dataset for the Stage-2 vehicle TYPE classifier.

Two-stage design (docs/two-stage-pipeline.md): v4 finds vehicles and its class
output is ignored; a separate classifier names the type of every box >= 48px.
The detector then never has to choose between 'Vehicle' and 'SUV' for one
object, so the taxonomy conflict that sank v5-v8 cannot occur.

Sources, all human-labeled:
  1. The review decisions in Labeling/review/decisions.sqlite -- clean Kaggle
     valid/test frames, 416x416 traffic cams. Latest active decision per crop,
     exactly as export_labels.py reads them.
  2. Subtype boxes >= 48px in the base dataset's TRAIN split.
  3. Subtype boxes >= 48px in the base VAL split -> test/, never trained on.
'Vehicle' boxes are skipped everywhere: it is the umbrella, not a type.

Splits (ultralytics ImageFolder layout):
  train/  base train + 80% of the Kaggle decisions
  val/    20% of the Kaggle decisions -- THE number that matters: traffic-cam
          scale, the domain the classifier must work in
  test/   base val crops -- a second, high-resolution domain

Leakage is controlled by construction, because the obvious splits leak:
  * Kaggle frames are named <intersection>-<direction>-<n>; the same camera
    shows the same road and often the same parked cars. The holdout is split by
    INTERSECTION, and among 500 seeded candidate splits the one whose class mix
    best matches the whole batch is kept.
  * base val contains consecutive video frames of base train (DJI_0005-0175 in
    val, -0174 in train). Val frames within MEAN_DIFF_DUP of any train frame are
    dropped from test/, so it does not reward memorising a vehicle.
  * the base TEST split is 108 byte-identical copies of train images. It is not
    used at all.

Two further corrections:
  * contradictory subtype pairs (two boxes, IoU >= 0.85, different subtypes) are
    dropped; agreeing pairs keep one box. Vehicle+subtype pairs keep the subtype.
  * scale gap: base subtype crops have a median long side of 200px+, the Kaggle
    ones 66px. Half the base TRAIN crops over 96px are downsampled to a random
    40-110px long side, so the classifier sees base vehicles at the resolution
    it will be deployed at. Val and test are never degraded.

Usage:
    python scripts/training/build_type_crops.py            # build images/type_cls
    python scripts/training/build_type_crops.py --stats    # counts only, write nothing
"""

import argparse
import csv
import json
import random
import re
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from _crops import MIN_TYPE_PX, PAD, TYPE_CLASSES, square_crop
from _paths import (CLASS_NAMES, KAGGLE, REPO, TYPE_CLS, VTD_TRAIN, VTD_VAL,
                    labels_for)
from labeling.build_crops import iou_matrix, read_boxes

REVIEW   = REPO / "Labeling" / "review"
MANIFEST = REVIEW / "manifest.json"
DB       = REVIEW / "decisions.sqlite"

HOLDOUT_FRAC  = 0.20
SPLIT_SEEDS   = 500
PAIR_IOU      = 0.85        # same threshold as tools/find_duplicate_boxes.py
MEAN_DIFF_DUP = 6.0         # /255 on 32x32 gray; see the leakage note above
MAX_SIDE      = 256         # stored crops; the classifier trains at 224
DEGRADE_P     = 0.5
DEGRADE_OVER  = 96          # only base crops with a long side above this
DEGRADE_TO    = (40, 110)
SEED          = 0

DIRECTION = re.compile(r"-(North|South|East|West)$", re.IGNORECASE)


def camera_of(stem):
    """'US-41-at-Delany-North-5_jpg.rf.<hash>' -> 'US-41-at-Delany-North'."""
    return re.sub(r"-\d+$", "", re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", "", stem))


def intersection_of(stem):
    return DIRECTION.sub("", camera_of(stem))


def latest_decisions():
    """crop_id -> (class_id, n_decisions, changed). Mirrors export_labels.py."""
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT crop_id, class_id FROM decisions "
                       "WHERE active = 1 ORDER BY row_id").fetchall()
    con.close()
    hist = defaultdict(list)
    for crop_id, cls in rows:
        hist[crop_id].append(cls)
    return {c: (h[-1], len(h), len(set(h)) > 1) for c, h in hist.items()}


def kaggle_records():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["classes"] == CLASS_NAMES, "manifest schema != configs/vehicle_7class.yaml"
    decisions = latest_decisions()
    out = []
    for c in manifest["crops"]:
        if c["id"] not in decisions:
            continue                              # undecided: no human label
        cls, n, changed = decisions[c["id"]]
        name = CLASS_NAMES[cls]
        if name not in TYPE_CLASSES:
            continue                              # 'Vehicle': the umbrella
        out.append({
            "source": "kaggle", "frame": str(KAGGLE / c["split"] / "images" / c["image"]),
            "box_norm": c["box_norm"], "size_px": c["size_px"], "cls": name,
            "group": intersection_of(c["stem"]), "n_decisions": n,
            "changed": int(changed), "prior": CLASS_NAMES[c["prior"]],
        })
    return out


def split_kaggle(recs):
    """Group split by intersection; keep the seed whose val class mix is closest."""
    by_group = defaultdict(list)
    for r in recs:
        by_group[r["group"]].append(r)
    groups = sorted(by_group)
    total = Counter(r["cls"] for r in recs)
    target = np.array([total[k] / len(recs) for k in TYPE_CLASSES])

    best = None
    for seed in range(SPLIT_SEEDS):
        order = groups[:]
        random.Random(seed).shuffle(order)
        val, n = set(), 0
        for g in order:
            if n >= HOLDOUT_FRAC * len(recs):
                break
            val.add(g)
            n += len(by_group[g])
        cnt = Counter(r["cls"] for g in val for r in by_group[g])
        dist = np.array([cnt[k] / n for k in TYPE_CLASSES])
        score = np.abs(dist - target).sum() + abs(n / len(recs) - HOLDOUT_FRAC)
        if best is None or score < best[0]:
            best = (score, seed, val)

    _, seed, val = best
    for r in recs:
        r["split"] = "val" if r["group"] in val else "train"
    return seed, len(val), len(groups)


def base_records(img_dir, split):
    """Subtype boxes >= MIN_TYPE_PX from one base split, contradictory pairs removed."""
    lbl_dir = labels_for(img_dir)
    out, n_contra, n_agree = [], 0, 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = next((img_dir / f"{lbl.stem}{e}" for e in (".jpg", ".jpeg", ".png")
                    if (img_dir / f"{lbl.stem}{e}").exists()), None)
        if img is None:
            continue
        w, h = Image.open(img).size
        cls_of = {i: int(line.split()[0])
                  for i, line in enumerate(lbl.read_text().splitlines()) if line.split()}
        boxes = [(i, x1, y1, x2, y2) for i, x1, y1, x2, y2 in read_boxes(lbl, w, h)
                 if CLASS_NAMES[cls_of[i]] in TYPE_CLASSES
                 and max(x2 - x1, y2 - y1) >= MIN_TYPE_PX]

        drop = set()
        if len(boxes) > 1:
            m = iou_matrix([b[1:] for b in boxes], [b[1:] for b in boxes])
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    if m[a, b] < PAIR_IOU:
                        continue
                    if cls_of[boxes[a][0]] != cls_of[boxes[b][0]]:
                        drop |= {a, b}
                        n_contra += 1
                    else:
                        drop.add(b)
                        n_agree += 1

        for k, (i, x1, y1, x2, y2) in enumerate(boxes):
            if k in drop:
                continue
            out.append({
                "source": f"base_{split}", "frame": str(img),
                "box_norm": [x1 / w, y1 / h, x2 / w, y2 / h],
                "size_px": round(max(x2 - x1, y2 - y1), 1),
                "cls": CLASS_NAMES[cls_of[i]], "group": camera_of(img.stem),
                "n_decisions": "", "changed": "", "prior": "",
                "split": "train" if split == "train" else "test",
            })
    return out, n_contra, n_agree


def near_duplicate_val_frames():
    """Base val frames that are near-copies of a base train frame (video neighbours)."""
    def vecs(d):
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        return paths, np.stack([np.asarray(Image.open(p).convert("L").resize((32, 32)),
                                           np.float32).ravel() for p in paths])
    _, tm = vecs(VTD_TRAIN)
    vp, vm = vecs(VTD_VAL)
    nearest = np.stack([np.abs(v - tm).mean(1).min() for v in vm])
    return {str(p) for p, d in zip(vp, nearest) if d < MEAN_DIFF_DUP}


def table(recs, title):
    print(f"\n  {title}")
    splits = ("train", "val", "test")
    print(f"    {'':14s}" + "".join(f"{s:>8s}" for s in splits))
    for k in TYPE_CLASSES:
        print(f"    {k:14s}" + "".join(
            f"{sum(r['split'] == s and r['cls'] == k for r in recs):8d}" for s in splits))
    print(f"    {'total':14s}" + "".join(
        f"{sum(r['split'] == s for r in recs):8d}" for s in splits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="counts only, write nothing")
    args = ap.parse_args()

    kag = kaggle_records()
    seed, n_val_groups, n_groups = split_kaggle(kag)
    print(f"Kaggle decisions : {len(kag)} typed crops, {n_groups} intersections; "
          f"holdout = {n_val_groups} intersections (split seed {seed})")

    base_tr, c1, a1 = base_records(VTD_TRAIN, "train")
    base_va, c2, a2 = base_records(VTD_VAL, "val")
    print(f"base train       : {len(base_tr)} crops >= {MIN_TYPE_PX}px "
          f"({c1} contradictory pairs dropped, {a1} agreeing duplicates merged)")

    dups = near_duplicate_val_frames()
    before = len(base_va)
    base_va = [r for r in base_va if r["frame"] not in dups]
    print(f"base val -> test : {len(base_va)} crops ({c2} contradictory pairs dropped; "
          f"{before - len(base_va)} crops from {len(dups)} near-duplicate frames excluded)")

    recs = kag + base_tr + base_va
    table(kag, "Kaggle decisions by split")
    table(recs, "ALL sources by split")
    if args.stats:
        return

    if TYPE_CLS.exists():
        shutil.rmtree(TYPE_CLS)
    # Every split gets every class folder, even empty. ImageFolder numbers classes
    # by the folders it finds, so val/ without Motorcycle/ would shift SUV from
    # index 2 to 1 and silently score every validation crop against the wrong class.
    for s in ("train", "val", "test"):
        for k in TYPE_CLASSES:
            (TYPE_CLS / s / k).mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    n_degraded, frames = 0, {}
    for n, r in enumerate(recs):
        if r["frame"] not in frames:
            frames = {r["frame"]: Image.open(r["frame"]).convert("RGB")}  # 1-frame cache
        im = frames[r["frame"]]
        bx = [r["box_norm"][0] * im.width, r["box_norm"][1] * im.height,
              r["box_norm"][2] * im.width, r["box_norm"][3] * im.height]
        crop = square_crop(im, bx, max_side=MAX_SIDE)

        r["degraded"] = 0
        if (r["source"] == "base_train" and r["size_px"] > DEGRADE_OVER
                and rng.random() < DEGRADE_P):
            # the crop is square_crop()'s square, so the vehicle's long side
            # inside it is width / (1 + 2*PAD) whatever max_side did to it
            side = int(round(rng.uniform(*DEGRADE_TO) * (1 + 2 * PAD)))
            if side < crop.width:
                crop = crop.resize((side, side), Image.BILINEAR)
                r["degraded"], n_degraded = 1, n_degraded + 1

        stem = Path(r["frame"]).stem
        r["file"] = f"{r['split']}/{r['cls']}/{r['source']}_{stem}_{n}.jpg"
        out = TYPE_CLS / r["file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out, quality=92)

    with (TYPE_CLS / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["file", "split", "cls", "source", "frame", "box_norm", "size_px", "group",
                "n_decisions", "changed", "prior", "degraded"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in recs:
            wr.writerow({**r, "box_norm": " ".join(f"{v:.6f}" for v in r["box_norm"])})

    print(f"\n  {len(recs)} crops -> {TYPE_CLS}  ({n_degraded} base train crops degraded)")
    print(f"  manifest -> {TYPE_CLS / 'manifest.csv'}")
    print("\nNext:  .\\myenv\\Scripts\\python.exe scripts\\training\\train_type_classifier.py")


if __name__ == "__main__":
    main()
