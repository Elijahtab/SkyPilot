"""
Explore vehicle colour naming (scripts/_color.py) on real crops.

There is no colour ground truth in this project, so this script does two jobs:

  1. render a fixed, seeded sample of crops as numbered contact sheets with the
     predicted colour beside each one, for eyeballing failure modes;
  2. score the predictions against a labels CSV once one exists
     (columns: file,color -- `file` as in images/type_cls/manifest.csv).

The sample is drawn from the type-classifier holdouts (Kaggle traffic cams and
base val), so every crop is >= 48px and its frame is on disk. Colour is computed
on the FULL frame, not the saved crop: the background ring needs real context.

Usage:
    python scripts/evaluation/explore_color.py                     # sheets + predictions
    python scripts/evaluation/explore_color.py --labels colors.csv # + accuracy
    python scripts/evaluation/explore_color.py --n 200 --seed 1
"""

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from _color import COLORS, name_color, white_balance_gains
from _paths import REPO, TYPE_CLS, require

OUT = REPO / "preds" / "color_explore"
SWATCH = {"white": (245, 245, 245), "gray": (128, 128, 128), "black": (15, 15, 15),
          "red": (200, 30, 30), "orange": (240, 140, 20), "yellow": (240, 220, 40),
          "green": (40, 150, 60), "blue": (40, 80, 200), "brown": (130, 90, 50)}


def sample(rows, n, seed):
    kag = [r for r in rows if r["split"] == "val"]
    base = [r for r in rows if r["split"] == "test"]
    rng = random.Random(seed)
    pick = rng.sample(kag, min(len(kag), n * 3 // 5)) + rng.sample(base, min(len(base), n - n * 3 // 5))
    return sorted(pick, key=lambda r: r["file"])


def sheets(rows, preds, per_sheet=50, cols=10, tile=128):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(rows), per_sheet):
        chunk = list(range(s, min(s + per_sheet, len(rows))))
        sheet = Image.new("RGB", (cols * tile, ((len(chunk) + cols - 1) // cols) * (tile + 16)), "white")
        d = ImageDraw.Draw(sheet)
        for j, i in enumerate(chunk):
            x, y = (j % cols) * tile, (j // cols) * (tile + 16)
            sheet.paste(Image.open(TYPE_CLS / rows[i]["file"]).convert("RGB").resize((tile, tile)), (x, y))
            color, conf = preds[i][0], preds[i][1]
            d.rectangle([x, y + tile, x + 14, y + tile + 14], fill=SWATCH.get(color, (255, 255, 255)),
                        outline="black")
            d.text((x + 18, y + tile + 2), f"#{i} {color} {conf:.2f}", fill="black")
        p = OUT / f"sheet_{s // per_sheet:02d}.jpg"
        sheet.save(p, quality=90)
        paths.append(p)
    return paths


def score(rows, preds, labels_csv):
    truth = {r["file"]: r["color"].strip().lower()
             for r in csv.DictReader(open(labels_csv, encoding="utf-8")) if r.get("color")}
    idx = [i for i, r in enumerate(rows) if r["file"] in truth and truth[r["file"]] in COLORS]
    if not idx:
        return print(f"[WARN] no sampled crop has a usable label in {labels_csv}")
    gt = [truth[rows[i]["file"]] for i in idx]
    pr = [preds[i][0] for i in idx]
    ok = np.array([g == p for g, p in zip(gt, pr)])
    print(f"\n  colour accuracy {ok.mean():.3f} on {len(idx)} labeled crops")
    print(f"\n  {'label':8s}{'n':>5s}{'recall':>8s}   most common wrong answers")
    for c in COLORS:
        m = [k for k, g in enumerate(gt) if g == c]
        if m:
            wrong = Counter(pr[k] for k in m if pr[k] != c).most_common(3)
            print(f"  {c:8s}{len(m):5d}{np.mean([pr[k] == c for k in m]):8.3f}   {wrong}")
    conf = np.array([preds[i][1] for i in idx])
    print(f"\n  {'conf >=':>8s}{'coverage':>10s}{'accuracy':>10s}")
    for t in (0.0, 0.3, 0.4, 0.5, 0.6):
        m = conf >= t
        print(f"  {t:8.2f}{m.mean():10.3f}{ok[m].mean() if m.any() else float('nan'):10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", help="CSV with file,color columns to score against")
    args = ap.parse_args()
    require(TYPE_CLS / "manifest.csv")

    rows = sample(list(csv.DictReader((TYPE_CLS / "manifest.csv").open(encoding="utf-8"))),
                  args.n, args.seed)
    preds, frames = [], {}
    for r in rows:
        if r["frame"] not in frames:
            f = np.asarray(Image.open(r["frame"]).convert("RGB"))
            frames = {r["frame"]: (f, white_balance_gains(f))}
        f, gains = frames[r["frame"]]
        b = [float(v) for v in r["box_norm"].split()]
        preds.append(name_color(f, (b[0] * f.shape[1], b[1] * f.shape[0],
                                    b[2] * f.shape[1], b[3] * f.shape[0]), gains))

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["idx", "file", "source", "type", "size_px", "color", "confidence", *COLORS])
        for i, (r, (c, conf, shares)) in enumerate(zip(rows, preds)):
            wr.writerow([i, r["file"], r["source"], r["cls"], r["size_px"], c, f"{conf:.3f}",
                         *[shares.get(k, 0) for k in COLORS]])

    print(f"  {len(rows)} crops  predicted: {dict(Counter(p[0] for p in preds).most_common())}")
    for p in sheets(rows, preds):
        print(f"  sheet -> {p}")
    print(f"  predictions -> {OUT / 'predictions.csv'}")
    if args.labels:
        score(rows, preds, args.labels)


if __name__ == "__main__":
    main()
