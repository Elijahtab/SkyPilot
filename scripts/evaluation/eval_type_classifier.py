"""
Evaluate the Stage-2 type classifier -- the numbers top-1 accuracy hides.

    python scripts/evaluation/eval_type_classifier.py                 # type_cls_v1, val + test
    python scripts/evaluation/eval_type_classifier.py type_cls_v2 --split val

Reports, per split:
  * top-1, balanced accuracy and macro-F1 against the majority-class baseline
  * per-class precision / recall with SUPPORT -- read the support column before
    trusting any row; Bus and Van have 5 and 10 crops in the Kaggle holdout
  * the confusion matrix
  * THE GATE: SUV vs Standard Car head-to-head. Of crops a human called one of
    the two, how often does the model rank the right one higher? SEPARABLE when
    the 95% Wilson lower bound clears "always answer the more common of the
    two"; NOT SEPARABLE -- merge into one 'Car' class, a finding rather than a
    failure (docs/v4-integration-plan.md §3) -- when even the upper bound sits
    within 10 points of it; otherwise INCONCLUSIVE, which is what a small or
    lopsided split produces.
  * confidence -> coverage: how many crops clear each threshold, and how accurate
    those are. The pipeline answers 'Vehicle' below its threshold.
  * Kaggle holdout only: accuracy on crops the human changed their mind about vs
    crops decided once, and v4's own class output as a baseline.

Writes <run>/eval_<split>.json and an error contact sheet <run>/eval_<split>_errors.jpg.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from _crops import TYPE_CLASSES
from _paths import TYPE_CLS, TYPE_RUNS, require

K = len(TYPE_CLASSES)
CHUNK = 64                      # explicit batches: a list source sets bs=len(list)
THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
SIZE_BUCKETS = ((48, 64), (64, 96), (96, 1e9))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def predict(model, rows):
    """(N, K) probabilities in TYPE_CLASSES order."""
    order = [list(model.names.values()).index(k) for k in TYPE_CLASSES]
    out = []
    for i in range(0, len(rows), CHUNK):
        paths = [str(TYPE_CLS / r["file"]) for r in rows[i:i + CHUNK]]
        for res in model.predict(paths, imgsz=224, verbose=False,
                                 device=0 if torch.cuda.is_available() else "cpu"):
            out.append(res.probs.data.cpu().numpy()[order])
    return np.stack(out)


def report(rows, probs, split, run_dir):
    gt = np.array([TYPE_CLASSES.index(r["cls"]) for r in rows])
    pred = probs.argmax(1)
    conf = probs.max(1)
    ok = pred == gt
    n = len(rows)
    summary = {"split": split, "n": n}

    cm = np.zeros((K, K), int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(K), where=support > 0)
    colsum = cm.sum(0)
    precision = np.divide(np.diag(cm), colsum, out=np.zeros(K), where=colsum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(K), where=(precision + recall) > 0)
    present = support > 0
    majority = support.max() / n

    lo, hi = wilson(ok.sum(), n)
    print(f"\n{'=' * 78}\n{split}: {n} crops\n{'=' * 78}")
    print(f"  top-1 accuracy    {ok.mean():.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  majority baseline {majority:.3f}   (always '{TYPE_CLASSES[support.argmax()]}')")
    print(f"  balanced accuracy {recall[present].mean():.3f}   macro-F1 {f1[present].mean():.3f}"
          f"   (over the {present.sum()} classes present)")
    summary.update(top1=float(ok.mean()), top1_ci=[lo, hi], majority=float(majority),
                   balanced_acc=float(recall[present].mean()), macro_f1=float(f1[present].mean()))

    print(f"\n  {'class':14s}{'support':>8s}{'precision':>11s}{'recall':>8s}{'F1':>7s}")
    for k in range(K):
        print(f"  {TYPE_CLASSES[k]:14s}{support[k]:8d}{precision[k]:11.3f}{recall[k]:8.3f}{f1[k]:7.3f}"
              + ("   <- too few to read" if 0 < support[k] < 20 else ""))
    summary["per_class"] = {TYPE_CLASSES[k]: dict(support=int(support[k]), precision=float(precision[k]),
                                                  recall=float(recall[k]), f1=float(f1[k])) for k in range(K)}

    print("\n  confusion  rows = HUMAN label, cols = MODEL, counts")
    print(" " * 16 + "".join(f"{c[:9]:>10s}" for c in TYPE_CLASSES))
    for k in range(K):
        if support[k]:
            print(f"  {TYPE_CLASSES[k]:14s}" + "".join(f"{v:10d}" for v in cm[k]))
    summary["confusion"] = cm.tolist()

    # ── the SUV / Standard Car gate ──────────────────────────
    s, c = TYPE_CLASSES.index("SUV"), TYPE_CLASSES.index("Standard Car")
    pair = np.isin(gt, [s, c])
    if pair.sum():
        h2h = np.where(probs[pair, s] >= probs[pair, c], s, c) == gt[pair]
        base = max((gt[pair] == s).mean(), (gt[pair] == c).mean())
        plo, phi = wilson(h2h.sum(), pair.sum())
        # three answers, not two: a small or lopsided sample can fail to prove
        # separability without being evidence against it
        if plo > base:
            verdict = "SEPARABLE — keep SUV and Standard Car"
        elif phi < base + 0.10:
            verdict = "NOT SEPARABLE — merge into 'Car'"
        else:
            verdict = "INCONCLUSIVE — too few crops to decide either way"
        print(f"\n  GATE  SUV vs Standard Car head-to-head, {pair.sum()} crops")
        print(f"    accuracy {h2h.mean():.3f}  95% CI [{plo:.3f}, {phi:.3f}]  vs baseline {base:.3f}")
        print(f"    any-class accuracy on those crops {ok[pair].mean():.3f}")
        print(f"    -> {verdict}")
        summary["gate"] = dict(n=int(pair.sum()), acc=float(h2h.mean()), ci=[plo, phi],
                               baseline=float(base), verdict=verdict)

    print("\n  confidence -> coverage  (below the threshold the pipeline answers 'Vehicle')")
    print(f"    {'conf >=':>8s}{'coverage':>10s}{'accuracy':>10s}")
    summary["coverage"] = []
    for t in THRESHOLDS:
        m = conf >= t
        acc = ok[m].mean() if m.any() else float("nan")
        print(f"    {t:8.2f}{m.mean():10.3f}{acc:10.3f}")
        summary["coverage"].append(dict(threshold=t, coverage=float(m.mean()), accuracy=float(acc)))

    size = np.array([float(r["size_px"]) for r in rows])
    print("\n  by box long side (native px)")
    for a, b in SIZE_BUCKETS:
        m = (size >= a) & (size < b)
        if m.any():
            print(f"    {a:>3d}-{'' if b > 1e8 else int(b):<4}  n={m.sum():4d}  accuracy {ok[m].mean():.3f}")

    if rows[0]["source"] == "kaggle":
        changed = np.array([r["changed"] == "1" for r in rows])
        print("\n  human consistency")
        print(f"    decided once          n={(~changed).sum():4d}  model accuracy {ok[~changed].mean():.3f}")
        if changed.any():
            print(f"    human changed answer  n={changed.sum():4d}  model accuracy {ok[changed].mean():.3f}")
        prior_ok = np.array([r["prior"] == r["cls"] for r in rows])
        prior_veh = np.mean([r["prior"] == "Vehicle" for r in rows])
        print(f"    v4's own class output on these crops: accuracy {prior_ok.mean():.3f} "
              f"({prior_veh:.0%} of the time it says 'Vehicle')")
        summary.update(acc_changed=float(ok[changed].mean()) if changed.any() else None,
                       acc_stable=float(ok[~changed].mean()), v4_prior_acc=float(prior_ok.mean()))

    errors_sheet(rows, gt, pred, conf, run_dir / f"eval_{split}_errors.jpg")
    (run_dir / f"eval_{split}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def errors_sheet(rows, gt, pred, conf, out, tile=112, cols=10, limit=60):
    wrong = [i for i in np.argsort(-conf) if pred[i] != gt[i]][:limit]
    if not wrong:
        return
    sheet = Image.new("RGB", (cols * tile, ((len(wrong) + cols - 1) // cols) * (tile + 26)), "white")
    d = ImageDraw.Draw(sheet)
    for j, i in enumerate(wrong):
        x, y = (j % cols) * tile, (j // cols) * (tile + 26)
        sheet.paste(Image.open(TYPE_CLS / rows[i]["file"]).convert("RGB").resize((tile, tile)), (x, y))
        d.text((x + 2, y + tile), f"H:{TYPE_CLASSES[gt[i]][:10]}", fill=(0, 110, 0))
        d.text((x + 2, y + tile + 12), f"M:{TYPE_CLASSES[pred[i]][:10]} {conf[i]:.2f}", fill=(170, 0, 0))
    sheet.save(out, quality=90)
    print(f"\n  {len(wrong)} most confident errors -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="type_cls_v1")
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    args = ap.parse_args()

    run_dir = TYPE_RUNS / args.run
    weights = run_dir / "weights" / "best.pt"
    require(weights, TYPE_CLS / "manifest.csv")
    model = YOLO(str(weights))
    if sorted(model.names.values()) != sorted(TYPE_CLASSES):
        sys.exit(f"[ERR] model classes {list(model.names.values())} != {TYPE_CLASSES}")

    rows = list(csv.DictReader((TYPE_CLS / "manifest.csv").open(encoding="utf-8")))
    splits = ("val", "test") if args.split == "both" else (args.split,)
    print(f"model: {weights}")
    print("val  = Kaggle intersection holdout (traffic-cam scale)  <- the number that matters")
    print("test = base val split, near-duplicate frames excluded (high-res domain)")
    for split in splits:
        sub = [r for r in rows if r["split"] == split]
        report(sub, predict(model, sub), split, run_dir)


if __name__ == "__main__":
    main()
