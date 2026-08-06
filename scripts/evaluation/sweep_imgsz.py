"""
Sweep inference resolution for a trained model and report accuracy vs speed.

    python scripts/evaluation/sweep_imgsz.py
    python scripts/evaluation/sweep_imgsz.py --run Vehicle_type_detection_v4 --split val

⚠ INTERPRETATION. This changes resolution at INFERENCE only. A model trained at
  640 has learned scale priors for 640; evaluating it at 1280 can help (small
  objects get more pixels) or hurt (object scales drift out of the trained
  distribution). A gain here is evidence that retraining at higher imgsz is worth
  trying — it is NOT the same result as retraining. A flat or negative curve is
  the more decisive outcome: it rules resolution out cheaply.

Reports class-agnostic recall alongside mAP, because "did we find the vehicle at
all" is the metric resolution is supposed to move. It comes free from the
confusion matrix: the background row counts GT boxes with no matching detection
of any class.

Speed is reported too — this detector feeds a real-time drone tracker, so a
resolution that wins on mAP but costs 4x latency may not be usable.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.models.yolo.detect import DetectionValidator

from _paths import (CLASS_NAMES, NC, VTD_VAL, VTD_TEST, best_weights,
                    build_data_yaml, labels_for)


def gt_support(split: str) -> dict:
    """GT box count per class, read straight from the label files."""
    counts = Counter()
    for txt in labels_for(VTD_TEST if split == "test" else VTD_VAL).glob("*.txt"):
        for line in txt.read_text().splitlines():
            p = line.split()
            if p:
                counts[int(p[0])] += 1
    return {CLASS_NAMES[i]: counts.get(i, 0) for i in range(NC)}

# batch scaled down as resolution climbs — 8GB card
SIZES = [(640, 16), (800, 12), (960, 8), (1280, 4)]


MIN_SUPPORT = 200          # a class needs this many GT boxes for its AP to mean much


def run_one(weights, data, split, imgsz, batch, out):
    v = DetectionValidator(args=dict(
        model=str(weights), data=str(data), split=split,
        imgsz=imgsz, batch=batch,
        # plots=True is REQUIRED: ultralytics only calls
        # confusion_matrix.process_batch() when it is set, and the background row
        # of that matrix is where class-agnostic recall comes from. With
        # plots=False the matrix stays all zeros and the metric silently reads 0.
        plots=True, verbose=False,
        project=str(out), name=f"imgsz{imgsz}", exist_ok=True,
    ))
    v()
    b = v.metrics.box
    m = v.confusion_matrix.matrix

    # class-agnostic: background row = GT boxes matched by nothing
    gt_total = m[:, :NC].sum()
    missed = m[NC, :NC].sum()
    agnostic_recall = 1 - (missed / gt_total) if gt_total else 0.0

    speed = sum(v.speed.values())     # ms/image, pre + inference + post

    return {
        "imgsz": imgsz,
        "mAP50": b.map50, "mAP50-95": b.map, "P": b.mp, "R": b.mr,
        "agnostic_recall": agnostic_recall,
        "ms": speed,
        "per_class": {CLASS_NAMES[c]: b.ap50[i] for i, c in enumerate(b.ap_class_index)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="Vehicle_type_detection_v4")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    weights = best_weights(args.run)
    if not weights.exists():
        sys.exit(f"[ERR] not found: {weights}")

    data = build_data_yaml("sweep", val=VTD_VAL, test=VTD_TEST)
    out = Path(__file__).resolve().parents[2] / "runs_sweep" / args.run

    print(f"model : {weights}")
    print(f"split : {args.split}")
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"NOTE  : this model was TRAINED at imgsz 640.\n")

    rows = []
    for imgsz, batch in SIZES:
        print(f"--- imgsz {imgsz} (batch {batch}) ---", flush=True)
        try:
            rows.append(run_one(weights, data, args.split, imgsz, batch, out))
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at batch {batch}, retrying at {max(1, batch // 2)}")
            torch.cuda.empty_cache()
            rows.append(run_one(weights, data, args.split, imgsz, max(1, batch // 2), out))
        torch.cuda.empty_cache()

    base = rows[0]
    print(f"\n{'=' * 88}\nRESOLUTION SWEEP — {args.run}, {args.split} split\n{'=' * 88}")
    print(f"{'imgsz':>7s}{'mAP50':>9s}{'mAP50-95':>10s}{'d50-95':>9s}"
          f"{'agn.recall':>12s}{'P':>8s}{'R':>8s}{'ms/img':>9s}{'speedup':>9s}")
    for r in rows:
        d = r["mAP50-95"] - base["mAP50-95"]
        print(f"{r['imgsz']:>7d}{r['mAP50']:>9.4f}{r['mAP50-95']:>10.4f}"
              f"{d:>+9.4f}{r['agnostic_recall']:>12.3f}{r['P']:>8.3f}{r['R']:>8.3f}"
              f"{r['ms']:>9.1f}{base['ms'] / r['ms']:>8.2f}x")

    support = gt_support(args.split)
    solid = [c for c in CLASS_NAMES if support[c] >= MIN_SUPPORT]
    thin = [c for c in CLASS_NAMES if support[c] < MIN_SUPPORT]

    print(f"\n--- per-class AP50  (n = GT boxes) ---")
    print(f"{'class':<16s}" + "".join(f"{r['imgsz']:>10d}" for r in rows) + f"{'n':>8s}")
    for c in CLASS_NAMES:
        cells = "".join(f"{r['per_class'][c]:>10.3f}" if c in r["per_class"]
                        else f"{'--':>10s}" for r in rows)
        flag = "" if support[c] >= MIN_SUPPORT else "  <- thin, AP is noisy"
        print(f"{c:<16s}{cells}{support[c]:>8d}{flag}")

    # mAP averages every class equally, so a handful of tiny classes can invert
    # the headline. Split it out rather than letting it mislead.
    def mean_ap(r, group):
        vals = [r["per_class"][c] for c in group if c in r["per_class"]]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n--- mean AP50 split by class support (threshold {MIN_SUPPORT} GT boxes) ---")
    print(f"  well-supported: {', '.join(solid)}")
    print(f"  thin          : {', '.join(thin)}")
    print(f"\n{'imgsz':>7s}{'all classes':>14s}{'well-supported':>17s}{'thin':>14s}")
    for r in rows:
        print(f"{r['imgsz']:>7d}{mean_ap(r, CLASS_NAMES):>14.4f}"
              f"{mean_ap(r, solid):>13.4f} {mean_ap(r, solid) - mean_ap(base, solid):>+.4f}"
              f"{mean_ap(r, thin):>9.4f} {mean_ap(r, thin) - mean_ap(base, thin):>+.4f}")

    best = max(rows, key=lambda r: r["mAP50-95"])
    best_solid = max(rows, key=lambda r: mean_ap(r, solid))
    print(f"\nBest overall mAP50-95 : imgsz {best['imgsz']} @ {best['mAP50-95']:.4f} "
          f"({best['mAP50-95'] - base['mAP50-95']:+.4f} vs 640), {best['ms']:.1f} ms/img")
    print(f"Best well-supported   : imgsz {best_solid['imgsz']} @ "
          f"{mean_ap(best_solid, solid):.4f} AP50 "
          f"({mean_ap(best_solid, solid) - mean_ap(base, solid):+.4f} vs 640), "
          f"{best_solid['ms']:.1f} ms/img ({base['ms'] / best_solid['ms']:.2f}x speed)")
    if best_solid["imgsz"] != base["imgsz"]:
        print("→ Resolution helps the classes that have enough data to measure.\n"
              "  This is inference-only; retrain at that imgsz to see the real ceiling.")
    else:
        print("→ 640 is best even on well-supported classes; resolution is not the bottleneck.")


if __name__ == "__main__":
    main()
