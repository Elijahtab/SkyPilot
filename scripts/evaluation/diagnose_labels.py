"""
Label-quality gate. Run this on any auto-labeled batch BEFORE merging it into
the training pool.

It separates two failure modes that a plain mAP number conflates:

  (A) DETECTION / COMPLETENESS — class-agnostic, so class names are ignored.
      low recall    → the reference model cannot find labeled objects
                      (out-of-domain imagery, or bad boxes)
      low precision → the model finds real objects the labels OMIT
                      (under-annotation — actively poisons training)

  (B) TAXONOMY — computed only over class-agnostically matched boxes, so
      detection failures cannot contaminate it. Low agreement means the two
      labelers describe the same objects with different vocabularies.

Usage:
    python scripts/evaluation/diagnose_labels.py <images_dir> [<labels_dir>]
    python scripts/evaluation/diagnose_labels.py --control      # base val baseline

Healthy reference (v4 on the base val split):
    class-agnostic recall 0.804 | precision 0.813 | class agreement 0.934
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from _paths import CLASS_NAMES, NC, POOL_IMG, VTD_VAL, best_weights, labels_for

REFERENCE_RUN = "Vehicle_type_detection_v4"     # strongest checkpoint
CONF = 0.25
IOU_MATCH = 0.5


def load_gt(txt: Path, w: int, h: int):
    """Return (boxes_xyxy, classes). Handles bbox(5) and polygon rows."""
    boxes, cls = [], []
    if not txt.exists():
        return np.zeros((0, 4)), np.zeros((0,), int)
    for line in txt.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        v = np.array(p[1:], float)
        if len(p) == 5:
            xc, yc, bw, bh = v
            x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
        else:                                   # segmentation polygon → its bbox
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
        cls.append(int(p[0]))
    return np.array(boxes, float).reshape(-1, 4), np.array(cls, int)


def diagnose(img_dir: Path, lbl_dir: Path, weights: Path):
    model = YOLO(str(weights))
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not imgs:
        sys.exit(f"[ERR] no images in {img_dir}")

    n_gt = n_pred = n_match = 0
    agree = np.zeros((NC, NC), int)
    matched_scale, missed_scale = [], []

    for r in model.predict(source=[str(p) for p in imgs], conf=CONF, imgsz=640,
                           stream=True, verbose=False,
                           device=0 if torch.cuda.is_available() else "cpu"):
        h, w = r.orig_shape
        gtb, gtc = load_gt(lbl_dir / (Path(r.path).stem + ".txt"), w, h)
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        pc = (r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None
              else np.zeros((0,), int))

        n_gt += len(gtb)
        n_pred += len(pb)
        scale = (np.sqrt(((gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])) / (w * h))
                 if len(gtb) else np.zeros(0))

        if len(gtb) == 0 or len(pb) == 0:
            missed_scale.extend(scale)
            continue

        iou = box_iou(torch.tensor(gtb, dtype=torch.float32),
                      torch.tensor(pb, dtype=torch.float32)).numpy()

        used_g, used_p = set(), set()
        for g, p in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
            if iou[g, p] < IOU_MATCH:
                break
            if g in used_g or p in used_p:
                continue
            used_g.add(int(g)); used_p.add(int(p))
            n_match += 1
            if gtc[g] < NC and pc[p] < NC:
                agree[gtc[g], pc[p]] += 1
            matched_scale.append(scale[g])
        missed_scale.extend(scale[g] for g in range(len(gtb)) if g not in used_g)

    # ── report ──────────────────────────────────────────────
    recall = n_match / n_gt if n_gt else 0.0
    precision = n_match / n_pred if n_pred else 0.0
    total = agree.sum()
    agreement = np.trace(agree) / total if total else 0.0

    print(f"\n{'=' * 78}\n{img_dir}  ({len(imgs)} images)\n{'=' * 78}")
    print(f"reference model: {weights.parent.parent.name}   conf>={CONF} IoU>={IOU_MATCH}\n")

    print("(A) DETECTION / COMPLETENESS  — class labels ignored")
    print(f"    GT boxes {n_gt}   predicted {n_pred}   matched {n_match}")
    print(f"    class-agnostic recall    : {recall:.3f}"
          f"{'   ⚠ imagery may be out-of-domain, or boxes are bad' if recall < 0.6 else ''}")
    print(f"    class-agnostic precision : {precision:.3f}"
          f"{'   ⚠ UNDER-ANNOTATED: model finds real objects the labels omit' if precision < 0.6 else ''}")

    print(f"\n(B) TAXONOMY — over the {total} matched boxes only")
    print(f"    class agreement : {agreement:.3f}"
          f"{'   ⚠ vocabularies conflict with configs/vehicle_7class.yaml' if agreement < 0.7 else ''}")
    print("\n    rows = GROUND TRUTH, cols = model PREDICTION, % of GT row\n")
    print(" " * 15 + "".join(f"{n[:11]:>12s}" for n in CLASS_NAMES) + f"{'n':>9s}")
    for i in range(NC):
        t = agree[i].sum()
        cells = "".join(f"{(100 * agree[i, j] / t):>11.1f}%" if t else f"{'-':>12s}"
                        for j in range(NC))
        print(f"{CLASS_NAMES[i]:<14s}|{cells}{t:>9d}")

    def pct(a):
        a = np.array(a)
        return (f"p10={np.percentile(a, 10):.3f} med={np.median(a):.3f} "
                f"p90={np.percentile(a, 90):.3f}") if len(a) else "n/a"

    print(f"\n(C) OBJECT SCALE  sqrt(box area / image area)")
    print(f"    matched : {pct(matched_scale)}")
    print(f"    missed  : {pct(missed_scale)}")
    print("    (similar matched/missed scale ⇒ misses are NOT a small-object problem)")

    return recall, precision, agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="?", default=str(POOL_IMG))
    ap.add_argument("labels", nargs="?", default=None)
    ap.add_argument("--control", action="store_true",
                    help="diagnose the base val split instead (healthy baseline)")
    ap.add_argument("--run", default=REFERENCE_RUN)
    args = ap.parse_args()

    img_dir = VTD_VAL if args.control else Path(args.images)
    lbl_dir = Path(args.labels) if args.labels else labels_for(img_dir)

    weights = best_weights(args.run)
    if not weights.exists():
        sys.exit(f"[ERR] reference weights not found: {weights}")

    diagnose(img_dir, lbl_dir, weights)


if __name__ == "__main__":
    main()
