"""
Compare two or more trained runs on the same split.

    python scripts/evaluation/compare_models.py                       # v4 vs v5
    python scripts/evaluation/compare_models.py v4 v5 v7 --split val

⚠ On the `test` split, 'Standard Car' has 0 GT boxes and 'SUV' has 3, so their
  per-class AP is noise that still gets averaged into the headline mAP. The
  per-class support column below makes that visible — read it before trusting
  a small mAP delta.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from _paths import (CLASS_NAMES, VTD_VAL, VTD_TEST, best_weights, build_data_yaml,
                    labels_for)

DEFAULT_RUNS = ["v4", "v5"]


def resolve(tag: str) -> Path:
    """Accept 'v4' or a full run name."""
    name = tag if tag.startswith("Vehicle_type_detection") else f"Vehicle_type_detection_{tag}"
    return best_weights(name)


def gt_support(split: str) -> dict:
    """
    GT box count per class, read straight from the label files.

    Ultralytics keeps this on the validator (`nt_per_class`), not on the metrics
    object that `model.val()` returns, and the attribute has moved between
    versions — counting the labels ourselves is version-proof.
    """
    img_dir = VTD_TEST if split == "test" else VTD_VAL
    counts = Counter()
    for txt in labels_for(img_dir).glob("*.txt"):
        for line in txt.read_text().splitlines():
            p = line.split()
            if p:
                counts[int(p[0])] += 1
    return {CLASS_NAMES[i]: counts.get(i, 0) for i in range(len(CLASS_NAMES))}


def evaluate(path: Path, tag: str, split: str):
    print(f"\n--- Evaluating {tag} ---")
    model = YOLO(str(path))
    m = model.val(data=str(build_data_yaml("compare", val=VTD_VAL, test=VTD_TEST)),
                  split=split, imgsz=640, batch=16, plots=False)
    per_class = {model.names[c]: m.box.ap50[i] for i, c in enumerate(m.box.ap_class_index)}
    return {"tag": tag, "mAP50": m.box.map50, "mAP50-95": m.box.map,
            "P": m.box.mp, "R": m.box.mr, "per_class": per_class}


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = "test"
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]
    tags = argv or DEFAULT_RUNS

    results = []
    for t in tags:
        p = resolve(t)
        if p.exists():
            results.append(evaluate(p, t, split))
        else:
            print(f"⚠ weights not found for '{t}': {p}")

    if not results:
        return

    print(f"\n=== Model Comparison ({split} split) ===")
    header = f"{'Model':<10} | {'mAP50':<10} | {'mAP50-95':<10} | {'Precision':<10} | {'Recall':<10}"
    print("\n" + header + "\n" + "-" * len(header))
    for r in results:
        print(f"{r['tag']:<10} | {r['mAP50']:<10.4f} | {r['mAP50-95']:<10.4f} "
              f"| {r['P']:<10.4f} | {r['R']:<10.4f}")

    print(f"\n=== Per-class AP50 (n = GT boxes in this split) ===")
    print(f"{'class':<16s}" + "".join(f"{r['tag']:>12s}" for r in results) + f"{'n':>8s}")
    support = gt_support(split)
    for c in CLASS_NAMES:
        n = support[c]
        cells = "".join(f"{r['per_class'][c]:>12.3f}" if c in r["per_class"]
                        else f"{'--':>12s}" for r in results)
        flag = "  ← too few GT to trust" if 0 < n < 20 else ("  ← ABSENT" if n == 0 else "")
        print(f"{c:<16s}{cells}{n:>8d}{flag}")

    if len(results) == 2:
        print(f"\n=== Improvement ({results[1]['tag']} - {results[0]['tag']}) ===")
        for k in ("mAP50", "mAP50-95", "P", "R"):
            diff = results[1][k] - results[0][k]
            print(f"{k:10s}: {'+' if diff >= 0 else ''}{diff:.4f}")


if __name__ == "__main__":
    main()
