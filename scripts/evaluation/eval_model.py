"""
Evaluate the most recently modified checkpoint against the vehicle-type val split.

    python scripts/evaluation/eval_model.py                 # newest run overall
    python scripts/evaluation/eval_model.py <run_name>      # a specific run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from _paths import (CLASS_NAMES, VTD_RUNS, VTD_VAL, VTD_TEST, best_weights,
                    build_data_yaml)


def newest_checkpoint() -> Path:
    """Newest best.pt under the runs tree, falling back to last.pt."""
    for pattern in ("*/weights/best.pt", "*/weights/last.pt"):
        found = sorted(VTD_RUNS.glob(pattern), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    raise FileNotFoundError(f"No best.pt or last.pt found under {VTD_RUNS}")


def main():
    weight_path = best_weights(sys.argv[1]) if len(sys.argv) > 1 else newest_checkpoint()
    if not weight_path.exists():
        raise FileNotFoundError(weight_path)

    print(f"Using weights: {weight_path}")
    model = YOLO(str(weight_path))

    metrics = model.val(
        data=str(build_data_yaml("eval", val=VTD_VAL, test=VTD_TEST)),
        split="val",
        imgsz=640,
        batch=16,
    )

    print(f"\nmAP50: {metrics.box.map50:.4f}   mAP50-95: {metrics.box.map:.4f}")
    print(f"{'class':<16s}{'AP50':>9s}{'AP50-95':>10s}")
    for idx, c in enumerate(metrics.box.ap_class_index):
        print(f"{model.names[c]:<16s}{metrics.box.ap50[idx]:>9.3f}{metrics.box.ap[idx]:>10.3f}")

    absent = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))
              if i not in metrics.box.ap_class_index]
    if absent:
        print(f"\n⚠ No GT boxes in this split, excluded from mAP: {', '.join(absent)}")

    # The checkpoint carries its own class names; flag a mismatch against configs/.
    ckpt_names = [model.names[i] for i in sorted(model.names)]
    if ckpt_names != CLASS_NAMES:
        print(f"\n⚠ Checkpoint class names differ from configs/vehicle_7class.yaml")
        print(f"    checkpoint: {ckpt_names}")
        print(f"    config    : {CLASS_NAMES}")
        print(f"  Class IDs still line up; run scripts/tools/retag_checkpoint.py to sync.")


if __name__ == "__main__":
    main()
