"""
Run inference with a fine-tuned model and apply the custom post-processing rule:

    • Keep non-Vehicle detections only when confidence >= THRESH
    • Suppress any generic 'Vehicle' box that overlaps a strong non-Vehicle
      detection (IoU >= IOU_THR)

The rule exists because class 1 'Vehicle' is a generic bucket that overlaps
SUV / Standard Car / Van by design, so the model often fires both on one object.

    python scripts/evaluation/predict_images.py
    python scripts/evaluation/predict_images.py --run v4 --source <dir> --out <dir>

Results are written to OUT_DIR with bounding boxes drawn.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from _paths import REPO, VEHICLE_ID, VTD_TEST, best_weights

# ────────────────────────────────────────────────────────────
# Defaults — override on the command line
# ────────────────────────────────────────────────────────────
DEFAULT_RUN = "Vehicle_type_detection_v5"

THRESH    = 0.55   # keep non-Vehicle only if conf >= THRESH
IOU_THR   = 0.10   # IoU threshold for "same object"
CONF_PRED = 0.25   # global detector threshold
IMG_SIZE  = 640    # must be <= training imgsz


def filter_boxes_single_img(boxes: Boxes) -> Boxes:
    """Apply confidence filter + generic-Vehicle suppression to one image."""
    if boxes is None or len(boxes) == 0:
        return boxes

    data = boxes.data.clone()          # [n, 6] (xyxy, conf, cls)
    cls, conf = data[:, 5].long(), data[:, 4]

    # 1) drop weak non-Vehicle detections outright
    keep = ~((cls != VEHICLE_ID) & (conf < THRESH))
    data, cls, conf = data[keep], cls[keep], conf[keep]

    if len(data) == 0:
        return Boxes(data, orig_shape=boxes.orig_shape)

    # 2) suppress generic boxes overlapping a strong specific detection
    generic_mask  = cls == VEHICLE_ID
    strong_specific = (cls != VEHICLE_ID) & (conf >= THRESH)

    if strong_specific.sum() == 0 or generic_mask.sum() == 0:
        return Boxes(data, orig_shape=boxes.orig_shape)

    iou = box_iou(data[generic_mask, :4], data[strong_specific, :4])
    suppress = (iou >= IOU_THR).any(dim=1)

    final_keep = torch.ones(len(data), dtype=torch.bool, device=data.device)
    final_keep[generic_mask.nonzero(as_tuple=False).flatten()[suppress]] = False

    return Boxes(data[final_keep], orig_shape=boxes.orig_shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",    default=DEFAULT_RUN, help="run name under Vehicle_type_detection/runs")
    ap.add_argument("--source", default=str(VTD_TEST), help="image folder or glob")
    ap.add_argument("--out",    default=None, help="output folder")
    args = ap.parse_args()

    weights = best_weights(args.run)
    if not weights.exists():
        raise FileNotFoundError(weights)

    out_dir = Path(args.out) if args.out else REPO / "preds" / args.run
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Weights :", weights)
    print("Source  :", args.source)
    print("Output  :", out_dir)
    print(f"Rule    : keep non-Vehicle only if conf >={THRESH}; "
          f"suppress generic Vehicle when IoU >={IOU_THR}\n")

    use_cuda = torch.cuda.is_available()
    model = YOLO(str(weights))

    for r in model.predict(source=args.source, conf=CONF_PRED, imgsz=IMG_SIZE,
                           save=False, stream=True,
                           device=0 if use_cuda else "cpu",
                           half=use_cuda):          # half is CUDA-only
        r.boxes = filter_boxes_single_img(r.boxes)
        cv2.imwrite(str(out_dir / Path(r.path).name), r.plot())
        print(f"{Path(r.path).name:25s} kept={len(r.boxes)}")

    print("\n✅ Done – check images in:", out_dir)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "6000000000")
    main()
