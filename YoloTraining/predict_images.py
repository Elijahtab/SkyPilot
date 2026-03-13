"""
 predict_images.py
 -----------------
 Run inference with your fine‑tuned YOLOv8 model **and** apply the custom
 post‑processing rule:
     • Keep non‑CAR detections only when confidence ≥ THRESH
     • Suppress any CAR box that overlaps a strong non‑CAR (IoU ≥ IOU_THR)

 Usage (PowerShell):
     python predict_images.py

 Results are written to `OUT_DIR` with bounding boxes drawn.
 Adjust the top‑of‑file constants as needed.
"""

from pathlib import Path
import shutil, cv2, torch, os
from torchvision.ops import box_iou
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# ────────────────────────────────────────────────────────────
# CONFIG — edit to fit your paths / dataset
# ────────────────────────────────────────────────────────────
WEIGHTS  = r"S:/GitHub/SkyPilot/Vehicle_type_detection/runs/Vehicle_type_detection_v4/weights/best.pt"
SOURCE   = r"S:/GitHub/SkyPilot/Vehicle_type_detection/images/images_test"  # folder or glob
OUT_DIR  = r"S:/GitHub/SkyPilot/preds_vehicle_v4"

CAR_ID   = 1       # 'CAR' index in [Bus, CAR, Motorcycle, …]
THRESH   = 0.55    # keep non‑CAR only if conf ≥ THRESH
IOU_THR  = 0.10    # IoU threshold for “same object”
CONF_PRED = 0.25   # global detector threshold
IMG_SIZE  = 640    # must be ≤ training imgsz

# ────────────────────────────────────────────────────────────


def filter_boxes_single_img(boxes: Boxes) -> Boxes:
    """Apply confidence filter + CAR‑suppression rule to a single image."""
    if boxes is None or len(boxes) == 0:
        return boxes  # nothing to do

    data = boxes.data.clone()  # [n, 6] (xyxy, conf, cls)
    cls   = data[:, 5].long()
    conf  = data[:, 4]

    # 1) Drop weak non‑CARs outright
    weak_noncar = (cls != CAR_ID) & (conf < THRESH)
    keep_mask   = ~weak_noncar
    data        = data[keep_mask]
    cls, conf   = cls[keep_mask], conf[keep_mask]

    if len(data) == 0:
        return Boxes(data, orig_shape=boxes.orig_shape)

    # 2) Suppress CAR boxes if they overlap a strong non‑CAR box (IoU ≥ IOU_THR)
    car_mask      = cls == CAR_ID
    strong_noncar = (cls != CAR_ID) & (conf >= THRESH)

    if strong_noncar.sum() == 0:
        return Boxes(data, orig_shape=boxes.orig_shape)

    iou = box_iou(data[car_mask, :4], data[strong_noncar, :4])
    suppress_car = (iou >= IOU_THR).any(dim=1)

    final_keep = torch.ones(len(data), dtype=torch.bool, device=data.device)
    final_keep[car_mask.nonzero(as_tuple=False).flatten()[suppress_car]] = False

    return Boxes(data[final_keep], orig_shape=boxes.orig_shape)


def main():
    out_dir = Path(OUT_DIR)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Weights :", WEIGHTS)
    print("Source  :", SOURCE)
    print("Output  :", out_dir)
    print(f"Rule    : keep non‑CAR only if conf ≥{THRESH}; suppress CAR when IoU ≥{IOU_THR}\n")

    model = YOLO(WEIGHTS)

    for r in model.predict(
            source=SOURCE,
            conf=CONF_PRED,
            imgsz=IMG_SIZE,
            save=False,
            stream=True,
            device=0 if torch.cuda.is_available() else "cpu",
            half=True):

        r.boxes = filter_boxes_single_img(r.boxes)

        vis = r.plot()
        cv2.imwrite(str(out_dir / Path(r.path).name), vis)
        print(f"{Path(r.path).name:25s} kept={len(r.boxes)}")

    print("\n✅ Done – check images in:", out_dir)


if __name__ == "__main__":
    # Ensure OpenCV doesn’t spam stdout when built without GUI support
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "6000000000")
    main()
