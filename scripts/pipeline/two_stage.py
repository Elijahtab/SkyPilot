"""
Two-stage vehicle pipeline: detect -> type -> colour, one frame at a time.

  Stage 1  v4 detector, class output IGNORED, class-agnostic NMS. Its job is
           finding vehicles, which it does well (0.80 class-agnostic recall on
           base val, against 0.40 for stock COCO yolov8m).
  Stage 2  type classifier on boxes >= MIN_TYPE_PX: Bus, Motorcycle, SUV,
           Standard Car, Truck, Van (scripts/training/train_type_classifier.py).
  Colour   pixel-rule colour naming on boxes >= MIN_COLOR_PX (scripts/_color.py).

Every detection gets a type. When the box is below the gate, or the classifier's
top probability is under type_conf, the answer is 'Vehicle' -- the umbrella
class, true-but-unspecific -- and `type_status` records which of the two it was,
with the classifier's best guess kept in `type_guess`.

Agnostic NMS matters: with per-class NMS v4 can return a 'Vehicle' box and an
'SUV' box on the same car (its training data does exactly that, see
tools/find_duplicate_boxes.py), and the pipeline would report the car twice.

Usage:
    from pipeline.two_stage import TwoStagePipeline
    pipe = TwoStagePipeline()
    dets, (w, h) = pipe("frame.jpg")
    for d in dets:
        print(d["type"], d["color"], d["box"])
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

from _color import MIN_COLOR_PX, name_color, white_balance_gains
from _crops import MIN_TYPE_PX, TYPE_CLASSES, square_crop
from _paths import TYPE_RUNS, best_weights

DETECTOR   = best_weights("Vehicle_type_detection_v4")
CLASSIFIER = TYPE_RUNS / "type_cls_v1" / "weights" / "best.pt"

DET_CONF  = 0.25
DET_IMGSZ = 640          # v4 was trained at 640
TYPE_CONF = 0.50         # Kaggle holdout: 94% of crops clear it, at 0.82 accuracy
CHUNK     = 64


class TwoStagePipeline:
    def __init__(self, detector=DETECTOR, classifier=CLASSIFIER, det_conf=DET_CONF,
                 type_conf=TYPE_CONF, device=None):
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.det = YOLO(str(detector))
        self.cls = YOLO(str(classifier))
        names = list(self.cls.names.values())
        if sorted(names) != sorted(TYPE_CLASSES):
            raise ValueError(f"classifier classes {names} != {TYPE_CLASSES}")
        self.order = [names.index(k) for k in TYPE_CLASSES]
        self.det_conf, self.type_conf = det_conf, type_conf
        self.versions = f"det={Path(detector).parents[1].name} cls={Path(classifier).parents[1].name}"

    def __call__(self, frame_path):
        """A path on disk -> (list of detection dicts, (width, height))."""
        # exif_transpose: cv2.imread, which the detector was validated through,
        # applies EXIF rotation; PIL does not unless asked
        im = ImageOps.exif_transpose(Image.open(frame_path)).convert("RGB")
        return self.from_image(im)

    def from_array(self, rgb):
        """An HxWx3 RGB uint8 array -> (dets, (width, height)).

        The entry point for live frames, e.g. the ROS 2 detector node. No EXIF
        handling here: a frame off a camera topic carries no EXIF orientation,
        and applying it would be wrong.
        """
        return self.from_image(Image.fromarray(rgb))

    def from_image(self, im):
        """A PIL RGB Image -> (dets, (width, height)). The shared body."""
        rgb = np.asarray(im)
        r = self.det.predict(im, conf=self.det_conf, imgsz=DET_IMGSZ,
                             agnostic_nms=True, verbose=False, device=self.device)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)

        dets = []
        for b, c in zip(boxes, confs):
            size = float(max(b[2] - b[0], b[3] - b[1]))
            dets.append({"box": [float(v) for v in b], "det_conf": float(c), "size_px": size,
                         "type": "Vehicle", "type_conf": None, "type_guess": None,
                         "type_status": "too_small", "color": None, "color_conf": None})

        # stage 2: type, batched
        typed = [i for i, d in enumerate(dets) if d["size_px"] >= MIN_TYPE_PX]
        for s in range(0, len(typed), CHUNK):
            idx = typed[s:s + CHUNK]
            crops = [square_crop(im, dets[i]["box"], max_side=256) for i in idx]
            for i, res in zip(idx, self.cls.predict(crops, imgsz=224, verbose=False, device=self.device)):
                p = res.probs.data.cpu().numpy()[self.order]
                k = int(p.argmax())
                d = dets[i]
                d["type_guess"], d["type_conf"] = TYPE_CLASSES[k], float(p[k])
                if p[k] >= self.type_conf:
                    d["type"], d["type_status"] = TYPE_CLASSES[k], "typed"
                else:
                    d["type_status"] = "unsure"

        # colour
        gains = white_balance_gains(rgb) if any(d["size_px"] >= MIN_COLOR_PX for d in dets) else None
        for d in dets:
            if d["size_px"] >= MIN_COLOR_PX:
                color, conf, _ = name_color(rgb, d["box"], gains)
                d["color"], d["color_conf"] = color, (conf if color else None)

        return dets, im.size
