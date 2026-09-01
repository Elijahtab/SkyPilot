"""
Manual vehicle-type labeler — hover a box, press a key, auto-advance.

Each image gets a set of candidate boxes; you assign the 7-class schema by
hovering a box and pressing 1-7, or drop a box with d. When every box on an
image is labeled or deleted the YOLO .txt is written and the next unlabeled
image loads automatically.

Where the boxes come from — --boxes:
    auto      (default) use the human boxes from the sibling labels/ dir if the
              frame has them, otherwise fall back to the yolov8m detector
    existing  always use the sibling labels/ boxes (their class is ignored — you
              assign the type). Best for the Kaggle set: it ships complete
              human-drawn boxes and a generic detector only finds ~20% of them.
    detect    always run the yolov8m detector. For imagery with no boxes yet.

    python scripts/labeling/manual_label.py                         # Kaggle set, human boxes
    python scripts/labeling/manual_label.py --source <dir> --out <dir> --no-dedupe
    python scripts/labeling/manual_label.py --boxes detect --conf 0.3

Keys
    1 Bus   2 Vehicle   3 Motorcycle   4 SUV   5 Standard Car   6 Truck   7 Van
    d / Del / Backspace   delete the hovered box
    left-drag             draw a new box (for the ones the boxes source missed)
    u                     undo the last change on this image
    n / Enter / Space     next image (only once fully resolved)
    b                     back to the previous image (re-opens it for editing)
    s                     skip this image, write nothing
    k                     confirm "nothing to label here" (writes an empty label)
    q / Esc               quit (writes the current image only if resolved)

Output is canonical schema ids (configs/vehicle_7class.yaml). Defaults write to
Labeling/kaggle_dataset/train/labels_gpt/, so the batch flows straight into
scripts/labeling/extract_good_kaggle.py then scripts/evaluation/diagnose_labels.py
— run that gate before training on it.

Note: the Kaggle train split is ~62% Roboflow augmentation duplicates. --dedupe
(on by default) shows only one image per unique source frame, i.e. the part of
the filename before "_jpg.rf.". Pass --no-dedupe to label every export.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from _paths import (CLASS_NAMES, KAGGLE_GPT, KAGGLE_IMG, KAGGLE_PREVIEW, PRETRAINED,
                    labels_for, require)

# ─── schema / key map ────────────────────────────────────────────────
# Keys 1..7 map to the schema in configs/vehicle_7class.yaml order, so the id
# written to disk is simply (key digit - 1). '2' is the generic 'Vehicle'
# bucket — use it for anything whose subtype isn't unmistakable, the same
# convention the base dataset follows (see scripts/README.md).
KEY_TO_ID = {ord(str(i + 1)): i for i in range(len(CLASS_NAMES))}

# COCO ids yolov8m should keep: car, motorcycle, bus, truck
COCO_VEHICLE = [2, 3, 5, 7]
COCO_HINT = {2: "car?", 3: "moto?", 5: "bus?", 7: "truck?"}

# BGR per schema id, plus white for an un-assigned box
CLASS_BGR = {
    0: (0, 140, 255),    # Bus          orange
    1: (185, 185, 185),  # Vehicle      grey
    2: (200, 0, 200),    # Motorcycle   magenta
    3: (0, 200, 0),      # SUV          green
    4: (255, 150, 0),    # Standard Car blue
    5: (0, 0, 255),      # Truck        red
    6: (0, 220, 220),    # Van          yellow
}
PENDING_BGR = (255, 255, 255)

DEL_KEYS = {8, 127, 255}          # Backspace / Delete across platforms
NEXT_KEYS = {13, 10, 32, ord("n")}
QUIT_KEYS = {27, ord("q")}
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Box:
    xyxy: tuple            # pixel coords in the ORIGINAL image
    conf: float
    coco: int
    state: str = "pending"          # pending | labeled | deleted
    cls_id: int = None              # schema id once labeled
    added: bool = False             # drawn by hand rather than detected

    def area(self):
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Frame:
    """One image the user will see: its path plus lazily-computed boxes."""
    path: Path
    boxes: list = None                       # None until detections have run
    img: "np.ndarray" = None                 # decoded once, then reused every tick
    saved: bool = False                      # written to disk and untouched since
    undo: list = field(default_factory=list)  # (box_index, prev_state, prev_cls)

    def base(self):
        if self.img is None:
            self.img = cv2.imread(str(self.path))
        return self.img

    def resolved(self) -> bool:
        return self.boxes is not None and all(b.state != "pending" for b in self.boxes)

    def labeled_boxes(self):
        return [b for b in (self.boxes or []) if b.state == "labeled"]


# ─── geometry helpers ────────────────────────────────────────────────
def fit_scale(h, w, view_long):
    return view_long / max(h, w)            # scale small images up, large ones down


def hovered_index(frame: Frame, mx, my, scale):
    """Smallest-area non-deleted box whose scaled rect contains the cursor."""
    best, best_area = None, None
    for i, b in enumerate(frame.boxes):
        if b.state == "deleted":
            continue
        x1, y1, x2, y2 = (v * scale for v in b.xyxy)
        if x1 <= mx <= x2 and y1 <= my <= y2:
            if best_area is None or b.area() < best_area:
                best, best_area = i, b.area()
    return best


# ─── rendering ───────────────────────────────────────────────────────
def scaled(base, scale):
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    h, w = base.shape[:2]
    return cv2.resize(base, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)


def draw_label(canvas, x, y, text, bgr, *, above=True):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    y0 = int(y) - th - 6 if above else int(y) + 4
    y0 = max(0, y0)
    cv2.rectangle(canvas, (int(x), y0), (int(x) + tw + 6, y0 + th + 6), bgr, -1)
    cv2.putText(canvas, text, (int(x) + 3, y0 + th + 2), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def render(frame: Frame, scale, hover_i, drag, msg, session):
    base = frame.base()
    if base is None:
        base = np.zeros((360, 640, 3), np.uint8)
    canvas = scaled(base, scale)

    for i, b in enumerate(frame.boxes or []):
        if b.state == "deleted":
            continue
        x1, y1, x2, y2 = (int(v * scale) for v in b.xyxy)
        if b.state == "labeled":
            bgr, tag = CLASS_BGR[b.cls_id], f"{b.cls_id + 1} {CLASS_NAMES[b.cls_id]}"
        else:
            bgr, tag = PENDING_BGR, COCO_HINT.get(b.coco, "?")
        thick = 3 if i == hover_i else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, thick)
        if i == hover_i or b.state == "labeled":
            draw_label(canvas, x1, y1, tag, bgr)

    if drag is not None:
        (dx1, dy1), (dx2, dy2) = drag
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 255, 255), 1)

    _hud(canvas, frame, msg, session)
    return canvas


def _hud(canvas, frame, msg, session):
    h, w = canvas.shape[:2]
    boxes = frame.boxes or []
    pend = sum(b.state == "pending" for b in boxes)
    lab = sum(b.state == "labeled" for b in boxes)
    dele = sum(b.state == "deleted" for b in boxes)
    stem = frame.path.stem.split("_jpg.rf.")[0]

    top = (f"[{session['idx'] + 1}/{session['total']}]  {stem[:52]}   "
           f"pending {pend} | labeled {lab} | deleted {dele}    "
           f"session: {session['written']} written")
    legend = ("1 Bus  2 Vehicle  3 Motorcycle  4 SUV  5 Standard Car  6 Truck  7 Van    "
              "d del  u undo  b back  s skip  q quit")

    cv2.rectangle(canvas, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(canvas, top, (8, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(canvas, legend, (8, h - 7), FONT, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    if msg:
        (tw, _), _ = cv2.getTextSize(msg[0], FONT, 0.7, 2)
        cv2.rectangle(canvas, (w // 2 - tw // 2 - 12, h // 2 - 24),
                      (w // 2 + tw // 2 + 12, h // 2 + 12), msg[1], -1)
        cv2.putText(canvas, msg[0], (w // 2 - tw // 2, h // 2 + 2),
                    FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


# ─── label IO ────────────────────────────────────────────────────────
def write_label(frame: Frame, out_dir: Path, preview_dir: Path, scale):
    base = frame.base()
    h, w = base.shape[:2]
    lines = []
    for b in frame.labeled_boxes():
        x1, y1, x2, y2 = b.xyxy
        lines.append(f"{b.cls_id} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    (out_dir / f"{frame.path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    frame.saved = True

    if preview_dir is not None:
        canvas = scaled(base, scale)
        for b in frame.labeled_boxes():
            x1, y1, x2, y2 = (int(v * scale) for v in b.xyxy)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), CLASS_BGR[b.cls_id], 2)
            draw_label(canvas, x1, y1, CLASS_NAMES[b.cls_id], CLASS_BGR[b.cls_id])
        preview_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_dir / f"{frame.path.stem}_preview.jpg"), canvas)


def read_yolo_txt(frame: Frame, txt: Path, *, labeled: bool):
    """
    Build Box objects from a YOLO label file.

    labeled=True  → boxes come back as already-classified (editing our own output)
    labeled=False → boxes come back pending, class discarded (seeding from a
                    single-class human-box file; the user assigns the real type)
    """
    h, w = frame.base().shape[:2]
    out = []
    for ln in txt.read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0]))
        xc, yc, bw, bh = (float(t) for t in p[1:5])
        xyxy = ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
        out.append(Box(xyxy, 1.0, -1, state="labeled", cls_id=cid) if labeled
                   else Box(xyxy, 1.0, -1))
    return out


# ─── image list ──────────────────────────────────────────────────────
def collect_images(source: Path, out_dir: Path, dedupe: bool, redo: bool):
    imgs = sorted(p for p in source.glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not imgs:
        sys.exit(f"[ERR] no images in {source}")

    if dedupe:
        seen, unique = set(), []
        for p in imgs:
            key = p.stem.split("_jpg.rf.")[0]
            if key not in seen:
                seen.add(key)
                unique.append(p)
        print(f"dedupe: {len(imgs)} images -> {len(unique)} unique source frames")
        imgs = unique

    if not redo:
        todo = [p for p in imgs if not (out_dir / f"{p.stem}.txt").exists()]
        done = len(imgs) - len(todo)
        if done:
            print(f"resume: {done} already labeled in {out_dir}, {len(todo)} to go")
        imgs = todo

    if not imgs:
        sys.exit("Nothing left to label. (pass --redo to revisit finished images)")
    return imgs


# ─── main loop ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=KAGGLE_IMG, help="folder of images")
    ap.add_argument("--out", type=Path, default=KAGGLE_GPT, help="where .txt labels go")
    ap.add_argument("--boxes", choices=("auto", "existing", "detect"), default="auto",
                    help="candidate boxes: sibling labels/ dir, the detector, or auto")
    ap.add_argument("--labels-dir", type=Path, default=None,
                    help="override the sibling labels/ dir for --boxes existing|auto")
    ap.add_argument("--weights", type=Path, default=PRETRAINED / "yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    ap.add_argument("--imgsz", type=int, default=960, help="detector input size")
    ap.add_argument("--view", type=int, default=1100, help="on-screen long edge (px)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N new labels")
    ap.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True,
                    help="show one image per unique source frame (default: on)")
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--redo", action="store_true", help="revisit already-labeled images")
    args = ap.parse_args()

    require(args.source)
    src_labels = args.labels_dir or labels_for(args.source)
    if args.boxes == "existing":
        require(src_labels)
    if args.boxes in ("auto", "detect") and not args.weights.exists():
        msg = f"[ERR] weights not found: {args.weights}"
        if args.boxes == "auto":
            msg += "\n      (use --boxes existing to skip the detector entirely)"
        sys.exit(msg)
    args.out.mkdir(parents=True, exist_ok=True)
    preview_dir = (KAGGLE_PREVIEW if args.out == KAGGLE_GPT else args.out.parent / "preview") \
        if args.preview else None

    paths = collect_images(args.source, args.out, args.dedupe, args.redo)
    frames = [Frame(p) for p in paths]

    device = ("mps" if torch.backends.mps.is_available()
              else 0 if torch.cuda.is_available() else "cpu")
    print(f"boxes source: {args.boxes}"
          + (f"  (labels/: {src_labels})" if args.boxes != "detect" else "")
          + (f"  (detector: {args.weights.name} on {device})" if args.boxes != "existing" else ""))

    model = None

    def get_model():
        nonlocal model
        if model is None:
            print(f"Loading {args.weights.name} on {device} ...")
            model = YOLO(str(args.weights))
        return model

    def run_detector(frame: Frame):
        r = get_model().predict(str(frame.path), conf=args.conf, imgsz=args.imgsz,
                                classes=COCO_VEHICLE, device=device, verbose=False)[0]
        boxes = []
        if r.boxes is not None:
            for xyxy, c, k in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int)):
                boxes.append(Box(tuple(map(float, xyxy)), float(c), int(k)))
        return boxes

    def ensure_boxes(frame: Frame):
        if frame.boxes is not None:
            return
        # our own finished output wins — reopen it for editing (--redo / b)
        prev = args.out / f"{frame.path.stem}.txt"
        if prev.exists():
            frame.saved = True                     # already on disk; don't re-advance on sight
            frame.boxes = read_yolo_txt(frame, prev, labeled=True)
            return
        # seed from the human boxes when we have them
        if args.boxes in ("existing", "auto"):
            src_txt = src_labels / f"{frame.path.stem}.txt"
            if src_txt.exists() and src_txt.read_text().strip():
                frame.boxes = read_yolo_txt(frame, src_txt, labeled=False)
                return
            if args.boxes == "existing":
                frame.boxes = []
                return
        # --boxes detect, or auto with no human boxes for this frame
        frame.boxes = run_detector(frame)

    win = "manual_label"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ms = {"xy": (0, 0), "down": None, "drag": None}

    def on_mouse(event, x, y, flags, _):
        ms["xy"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            ms["down"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ms["down"] and (flags & cv2.EVENT_FLAG_LBUTTON):
            ms["drag"] = (ms["down"], (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            if ms["drag"]:
                (x1, y1), (x2, y2) = ms["drag"]
                if abs(x2 - x1) > 6 and abs(y2 - y1) > 6:
                    on_mouse.new_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            ms["down"] = ms["drag"] = None
    on_mouse.new_box = None
    cv2.setMouseCallback(win, on_mouse)

    idx, written, msg, msg_until = 0, 0, None, 0
    tick = 0

    while 0 <= idx < len(frames):
        # keep only a small window of decoded images in RAM (photos can be ~30MB each)
        for j, fr in enumerate(frames):
            if abs(j - idx) > 2:
                fr.img = None

        frame = frames[idx]
        if frame.base() is None:
            print(f"unreadable, skipping: {frame.path.name}")
            idx += 1
            continue
        ensure_boxes(frame)
        h, w = frame.base().shape[:2]
        scale = fit_scale(h, w, args.view)

        session = {"idx": idx, "total": len(frames), "written": written}
        mx, my = ms["xy"]
        hover_i = hovered_index(frame, mx, my, scale) if frame.boxes else None
        if tick < msg_until:
            shown_msg = msg
        else:
            shown_msg = None
            if not frame.boxes:
                shown_msg = ("no boxes here - k = confirm empty, drag = add, s = skip",
                             (60, 160, 60))
        cv2.imshow(win, render(frame, scale, hover_i, ms["drag"], shown_msg, session))
        tick += 1
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed
            break

        # a hand-drawn box arrived from the mouse callback
        if on_mouse.new_box is not None:
            x1, y1, x2, y2 = (v / scale for v in on_mouse.new_box)
            frame.boxes.append(Box((x1, y1, x2, y2), 1.0, -1, added=True))
            frame.undo.append((len(frame.boxes) - 1, "__new__", None))
            frame.saved = False
            on_mouse.new_box = None
            continue

        key = cv2.waitKeyEx(20)
        if key == -1:
            # auto-advance the moment every detected box is resolved — but not on a
            # frame the user navigated back to and hasn't re-touched (frame.saved)
            if frame.boxes and frame.resolved() and not frame.saved:
                write_label(frame, args.out, preview_dir, scale)
                written += 1
                msg, msg_until = (f"saved {len(frame.labeled_boxes())} boxes", (60, 200, 60)), tick + 25
                if args.limit and written >= args.limit:
                    print(f"limit reached ({args.limit})")
                    break
                idx += 1
            continue

        key &= 0xFFFFFF
        if key in QUIT_KEYS:
            if frame.boxes and frame.resolved() and not frame.saved:
                write_label(frame, args.out, preview_dir, scale)
                written += 1
            break

        if key in KEY_TO_ID and hover_i is not None:
            b = frame.boxes[hover_i]
            frame.undo.append((hover_i, b.state, b.cls_id))
            b.state, b.cls_id = "labeled", KEY_TO_ID[key]
            frame.saved = False
        elif (key in DEL_KEYS or key == ord("d")) and hover_i is not None:
            b = frame.boxes[hover_i]
            frame.undo.append((hover_i, b.state, b.cls_id))
            b.state = "deleted"
            frame.saved = False
        elif key == ord("u") and frame.undo:
            i, st, cl = frame.undo.pop()
            if st == "__new__":
                frame.boxes.pop(i)
            else:
                frame.boxes[i].state, frame.boxes[i].cls_id = st, cl
            frame.saved = False
        elif key == ord("k") and not frame.boxes:
            write_label(frame, args.out, preview_dir, scale)      # empty negative
            written += 1
            msg, msg_until = ("saved (no vehicles)", (60, 200, 60)), tick + 25
            idx += 1
        elif key == ord("s"):
            idx += 1
        elif key == ord("b"):
            idx = max(0, idx - 1)
        elif key in NEXT_KEYS:
            if frame.boxes and frame.resolved():
                if not frame.saved:
                    write_label(frame, args.out, preview_dir, scale)
                    written += 1
                idx += 1
            else:
                msg, msg_until = ("still pending - label or delete every box", (60, 60, 200)), tick + 25

    cv2.destroyAllWindows()
    print(f"\nDone. {written} label file(s) written to {args.out}")
    if written:
        print("Next: python scripts/labeling/extract_good_kaggle.py")
        print("      python scripts/evaluation/diagnose_labels.py   # GATE before training")


if __name__ == "__main__":
    main()
