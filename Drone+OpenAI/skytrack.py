# skytrack.py
# ────────────────────────────────────────────────────────────────────
import os, cv2, base64, re, json, time, math
import numpy as np
import datetime

is_person = False

# ───────────────── OpenAI (center+shape) ────────────────────────────
try:
    from openai import OpenAI
    _oa_client = OpenAI(timeout=10, max_retries=0)
except Exception:
    _oa_client = None

SYSTEM_CENTER_SHAPE = (
    "You are a drone pilot on a critical mission. Return ONLY JSON with either:\n"
    '  {"cx":int,"cy":int,"scale":float}  # scale = fraction of image area (e.g., 0.08)\n'
    "OPTIONALLY you may also include:\n"
    '  {"aspect":float}  # width/height ratio if known\n'
    "If you prefer, you may instead return an explicit pixel box:\n"
    '  {"x":int,"y":int,"w":int,"h":int}\n'
    "Image origin is top-left (0,0); y increases downward. Do not return text besides JSON."
)

def _b64_frame(cap, max_w=512, jpeg_quality=55):
    """Read one frame, optionally downscale, return (frame_bgr, data_uri, jpg_bytes)."""
    ok, f = cap.read()
    if not ok or f is None:
        raise RuntimeError("no frame")
    H, W = f.shape[:2]
    if W > max_w:
        s = max_w / float(W)
        f = cv2.resize(f, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b64 = base64.b64encode(jpg).decode()
    return f, f"data:image/jpeg;base64,{b64}", jpg

def _parse_json_obj(txt: str):
    m = re.search(r"\{[^\}]*\}", txt, re.S)
    if not m:
        raise ValueError(f"No JSON object found in model response: {txt!r}")
    return json.loads(m.group(0))

def acquire_center_and_shape(cap, desc: str, return_frame=False):
    """Call GPT (text+image) once. Returns (frame, dict) if return_frame else dict."""
    if _oa_client is None:
        raise RuntimeError("OpenAI client unavailable")
    frame, b64, jpg = _b64_frame(cap)
    msgs = [
        {"role": "system", "content": SYSTEM_CENTER_SHAPE},
        {"role": "user", "content": [
            {"type": "text", "text": desc},
            {"type": "image_url", "image_url": {"url": b64}},
        ]},
    ]
    resp = _oa_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0)
    txt = (resp.choices[0].message.content or "").strip()
    data = _parse_json_obj(txt)
    if return_frame:
        return frame, data
    return data

def box_from_center_shape(result: dict, W: int, H: int, area_multiplier: float = 1.0):
    """
    Build a pixel box (x,y,w,h) from center+shape (or explicit box) and clamp.
    - If explicit w/h are present, scale them by sqrt(area_multiplier).
    - Else if scale (fraction of area) given, compute side (or w/h via aspect) and scale area.
    """
    if all(k in result for k in ("x", "y", "w", "h")):
        cx = result["x"] + result["w"] / 2.0
        cy = result["y"] + result["h"] / 2.0
        lin = math.sqrt(max(0.0, area_multiplier))
        w = float(result["w"]) * lin
        h = float(result["h"]) * lin
    else:
        cx = float(result["cx"]); cy = float(result["cy"])
        aspect = result.get("aspect")  # w/h
        scale  = float(result.get("scale", 0.08)) * float(max(0.0, area_multiplier))
        scale  = max(1e-6, min(0.99, scale))
        area   = scale * (W * H)
        if aspect is None or aspect <= 0:
            side = max(8.0, math.sqrt(area))
            w = h = side
        else:
            w = max(8.0, math.sqrt(area * aspect))
            h = max(8.0, w / max(1e-6, aspect))

    x = int(round(cx - w/2.0)); y = int(round(cy - h/2.0))
    w = int(round(w)); h = int(round(h))
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > W: w = max(1, W - x)
    if y + h > H: h = max(1, H - y)
    return x, y, w, h

# ───────────────── YOLO (tiny, MPS/CPU) ─────────────────────────────
try:
    import torch
    from ultralytics import YOLO
except Exception:
    torch = None
    YOLO  = None

_YOLO_MODEL  = None
_YOLO_DEVICE = None

def _select_device():
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"

def _get_yolo(model_name: str = "yolov8n.pt"):
    global _YOLO_MODEL, _YOLO_DEVICE
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO not installed.  pip install ultralytics torch torchvision")
    if _YOLO_MODEL is None:
        _YOLO_DEVICE = _select_device()
        _YOLO_MODEL  = YOLO(model_name)
        # warm-up once
        try:
            dummy = np.zeros((384, 384, 3), dtype=np.uint8)
            _YOLO_MODEL.predict(dummy, imgsz=384, conf=0.25, iou=0.5, classes=[0],
                                device=_YOLO_DEVICE, verbose=False)
        except Exception:
            pass
    return _YOLO_MODEL

def prewarm_yolo():
    try:
        _get_yolo()
    except Exception as e:
        print(f"[skytrack] YOLO pre-warm skipped: {e}")

def acquire_person_local(cap, conf: float = 0.45, imgsz: int = 384, **_):
    """
    Fast local 'person' detector using tiny YOLO. Returns (frame_bgr, box_dict).
    """
    ok, f = cap.read()
    if not ok or f is None:
        raise RuntimeError("no frame")

    model = _get_yolo("yolov8n.pt")
    res   = model.predict(
                f, imgsz=int(imgsz), conf=float(conf), iou=0.5,
                classes=[0], device=_YOLO_DEVICE, verbose=False
            )[0]

    best = None; best_conf = -1.0
    for b in res.boxes:
        if int(b.cls[0]) != 0:
            continue
        c = float(getattr(b, "conf", [0.0])[0])
        if c > best_conf:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            best, best_conf = (x1, y1, x2, y2), c

    if best is None:
        raise RuntimeError("no person detected")

    x1, y1, x2, y2 = best
    return f, {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

# ── Heuristic: does the description sound like a person? ────────────
def is_person_like_desc(desc: str) -> bool:
    if not desc:
        return False
    s = desc.lower()
    kws = (
        "person","people","human","someone","pedestrian","man","woman","boy","girl","kid","child",
        "runner","biker","cyclist","face","torso","body","hoodie","jacket","shirt","pants","dress","coat","hat"
    )
    return any(k in s for k in kws)

# ───────────────── target-acquisition helper ────────────────────────
def acquire_target_box(cap,
                       desc: str,
                       *,
                       return_frame: bool = True,
                       yolo_conf: float = 0.40,
                       yolo_imgsz: int = 384,
                       gpt_area_mult: float = 1.0):
    """
    One call → one (frame, box) result.

    • If the text description *looks* like a person, try YOLO first.
    • Otherwise use GPT-4o (center+shape).  If GPT fails, fall back to YOLO.
    """
    #open_video_logger("flight_logs", fps=30)
    global is_person
    is_person = is_person_like_desc(desc)

    def _try_yolo():
        return acquire_person_local(cap, conf=yolo_conf, imgsz=yolo_imgsz)

    def _try_gpt():
        frm, res = acquire_center_and_shape(cap, desc, return_frame=True)
        H, W = frm.shape[:2]
        x, y, w, h = box_from_center_shape(res, W, H, area_multiplier=gpt_area_mult)
        return frm, {"x": x, "y": y, "w": w, "h": h}

    errors = []
    for fn in (_try_yolo, _try_gpt) if is_person else (_try_gpt, _try_yolo):
        try:
            return fn()
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError("Both detection methods failed:\n" + "\n".join(errors))

# ───────────────── tracker helpers ──────────────────────────────────
def init_tracker(cap, box: dict, *, frame=None):
    """
    Return an OpenCV CSRT tracker initialised on `frame`
    (or on cap.read() if frame=None).
    """
    if frame is None:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("no frame for tracker init")

    if hasattr(cv2, "legacy"):
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = cv2.TrackerCSRT_create()

    tracker.init(frame, (box["x"], box["y"], box["w"], box["h"]))
    return tracker


# ───────── video-logging globals & helpers ─────────────────────────
_VIDEO_WRITER = None
_VIDEO_FPS    = 30.0   # until first real frame establishes size
_VIDEO_PATH   = None

def open_video_logger(output_dir="logs", fps: float = 30.0):
    """
    Call ONCE (e.g. right after take-off). Creates output_dir if needed
    and starts a cv2.VideoWriter.  The actual writer is finalised the
    first time _save_frame() sees a frame (so it knows HxW).
    """
    global _VIDEO_WRITER, _VIDEO_FPS, _VIDEO_PATH
    if _VIDEO_WRITER is not None:
        return  # already open
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"tello_{ts}.mp4")
    _VIDEO_FPS  = fps
    _VIDEO_PATH = path
    _VIDEO_WRITER = "pending"      # sentinel until first frame
    print(f"[skytrack] video logging → {path}")

def _save_frame(frame):
    """Write BGR frame to disk iff logging is enabled."""
    global _VIDEO_WRITER
    if _VIDEO_WRITER is None:
        return                       # logging disabled

    if _VIDEO_WRITER == "pending":   # first frame decides size
        H, W = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _VIDEO_WRITER = cv2.VideoWriter(_VIDEO_PATH, fourcc,
                                        _VIDEO_FPS, (W, H))
    _VIDEO_WRITER.write(frame)

def close_video_logger():
    """Call when landing. Flush & release the VideoWriter."""
    global _VIDEO_WRITER
    if _VIDEO_WRITER and _VIDEO_WRITER not in ("pending",):
        _VIDEO_WRITER.release()
        print(f"[skytrack] saved video → {_VIDEO_PATH}")
    _VIDEO_WRITER = None

# ───────────────── per-frame update (tracker **or** YOLO) ───────────
def track_step(cap,
               tracker,
               *,
               yolo_conf: float = 0.45,
               yolo_imgsz: int = 384,
               reconnect_cb=None,
               telemetry_q=None,
               parent=None,
               max_fail: int = 3):
    """
    One iteration of tracking.

    Returns  (percent_cover, x_offset)   or  (None, None) on failure.

      • percent_cover – object area as % of frame
      • x_offset      – pixels, R+ / L–, where 0 = centred
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        track_step._miss = getattr(track_step, "_miss", 0) + 1
        if track_step._miss >= max_fail and reconnect_cb:
            reconnect_cb(); track_step._miss = 0
        return None, None
    track_step._miss = 0  # reset

    #_save_frame(frame) # for video logging

    # ── choose engine per frame ────────────────────────────────────
    if is_person:
        model  = _get_yolo("yolov8n.pt")
        result = model.predict(
                     frame, imgsz=int(yolo_imgsz), conf=float(yolo_conf),
                     classes=[0], device=_YOLO_DEVICE, verbose=False
                 )[0]
        best, best_conf = None, -1.0
        for b in result.boxes:
            if int(b.cls[0]) != 0:
                continue
            c = float(getattr(b, "conf", [0.0])[0])
            if c > best_conf:
                best = list(map(int, b.xyxy[0])); best_conf = c
        if best is None:
            return None, None  # lost
        x1, y1, x2, y2 = best
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
    else:
        ok, box = tracker.update(frame)
        if not ok:
            return None, None  # tracker lost
        x, y, w, h = map(int, box)

    # ── derive % cover + x-offset ──────────────────────────────────
    H, W = frame.shape[:2]
    pct  = (w * h) / (W * H) * 100.0
    off  = (x + w/2) - (W / 2)

    # telemetry / overlay hooks
    if parent is not None:
        parent.cur_box = (x, y, w, h)
        parent.cur_pct = pct
    if telemetry_q is not None:
        telemetry_q.put(pct)

    return pct, off
