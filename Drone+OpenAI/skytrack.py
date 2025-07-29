# skytrack.py
import os, cv2, base64, re, json, time, math
import numpy as np

# ───────────────── OpenAI (center+shape) ─────────────────
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

# ───────────────── YOLO (tiny, MPS) ─────────────────
try:
    import torch
    from ultralytics import YOLO
except Exception:
    torch = None
    YOLO = None

_YOLO_MODEL = None
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
        raise RuntimeError("Ultralytics YOLO not installed. pip install ultralytics torch torchvision")
    if _YOLO_MODEL is None:
        _YOLO_DEVICE = _select_device()
        _YOLO_MODEL = YOLO(model_name)
        # warm up once
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
        print(f"[skytrack] YOLO prewarm skipped: {e}")

def acquire_person_local(cap, conf: float = 0.45, imgsz: int = 384, **_):
    """
    Fast local 'person' detector using tiny YOLO. Accepts imgsz to control speed.
    Returns (frame_bgr, {'x','y','w','h'}).
    """
    ok, f = cap.read()
    if not ok or f is None:
        raise RuntimeError("no frame")

    model = _get_yolo("yolov8n.pt")
    try:
        res = model.predict(
            f, imgsz=int(imgsz), conf=float(conf), iou=0.5,
            classes=[0], device=_YOLO_DEVICE, verbose=False
        )[0]
    except TypeError:
        # older ultralytics signature without imgsz/device
        res = model.predict(
            f, conf=float(conf), iou=0.5, classes=[0], verbose=False
        )[0]

    best = None; best_conf = -1.0
    for b in res.boxes:
        if int(b.cls[0]) != 0:
            continue
        c = float(getattr(b, "conf", [0.0])[0])
        if c > best_conf:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            best = (x1, y1, x2, y2); best_conf = c

    if best is None:
        raise RuntimeError("no person detected")

    x1, y1, x2, y2 = best
    return f, {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

# ── Optional text-only heuristic to auto-route person targets ──
def is_person_like_desc(desc: str) -> bool:
    if not desc:
        return False
    s = desc.lower()
    kws = (
        "person","people","human","someone","pedestrian","man","woman","boy","girl","kid","child",
        "runner","biker","cyclist","face","torso","body","hoodie","jacket","shirt","pants","dress","coat","hat"
    )
    return any(k in s for k in kws)
