# skytrack.py
import os
import cv2, base64, re, json, time
import numpy as np
from openai import OpenAI

# Create a client that uses env vars (OPENAI_API_KEY, etc.)
client = OpenAI(timeout=20, max_retries=1)

SYSTEM = (
    "You are a drone pilot on a critical mission. Locate the CENTER of the described object "
    "and estimate its size. Return ONLY JSON in exactly this schema:\n"
    '{ "cx": <int>, "cy": <int>, "scale": <float> }\n'
    "- (cx, cy): center pixel coordinates relative to THIS image (origin top-left (0,0)).\n"
    "- scale: either a fraction of the image AREA (e.g., 0.05 for 5% of image area), or a size multiplier if area is unknown.\n"
    "Pixel integers only for cx/cy. Values must be within image bounds. No extra keys, no prose, no code fences."
)

# ────────────────────────────────────────────────────────────────
def _b64_frame(cap, max_w=768, jpeg_quality=85):
    """
    Grab a recent frame (works with real VideoCapture or your _CapProxy),
    optionally downscale to width<=max_w, return (frame_bgr, data_uri, jpg_bytes).
    """
    ok, f = False, None
    # Read a few frames to get a 'fresh' one from the proxy/decoder
    for _ in range(8):
        ok, f = cap.read()
        if ok and f is not None:
            break
        time.sleep(0.01)
    if not ok or f is None:
        raise RuntimeError("no frame")

    H, W = f.shape[:2]
    if W > max_w:
        scale = max_w / float(W)
        f = cv2.resize(f, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

    ok, jpg = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")

    b64 = base64.b64encode(jpg).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    return f, data_uri, jpg

# ────────────────────────────────────────────────────────────────
def _parse_json_obj(txt: str) -> dict:
    """
    Extract the first { ... } JSON object from txt and parse it.
    Strips code fences if present.
    """
    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError(f"No JSON object found in model response: {txt!r}")
    return json.loads(m.group(0))

# ────────────────────────────────────────────────────────────────
def _draw_center_and_box(img, cx, cy, x, y, w, h, out_path="model_input_with_center.jpg"):
    vis = img.copy()
    # center mark
    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                   markerSize=14, thickness=2, line_type=cv2.LINE_AA)
    # box
    cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)

# ────────────────────────────────────────────────────────────────
def acquire_center_and_scale(cap, desc: str, return_frame: bool = False):
    """
    Call GPT‑4o Vision once.
    Returns dict: {"cx": int, "cy": int, "scale": float} (or may include "multiplier" instead).
    If return_frame=True, returns (frame, dict).
    Also writes debug images next to the script.
    """
    t0 = time.time()
    print("[skytrack] acquire_center_and_scale begin; cwd:", os.getcwd())

    frame, b64, jpg = _b64_frame(cap)

    # Debug artifacts: input images
    cv2.imwrite("seed_from_skytrack.jpg", frame)
    with open(os.path.abspath("model_input.jpg"), "wb") as f:
        f.write(jpg.tobytes())
    print("[skytrack] wrote seed_from_skytrack.jpg and model_input.jpg")

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text",      "text": desc},
            {"type": "image_url", "image_url": {"url": b64}}
        ]}
    ]
    print("[skytrack] calling OpenAI…")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs,
        temperature=0,
    )
    print("[skytrack] OpenAI returned")
    txt = (resp.choices[0].message.content or "").strip()

    data = _parse_json_obj(txt)

    # Accept legacy bbox format as fallback and convert to center/scale
    if all(k in data for k in ("x", "y", "w", "h")) and not all(k in data for k in ("cx", "cy")):
        x, y, w, h = map(float, (data["x"], data["y"], data["w"], data["h"]))
        cx = x + w / 2.0
        cy = y + h / 2.0
        H, W = frame.shape[:2]
        scale = (w * h) / float(W * H)  # fraction of image area
        result = {"cx": int(round(cx)), "cy": int(round(cy)), "scale": float(scale)}
    else:
        # Prefer center schema
        cx = float(data["cx"])
        cy = float(data["cy"])
        result = {"cx": int(round(cx)), "cy": int(round(cy))}
        if "scale" in data:
            result["scale"] = float(data["scale"])
        if "multiplier" in data:
            result["multiplier"] = float(data["multiplier"])

    # Draw a diagnostic box using our conversion (area-based if 'scale' in result)
    H, W = frame.shape[:2]
    x, y, w, h = box_from_center_result(result, W, H)
    _draw_center_and_box(
        cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR),
        result["cx"], result["cy"], x, y, w, h,
        out_path="model_input_with_center.jpg"
    )

    dt = time.time() - t0
    print(f"[skytrack] acquire_center_and_scale OK in {dt:.2f}s -> {result}")
    return (frame, result) if return_frame else result

# ────────────────────────────────────────────────────────────────
def box_from_center_result(result: dict, W: int, H: int):
    """
    Convert {'cx','cy','scale' or 'multiplier'} to a SQUARE (x,y,w,h), clamped to frame.
    - If 0<scale<=1: interpret as fraction of image area → side = sqrt(scale * W*H)
    - Else or if 'multiplier' present: side = base * multiplier (base ~= 0.15 * min(W,H))
    """
    cx = float(result.get("cx"))
    cy = float(result.get("cy"))
    scale = result.get("scale")
    mult  = result.get("multiplier")

    if scale is not None:
        scale = float(scale)
        if 0.0 < scale <= 1.0:
            area = scale * (W * H)
            side = max(8.0, area ** 0.5)
        else:
            # If scale is >1, treat as multiplier
            base = max(8.0, min(W, H) * 0.15)
            side = max(8.0, base * max(1.0, scale))
    elif mult is not None:
        mult = float(mult)
        base = max(8.0, min(W, H) * 0.15)
        side = max(8.0, base * max(0.5, mult))
    else:
        # fallback if neither provided
        base = max(8.0, min(W, H) * 0.15)
        side = base

    x = int(round(cx - side / 2.0))
    y = int(round(cy - side / 2.0))
    w = int(round(side))
    h = int(round(side))

    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > W: w = max(1, W - x)
    if y + h > H: h = max(1, H - y)
    return x, y, w, h

# ────────────────────────────────────────────────────────────────
def init_tracker(cap, box):
    """
    Create a CSRT tracker and init on the NEXT frame read from cap with the provided
    top-left pixel box dict {'x','y','w','h'}.
    """
    ok, f = cap.read()
    if not ok or f is None:
        raise RuntimeError("init_tracker: no frame")
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    if hasattr(cv2, "legacy"):
        t = cv2.legacy.TrackerCSRT_create()
    else:
        t = cv2.TrackerCSRT_create()
    t.init(f, (x, y, w, h))
    return t

# ────────────────────────────────────────────────────────────────
def track_step(cap, tracker, reconnect_cb=None, telemetry_q=None, parent=None):
    """
    Read 1 frame, update tracker.
    Returns (percent_cover, x_offset) or (None, None) on failure.
      - percent_cover: float (% of frame area), None if update failed
      - x_offset     : center_x - frame_center_x (pixels), None if update failed
    If 3 failed reads in a row, calls reconnect_cb() once (if provided).
    Also mirrors the box & pct to parent.cur_box / parent.cur_pct for overlay (ints).
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        track_step.fail_cnt = getattr(track_step, "fail_cnt", 0) + 1
        if track_step.fail_cnt >= 3 and reconnect_cb:
            print("🔄 Video lost – reconnecting …")
            reconnect_cb()
            track_step.fail_cnt = 0
        return None, None
    track_step.fail_cnt = 0

    ok, b = tracker.update(frame)
    if not ok:
        return None, None

    # Keep float for math, but publish ints for drawing
    x, y, w, h = b
    H, W = frame.shape[:2]
    pct = (w * h) / float(W * H) * 100.0
    off = (x + w / 2.0) - (W / 2.0)

    if parent is not None:
        parent.cur_box = (int(x), int(y), int(w), int(h))
        parent.cur_pct = pct
    if telemetry_q:
        telemetry_q.put(pct)

    return pct, off
