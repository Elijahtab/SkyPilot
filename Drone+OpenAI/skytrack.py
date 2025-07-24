# skytrack.py
import cv2
import time
import queue

def _create_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not available in this OpenCV build.")

def _safe_put(q, val):
    if q is None:
        return
    try:
        q.put_nowait(val)
    except queue.Full:
        pass

def acquire_box(cap, desc: str = "", timeout_sec: float = 2.0, out_q=None):
    """
    Dummy box acquisition: center 25% of the first good frame.
    Replace with your LLM-based detector if desired.
    """
    t0 = time.time()
    frame = None
    while time.time() - t0 < timeout_sec:
        ok, frm = cap.read()
        if ok and frm is not None:
            frame = frm
            break
    if frame is None:
        print("❌ acquire_box: no frame.")
        return None

    h, w = frame.shape[:2]
    bw, bh = int(w * 0.25), int(h * 0.25)
    x = (w - bw) // 2
    y = (h - bh) // 2
    box = {'x': x, 'y': y, 'w': bw, 'h': bh}

    # push a preview frame (with bbox) to the viewer queue if provided
    if out_q is not None:
        vis = frame.copy()
        cv2.rectangle(vis, (x, y), (x + bw, y + h), (0, 255, 0), 2)
        _safe_put(out_q, vis)

    return box

def init_tracker(frame, box_dict):
    tracker = _create_csrt()
    x, y, w, h = box_dict['x'], box_dict['y'], box_dict['w'], box_dict['h']
    tracker.init(frame, (x, y, w, h))
    return tracker

def track_step(cap, tracker, telem_q=None, out_q=None):
    """
    Returns (pct, offset). Sends drawn frame to out_q if provided.
    pct    = % of frame covered by bbox
    offset = horizontal px offset of bbox center from frame center (+ right)
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None

    ok, bbox = tracker.update(frame)
    if not ok or bbox is None:
        return None, None

    x, y, w, h = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    area_pct = 100.0 * (w * h) / (fw * fh)
    offset_x = (x + w / 2) - (fw / 2)

    _safe_put(telem_q, area_pct)

    if out_q is not None:
        vis = frame.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.line(vis, (fw // 2, 0), (fw // 2, fh), (255, 0, 0), 1)
        _safe_put(out_q, vis)

    return area_pct, offset_x
