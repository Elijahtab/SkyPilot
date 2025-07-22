# skytrack.py
import cv2, base64, re, json
from openai import OpenAI

client = OpenAI()  # assumes env var OPENAI_API_KEY set


def _b64_frame(cap):
    # skip buffered frames
    for _ in range(30):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        raise RuntimeError("no frame")
    _, jpg = cv2.imencode('.jpg', frame)
    b64 = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()
    return frame, b64


def acquire_box(cap, desc):
    """Ask the model for an initial bbox. Returns dict or None."""
    frame, b64 = _b64_frame(cap)
    msgs = [
        {"role": "system",
         "content": 'Return ONLY JSON like {"x":int,"y":int,"w":int,"h":int}.'},
        {"role": "user",
         "content": [
             {"type": "text", "text": desc},
             {"type": "image_url", "image_url": {"url": b64}},
         ]},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0
    ).choices[0].message.content

    if not resp:
        return None

    # try straight JSON first
    try:
        return json.loads(resp.strip())
    except json.JSONDecodeError:
        # fallback: find first {...} block without nested braces
        m = re.search(r"\{[^{}]+\}", resp)
        if not m:
            return None
        return json.loads(m.group(0))


def track_step(cap, tracker, telemetry_q=None, reconnect_cb=None):
    """Update tracker, compute %cover + center offset, overlay HUD.
       Returns (pct, offset) or (None, None) if frame/tracker failed."""
    ok, frame = cap.read()
    if not ok:
        track_step.fail_cnt = getattr(track_step, "fail_cnt", 0) + 1
        if track_step.fail_cnt >= 3 and reconnect_cb:
            print("🔄 Lost video - trying reconnect …")
            reconnect_cb()
            track_step.fail_cnt = 0
        return None, None

    track_step.fail_cnt = 0
    ok, box = tracker.update(frame)
    if not ok:
        return None, None

    x, y, w, h = map(int, box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    pct = (w * h) / (frame.shape[0] * frame.shape[1]) * 100.0
    off = (x + w // 2) - (frame.shape[1] // 2)

    cv2.putText(frame, f"{pct:.1f}% cover", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Follow", frame)
    cv2.waitKey(1)

    if telemetry_q:
        telemetry_q.put(pct)

    return pct, off
