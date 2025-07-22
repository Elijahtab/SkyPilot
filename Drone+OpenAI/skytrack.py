# skytrack.py
import cv2, base64, re, json, time
from openai import OpenAI

client = OpenAI()

def _b64_frame(cap):
    for _ in range(30): cap.grab()
    ok, f = cap.retrieve(); assert ok, "no frame"
    _, jpg = cv2.imencode('.jpg', f)
    return f, "data:image/jpeg;base64,"+base64.b64encode(jpg).decode()

def acquire_box(cap, desc):
    frame, b64 = _b64_frame(cap)
    msgs=[{"role":"system","content":"JSON {\"x\":int,\"y\":int,\"w\":int,\"h\":int} only."},
          {"role":"user","content":[{"type":"text","text":desc},
                                    {"type":"image_url","image_url":{"url":b64}}]}]
    txt = client.chat.completions.create(model="gpt-4o",messages=msgs,temperature=0
          ).choices[0].message.content
    box = json.loads(re.search(r"\{.*\}",txt).group(0))
    return box

def init_tracker(cap, box):
    ok, f = cap.read(); assert ok
    t = cv2.legacy.TrackerCSRT_create() if hasattr(cv2,"legacy") \
        else cv2.TrackerCSRT_create()
    t.init(f, (box['x'],box['y'],box['w'],box['h']))
    return t

def track_step(cap, tracker, reconnect_cb=None, telemetry_q=None):
    """
    Read 1 frame, update tracker, return (percent_cover, x_offset)
      • percent_cover : float or None
      • x_offset      : pixels (object center – frame center) or None
    If 3 frames in a row fail, calls reconnect_cb() once.
    """
    ok, frame = cap.read()
    if not ok:
        track_step.fail_cnt = getattr(track_step, "fail_cnt", 0) + 1
        if track_step.fail_cnt >= 3 and reconnect_cb:
            print("🔄 Lost video – trying reconnect …")
            reconnect_cb()
            track_step.fail_cnt = 0
        return None, None

    track_step.fail_cnt = 0
    ok, box = tracker.update(frame)
    if not ok:
        return None, None                # tracker lost

    x, y, w, h = map(int, box)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    pct = (w * h) / (frame.shape[0] * frame.shape[1]) * 100.0
    off = (x + w // 2) - (frame.shape[1] // 2)

    # overlay HUD
    cv2.putText(frame, f"{pct:.1f}% cover", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Follow", frame)
    cv2.waitKey(1)

    if telemetry_q:
        telemetry_q.put(pct)

    return pct, off


