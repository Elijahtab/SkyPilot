# follow_seed_test.py
import argparse, time, cv2, json, threading, queue, math
from drone_controller import DroneController
import skytrack

def wait_for_frame(dc, timeout=8.0):
    t0 = time.time()
    while dc.latest_frame is None and time.time() - t0 < timeout:
        time.sleep(0.02)
    if dc.latest_frame is None:
        raise RuntimeError("Timed out waiting for first video frame")

def call_acquire_center_with_timeout(cap, desc, timeout_s=20.0):
    q = queue.Queue(maxsize=1)
    def worker():
        try:
            frame, res = skytrack.acquire_center_and_shape(cap, desc, return_frame=True)
            q.put(("ok", (frame, res)))
        except Exception as e:
            q.put(("err", e))
    t = threading.Thread(target=worker, daemon=True); t.start()
    try:
        status, payload = q.get(timeout=timeout_s)
    except queue.Empty:
        return None, None, "timeout"
    if status == "ok":
        frame, res = payload
        return frame, res, None
    return None, None, str(payload)

def scale_explicit_box_area(x, y, w, h, mult, W, H):
    if mult is None or mult == 1.0:
        return int(x), int(y), int(w), int(h)
    lin = math.sqrt(max(0.0, mult))
    cx = x + w / 2.0; cy = y + h / 2.0
    w2 = max(1.0, w * lin); h2 = max(1.0, h * lin)
    x2 = int(round(cx - w2 / 2.0)); y2 = int(round(cy - h2 / 2.0))
    w2 = int(round(w2)); h2 = int(round(h2))
    if x2 < 0: x2 = 0
    if y2 < 0: y2 = 0
    if x2 + w2 > W: w2 = max(1, W - x2)
    if y2 + h2 > H: h2 = max(1, H - y2)
    return x2, y2, w2, h2

def main():
    ap = argparse.ArgumentParser(
        description="Start Tello stream (no takeoff), get one bbox on the SAME inference frame, save image."
    )
    ap.add_argument("--desc", help="Target description. Required unless --force_yolo.")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--model_timeout", type=float, default=20.0)
    ap.add_argument("--mult", type=float, default=1.0, help="Area multiplier for final box")
    ap.add_argument("--out",  default="boxed_seed.jpg")
    ap.add_argument("--force_yolo", action="store_true", help="Skip GPT and use local YOLO person detector")
    ap.add_argument("--yolo_conf", type=float, default=0.45)
    ap.add_argument("--yolo_imgsz", type=int, default=384)
    ap.add_argument("--no_auto_person", action="store_true",
                    help="Disable auto routing to YOLO when desc looks like a person")
    args = ap.parse_args()

    if not args.force_yolo and not args.desc:
        ap.error("--desc is required unless --force_yolo is used.")

    # optional: prewarm YOLO to hide first-call latency
    try:
        skytrack.prewarm_yolo()
    except Exception:
        pass

    dc = DroneController()
    try:
        dc.start()
        wait_for_frame(dc, timeout=args.timeout)
        cap = dc._cap_proxy

        ok, seed = cap.read()
        if not ok or seed is None:
            raise RuntimeError("Could not read seed frame from proxy")
        cv2.imwrite("seed_frame.jpg", seed)

        # Decide path
        auto_person = (not args.no_auto_person) and args.desc and skytrack.is_person_like_desc(args.desc)
        use_yolo = args.force_yolo or auto_person

        if use_yolo:
            try:
                frame0, box = skytrack.acquire_person_local(cap, conf=args.yolo_conf, imgsz=args.yolo_imgsz)
            except TypeError:
                # older skytrack signature without imgsz (shouldn't happen with this file, but safe)
                frame0, box = skytrack.acquire_person_local(cap, conf=args.yolo_conf)
            except Exception as e:
                print(f"⚠️ YOLO detection failed: {e}")
                cv2.imwrite(args.out, seed); print(f"✓ saved fallback: {args.out}")
                return

            H, W = frame0.shape[:2]
            x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
            x, y, w, h = scale_explicit_box_area(x, y, w, h, args.mult, W, H)

            img = frame0.copy()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.imwrite(args.out, img)
            print(f"✓ saved {args.out} (YOLO path)")
            print("bbox:", (x, y, w, h))
            return

        # GPT center+shape path
        frame0, result, err = call_acquire_center_with_timeout(cap, args.desc, timeout_s=args.model_timeout)
        if err is not None or frame0 is None or result is None:
            print(f"⚠️ Model call failed: {err}")
            cv2.imwrite(args.out, seed); print(f"✓ saved fallback: {args.out}")
            return

        H, W = frame0.shape[:2]
        x, y, w, h = skytrack.box_from_center_shape(result, W, H, area_multiplier=args.mult)

        img = frame0.copy()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.imwrite(args.out, img)
        print(f"✓ saved {args.out}")
        print("raw center/shape:", json.dumps(result))
        print("computed box     :", (x, y, w, h))

    finally:
        dc.stop()

if __name__ == "__main__":
    main()
