#!/usr/bin/env python3
# follow_test.py
#
# Minimal follow-mode test using ONLY skytrack from your code.
# - Connects to Tello over UDP
# - Enters SDK, streamon, takeoff
# - Grabs a frame, asks OpenAI (via skytrack.acquire_box) for a bbox
# - Starts a CSRT tracker
# - Runs naive follow logic and shows a preview window
#
# macOS notes:
#   * Use python from a venv (not the system one)
#   * pip install opencv-contrib-python openai
#   * If the window doesn't appear, run from Terminal.app (not VSCode's internal console)
#
# Quit with 'q' or Ctrl-C.

import argparse
import socket
import sys
import time
import signal
import cv2
import skytrack

# ------------------ Config ------------------
TELLO_ADDR        = ("192.168.10.1", 8889)
LOCAL_PORT        = 9000
VIDEO_URL         = ("udp://0.0.0.0:11111"
                     "?fifo_size=0&overrun_nonfatal=1&fflags=nobuffer&flags=low_delay")

INTER_CMD_DELAY   = 0.3    # sec between movement cmds
OFFSET_THRESH_PX  = 50     # px left/right
PCT_MIN           = 5.0    # % too small => forward
PCT_MAX           = 15.0   # % too big   => back

# ------------------ Helpers ------------------
def send_cmd(sock, cmd, wait=True, timeout=8):
    sock.sendto(cmd.encode("ascii"), TELLO_ADDR)
    print(f"[TX] {cmd}")
    if not wait:
        return None

    sock.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            break
        if addr != TELLO_ADDR:
            continue  # ignore junk
        msg = data.decode("ascii", errors="ignore").strip()
        if msg:
            print(f"[RX] {msg}")
        if msg in ("ok", "error"):
            return msg
    print("⚠️ timeout waiting for reply")
    return None



def land_and_cleanup(sock, cap):
    # try to land/streamoff but don't block
    try: send_cmd(sock, "land", wait=False)
    except: pass
    time.sleep(1.0)
    try: send_cmd(sock, "streamoff", wait=False)
    except: pass
    try: sock.close()
    except: pass
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

def ctrl_c_handler(signum, frame):
    raise KeyboardInterrupt

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc",  default="person", help="Target description for GPT")
    parser.add_argument("--video", default=None,     help="Local video file for testing")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, ctrl_c_handler)

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", LOCAL_PORT))

    # Init Tello (skip if using local video but still want full flow? You can comment takeoff out.)
    if args.video is None:
        if send_cmd(sock, "command") != "ok":
            print("❌ SDK init failed"); sock.close(); sys.exit(1)
        send_cmd(sock, "streamon")   # don't care if reply lost
        time.sleep(1.0)
        if send_cmd(sock, "takeoff") != "ok":
            print("❌ takeoff failed"); land_and_cleanup(sock, None); sys.exit(1)
    else:
        print("▶ Using local video; skipping Tello takeoff.")

    # Video source
    src = args.video if args.video else VIDEO_URL
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if args.video is None else 0)
    if not cap.isOpened():
        print("❌ Could not open video source."); land_and_cleanup(sock, cap); sys.exit(1)

    # Get initial box via skytrack
    try:
        box = skytrack.acquire_box(cap, args.desc)
    except Exception as e:
        print(f"❌ acquire_box failed: {e}")
        land_and_cleanup(sock, cap); sys.exit(1)

    if not box:
        print("❌ No box from GPT"); land_and_cleanup(sock, cap); sys.exit(1)

    # Re-read a frame to init tracker
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame after acquire_box"); land_and_cleanup(sock, cap); sys.exit(1)

    # Clamp bbox in case of resolution mismatch
    H, W = frame.shape[:2]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, (x, y, w, h))
    print(f"📦 Box: {x,y,w,h} | Frame: {W}x{H}")

    cv2.namedWindow("follow_test", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame, continuing...")
                continue

            ok, r = tracker.update(frame)
            if not ok:
                cv2.putText(frame, "TRACK LOST", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("follow_test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            bx, by, bw, bh = map(int, r)
            # draw box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # compute coverage % and horizontal offset (center - image_center)
            target_area = bw * bh
            pct = 100.0 * target_area / (W * H)
            offset = (bx + bw // 2) - (W // 2)

            cv2.putText(frame, f"{pct:4.1f}% | off {offset:+d}px",
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            cv2.imshow("follow_test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 q pressed"); break

            # naive control logic (skip if using local video)
            if args.video is None:
                if abs(offset) > OFFSET_THRESH_PX:
                    send_cmd(sock, "right 20" if offset > 0 else "left 20", wait=False)
                    time.sleep(INTER_CMD_DELAY)
                if pct < PCT_MIN:
                    send_cmd(sock, "forward 20", wait=False); time.sleep(INTER_CMD_DELAY)
                elif pct > PCT_MAX:
                    send_cmd(sock, "back 20",    wait=False); time.sleep(INTER_CMD_DELAY)

    except KeyboardInterrupt:
        print("🛑 Ctrl-C")
    finally:
        land_and_cleanup(sock, cap)

if __name__ == "__main__":
    main()
