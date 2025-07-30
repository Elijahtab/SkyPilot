# drone_controller.py
import socket, queue, threading, time, cv2
from datetime import datetime
import skytrack

TELLO_ADDR = ("192.168.10.1", 8889)        # real drone
# TELLO_ADDR = ("127.0.0.1", 8889)         # fake‑tello tests

VIDEO_URL = (
    "udp://0.0.0.0:11111"
    "?fifo_size=0&overrun_nonfatal=1&fflags=nobuffer&flags=low_delay"
)

class DroneController:
    HEARTBEAT_SEC = 8
    QUEUE_MAXSIZE = 5

    def __init__(self):
        # ── misc state ────────────────────────────────
        self.inter_command_delay = 0.3
        self.running            = False
        self.video_src          = VIDEO_URL
        self.latest_frame       = None          # most recent frame for proxy
        self.cur_box            = None          # (x,y,w,h) from tracker
        self.cur_pct            = None          # latest % cover
        # flight‑ack tracking
        self.waiting_flight_ack = None

        # ── networking / queues ───────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self.sock.bind(('', 8889))
        self.address = TELLO_ADDR

        self.cmd_q       = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self.frame_q     = queue.Queue(maxsize=2)
        self.telemetry_q = queue.Queue()

        # ── threading primitives ──────────────────────
        self.state_lock  = threading.RLock()
        self.stop_event  = threading.Event()
        self.follow_stop = threading.Event()
        self.ack_event   = threading.Event(); self.ack_event.set()

        self._workers   = []        # RX / CMD / video / heartbeat threads
        self.follow_thr = None      # follow thread

        # misc
        self.log_file      = None
        self.battery_level = None
        self._batt_event   = threading.Event()

        # proxy object for safe frame access
        self._cap_proxy = self._CapProxy(self)

    # ───────────────────────── lifecycle ─────────────────────────
    def start(self):
        with self.state_lock:
            if self.running:
                print("⚠️ Controller already running."); return
            self.running = True
            self.stop_event.clear(); self.follow_stop.clear()

        self.log_file = open("commands.txt", "w", buffering=1)

        # spawn workers (RX first!)
        self._spawn(self._recv_loop,        "rx")
        self._spawn(self._telemetry_worker, "telemetry")
        self._spawn(self._cmd_worker,       "cmd")
        self._spawn(self._heartbeat,        "heartbeat")

        # SDK / stream
        self._tx(b"command",  track_ack=False); time.sleep(0.3)
        self._tx(b"streamon", track_ack=False); time.sleep(1.0)
        self._spawn(self._video_reader_loop, "video_reader")

        # battery check
        lvl = self.query_battery(timeout=2.0)
        if lvl is not None: print(f"🔋 Battery: {lvl}%")
        else:               print("⚠️ Battery query timeout")

        print("🚀 Controller started")

    def stop(self):
        self.stop_event.set(); self.follow_stop.set()
        try: self._tx(b"streamoff", track_ack=False)
        except Exception: pass

        for t in self._workers: t.join(timeout=2)
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)

        try: self.sock.close()
        except OSError: pass
        if self.log_file: self.log_file.close()

        with self.state_lock: self.running = False
        print("🛑 Controller stopped.")

    def immediate_land(self):
        """Clear queue & force LAND now."""
        self.stop_event.set(); self.follow_stop.set()
        while True:
            try: self.cmd_q.get_nowait()
            except queue.Empty: break
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self._tx(b"land", track_ack=True)
        print("🛬 Emergency landing.")

    # ───────────── public helpers (enqueue wrapper) ──────────────
    def enqueue(self, cmd: str) -> bool:
        try:
            self.cmd_q.put_nowait(cmd); return True
        except queue.Full:
            print("⚠️ Queue full."); return False

    # ───────────────────── follow API ───────────────────────────
    def start_follow(self, desc: str):
        if self.follow_thr and self.follow_thr.is_alive():
            print("⚠️ Follow already running."); return
        self.follow_stop.clear()
        self.follow_thr = threading.Thread(
            target=self._follow_loop, args=(desc,), daemon=True, name="follow"
        )
        self.follow_thr.start(); print("🎯 Follow thread started.")

    def stop_follow(self):
        self.follow_stop.set()
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None
        self.cur_box = self.cur_pct = None
        print("🧱 Follow stopped.")

    # ───────────────────── internal helpers ─────────────────────
    def _spawn(self, fn, name):
        t = threading.Thread(target=fn, daemon=True, name=name); t.start()
        self._workers.append(t)

    def _tx(self, cmd: bytes, track_ack=True):
        """Send raw bytes to Tello + minimal logging/ack tracking."""
        cmd_str = cmd.decode().strip()
        if track_ack and cmd_str in ("takeoff", "land"):
            with self.state_lock: self.waiting_flight_ack = cmd_str
        try: self.sock.sendto(cmd, self.address)
        except OSError as e:
            print(f"[TX] socket error: {e}"); return
        with self.state_lock: self.last_cmd_sent = cmd_str
        print(f"[TX] {cmd_str}")
        if self.log_file and cmd_str != "command":
            ts = datetime.now().strftime("%F %T")
            self.log_file.write(f"{ts} - {cmd_str}\n")

    # ───────────────────────── RX / CMD / HB loops ──────────────
    def _recv_loop(self):
        while not self.stop_event.is_set():
            try: data, _ = self.sock.recvfrom(1024)
            except OSError:
                if self.stop_event.is_set(): break
                continue
            reply = data.decode(errors="ignore").strip(); print(f"[RX] {reply}")
            with self.state_lock:
                if reply.isdigit() and getattr(self, "_last_query", "") == "battery?":
                    self.battery_level = int(reply); self._batt_event.set()
                if reply == "ok":
                    if self.waiting_flight_ack:
                        print(f"✅ {self.waiting_flight_ack} ok")
                        self.waiting_flight_ack = None
                    self.ack_event.set()
                elif reply.startswith("error") or reply.startswith("out of range"):
                    print(f"❌ '{self.last_cmd_sent}' failed: {reply}")
                    self.waiting_flight_ack = None; self.ack_event.set()

    def _cmd_worker(self):
        while not self.stop_event.is_set():
            try: cmd = self.cmd_q.get(timeout=0.1)
            except queue.Empty: continue
            self.ack_event.wait(); self.ack_event.clear()
            time.sleep(0.05)                       # small gap
            self._tx(cmd.encode(), track_ack=(cmd in ("takeoff", "land")))
            time.sleep(self.inter_command_delay)

    def _telemetry_worker(self):
        while not self.stop_event.is_set():
            try: pct = self.telemetry_q.get(timeout=0.5)
            except queue.Empty: pct = None
            # you can log pct here if desired

    def _heartbeat(self):
        while not self.stop_event.is_set():
            self._tx(b"command", track_ack=False)
            time.sleep(self.HEARTBEAT_SEC)

    # ───────────────────────── video reader ─────────────────────
    def _video_reader_loop(self):
        cap = cv2.VideoCapture(self.video_src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("❌ Could not open video stream."); return
        try:
            while not self.stop_event.is_set():
                ok, frm = cap.read()
                if not ok: time.sleep(0.01); continue
                self.latest_frame = frm
                # draw overlay if tracker is active
                if self.cur_box:
                    x, y, w, h = self.cur_box
                    cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 2)
                    if self.cur_pct is not None:
                        txt = f"{self.cur_pct:4.1f}%"
                        ty  = max(20, y-10)
                        cv2.putText(frm, txt, (x,ty), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,255,0), 2, cv2.LINE_AA)
                try:
                    if self.frame_q.full(): self.frame_q.get_nowait()
                    self.frame_q.put_nowait(frm)
                except queue.Full: pass
        finally: cap.release()

    # ───────────────────────── follow loop ─────────────────────
    def _follow_loop(self, desc: str):
        cap = self._cap_proxy
        while self.latest_frame is None and not self.stop_event.is_set():
            time.sleep(0.01)

        try:
            # 1) one‑shot GPT call -> center+shape, SAME frame
            frame_inf, result = skytrack.acquire_center_and_shape(cap, desc, return_frame=True)
            H,W = frame_inf.shape[:2]
            x,y,w,h = skytrack.box_from_center_shape(result, W, H, area_multiplier=1.0)

            dbg = frame_inf.copy()
            cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.imwrite("boxed_first_frame.jpg", dbg)
            self.cur_box = (x,y,w,h)

            # 2) init tracker ON THAT FRAME
            tracker = skytrack.init_tracker(cap, {"x":x,"y":y,"w":w,"h":h}, frame=frame_inf)

            # 3) tracking loop (your control logic commented out; re‑enable as needed)
            
            DESIRED_PCT=6.0; PCT_TOL=2.0; OFFSET_TOL=20
            STEP_FB=20; STEP_YAW=10; COOLDOWN=3; last=0
            while not (self.stop_event.is_set() or self.follow_stop.is_set()):
                print(off)
                pct, off = skytrack.track_step(cap, tracker,
                                               telemetry_q=self.telemetry_q,
                                               parent=self)
                if pct is None: time.sleep(0.01); continue
                now=time.time()
                if now-last < COOLDOWN: time.sleep(0.01); continue
                if pct < DESIRED_PCT-PCT_TOL: self.enqueue(f"forward {STEP_FB}"); last=now
                elif pct > DESIRED_PCT+PCT_TOL: self.enqueue(f"back {STEP_FB}"); last=now
                if off > OFFSET_TOL: self.enqueue(f"cw {STEP_YAW}"); last=now
                elif off < -OFFSET_TOL: self.enqueue(f"ccw {STEP_YAW}"); last=now
                time.sleep(0.01)
            
        finally:
            self.cur_box = None; self.cur_pct = None

    # ─────────────────────── proxy & viewer ────────────────────
    class _CapProxy:
        def __init__(self, parent): self.p=parent
        def read(self):
            f=self.p.latest_frame
            return (False,None) if f is None else (True, f.copy())

    def viewer_mainloop(self, window_name="Follow"):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while not self.stop_event.is_set():
                try: frm = self.frame_q.get(timeout=0.05)
                except queue.Empty:
                    if cv2.waitKey(1)&0xFF==ord('q'): self.stop_event.set()
                    continue
                cv2.imshow(window_name, frm)
                if cv2.waitKey(1)&0xFF==ord('q'): self.stop_event.set(); break
        finally: cv2.destroyAllWindows()

    # ─────────────────────── misc helpers ─────────────────────
    def query_battery(self, timeout=2.0):
        self._batt_event.clear()
        self._tx(b"battery?", track_ack=False)
        return self.battery_level if self._batt_event.wait(timeout) else None
