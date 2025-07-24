# drone_controller.py
import socket, queue, threading, time, cv2
from datetime import datetime
import skytrack

TELLO_ADDR = ("192.168.10.1", 8889)
# TELLO_ADDR = ("127.0.0.1", 8889)  # for fake_tello tests

VIDEO_URL = ("udp://0.0.0.0:11111"
             "?fifo_size=0&overrun_nonfatal=1&fflags=nobuffer&flags=low_delay")


class DroneController:
    """Thin wrapper around the Tello command/telemetry loop."""

    HEARTBEAT_SEC = 8
    QUEUE_MAXSIZE = 5

    def __init__(self):
        self.inter_command_delay = .3
        self.frame_q = queue.Queue(maxsize=2)
        self.latest_frame = None

        # ---- state ----
        self.video_src           = VIDEO_URL
        self.running             = False
        self.waiting_flight_ack  = None   # "takeoff" or "land"
        self.last_cmd_sent       = None
        self.last_cmd            = None

        # ---- infra ----
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self.sock.bind(('', 8889))        # receive replies

        self.cmd_q       = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self.telemetry_q = queue.Queue()

        self.state_lock  = threading.RLock()
        self.stop_event  = threading.Event()
        self.follow_stop = threading.Event()

        self._workers    = []
        self.follow_thr  = None
        self.log_file    = None
        self.address     = TELLO_ADDR

        self.ack_event = threading.Event()
        self.ack_event.set()

        self._cap_proxy = self._CapProxy(self)
        # battery
        self.battery_level = None
        self._batt_event   = threading.Event()

    # ----------------------- lifecycle -----------------------

    def start(self):
        with self.state_lock:
            if self.running:
                print("⚠️ Controller already running.")
                return
            self.running = True
            self.stop_event.clear()
            self.follow_stop.clear()

        self.log_file = open("commands.txt", "w", buffering=1)

        # workers (RX first)
        self._spawn(self._recv_loop,        "rx")
        self._spawn(self._telemetry_worker, "telemetry")
        self._spawn(self._cmd_worker,       "cmd")
        self._spawn(self._heartbeat,        "heartbeat")

        # enter SDK + start video
        self._tx(b'command',  track_ack=False); time.sleep(0.3)
        self._tx(b'streamon', track_ack=False); time.sleep(1.0)

        # start grabbing frames immediately
        self._spawn(self._video_reader_loop, "video_reader")

        print("🚀 Controller started")
        # query battery right away
        lvl = self.query_battery(timeout=2.0)
        if lvl is not None:
            print(f"🔋 Battery: {lvl}%")
        else:
            print("⚠️ Battery query timeout")

        print("🚀 Controller started")

    def stop(self):
        self.stop_event.set()
        self.follow_stop.set()

        try:
            self._tx(b'streamoff', track_ack=False)
        except Exception:
            pass

        for t in self._workers:
            t.join(timeout=2)
        self._workers.clear()

        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None

        try:
            self.sock.close()
        except OSError:
            pass

        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass

        with self.state_lock:
            self.running = False

        print("🛑 Controller stopped.")

    def immediate_land(self):
        """Bypass queue and force a land now."""
        self.stop_event.set()
        self.follow_stop.set()

        try:
            while True:
                self.cmd_q.get_nowait()
        except queue.Empty:
            pass

        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None

        self._tx(b'land', track_ack=True)
        print("🛬 Emergency landing initiated.")

    # ----------------------- public API -----------------------

    def enqueue(self, cmd: str) -> bool:
        try:
            self.cmd_q.put_nowait(cmd)
            return True
        except queue.Full:
            print("⚠️ Queue full.")
            return False

    def takeoff(self):
        self.enqueue("takeoff")
        print("🛫 Takeoff requested.")

    def land(self):
        self.enqueue("land")
        print("🛬 Landing requested.")

    def start_follow(self, desc: str):
        if self.follow_thr and self.follow_thr.is_alive():
            print("⚠️ Follow already running.")
            return
        self.follow_stop.clear()
        self.follow_thr = threading.Thread(
            target=self._follow_loop, args=(desc,), daemon=True, name="follow"
        )
        self.follow_thr.start()
        print("🎯 Follow thread started.")

    def stop_follow(self):
        self.follow_stop.set()
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None
        print("🧱 Follow stopped.")

    # ----------------------- internals -----------------------

    def _spawn(self, fn, name):
        t = threading.Thread(target=fn, daemon=True, name=name)
        t.start()
        self._workers.append(t)

    def _tx(self, cmd: bytes, track_ack: bool = True):
        cmd_str = cmd.decode('utf-8').strip()

        if track_ack and cmd_str in ("takeoff", "land"):
            with self.state_lock:
                self.waiting_flight_ack = cmd_str

        self.last_cmd = cmd
        if cmd_str.endswith("?"):
            self._last_query = cmd_str
        try:
            self.sock.sendto(cmd, self.address)
        except OSError as e:
            print(f"[TX] socket error: {e}")
            return

        with self.state_lock:
            self.last_cmd_sent = cmd_str

        print(f"[TX] {cmd_str}")
        if self.log_file and not self.log_file.closed and cmd_str != "command":
            try:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"{ts} - {cmd_str}\n")
            except Exception:
                pass

    def _recv_loop(self):
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(1024)
            except OSError:
                if self.stop_event.is_set():
                    break
                continue

            reply = data.decode("utf-8", errors="ignore").strip()
            print(f"[RX] {reply}")

            with self.state_lock:
                # battery? returns a plain integer string
                if reply.isdigit() and self._last_query == "battery?":
                    self.battery_level = int(reply)
                    self._batt_event.set()
                    # don't return; Tello also sends "ok" sometimes after queries
                if reply == "ok":
                    if self.waiting_flight_ack:
                        print(f"✅ {self.waiting_flight_ack} ok")
                        self.waiting_flight_ack = None
                    self.ack_event.set()
                elif reply.startswith("error") or reply.startswith("out of range"):
                    if self.waiting_flight_ack:
                        print(f"❌ '{self.waiting_flight_ack}' failed: {reply}")
                        self.waiting_flight_ack = None
                    else:
                        print(f"❌ '{self.last_cmd_sent}' failed: {reply}")
                    self.ack_event.set()

    def _telemetry_worker(self):
        while not self.stop_event.is_set():
            try:
                pct = self.telemetry_q.get(timeout=0.5)
                #print(f"📊 Cover: {pct:.1f}%")
            except queue.Empty:
                pass

    def _cmd_worker(self):
        while not self.stop_event.is_set():
            try:
                cmd = self.cmd_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self.ack_event.wait()
            self.ack_event.clear()
            time.sleep(0.05)

            track = cmd in ("takeoff", "land")
            self._tx(cmd.encode(), track_ack=track)
            time.sleep(self.inter_command_delay)

    def _heartbeat(self):
        while not self.stop_event.is_set():
            self._tx(b'command', track_ack=False)
            time.sleep(self.HEARTBEAT_SEC)

    # ----------------------- follow logic -----------------------
    def _follow_loop(self, desc: str):
        cap = self._cap_proxy
        out_q = self.frame_q

        # wait until we have at least one frame
        while self.latest_frame is None and not self.stop_event.is_set():
            time.sleep(0.01)

        try:
            box = skytrack.acquire_box(cap, desc, out_q=out_q)
            ok, frame = cap.read()
            if not box or not ok or frame is None:
                print("❌ No box/frame; aborting follow.")
                return

            tracker = skytrack.init_tracker(frame, box)

            while not (self.stop_event.is_set() or self.follow_stop.is_set()):
                try:
                    pct, offset = skytrack.track_step(
                        cap, tracker, telem_q=self.telemetry_q, out_q=out_q
                    )
                except cv2.error as e:
                    print(f"OpenCV error in follow loop: {e}")
                    break

                if pct is None:
                    continue

                # control logic here if you want
                time.sleep(0.01)

        finally:
            pass  # nothing to release

    class _CapProxy:
        """cv2.VideoCapture.read()-like shim using latest_frame."""
        def __init__(self, parent):
            self.p = parent
        def read(self):
            f = self.p.latest_frame
            if f is None:
                return False, None
            return True, f.copy()

    def viewer_mainloop(self, window_name="Follow"):
        """Run this on the MAIN THREAD (macOS OpenCV requirement)."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_q.get(timeout=0.05)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_event.set()
                    continue

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
        finally:
            cv2.destroyAllWindows()

    def _video_reader_loop(self):
        cap = cv2.VideoCapture(self.video_src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("❌ Could not open video stream.")
            return
        try:
            while not self.stop_event.is_set():
                ok, frm = cap.read()
                if not ok or frm is None:
                    time.sleep(0.01)
                    continue
                self.latest_frame = frm
                try:
                    if self.frame_q.full():
                        self.frame_q.get_nowait()
                    self.frame_q.put_nowait(frm)
                except queue.Full:
                    pass
        finally:
            cap.release()

    def query_battery(self, timeout=2.0):
        """Send 'battery?' and wait for integer reply."""
        self._batt_event.clear()
        self._tx(b"battery?", track_ack=False)
        if self._batt_event.wait(timeout):
            return self.battery_level
        return None

