# drone_controller.py
import socket, queue, threading, time, cv2
from datetime import datetime
import skytrack
from collections import deque

#TELLO_ADDR = ("192.168.10.1", 8889)
TELLO_ADDR = ("127.0.0.1", 8889)
VIDEO_URL  = ("udp://0.0.0.0:11111"
              "?fifo_size=0&overrun_nonfatal=1&fflags=nobuffer&flags=low_delay") 


class DroneController:
    """Thin-ish wrapper around the Tello command/telemetry loop."""

    HEARTBEAT_SEC = 8
    QUEUE_MAXSIZE = 5

    def __init__(self):
        # --- state ---
        self.video_src = VIDEO_URL  # default to Tello stream
        self._pending_acks = deque()
        self.running      = False
        self.is_flying    = False          # our belief about state
        self.cmd_gate     = False          # block new commands when True (e.g. landing)
        self._last_cmd    = None

        # --- infra ---
        self.sock         = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)

        self.cmd_q        = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self.telemetry_q  = queue.Queue()

        self.state_lock   = threading.RLock()
        self.stop_event   = threading.Event()
        self.follow_stop  = threading.Event()

        self._workers     = []
        self.follow_thr   = None
        self.log_file     = None
        self.address      = TELLO_ADDR

    # ----------------------- lifecycle -----------------------

    def start(self):
        with self.state_lock:
            if self.running:
                print("⚠️ Controller already running.")
                return
            self.running = True
            self.stop_event.clear()
            self.follow_stop.clear()
            self.cmd_gate = True

        # log
        self.log_file = open("commands.txt", "w", buffering=1)

        # enter SDK + start video
        self._tx(b'command');  time.sleep(0.3)
        self._tx(b'streamon'); time.sleep(1.0)

        # spawn workers
        self._spawn(self._recv_loop,        "rx")
        self._spawn(self._telemetry_worker, "telemetry")
        self._spawn(self._cmd_worker,       "cmd")
        self._spawn(self._heartbeat,        "heartbeat")

        print("🚀 Controller started")

    def stop(self):
        # signal all threads
        self.stop_event.set()
        self.follow_stop.set()

        # Optional: stop video nicely
        try:
            self._tx(b'streamoff')
        except Exception:
            pass

        # join workers
        for t in self._workers:
            t.join(timeout=2)
        self._workers.clear()

        # join follow thread
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None

        # close socket
        try:
            self.sock.close()
        except OSError:
            pass

        # close log
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass

        with self.state_lock:
            self.running = False

        print("🛑 Controller stopped.")

    # emergency immediate land (bypass queue)
    def immediate_land(self):
        self.stop_event.set()
        self.follow_stop.set()

        # clear queue
        try:
            while True:
                self.cmd_q.get_nowait()
        except queue.Empty:
            pass

        # stop follow thread if needed
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None

        # send land directly
        self._tx(b'land')
        with self.state_lock:
            self.is_flying = False
            self.cmd_gate = True
        print("🛬 Emergency landing initiated.")

    # ----------------------- public API -----------------------

    def enqueue(self, cmd: str) -> bool:
        """Safe enqueue with gating, full check, and duplicate check."""
        with self.state_lock:
            if self.cmd_gate:
                print("🚧 Command gate closed.")
                return False
            if self.cmd_q.full():
                print("⚠️ Queue full.")
                return False
            self._last_cmd = cmd
        try:
            self.cmd_q.put_nowait(cmd)
            return True
        except queue.Full:
            return False

    def takeoff(self):
        with self.state_lock:
            if self.is_flying:
                print("⚠️ Already flying.")
                return
            self.cmd_gate = False
        self.enqueue("takeoff")
        print("🛫 Takeoff requested.")

    def land(self):
        self.enqueue("land")
        with self.state_lock:
            if not self.is_flying:
                print("⚠️ Already landed.")
                return
            self.cmd_gate = True
        print("🛬 Landing requested.")

    # follow target via CV/LLM
    def start_follow(self, desc: str):
        with self.state_lock:
            if self.cmd_gate:
                print("⚠️ Landed cannot begin follow sequence.")
                return
        if self.follow_thr and self.follow_thr.is_alive():
            print("⚠️ Follow already running.")
            return
        self.follow_stop.clear()
        self.follow_thr = threading.Thread(
            target=self._follow_loop, args=(desc,), daemon=True, name="follow")
        self.follow_thr.start()
        print("🎯 Follow thread started.")

    def stop_follow(self):
        self.follow_stop.set()
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None
        with self.state_lock:
            self.cmd_gate = False
        print("🧱 Follow stopped.")

    # ----------------------- internals -----------------------

    def _spawn(self, fn, name):
        t = threading.Thread(target=fn, daemon=True, name=name)
        t.start()
        self._workers.append(t)

    def _tx(self, cmd: bytes):
        """Low-level UDP send + remember last command for state updates."""
        cmd_str = cmd.decode("utf-8").strip()
        print(cmd_str)
        if cmd_str in ("takeoff", "land"):          # only the ones that change flight state
            self._pending_acks.append(cmd_str)
        self.last_cmd = cmd
        
        try:
            self.sock.sendto(cmd, self.address)
        except OSError as e:
            print(f"[TX] socket error: {e}")
            return
        with self.state_lock:
            self.last_cmd_sent = cmd_str
        print(f"[TX] {cmd.decode()}")
        if self.log_file and not self.log_file.closed:
            try:
                if(cmd_str != "command"):
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_file.write(f"{ts} - {cmd_str}\n")
            except Exception:
                pass

    def _recv_loop(self):
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(1024)
            except OSError:          # or socket.timeout if you set one
                if self.stop_event.is_set():
                    break
                continue

            reply = data.decode("utf-8").strip()
            print(f"[RX] {reply}")

            with self.state_lock:
                if reply == "ok" and self._pending_acks:
                    cmd = self._pending_acks.popleft()
                    print(cmd)
                    if cmd == "takeoff":
                        self.is_flying = True
                        print("✅ Now flying.")
                    elif cmd == "land":
                        self.is_flying = False
                        print("✅ Landed.")
                elif reply == "error":
                    failed = self._pending_acks.popleft() if self._pending_acks else self.last_cmd_sent
                    print(f"❌ Command '{failed}' failed.")


    def _telemetry_worker(self):
        while not self.stop_event.is_set():
            try:
                pct = self.telemetry_q.get(timeout=0.5)
                print(f"📊 Cover: {pct:.1f}%")
            except queue.Empty:
                pass

    def _cmd_worker(self):
        while not self.stop_event.is_set():
            try:
                cmd = self.cmd_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._tx(cmd.encode())

    def _heartbeat(self):
        while not self.stop_event.is_set():
            self._tx(b'command')
            time.sleep(self.HEARTBEAT_SEC)

    # ----------------------- follow logic -----------------------

    def _follow_loop(self, desc: str):
        cap = cv2.VideoCapture(self.video_src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("❌ Could not open video stream.")
            return

        tracker = None
        try:
            box = skytrack.acquire_box(cap, desc)
            _, frame = cap.read()
            if box:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("boxed_frame.jpg", frame)  # or cv2.imshow(...) for preview
            if box:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(cap.read()[1], tuple(box.values()))
            else:
                print("❌ No box acquired; aborting follow.")
                cap.release()
                return
        except Exception as e:
            print(f"❌ acquire_box failed: {e}")
            cap.release()
            return

        while not (self.stop_event.is_set() or self.follow_stop.is_set()):
            pct, offset = skytrack.track_step(cap, tracker, self.telemetry_q)
            if pct is None:
                continue
            print("Following")
            """
            # naive control logic
            if abs(offset) > 50:
                self.enqueue('right 20' if offset > 0 else 'left 20')
            if pct < 5:
                self.enqueue('forward 20')
            elif pct > 15:
                self.enqueue('back 20')
            """
        cap.release()
