# drone_controller.py
import socket, queue, threading, time, cv2, skytrack

TELLO = ("192.168.10.1", 8889)
URL   = "udp://0.0.0.0:11111?fifo_size=0&overrun_nonfatal=1&fflags=nobuffer&flags=low_delay"


class DroneController:
    def __init__(self):
        self.is_flying  = False      # physical state (what we believe)
        self.cmd_gate   = False      # True = block *new* commands (e.g. while landing)
        self.cmd_q       = queue.Queue()
        self.telemetry_q = queue.Queue()       # % coverage, etc.
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 0))
        self.sock.settimeout(2)      # 2️permanent 2‑second timeout 
        self.keepalive   = 8
        self.follow_thr  = None
        self.stop_event  = threading.Event()
        

    # ─ SDK helpers ─
    def _tx(self, data: bytes):
        try:
            self.sock.sendto(data, TELLO)
            self.sock.recv(16)        # blocks ≤ 2 s because of the timeout set in __init__
        except socket.timeout:
            print("⚠️ No reply from drone (timeout)")
        except OSError as e:
            print(f"⚠️ UDP error: {e}")


    def start(self):
        # 1) build a NEW event so old threads (if any) stay stopped
        if not self.stop_event.is_set():
            print("⚠️ Controller already running.")
            return

        self.stop_event = threading.Event()    #  recreate fresh event
        self.cmd_gate   = False

        self._tx(b'command');  time.sleep(0.3)
        self._tx(b'streamon'); time.sleep(1.0)

        # 2) launch fresh worker threads that watch the new event
        threading.Thread(target=self._telemetry_worker,
                         daemon=True).start()
        threading.Thread(target=self._cmd_worker,
                         daemon=True).start()
        threading.Thread(target=self._heartbeat,
                         daemon=True).start()
        print("🚀 Controller started")

    def immediate_land(self):
        # 1) clear any queued commands
        with self.cmd_q.mutex:
            self.cmd_q.queue.clear()

        # 2) stop follow thread + flag all workers to exit
        self.stop_event.set()
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join()
            self.follow_thr = None

        # 3) land NOW, bypassing queue
        self._tx(b'land')
        print("🛬 Emergency landing initiated")

        # 4) update state / gate
        self.is_flying = False
        self.cmd_gate  = True      # block new commands until next start()


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
            #log commands sent
            with open("commands.txt", "a") as log:
                log.write(f"{time.strftime('%H:%M:%S')} {cmd}\n")
            # update state:
            if cmd == "takeoff":
                self.is_flying = True
            elif cmd == "land":
                self.is_flying = False


    def _heartbeat(self):
        while not self.stop_event.is_set():
            self._tx(b'rc 0 0 0 0'); time.sleep(self.keepalive)

    # external API ---------------------------------------------------
    def enqueue(self, raw_cmd: str):
        self.cmd_q.put(raw_cmd)

    def follow(self, description: str):
        if self.follow_thr and self.follow_thr.is_alive():
            print("Already following; stop first")
            return
        self.follow_thr = threading.Thread(
            target=self._follow_loop, args=(description,), daemon=True)
        self.follow_thr.start()

    def _follow_loop(self, desc):
        cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
        box = skytrack.acquire_box(cap, desc)        # GPT‑4o → bbox
        tracker = skytrack.init_tracker(cap, box)

        while not self.stop_event.is_set():
            pct, offset = skytrack.track_step(cap, tracker)
            if pct is None: continue
            self.telemetry_q.put(pct)                # optional feedback
            # very dumb control logic
            if abs(offset) > 50:
                self.enqueue('right 20' if offset > 0 else 'left 20')
            if pct < 5:  self.enqueue('forward 20')
            elif pct > 15: self.enqueue('back 20')
        cap.release()

    def stop_follow(self):
        self.stop_event.set()
        if self.follow_thr:
            self.follow_thr.join()
            self.follow_thr = None  
