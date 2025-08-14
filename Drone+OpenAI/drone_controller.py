# drone_controller.py
import socket, queue, threading, time, cv2, os
from datetime import datetime
import skytrack
from functools import partial         

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
        self.cur_box = None          # latest (x, y, w, h) or None, used to get box from skytrack to drone_controller
        self.cur_pct = None          # used for pct of screen from skytrack

        self._cap_proxy = self._CapProxy(self)
        # battery
        self.battery_level = None
        self._batt_event   = threading.Event()

        # ─── RC streaming state ────────────────────────────
        self._rc_lock   = threading.Lock()
        self._rc_state  = {"lr": 0, "fb": 0, "ud": 0, "yaw": 0}
        self._rc_stop   = threading.Event()
        self._rc_pause  = threading.Event()

        self.voice_thr = None
        self.stop_voice = None
        self.voice_thr_should_join = False
        self._stopping   = threading.Event()
        self._video_cap   = None
        self._local_writer = None
        self._local_writer_path = None
        self._local_size = None
        self._local_frames = 0


    # ----------------------- lifecycle -----------------------

    def start(self):
        skytrack.open_video_logger('flight_logs', fps=30)

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
        self._spawn(lambda: self._rc_pump_loop(hz=50), "rc_pump")
        self.pause_rc()

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
            print("⚠️ Battery query timeout - check 📶 Wifi and 🪫 Tello Power")

        print("🚀 Controller started")

    def stop(self) -> None:
        """Clean, idempotent shutdown. Safe to call from any thread."""
        # Prevent re-entry
        if self._stopping.is_set():
            return
        self._stopping.set()

        # Signal everyone to stop
        self.stop_event.set()
        self.follow_stop.set()
        self._rc_stop.set()         # stop RC pump early

        # Wake the viewer (main thread) so viewer_mainloop breaks immediately
        self._poke_viewer()

        # Best-effort tell Tello to stop the stream (don’t block on ack)
        try:
            self._tx(b"streamoff", track_ack=False)
        except Exception:
            pass

        # Just give it a moment to notice stop_event and exit.
        for t in list(self._workers):
            try:
                t.join(timeout=2)
            except Exception:
                pass
        self._workers.clear()

        # Follow thread (if any)
        if self.follow_thr and self.follow_thr.is_alive():
            try:
                self.follow_thr.join(timeout=2)
            except Exception:
                pass
        self.follow_thr = None

        # Stop voice (if registered); don’t join here if this thread IS the voice thread
        if self.stop_voice:
            try:
                self.stop_voice()
            except Exception:
                pass
            self.stop_voice = None
            self.voice_thr_should_join = True

        # Close UDP socket so recvfrom() unblocks
        try:
            self.sock.close()
        except OSError:
            pass

        # Flush command log
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass
        self.log_file = None

        # Close the video logger AFTER the reader has exited, so MP4 is finalized
        try:
            skytrack.close_video_logger()
        except Exception:
            pass

        with self.state_lock:
            self.running = False

        print("🛑 Controller stopped.")





    def join_threads(self):
        if self.voice_thr_should_join and self.voice_thr:
            if threading.current_thread() != self.voice_thr:
                self.voice_thr.join(timeout=2)
            self.voice_thr = None
            self.voice_thr_should_join = False
    def register_voice_control(self, voice_thread, stop_fn):
        self.voice_thr = voice_thread
        self.stop_voice = stop_fn
        
    def immediate_land(self):
        """Bypass queue and force a land now."""
        self.set_rc(lr=0, fb=0, ud=0, yaw=0)
        self.stop_event.set()
        self.follow_stop.set()
        self._poke_viewer()
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
        self.enqueue("command")
        print("🛫 Takeoff requested.")

    def land(self):
        self.enqueue("land")
        print("🛬 Landing requested.")

    def start_follow(self, desc: str):
        self.resume_rc() 
        skytrack.open_video_logger("flight_logs", fps=30) 

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
        self.pause_rc()
        self.follow_stop.set()
        if self.follow_thr and self.follow_thr.is_alive():
            self.follow_thr.join(timeout=2)
        self.follow_thr = None

        # ── clear overlays ─────────────────────────────
        self.cur_box = None
        self.cur_pct = None
        #clear rc
        self.set_rc(lr=0, fb=0, ud=0, yaw=0)

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
    
    def set_rc(self, *, lr=None, fb=None, ud=None, yaw=None, flush=True):
        """Update any subset of the four stick axes (−100…100)."""
        with self._rc_lock:
            if lr  is not None: self._rc_state["lr"]  = int(max(-100, min(100, lr)))
            if fb  is not None: self._rc_state["fb"]  = int(max(-100, min(100, fb)))
            if ud  is not None: self._rc_state["ud"]  = int(max(-100, min(100, ud)))
            if yaw is not None: self._rc_state["yaw"] = int(max(-100, min(100, yaw)))
            
            if flush and not self._rc_pause.is_set():
                pkt = f"rc {self._rc_state['lr']} {self._rc_state['fb']} " \
                    f"{self._rc_state['ud']} {self._rc_state['yaw']}"
                try:
                    self.sock.sendto(pkt.encode(), self.address)
                except OSError:
                    pass
    # ─────────── pause / resume RC streaming ────────────
    def pause_rc(self):
        """Freeze RC output (the last packet stays in effect on the drone)."""
        self._rc_pause.set()

    def resume_rc(self):
        """Re-enable continuous RC streaming."""
        self._rc_pause.clear()

    def _rc_pump_loop(self, hz: int = 20):
        """
        Stream an `rc a b c d` packet at `hz` Hz (default 20 Hz ⇒ every 50 ms).

        • Uses self._rc_state  (dict with keys 'lr','fb','ud','yaw')
        • Pauses if self._rc_pause  is set   (resume with .resume_rc())
        • Terminates when self.stop_event  or self._rc_stop are set
        """
        period = 1.0 / float(hz)
        last_packet = None

        while not (self.stop_event.is_set() or self._rc_stop.is_set()):
            # If paused, just wait for one period and continue
            if self._rc_pause.is_set():
                time.sleep(period)
                continue

            # Build the current packet
            with self._rc_lock:                       # protect shared dict
                pkt = f"rc {self._rc_state['lr']} {self._rc_state['fb']} " \
                    f"{self._rc_state['ud']} {self._rc_state['yaw']}"

            # Avoid spamming the console with duplicates
            if pkt != last_packet:
                print("→", pkt)
                last_packet = pkt

            # Send to the drone
            try:
                self.sock.sendto(pkt.encode(), self.address)
            except OSError:
                pass                                   # ignore socket hiccups

            time.sleep(period)

        # Cleanup (optional): send neutral sticks once on exit
        try:
            self.sock.sendto(b"rc 0 0 0 0", self.address)
        except OSError:
            pass

        print("🛑 RC pump thread exited.")


    # ----------------------- follow logic -----------------------
    def _follow_loop(self, desc: str):
        cap = self._cap_proxy

        # wait until first frame arrives
        while self.latest_frame is None and not self.stop_event.is_set():
            time.sleep(0.01)

        # 1) one‑shot detection (skytrack decides GPT vs YOLO)
        try:
            frame_inf, box = skytrack.acquire_target_box(cap, desc)
        except Exception as e:
            print(f"❌ initial detection failed: {e}")
            return

        # 2) debug overlay & store
        x, y, w, h = map(int, (box["x"], box["y"], box["w"], box["h"]))
        dbg = frame_inf.copy()
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("boxed_first_frame.jpg", dbg)
        self.cur_box = (x, y, w, h)

        # 3) initialise tracker on the SAME frame
        tracker = skytrack.init_tracker(cap, box, frame=frame_inf)

        pct, off = skytrack.track_step(cap, tracker, telemetry_q=self.telemetry_q, parent=self)
        if pct is None:
            print("⚠️ initial track_step returned None; aborting follow")
            return
        DES_PCT, TOL = pct, pct * .25

        pixels_per_deg = 11.62
        Kp             = 0.9    

        while not (self.stop_event.is_set() or self.follow_stop.is_set()):
            pct, off = skytrack.track_step(
                cap, tracker, telemetry_q=self.telemetry_q, parent=self
            )
            if pct is None:
                time.sleep(0.01)
                continue
            
            deg_error = off / pixels_per_deg
            
            # yaw
            cmd_degrees = int(Kp * deg_error) # round to an int Tello accepts
            cmd_degrees = max(-30, min(30, cmd_degrees))  # clamp for safety

            if abs(cmd_degrees) >= 5:
                yaw_rate = int(max(-100, min(100, cmd_degrees * 3)))  # scale as you like
                self.set_rc(yaw=yaw_rate)
            else:
                self.set_rc(yaw=0)
            
            FB_RATE  = 40

            # ───────── forward / back via RC stick ─────────
            if pct < DES_PCT - TOL:            # target too small → move forward
                self.set_rc(fb=FB_RATE)
            elif pct > DES_PCT + TOL:          # target too big → move backward
                self.set_rc(fb=-FB_RATE)
            else:                              # inside tolerance → hover forward axis
                self.set_rc(fb=0)
            
           
            time.sleep(0.01)
        

        self.set_rc(lr=0, fb=0, ud=0, yaw=0)
        self.cur_box = self.cur_pct = None


    class _CapProxy:
        """cv2.VideoCapture.read()-like shim using latest_frame."""
        def __init__(self, parent):
            self.p = parent
        def read(self):
            f = self.p.latest_frame
            if f is None:
                return False, None
            return True, f.copy()

    def _poke_viewer(self):
        # Wake viewer_mainloop if it’s blocked on .get()
        try:
            self.frame_q.put_nowait(None)   # None = sentinel
        except queue.Full:
            # Replace oldest and insert sentinel
            try:
                _ = self.frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_q.put_nowait(None)
            except queue.Full:
                pass

    def _open_local_writer(self, frm):
        os.makedirs("flight_logs", exist_ok=True)
        h, w = frm.shape[:2]

        # H.264 likes even dims; MJPG doesn't care, but cropping is harmless.
        if (w % 2) or (h % 2):
            frm = frm[:h - (h % 2), :w - (w % 2)]
            h, w = frm.shape[:2]

        # Allow forcing the container/codec via env var:
        # SKY_VIDEO_FORCE=avi | mjpg | mp4 | h264 | avc1
        force = os.getenv("SKY_VIDEO_FORCE", "").lower()
        if force in ("avi", "mjpg", "mjpeg"):
            candidates = [(".avi", "MJPG")]
        elif force in ("mp4", "h264", "avc1"):
            candidates = [(".mp4", "avc1"), (".mp4", "mp4v")]
        else:
            # Prefer AVI/MJPG first for maximum compatibility on macOS builds.
            candidates = [
                (".avi", "MJPG"),   # ✅ safest
                (".mp4", "avc1"),   # H.264 (often not actually available)
                (".mp4", "mp4v"),   # MPEG-4 Part 2
            ]

        ts = datetime.now().strftime("tello_%Y%m%d_%H%M%S")
        fps = 30

        for ext, four in candidates:
            path = os.path.join("flight_logs", f"{ts}{ext}")
            fourcc = cv2.VideoWriter_fourcc(*four)
            wr = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if wr is not None and wr.isOpened():
                self._local_writer = wr
                self._local_writer_path = path
                self._local_size = (w, h)
                self._local_frames = 0
                print(f"🎥 local video logging → {path} [{four}]")
                return frm
            try:
                wr.release()
            except Exception:
                pass

        print("⚠️ No working local codec; disabling local video save.")
        return frm



    def viewer_mainloop(self, window_name="Follow"):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_q.get(timeout=0.05)
                except queue.Empty:
                    # still pump GUI so window is responsive
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop_event.set()
                    continue

                # ← sentinel means “please exit the viewer”
                if frame is None:
                    break

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
        finally:
            cv2.destroyAllWindows()
            print("🎥 viewer exited")


    def _video_reader_loop(self):
        cap = cv2.VideoCapture(self.video_src, cv2.CAP_FFMPEG)
        self._video_cap = cap
        try:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                print("❌ Could not open video stream.")
                return

            while True:
                if self.stop_event.is_set():
                    break

                ok, frm = cap.read()

                
                if not ok or frm is None:
                    time.sleep(0.01)
                    continue

                self.latest_frame = frm
                # init local writer on first good frame
                if self._local_writer is None:
                    frm = self._open_local_writer(frm)  # may crop to even dims
                    # keep latest_frame raw size; if _open_local_writer cropped frm,
                    # that's only for the writer path

                # ensure size matches writer (some streams can shift)
                if self._local_size and (frm.shape[1], frm.shape[0]) != self._local_size:
                    frm = cv2.resize(frm, self._local_size)

                # draw overlays on a COPY so tracker always sees raw latest_frame
                vis = frm.copy()
                if self.cur_box is not None:
                    x, y, w, h = self.cur_box
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if self.cur_pct is not None:
                        text = f"{self.cur_pct:4.1f}%"
                        ty = max(20, y - 10)
                        cv2.putText(vis, text, (x, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # 1) write the OVERLAYED frame to the global skytrack logger
                try:
                    skytrack._save_frame(vis)  # same writer used by follow
                except Exception:
                    pass

                # 2) write the OVERLAYED frame to your local file (AVI/MP4)
                if self._local_writer is not None and self._local_writer.isOpened():
                    try:
                        self._local_writer.write(vis)
                        self._local_frames += 1
                    except Exception:
                        pass

                # 3) show the OVERLAYED frame in the viewer
                try:
                    if self.frame_q.full():
                        self.frame_q.get_nowait()
                    self.frame_q.put_nowait(vis)
                except queue.Full:
                    pass
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self._video_cap = None

            # finalize local writer
            if self._local_writer is not None:
                try:
                    self._local_writer.release()
                except Exception:
                    pass
                if self._local_writer_path:
                    print(f"💾 local video saved → {self._local_writer_path} ({self._local_frames} frames)")
                self._local_writer = None
                self._local_writer_path = None
                self._local_size = None
                self._local_frames = 0

            print("🎥 video_reader exited")




    def register_voice_control(self, voice_thread, stop_fn):
        self.voice_thr = voice_thread
        self.stop_voice = stop_fn

    def schedule_stop(self):
        # run stop() on its own daemon thread
        threading.Thread(target=self.stop, name="dc_stop", daemon=True).start()


    def query_battery(self, timeout=2.0):
        """Send 'battery?' and wait for integer reply."""
        self._batt_event.clear()
        self._tx(b"battery?", track_ack=False)
        if self._batt_event.wait(timeout):
            return self.battery_level
        return None

    