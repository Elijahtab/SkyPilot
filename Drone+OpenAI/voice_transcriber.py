# voice_transcriber.py
import queue, threading, time, typing as _t
import numpy as np, sounddevice as sd, webrtcvad
from faster_whisper import WhisperModel

_SAMPLE_RATE   = 16_000
_FRAME_MS      = 20
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_MS // 1000
_SILENCE_MS    = 700
_MAX_UTTER_MS  = 5_000
_whisper       = WhisperModel("base.en", device="cpu")

def start_transcriber(
    on_result: _t.Callable[[str], _t.Any],
    *,
    vad_level: int = 1,                 # slightly less aggressive
    device: str|int|None = None,
    debug: bool = False,
) -> tuple[threading.Thread, _t.Callable[[], None]]:
    """
    Background mic → (VAD + energy gate) → Whisper → on_result(text)
    Returns (thread, stop_fn). Thread is daemon.
    """

    stop_evt  = threading.Event()
    pcm_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
    vad = webrtcvad.Vad(vad_level)

    # simple high-pass (pre-emphasis) to knock down low-freq prop hum
    def preemphasis(x: np.ndarray, a: float = 0.97) -> np.ndarray:
        # x is float32 mono
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - a * x[:-1]
        return y

    def _audio_cb(indata, frames, time_info, status):
        if not stop_evt.is_set():
            try:
                pcm_q.put_nowait(indata.copy())
            except queue.Full:
                # drop oldest to avoid growing latency
                try: pcm_q.get_nowait()
                except queue.Empty: pass
                try: pcm_q.put_nowait(indata.copy())
                except queue.Full: pass

    stream = sd.InputStream(
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=_FRAME_SAMPLES,
        callback=_audio_cb,
        device=device,
    )
    stream.start()

    def _flush(buf: list[np.ndarray]):
        if not buf:
            return
        pcm16 = b"".join(b.tobytes() for b in buf)
        buf.clear()
        audio = np.frombuffer(pcm16, np.int16).astype(np.float32) / 32768.0
        segments, _ = _whisper.transcribe(audio)
        text = " ".join(s.text for s in segments).strip()
        print(f"🎤 {text if text else '(no text)'}")
        if text:
            try:
                on_result(text)
            except Exception as e:
                print("on_result error:", e)

    def _worker():
        # --- calibrate noise floor for ~0.8 s
        calib_until = time.time() + 0.8
        rms_samples = []
        while time.time() < calib_until and not stop_evt.is_set():
            try:
                f = pcm_q.get(timeout=0.1)
            except queue.Empty:
                continue
            rms_samples.append(float(np.sqrt(np.mean(f**2))))
        base_rms = (np.median(rms_samples) if rms_samples else 0.01)
        if debug:
            print(f"🎚️  noise floor ~ {base_rms:.5f}")

        # dynamic thresholds
        min_gate   = 0.012          # absolute floor (tune if needed)
        rel_gate_k = 2.0            # must exceed k×baseline to count as speech

        buffer: list[np.ndarray] = []
        lead_in = []                # keep 200ms before speech for context
        max_lead_frames = int(0.2 * 1000 / _FRAME_MS)
        started   = False
        utter_ms  = 0
        silence_ms = 0

        last_adapt = time.time()

        while not stop_evt.is_set():
            try:
                frame = pcm_q.get(timeout=0.1)     # float32, 20ms
            except queue.Empty:
                continue

            # preemphasis before VAD/energy
            f_hp = preemphasis(frame)

            # energy
            rms = float(np.sqrt(np.mean(f_hp**2)))

            # slow baseline adaptation (toward current rms) during non-speech
            now = time.time()
            if not started and (now - last_adapt) > 0.5:
                base_rms = 0.95*base_rms + 0.05*rms
                last_adapt = now

            # VAD
            f_i16 = (f_hp * 32767).astype(np.int16)
            vad_flag = False
            try:
                vad_flag = vad.is_speech(f_i16.tobytes(), _SAMPLE_RATE)
            except Exception:
                pass

            # Hybrid decision: accept if VAD says speech OR energy jumps above gate
            energy_gate = max(min_gate, rel_gate_k * base_rms)
            voiced = vad_flag or (rms > energy_gate)

            if debug and not started:
                # light debug so we can see life while motors are on
                print(f"mic rms={rms:.5f} gate={energy_gate:.5f} vad={int(vad_flag)} voiced={int(voiced)}")

            if voiced:
                if not started:
                    # prepend a little lead-in for Whisper context
                    buffer.extend(lead_in)
                    lead_in.clear()
                    started  = True
                    utter_ms = 0
                    silence_ms = 0
                buffer.append(f_i16)
                utter_ms += _FRAME_MS
            else:
                if started:
                    silence_ms += _FRAME_MS
                    buffer.append(f_i16)  # include trailing context during tail
                    utter_ms += _FRAME_MS
                    if silence_ms >= _SILENCE_MS:
                        _flush(buffer); started = False
                        utter_ms = silence_ms = 0
                else:
                    # build lead-in ring buffer
                    lead_in.append(f_i16)
                    if len(lead_in) > max_lead_frames:
                        lead_in.pop(0)

            # hard cap utterance
            if started and utter_ms >= _MAX_UTTER_MS:
                _flush(buffer); started = False
                utter_ms = silence_ms = 0

        # drain on exit
        if buffer:
            _flush(buffer)
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

    thr = threading.Thread(target=_worker, daemon=True, name="voice")
    thr.start()

    def _stop():
        stop_evt.set()

    return thr, _stop
