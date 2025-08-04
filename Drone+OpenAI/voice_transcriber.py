# voice_transcriber.py
"""
Minimal “fire-and-forget” voice-to-text helper.

Usage from any script (e.g. openapi.py):

    from voice_transcriber import start_transcriber

    # start a background listener; feed every transcript to promptgpt()
    voice_thr, stop_voice = start_transcriber(promptgpt)

    ...
    # on shutdown:
    stop_voice()          # signal the thread to exit
    voice_thr.join(1)     # optional: wait a moment
"""

from __future__ import annotations
import queue, threading, time, typing as _t

import numpy as np, sounddevice as sd, webrtcvad           # pip install sounddevice webrtcvad
from faster_whisper import WhisperModel                    # pip install faster-whisper


# ─── constants ────────────────────────────────────────────────────────────────
_SAMPLE_RATE   = 16_000            # Hz  (what Whisper expects)
_FRAME_MS      = 20                # VAD-legal frame: 10 / 20 / 30 ms
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_MS // 1000     # 320 for 20 ms
_SILENCE_MS    = 700               # gap that ends an utterance
_MAX_UTTER_MS  = 5_000             # hard cap (safety)

# ─── model is loaded exactly once, then reused by every thread ───────────────
_whisper = WhisperModel("base.en", device="cpu")           # or "tiny.en"


# ─────────────────────────────────────────────────────────────────────────────
def start_transcriber(
    on_result: _t.Callable[[str], _t.Any],
    *,
    vad_level: int       = 2,          # 0=aggressive, 3=lenient
    device:    str|int  = None,       # pick a sounddevice.InputStream device
) -> tuple[threading.Thread, _t.Callable[[], None]]:
    """
    Launch a background thread that pushes each recognised utterance
    to `on_result(text)`.  Returns (thread, stop_fn).
    """

    stop_evt  = threading.Event()
    pcm_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
    vad       = webrtcvad.Vad(vad_level)

    # ── audio callback → put float32 frames in queue ─────────────────────────
    def _audio_cb(indata, frames, time_info, status):
        if not stop_evt.is_set():
            pcm_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=_FRAME_SAMPLES,
        callback=_audio_cb,
        device=device,
    )
    stream.start()

    # ── worker thread: VAD + Whisper ────────────────────────────────────────
    def _worker():
        buffer: list[np.ndarray] = []
        started   = False
        start_ts  = 0.0

        while not stop_evt.is_set():
            try:
                frame = pcm_q.get(timeout=0.1)  # float32, 20 ms
            except queue.Empty:
                continue

            frame_i16 = (frame * 32767).astype(np.int16)
            is_speech = vad.is_speech(frame_i16.tobytes(), _SAMPLE_RATE)

            if is_speech:
                if not started:
                    started  = True
                    start_ts = time.time()
                buffer.append(frame_i16)
            elif started and (time.time() - start_ts)*1000 >= _SILENCE_MS:
                _flush(buffer, on_result)
                started = False

            # safety cap
            if started and (time.time() - start_ts)*1000 >= _MAX_UTTER_MS:
                _flush(buffer, on_result)
                started = False

        # drain on exit
        if buffer:
            _flush(buffer, on_result)
        stream.stop()
        stream.close()

    def _flush(buf: list[np.ndarray], cb):
        if not buf:
            return
        pcm16 = b"".join(b.tobytes() for b in buf)
        buf.clear()
        # Whisper expects float32 -1…1
        audio = np.frombuffer(pcm16, np.int16).astype(np.float32) / 32768.0
        segments, _ = _whisper.transcribe(audio)
        text = " ".join(s.text for s in segments).strip()
        if text:
            cb(text)

    thr = threading.Thread(target=_worker, daemon=True, name="voice")
    thr.start()

    def _stop():
        stop_evt.set()

    return thr, _stop
