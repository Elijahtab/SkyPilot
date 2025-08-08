# voice_transcriber.py
"""
End-of-speech (EOS) voice transcriber.

Goal
----
Immediately send a transcript **as soon as the user stops speaking**. This module
does small-frame audio capture with VAD-based endpointing, then calls a callback
with the final text for each utterance.

Design
------
- Captures mono 16 kHz audio in short frames (default 30 ms).
- Uses WebRTC VAD if available (py-webrtcvad). Falls back to a robust
  energy-based VAD.
- Detects EOS when consecutive silence exceeds `silence_ms` (default 400 ms).
- On EOS (or when `max_utterance_ms` is reached), transcribes the buffered audio
  and calls the provided `callback(text)` immediately.
- Threaded, "fire-and-forget" API with `start_transcriber(callback)`.

Dependencies
------------
- numpy
- sounddevice
- One of: faster-whisper OR openai-whisper (installed as `whisper`).
  We'll try faster-whisper first, then fall back to openai-whisper.
  If neither is available, we raise at startup.

Public API
----------
    voice_thr, stop_voice = start_transcriber(
        callback,
        sample_rate=16000,
        frame_ms=30,
        silence_ms=400,
        min_utterance_ms=220,
        max_utterance_ms=15000,
        vad_aggressiveness=2,
        model_name="base",
        device=None,
    )

    # Later on shutdown:
    stop_voice()
    voice_thr.join(1)

Notes
-----
- If you want partial streaming while speaking, you can extend `maybe_emit_partial`
  (left as a stub). For now we only emit a **final** transcript on EOS.
- If you previously imported an earlier version of this module, the API remains
  identical: `start_transcriber(cb)` returns `(thread, stop_fn)`.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    raise RuntimeError("sounddevice is required for microphone capture") from e


# ---------- Whisper loader (singleton) ----------
class _Whisper:
    _lock = threading.Lock()
    _model = None
    _use_faster = False

    @classmethod
    def load(cls, model_name: str = "base", device: Optional[str] = None):
        with cls._lock:
            if cls._model is not None:
                return cls._model
            # Try faster-whisper first
            try:
                from faster_whisper import WhisperModel  # type: ignore
                cls._model = WhisperModel(model_name, device=device or "auto")
                cls._use_faster = True
                return cls._model
            except Exception:
                pass
            # Fallback to openai-whisper
            try:
                import whisper  # type: ignore
                cls._model = whisper.load_model(model_name, device=device or None)
                cls._use_faster = False
                return cls._model
            except Exception as e:
                raise RuntimeError(
                    "Neither faster-whisper nor openai-whisper is available. "
                    "Install one of them to enable transcription."
                ) from e

    @classmethod
    def transcribe(cls, audio_f32_mono: np.ndarray):
        m = cls._model or cls.load()
        if cls._use_faster:
            # faster-whisper expects float32 array in -1..1
            segments, info = m.transcribe(audio_f32_mono, vad_filter=False)
            text = " ".join(seg.text for seg in segments).strip()
            return text
        else:
            # openai-whisper expects numpy array or torch tensor
            import whisper  # type: ignore
            # Create a 16k sample rate log-mel from raw float
            # Simpler: use transcribe with fp array
            r = m.transcribe(audio_f32_mono, fp16=False)
            return (r.get("text") or "").strip()


# ---------- VAD helpers ----------
def _has_webrtcvad():
    try:
        import webrtcvad  # type: ignore
        return True
    except Exception:
        return False


@dataclass
class VadCfg:
    sample_rate: int = 16000
    frame_ms: int = 30            # valid for webrtcvad: 10, 20, or 30
    aggressiveness: int = 2       # 0..3
    energy_floor: float = 0.0005  # fallback VAD RMS floor (~-66 dBFS)


class _VAD:
    def __init__(self, cfg: VadCfg):
        self.cfg = cfg
        self._use_webrtc = _has_webrtcvad()
        if self._use_webrtc:
            import webrtcvad  # type: ignore
            self._vad = webrtcvad.Vad(cfg.aggressiveness)
            if cfg.frame_ms not in (10, 20, 30):
                raise ValueError("frame_ms must be 10/20/30 when using WebRTC VAD")

    def is_speech(self, pcm16: bytes) -> bool:
        """Return True if frame contains speech. `pcm16` is little-endian mono 16-bit."""
        if self._use_webrtc:
            try:
                return self._vad.is_speech(pcm16, self.cfg.sample_rate)
            except Exception:
                return False
        # Energy-based fallback
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        return rms >= self.cfg.energy_floor


# ---------- Core transcriber worker ----------
@dataclass
class TranscriberCfg:
    sample_rate: int = 16000
    frame_ms: int = 30
    silence_ms: int = 400
    min_utterance_ms: int = 220
    max_utterance_ms: int = 15000
    vad_aggressiveness: int = 2
    model_name: str = "base"
    device: Optional[str] = None


def start_transcriber(
    callback: Callable[[str], None],
    sample_rate: int = 16000,
    frame_ms: int = 30,
    silence_ms: int = 400,
    min_utterance_ms: int = 220,
    max_utterance_ms: int = 15000,
    vad_aggressiveness: int = 2,
    model_name: str = "base",
    device: Optional[str] = None,
):
    """
    Spawn a background thread that listens to the default microphone,
    detects end-of-speech, and invokes `callback(text)` right away.
    Returns (thread, stop_fn).
    """
    cfg = TranscriberCfg(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        silence_ms=silence_ms,
        min_utterance_ms=min_utterance_ms,
        max_utterance_ms=max_utterance_ms,
        vad_aggressiveness=vad_aggressiveness,
        model_name=model_name,
        device=device,
    )

    # Prepare VAD and Whisper
    vad = _VAD(VadCfg(sample_rate=cfg.sample_rate, frame_ms=cfg.frame_ms, aggressiveness=cfg.vad_aggressiveness))
    _Whisper.load(cfg.model_name, cfg.device)  # lazy-safe; guarantees model is loaded once

    # Frame & timing math
    frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
    bytes_per_frame = frame_samples * 2  # int16
    max_frames = int(cfg.max_utterance_ms / cfg.frame_ms + 0.5)
    min_frames = max(1, int(cfg.min_utterance_ms / cfg.frame_ms + 0.5))
    silence_frames_needed = max(1, int(cfg.silence_ms / cfg.frame_ms + 0.5))

    # Buffers/state
    stop_evt = threading.Event()
    pcm_ring: Deque[bytes] = collections.deque(maxlen=max_frames + 5)
    utter_active = False
    consec_silence = 0
    utter_frames = 0

    def _audio_callback(indata, frames, time_info, status):
        # indata is float32 -1..1; convert to pcm16 bytes per frame_ms block size
        nonlocal utter_active, consec_silence, utter_frames
        if stop_evt.is_set():
            raise sd.CallbackAbort

        # collapse to mono
        x = indata
        if x.ndim == 2 and x.shape[1] > 1:
            x = np.mean(x, axis=1, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        pcm16 = (x * 32767.0).astype(np.int16).tobytes()

        # VAD
        speech = vad.is_speech(pcm16)

        if speech:
            consec_silence = 0
            if not utter_active:
                utter_active = True
                utter_frames = 0
        else:
            consec_silence += 1

        if utter_active:
            pcm_ring.append(pcm16)
            utter_frames += 1

        # EOS conditions
        hit_timeout = utter_active and utter_frames >= max_frames
        hit_silence = utter_active and consec_silence >= silence_frames_needed

        if hit_timeout or hit_silence:
            # collect utterance and reset state
            pcm_bytes = b"".join(pcm_ring)
            pcm_ring.clear()
            utter_active = False
            consec_silence = 0
            utter_frames = 0

            if len(pcm_bytes) >= bytes_per_frame * min_frames:
                # transcribe on a worker thread so we don't block audio callback
                threading.Thread(target=_transcribe_and_emit, args=(pcm_bytes,), daemon=True).start()
            else:
                # too short/noisy; drop
                pass

    def _transcribe_and_emit(pcm_bytes: bytes):
        # Convert PCM16 LE bytes -> float32 -1..1 mono
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            text = _Whisper.transcribe(audio)
        except Exception as e:
            text = ""
        if text:
            try:
                callback(text)
            except Exception:
                # don't crash the thread on user callback errors
                pass

    # Open input stream with fixed blocksize matching one frame
    blocksize = frame_samples
    stream = sd.InputStream(
        channels=1,
        samplerate=cfg.sample_rate,
        dtype="float32",
        blocksize=blocksize,
        callback=_audio_callback,
    )

    def _worker():
        with stream:
            while not stop_evt.is_set():
                time.sleep(0.05)

    thr = threading.Thread(target=_worker, daemon=True, name="voice")
    thr.start()

    def _stop():
        stop_evt.set()

    return thr, _stop
