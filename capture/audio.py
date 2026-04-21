"""
Audio capture with hardened VAD and Faster Whisper STT.
RESTORED: Multi-device hardware binding (Mic/System/WASAPI).
LAYER 6: Integrated restart() for hot-swapping audio modes in settings.
"""

import collections
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Suppress HuggingFace symlinks warning on Windows.
# The cache works fine without symlinks (copies files instead) — this just
# silences the noisy warning that appears every time Whisper is loaded.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioCapture(QObject):
    transcription_ready = pyqtSignal(str)
    level = pyqtSignal(float)

    def __init__(self, config, state=None):
        super().__init__()
        self.config = config
        self._running = False
        self._paused = False
        self._muted = False
        self._capture_thread = None
        self._process_thread = None
        self.q = queue.Queue(maxsize=100)
        self.transcripts = collections.deque(
            maxlen=config.get("performance.max_history", 50)
        )
        self.sr = config.get("capture.audio.sample_rate", 16000)
        self.capture_mode = config.get("capture.audio.mode", "system")
        self.last_mode = self.capture_mode

        self.block_ms = 200
        self.block_size = int(self.sr * self.block_ms / 1000)
        self.silence_blocks = int(900 / self.block_ms)  # 900ms balanced default

        self.model = None
        self._model_name = config.get("capture.audio.whisper_model", "tiny")
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._current_rms = 0.0
        self._last_transcription_metrics = {}

        # Single-worker pool: Whisper transcription runs off the VAD thread.
        # This means the VAD loop can immediately resume listening while the
        # previous speech segment is being transcribed in the background.
        self._transcribe_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper"
        )

        self._active_streams = []
        self._lock = threading.RLock()

        if state is None:
            from core.state import AppState

            state = AppState(config)

        self._state = state
        self._state.muted_changed.connect(self._on_state_mute_changed)
        self._muted = self._state.is_muted
        self._paused = self._muted

    def _on_state_mute_changed(self, muted: bool):
        logger.debug(f"Audio: State sync -> Muted={muted}")
        self._muted = muted
        self._paused = muted
        if not muted:
            self._drain_queue()
            logger.debug("Audio: Buffer flushed on unmute.")

    def _ensure_whisper_loaded(self):
        with self._model_lock:
            if self._model_loaded:
                return
            try:
                from faster_whisper import WhisperModel

                # GPU auto-detection: CUDA float16 is 3–5x faster than CPU int8.
                # Falls back gracefully to CPU if no CUDA-capable GPU is found.
                try:
                    import torch
                    if torch.cuda.is_available():
                        device, compute = "cuda", "float16"
                        logger.info(f"Whisper: CUDA GPU detected — using float16")
                    else:
                        device, compute = "cpu", "int8"
                except ImportError:
                    device, compute = "cpu", "int8"

                self.model = WhisperModel(
                    self._model_name, device=device, compute_type=compute
                )
                self._model_loaded = True
                logger.info(f"✅ Whisper Ready: {self._model_name} on {device}")
            except Exception as e:
                logger.error(f"Whisper Error: {e}")
                self._model_loaded = False

    def start(self):
        with self._lock:
            if self._running:
                return
            if not self.config.get("capture.audio.enabled", True):
                logger.info("Audio capture disabled in config, skipping start")
                return

            self.capture_mode = self.config.get("capture.audio.mode", self.capture_mode)
            self.last_mode = self.capture_mode
            self._running = True
            self._paused = self._muted
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="audio-cap"
            )
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True, name="audio-proc"
            )
            self._capture_thread.start()
            self._process_thread.start()

    def stop(self):
        """Stop capture cleanly and release active hardware handles."""
        with self._lock:
            self._running = False
            self._close_streams()
            self._drain_queue()
            logger.info("🎙️ Audio Capture Stopped.")

    def toggle(self) -> bool:
        """Toggle muted state and return the new mute status."""
        self._muted = not self._muted
        self._paused = self._muted
        if not self._muted:
            self._drain_queue()
        logger.info(f"🎤 Audio {'Muted' if self._muted else 'Unmuted'}")
        return self._muted

    def restart(self):
        """
        Full hot-restart for audio mode changes.
        We stop and reopen streams instead of trying to patch live state.
        """
        new_mode = self.config.get("capture.audio.mode", "system")
        with self._lock:
            was_running = self._running
            if was_running and new_mode == self.last_mode:
                logger.debug(f"🎙️ Audio: Mode '{new_mode}' already active. Skipping restart.")
                return

            logger.info(f"🎤 Restarting Audio Pipeline: {new_mode}...")
            self._running = False
            self._close_streams()
            self._drain_queue()
            self.capture_mode = new_mode or "system"
            self.last_mode = self.capture_mode

        if was_running:
            time.sleep(0.4)
            self.start()

    def _close_streams(self):
        for s in list(self._active_streams):
            try:
                if s and hasattr(s, "stop"):
                    s.stop()
                if s and hasattr(s, "close"):
                    s.close()
            except Exception as e:
                logger.warning(f"Audio stream close error: {e}")
        self._active_streams.clear()

    def _drain_queue(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def _find_system_audio_source(self):
        import sounddevice as sd

        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0 and any(
                    x in d["name"].lower()
                    for x in ["stereo mix", "cable", "vb-audio", "what u hear"]
                ):
                    return i, d["name"], False

            apis = sd.query_hostapis()
            wasapi_idx = next(
                (i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None
            )
            if wasapi_idx is not None:
                for i, d in enumerate(devices):
                    if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
                        return i, d["name"], True

            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0 and any(
                    x in d["name"].lower()
                    for x in ["microphone", "mic", "headset", "input"]
                ):
                    return i, d["name"], False
        except Exception as e:
            logger.debug(f"Source discovery error: {e}")
        return None, "Default", False

    def _resample_to_target_rate(self, indata, source_rate):
        if source_rate == self.sr:
            return indata.astype(np.float32, copy=False)

        if indata.ndim == 1:
            indata = indata.reshape(-1, 1)

        frame_count = indata.shape[0]
        if frame_count <= 1:
            return indata.astype(np.float32, copy=False)

        target_frames = max(1, int(round(frame_count * self.sr / source_rate)))
        src_x = np.linspace(0.0, 1.0, frame_count, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, target_frames, endpoint=False)

        resampled = np.empty((target_frames, indata.shape[1]), dtype=np.float32)
        for ch in range(indata.shape[1]):
            resampled[:, ch] = np.interp(dst_x, src_x, indata[:, ch])
        return resampled

    def _capture_loop(self):
        import sounddevice as sd

        def make_cb(source_rate):
            def cb(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Audio Status: {status}")
                if not self._paused and self._running:
                    try:
                        normalized = self._resample_to_target_rate(indata, source_rate)
                        data = (
                            np.mean(normalized, axis=1, keepdims=True).astype(np.float32)
                            if normalized.shape[1] > 1
                            else normalized.copy().astype(np.float32)
                        )
                        self.q.put_nowait(data)
                        self._current_rms = float(np.sqrt(np.mean(normalized**2)))
                    except queue.Full:
                        logger.debug("Audio queue full; dropping frame")

            return cb

        mode = self.capture_mode
        try:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if not self._running:
                        return

                    if mode in ["mic", "both"]:
                        default_input = sd.query_devices(None, "input")
                        mic_rate = int(default_input.get("default_samplerate", self.sr))
                        s_mic = sd.InputStream(
                            samplerate=mic_rate, channels=1, callback=make_cb(mic_rate)
                        )
                        self._active_streams.append(s_mic)
                        s_mic.start()

                    if mode in ["system", "both"]:
                        idx, name, is_loopback = self._find_system_audio_source()
                        if idx is not None:
                            d = sd.query_devices(idx)
                            logger.info(
                                f"🎤 Binding to System Audio: {name} (Loopback: {is_loopback})"
                            )
                            native_rate = int(d.get("default_samplerate", self.sr))
                            input_channels = int(d.get("max_input_channels", 0))
                            output_channels = int(d.get("max_output_channels", 0))
                            kwargs = {
                                "device": idx,
                                "samplerate": native_rate,
                                "channels": max(
                                    1, min(2, input_channels or output_channels or 1)
                                ),
                                "callback": make_cb(native_rate),
                            }
                            if is_loopback:
                                try:
                                    kwargs["loopback"] = True
                                    kwargs["channels"] = max(
                                        1, min(2, output_channels or 2)
                                    )
                                    s_sys = sd.InputStream(**kwargs)
                                except TypeError:
                                    raise RuntimeError(
                                        "Installed sounddevice build has no WASAPI loopback support"
                                    )
                            else:
                                s_sys = sd.InputStream(**kwargs)
                            self._active_streams.append(s_sys)
                            s_sys.start()

                    logger.info("🎙️ Audio Hardware Successfully Synchronized.")
                    break
                except Exception as e:
                    self._close_streams()
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⚠️ Hardware Busy (Attempt {attempt + 1}/{max_retries}). Retrying WASAPI reset..."
                        )
                        time.sleep(2.0)
                    else:
                        raise e

            while self._running:
                self.level.emit(self._current_rms)
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"❌ Final Capture Loop Failure: {e}")
        finally:
            self._close_streams()

    def set_vad_silence_ms(self, ms: int) -> None:
        """Update the VAD silence window live (no restart needed).

        Called from app.py when the mode switches so interview/meeting modes
        get tighter silence detection without requiring an audio pipeline restart.

        Args:
            ms: Milliseconds of silence before the speech segment is sent to
                Whisper. Clamped to [200, 2000] for safety.
        """
        ms = max(200, min(2000, int(ms)))
        new_blocks = int(ms / self.block_ms)
        if new_blocks != self.silence_blocks:
            self.silence_blocks = new_blocks
            logger.info(f"🎙️ VAD silence window updated → {ms}ms ({new_blocks} blocks)")

    def _process_loop(self):
        speech_buffer = []
        is_speaking = False
        silence_count = 0
        speech_started_at = None
        while self._running:
            try:
                data = self.q.get(timeout=0.5)
            except Exception as e:
                if self._running:
                    logger.debug(f"Audio queue timeout: {e}")
                continue

            rms = np.sqrt(np.mean(data**2))
            has_speech = rms > 0.005
            if has_speech or is_speaking:
                speech_buffer.append(data)
            if has_speech:
                silence_count = 0
                if not is_speaking:
                    speech_started_at = time.time()
                is_speaking = True
            elif is_speaking:
                silence_count += 1
                if silence_count >= self.silence_blocks:
                    # ── Dispatch transcription to pool so VAD loop is not blocked ──
                    # Previously: self._transcribe() ran synchronously here (~300ms).
                    # Now: submitted to a single-worker ThreadPoolExecutor so the VAD
                    # loop returns immediately and can process the next audio frame.
                    self._transcribe_pool.submit(
                        self._transcribe, list(speech_buffer), speech_started_at
                    )
                    speech_buffer = []
                    is_speaking = False
                    speech_started_at = None

    def _transcribe(self, buffer, speech_started_at=None):
        self._ensure_whisper_loaded()
        if not self.model or len(buffer) < 5:
            return
        transcribe_started_at = time.time()
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            segments, _ = self.model.transcribe(audio, language="en")
            text = " ".join(
                [s.text.strip() for s in segments if not self._is_hall(s.text)]
            )
            if text.strip():
                cleaned = text.strip()
                transcribe_finished_at = time.time()
                audio_duration_ms = (len(audio) / float(self.sr)) * 1000.0 if self.sr else 0.0
                self._last_transcription_metrics = {
                    "speech_started_at": speech_started_at,
                    "transcribe_started_at": transcribe_started_at,
                    "transcribe_finished_at": transcribe_finished_at,
                    "speech_to_transcript_ms": (
                        (transcribe_finished_at - speech_started_at) * 1000.0
                        if speech_started_at
                        else None
                    ),
                    "transcribe_only_ms": (transcribe_finished_at - transcribe_started_at) * 1000.0,
                    "audio_duration_ms": audio_duration_ms,
                    "text_length": len(cleaned),
                }
                self.transcripts.append(cleaned)
                self.transcription_ready.emit(cleaned)
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")

    def _is_hall(self, t):
        return any(x in t.lower() for x in ["thank you", "watching", "♪", "[music]"])

    def get_transcript(self):
        return " ".join(self.transcripts)

    def get_last_transcription_metrics(self):
        return dict(self._last_transcription_metrics)

    def clear(self):
        self.transcripts.clear()
