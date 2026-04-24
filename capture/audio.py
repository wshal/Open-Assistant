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
import warnings
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
    interim_transcription_ready = pyqtSignal(str)
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
        self._model_name = config.get("capture.audio.whisper_model", "base.en")
        # P2.3: Language hint for Faster Whisper (empty string = auto-detect)
        self._language = config.get("capture.audio.language", "") or None
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._current_rms = 0.0
        self._last_transcription_metrics = {}

        # Interim transcription (Option 2): best-effort partial ASR while speaking.
        # Guardrails are enforced at the detector/app layer so this never auto-fires by itself.
        self._interim_enabled = bool(config.get("capture.audio.interim.enabled", True))
        self._interim_interval_s = float(
            (config.get("capture.audio.interim.interval_ms", 900) or 900) / 1000.0
        )
        self._interim_min_speech_s = float(
            (config.get("capture.audio.interim.min_speech_ms", 1200) or 1200) / 1000.0
        )
        self._interim_tail_s = float(
            (config.get("capture.audio.interim.tail_ms", 3500) or 3500) / 1000.0
        )
        self._last_interim_at = 0.0
        self._interim_epoch = 0
        self._interim_inflight = False

        # WebRTC VAD (optional but recommended): improves real-time speech detection
        # without changing Whisper model accuracy.
        self._vad = None
        self._vad_frame_ms = int(config.get("capture.audio.vad.frame_ms", 20) or 20)
        if self._vad_frame_ms not in (10, 20, 30):
            self._vad_frame_ms = 20
        self._vad_mode = int(config.get("capture.audio.vad.mode", 2) or 2)
        self._vad_mode = max(0, min(3, self._vad_mode))
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"pkg_resources is deprecated as an API\.",
                    category=UserWarning,
                )
                import webrtcvad  # type: ignore

            self._vad = webrtcvad.Vad(self._vad_mode)
            logger.info(f"🎙️ WebRTC VAD enabled (mode={self._vad_mode}, frame_ms={self._vad_frame_ms})")
        except Exception as e:
            self._vad = None
            logger.debug(f"Audio: WebRTC VAD unavailable, falling back to RMS gate ({e})")

        # Serialize Whisper inference calls (final + interim).
        self._infer_lock = threading.Lock()

        # Single-worker pool: Whisper transcription runs off the VAD thread.
        # This means the VAD loop can immediately resume listening while the
        # previous speech segment is being transcribed in the background.
        self._transcribe_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper"
        )
        # Separate pool for interim ASR so final transcription isn't queued behind interim jobs.
        self._interim_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper-live"
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
            err = str(e)
            if "9996" in err or "Invalid device" in err or "Busy" in err.lower():
                logger.error(
                    f"❌ Audio device unavailable: {e}\n"
                    "   → Close any app holding exclusive audio access "
                    "(Discord, Zoom, Teams, another OpenAssist instance) and restart."
                )
            else:
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
            # Speech detection:
            # - Prefer WebRTC VAD (trained speech detector) for robust segmentation.
            # - Fall back to RMS gate if VAD is unavailable.
            has_speech = self._webrtc_vad_has_speech(data) if self._vad else (rms > 0.001)
            if has_speech or is_speaking:
                speech_buffer.append(data)
            if has_speech:
                silence_count = 0
                if not is_speaking:
                    speech_started_at = time.time()
                    self._last_interim_at = 0.0
                    self._interim_epoch += 1
                is_speaking = True
            elif is_speaking:
                silence_count += 1
                if silence_count >= self.silence_blocks:
                    # Speech ended: bump epoch to invalidate any in-flight interim work.
                    self._interim_epoch += 1
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

            # Best-effort interim transcription while speaking (never blocks VAD loop).
            if (
                self._interim_enabled
                and is_speaking
                and speech_started_at
                and (time.time() - speech_started_at) >= self._interim_min_speech_s
            ):
                now = time.time()
                if (now - self._last_interim_at) >= max(self._interim_interval_s, 0.2):
                    self._last_interim_at = now
                    self._submit_interim(list(speech_buffer), speech_started_at, self._interim_epoch)

    def _webrtc_vad_has_speech(self, block: np.ndarray) -> bool:
        """Return True if WebRTC VAD detects speech in this audio block.

        Expects `block` as float32 mono samples at self.sr with length ~= block_size.
        WebRTC VAD requires 16-bit PCM in 10/20/30ms frames.
        """
        vad = self._vad
        if not vad:
            return False
        try:
            if block is None or len(block) == 0:
                return False
            # Ensure mono float array.
            samples = np.asarray(block, dtype=np.float32).reshape(-1)
            # Convert to 16-bit PCM bytes.
            pcm16 = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16, copy=False).tobytes()

            frame_len = int(self.sr * (self._vad_frame_ms / 1000.0))
            if frame_len <= 0:
                return False
            frame_bytes = frame_len * 2  # int16
            if len(pcm16) < frame_bytes:
                return False

            # Mark as speech if ANY frame contains speech.
            # This is intentionally permissive; downstream buffering + question
            # detection prevents most false triggers.
            for i in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
                if vad.is_speech(pcm16[i : i + frame_bytes], self.sr):
                    return True
            return False
        except Exception:
            return False

    def _submit_interim(self, buffer, speech_started_at: float, epoch: int) -> None:
        if self._interim_inflight:
            return
        if not buffer or len(buffer) < 3:
            return
        self._interim_inflight = True

        def _run():
            try:
                self._transcribe_interim(buffer, speech_started_at, epoch)
            finally:
                self._interim_inflight = False

        try:
            self._interim_pool.submit(_run)
        except Exception:
            self._interim_inflight = False

    def _transcribe(self, buffer, speech_started_at=None):
        self._ensure_whisper_loaded()
        if not self.model or len(buffer) < 5:
            return
        transcribe_started_at = time.time()
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            # P2.3: Use configured language hint; None = Whisper auto-detects
            # P0.5: Enable highly-accurate Silero VAD to cleanly separate speech from noise/breathing.
            with self._infer_lock:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self._language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
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

    def _transcribe_interim(self, buffer, speech_started_at: float, epoch: int) -> None:
        """Best-effort interim transcription while the user is still speaking."""
        if epoch != self._interim_epoch:
            return
        self._ensure_whisper_loaded()
        if not self.model or not buffer:
            return
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            # Only transcribe the most recent tail to keep latency down.
            tail_samples = int(self.sr * max(self._interim_tail_s, 0.5))
            if tail_samples > 0 and len(audio) > tail_samples:
                audio = audio[-tail_samples:]
            if epoch != self._interim_epoch:
                return
            with self._infer_lock:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self._language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
            if epoch != self._interim_epoch:
                return
            text = " ".join(
                [s.text.strip() for s in segments if not self._is_hall(s.text)]
            )
            cleaned = (text or "").strip()
            if cleaned and epoch == self._interim_epoch:
                self.interim_transcription_ready.emit(cleaned)
        except Exception as e:
            logger.debug(f"Whisper interim transcription error (non-fatal): {e}")

    def get_transcript(self):
        return " ".join(self.transcripts)

    def get_last_transcription_metrics(self):
        return dict(self._last_transcription_metrics)

    def clear(self):
        self.transcripts.clear()
