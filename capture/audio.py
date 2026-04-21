"""
Audio capture with hardened VAD and Faster Whisper STT.
RESTORED: Multi-device hardware binding (Mic/System/WASAPI).
LAYER 6: Integrated restart() for hot-swapping audio modes in settings.
"""

import collections
import queue
import threading
import time

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
        self.silence_blocks = int(1500 / self.block_ms)

        self.model = None
        self._model_name = config.get("capture.audio.whisper_model", "base")
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._current_rms = 0.0

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

                self.model = WhisperModel(
                    self._model_name, device="auto", compute_type="int8"
                )
                self._model_loaded = True
                logger.info(f"✅ Whisper Ready: {self._model_name}")
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
            apis = sd.query_hostapis()
            wasapi_idx = next(
                (i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None
            )

            if wasapi_idx is not None:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
                        return i, d["name"], True

            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0 and any(
                    x in d["name"].lower() for x in ["cable", "vb-audio", "stereo mix"]
                ):
                    return i, d["name"], False
        except Exception as e:
            logger.debug(f"Source discovery error: {e}")
        return None, "Default", False

    def _capture_loop(self):
        import sounddevice as sd

        def make_cb():
            def cb(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Audio Status: {status}")
                if not self._paused and self._running:
                    try:
                        data = (
                            np.mean(indata, axis=1, keepdims=True).astype(np.float32)
                            if indata.shape[1] > 1
                            else indata.copy().astype(np.float32)
                        )
                        self.q.put_nowait(data)
                        self._current_rms = float(np.sqrt(np.mean(indata**2)))
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
                        s_mic = sd.InputStream(
                            samplerate=self.sr, channels=1, callback=make_cb()
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
                            kwargs = {
                                "device": idx,
                                "samplerate": self.sr,
                                "channels": 1,
                                "callback": make_cb(),
                            }
                            if is_loopback:
                                try:
                                    kwargs["loopback"] = True
                                    kwargs["channels"] = d.get("max_output_channels", 2)
                                    s_sys = sd.InputStream(**kwargs)
                                except TypeError:
                                    kwargs.pop("loopback", None)
                                    kwargs["channels"] = d.get("max_output_channels", 2)
                                    s_sys = sd.InputStream(**kwargs)
                                    logger.warning(
                                        f"Audio: Using {kwargs['channels']}ch capture fallback"
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

    def _process_loop(self):
        speech_buffer = []
        is_speaking = False
        silence_count = 0
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
                is_speaking = True
            elif is_speaking:
                silence_count += 1
                if silence_count >= self.silence_blocks:
                    self._transcribe(speech_buffer)
                    speech_buffer = []
                    is_speaking = False

    def _transcribe(self, buffer):
        self._ensure_whisper_loaded()
        if not self.model or len(buffer) < 5:
            return
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            segments, _ = self.model.transcribe(audio, language="en")
            text = " ".join(
                [s.text.strip() for s in segments if not self._is_hall(s.text)]
            )
            if text.strip():
                cleaned = text.strip()
                self.transcripts.append(cleaned)
                self.transcription_ready.emit(cleaned)
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")

    def _is_hall(self, t):
        return any(x in t.lower() for x in ["thank you", "watching", "♪", "[music]"])

    def get_transcript(self):
        return " ".join(self.transcripts)

    def clear(self):
        self.transcripts.clear()
