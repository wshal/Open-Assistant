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
        # 700ms config → 3 blocks × 200ms = 600ms effective (reduced from 800ms to cut EOS lag)
        self.silence_blocks = int(700 / self.block_ms)
        self._base_silence_ms = int(self.silence_blocks * self.block_ms)

        # ── Phase 2: Hybrid Micro-Pause VAD Chunking ─────────────────────────
        self._chunking_enabled = bool(config.get("capture.audio.chunking.enabled", True))
        self._chunk_min_s = float(config.get("capture.audio.chunking.min_chunk_s", 2.0))
        self._chunk_max_s = float(config.get("capture.audio.chunking.max_chunk_s", 4.0))

        # ── Phase 2: Transcription Provider ──────────────────────────────────
        self._transcription_provider = str(
            config.get("capture.audio.transcription_provider", "local")
        ).lower()

        # ── Phase 2: Adaptive Ambient VAD Calibration ─────────────────────────
        # Calibration samples the first N ms of audio on session start to build
        # a dynamic noise floor, preventing breathing/fan noise from being
        # misclassified as speech.
        _calib_ms = int(config.get("capture.audio.ambient_calibration_ms", 500) or 500)
        self._ambient_calib_blocks = max(1, int(_calib_ms / self.block_ms))
        self._ambient_calib_remaining = self._ambient_calib_blocks
        self._ambient_rms_samples: list = []
        self._dynamic_rms_floor = 0.0  # 0 = disabled until first calibration completes

        self.model = None
        self._model_name = config.get("capture.audio.whisper_model", "small.en")
        # P2.3: Language hint for Faster Whisper (empty string = auto-detect)
        self._language = config.get("capture.audio.language", "") or None
        self._model_loaded = False
        # Beam size: sweep shows beam=3 matches beam=5 WER with ~30% lower latency.
        self._beam_size = max(1, min(5, int(config.get("capture.audio.whisper_beam_size", 3) or 3)))
        self._model_lock = threading.Lock()
        self._current_rms = 0.0
        self._last_transcription_metrics = {}

        # ── Whisper Vocabulary Bias (initial_prompt) ──────────────────────────
        # Primes Whisper's decoder with technical vocabulary so it strongly
        # prefers "React" over "the act", "hooks" over "books", etc.
        # Can be overridden via config; empty string disables the hint.
        # Vocabulary bias — most-confused terms come first so Whisper weights them highest.
        # useReducer leads because it is the single most frequently misrecognized hook name.
        _default_prompt = (
            "useReducer, useState, useEffect, useCallback, useMemo, useRef, useContext, "
            "React hooks, Context API, prop drilling, reconciliation, "
            "React, Redux, JavaScript, TypeScript, Node.js, npm, "
            "difference, API, component, props, state, render, "
            "async, await, Promise, import, export, class, function, "
            "var, let, const, closure, prototype, arrow function, "
            "performance, optimization, Docker, Kubernetes, "
            "Python, FastAPI, Flask, Django, "
            "PostgreSQL, MongoDB, WebSocket, REST, GraphQL, "
            "Tailwind, Vite, webpack, interface, generic, decorator, middleware"
        )
        self._whisper_initial_prompt: str = str(
            config.get("capture.audio.whisper_initial_prompt", _default_prompt) or _default_prompt
        )
        # The static base prompt is kept separately so context injection can
        # always restore it as a fallback.
        self._whisper_base_prompt: str = self._whisper_initial_prompt

        # ── Context-Aware Prompt Injection ────────────────────────────────────
        # Keeps a short ring-buffer of (transcript, terms) from recent utterances.
        # After each transcription, topic-relevant technical terms from recent
        # history are prepended to the Whisper prompt — but ONLY when the new
        # question is still on the same topic.  Terms from unrelated turns are
        # automatically faded so asking about async/await doesn't keep injecting
        # useReducer from a previous question.
        #
        # Ring buffer: list of dicts {"text": str, "terms": list[str], "age": int}
        # "age" counts turns since that entry; entries with age > MAX_PROMPT_AGE are dropped.
        self._recent_transcripts: list[dict] = []   # ring buffer, max PROMPT_RING_SIZE
        self._prompt_context_lock = threading.Lock()
        self.PROMPT_RING_SIZE = 4      # remember last 4 utterances
        self.PROMPT_MAX_AGE   = 2      # inject terms only from last 2 on-topic turns
        self.PROMPT_TERM_CAP  = 8      # max injected terms to avoid bloating the 224-token window

        # ── Session Chunk Accumulator ───────────────────────────────────────
        # When hybrid chunking fires mid-utterance, each chunk's text is stored
        # here. Only when real end-of-speech (silence_blocks) fires do we join
        # all parts into a single transcription_ready signal — so the UI always
        # receives the complete query, not just the last 3 seconds of it.
        self._session_transcript_parts: list = []
        self._session_parts_lock = threading.Lock()

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

        # Short-question tuning: keep end-of-speech tighter for brief prompts.
        self._short_utterance_max_s = float(
            config.get("capture.audio.vad.short_utterance_max_s", 2.8) or 2.8
        )
        self._short_silence_ms = int(
            config.get("capture.audio.vad.short_silence_ms", 500) or 500
        )
        self._short_silence_blocks = max(1, int(self._short_silence_ms / self.block_ms))
        self._post_chunk_silence_ms = int(
            config.get("capture.audio.vad.post_chunk_silence_ms", 500) or 500
        )
        self._post_chunk_silence_blocks = max(
            1, int(self._post_chunk_silence_ms / self.block_ms)
        )
        # Warn when configured ms values floor to a different effective duration.
        # This prevents silent config/behaviour discrepancies during EOS tuning.
        for _cfg_key, _cfg_ms, _blocks in [
            ("short_silence_ms", self._short_silence_ms, self._short_silence_blocks),
            ("post_chunk_silence_ms", self._post_chunk_silence_ms, self._post_chunk_silence_blocks),
        ]:
            _eff = _blocks * self.block_ms
            if _eff != _cfg_ms:
                logger.debug(
                    f"VAD: {_cfg_key}={_cfg_ms}ms adjusted to {_eff}ms "
                    f"({_blocks}\u00d7{self.block_ms}ms blocks)."
                )
        self._vad_backend_name = "rms"

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

        if self._vad is not None:
            self._vad_backend_name = "webrtc"
        else:
            self._vad_backend_name = "rms"

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

    # ── Hardware-Aware Whisper Loader ────────────────────────────────────────
    # VRAM → (model_upgrade, compute_type) mapping.
    # Only applied when the user has NOT overridden capture.audio.whisper_model
    # in config.yaml.  Headroom is 0.7× usable VRAM to leave room for the OS,
    # CUDA runtime, and intermediate buffers.
    #
    # Approximate faster-whisper VRAM footprints (float16 / int8):
    #   large-v3-turbo : ~2.5GB float16,  ~1.5GB int8
    #   medium.en      : ~3.0GB float16,  ~2.0GB int8
    #   small.en       : ~1.0GB float16,  ~0.6GB int8
    _GPU_TIERS: list[tuple[float, str, str]] = [
        # (min_free_vram_gb, model, compute_type)   — checked largest-first
        (3.5, "large-v3-turbo", "float16"),   # 4GB+ GPU — best accuracy
        (2.5, "large-v3-turbo", "int8"),      # 3–3.5GB  — still great
        (1.8, "medium.en",      "int8"),      # 2–2.5GB  — solid upgrade
        (0.8, "small.en",       "float16"),   # 1–1.8GB  — GPU speedup only
        # below 0.8GB free → stay on CPU
    ]
    # Config key sentinel: if user explicitly sets a model, we respect it.
    _DEFAULT_MODEL_NAME = "small.en"

    @staticmethod
    def _probe_gpu() -> tuple[str, str, str]:
        """Return (device, compute_type, model_override_or_empty).

        Probes CUDA availability and free VRAM.  Returns:
        - device      : "cuda" | "cpu"
        - compute     : "float16" | "int8"
        - model_hint  : suggested model name, or "" if no upgrade is warranted
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu", "int8", ""
        except ImportError:
            return "cpu", "int8", ""

        # Try to get free VRAM via pynvml (most accurate)
        free_gb = 0.0
        gpu_name = "unknown GPU"
        try:
            import pynvml  # type: ignore[import-untyped]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem.free / (1024 ** 3)
        except Exception:
            # Fallback: use torch's own memory query (less accurate — excludes
            # CUDA runtime overhead, so apply a 0.75 safety factor)
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_name = props.name
                total_gb = props.total_memory / (1024 ** 3)
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                free_gb = (total_gb - allocated_gb) * 0.75
            except Exception:
                free_gb = 0.0

        logger.info(f"[GPU] {gpu_name} detected — free VRAM ≈ {free_gb:.1f} GB")
        return "cuda", "float16", free_gb, gpu_name

    def _ensure_whisper_loaded(self):
        with self._model_lock:
            if self._model_loaded:
                return
            try:
                from faster_whisper import WhisperModel

                # Determine if the user has overridden the model explicitly.
                user_set_model = (self._model_name != self._DEFAULT_MODEL_NAME)

                # Probe GPU
                probe = self._probe_gpu()
                if probe[0] == "cuda":
                    _, _, free_gb, gpu_name = probe
                    device = "cuda"

                    # Choose compute_type and optional model upgrade
                    compute = "int8"   # safe default on CUDA
                    model_hint = ""
                    for min_gb, hint_model, hint_compute in self._GPU_TIERS:
                        if free_gb >= min_gb:
                            compute = hint_compute
                            model_hint = hint_model
                            break

                    # Apply model upgrade only if user hasn't overridden
                    if model_hint and not user_set_model:
                        logger.info(
                            f"[GPU] Auto-upgrading model: {self._model_name} → {model_hint} "
                            f"({compute}, {free_gb:.1f}GB free on {gpu_name})"
                        )
                        self._model_name = model_hint
                    else:
                        if user_set_model:
                            logger.info(
                                f"[GPU] CUDA available ({gpu_name}, {free_gb:.1f}GB free) — "
                                f"using user-configured model {self._model_name!r} on GPU"
                            )
                        else:
                            logger.info(
                                f"[GPU] CUDA available but low VRAM ({free_gb:.1f}GB) — "
                                f"staying on {self._model_name} with int8"
                            )
                else:
                    device, compute = "cpu", "int8"

                # Load model — with OOM guard for CUDA
                try:
                    self.model = WhisperModel(
                        self._model_name, device=device, compute_type=compute
                    )
                    self._model_loaded = True
                    logger.info(
                        f"✅ Whisper Ready: {self._model_name} on {device} ({compute})"
                    )
                except Exception as load_err:
                    if device == "cuda":
                        # OOM or driver error — fall back to CPU gracefully
                        logger.warning(
                            f"[GPU] Failed to load {self._model_name} on CUDA "
                            f"({load_err!r}) — falling back to small.en on CPU"
                        )
                        self._model_name = self._DEFAULT_MODEL_NAME
                        self.model = WhisperModel(
                            self._model_name, device="cpu", compute_type="int8"
                        )
                        self._model_loaded = True
                        logger.info(
                            f"✅ Whisper Ready (CPU fallback): {self._model_name} int8"
                        )
                    else:
                        raise
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
            # Phase 2: Reset ambient calibration so every new session recalibrates
            self._ambient_calib_remaining = self._ambient_calib_blocks
            self._ambient_rms_samples = []
            self._dynamic_rms_floor = 0.0
            # Reset session chunk accumulator
            with self._session_parts_lock:
                self._session_transcript_parts = []
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
            self._base_silence_ms = ms
            logger.info(f"🎙️ VAD silence window updated → {ms}ms ({new_blocks} blocks)")

    def _required_silence_blocks(self, speech_started_at, had_mid_utterance_slice: bool = False) -> int:
        """Choose a silence window based on utterance shape.

        After a chunk-slice, `speech_started_at` is reset so `elapsed` counts
        from the new chunk start.  We apply the aggressive post-chunk tail ONLY
        while the *new* chunk is still short (<=short_utterance_max_s).  If the
        user keeps talking past that threshold the window reverts to the full
        silence_blocks so multi-clause prompts are not prematurely cut.
        """
        if speech_started_at is None:
            return self.silence_blocks
        elapsed = max(0.0, time.time() - speech_started_at)
        if had_mid_utterance_slice and elapsed <= self._short_utterance_max_s:
            # New chunk is still short — use aggressive post-chunk tail.
            return min(self.silence_blocks, self._post_chunk_silence_blocks)
        if elapsed <= self._short_utterance_max_s:
            return min(self.silence_blocks, self._short_silence_blocks)
        return self.silence_blocks

    def _detect_speech(self, block: np.ndarray, rms: float) -> tuple[bool, str]:
        """Choose the best available local speech detector."""
        if self._vad is not None:
            return self._webrtc_vad_has_speech(block), "webrtc"
        return rms > 0.001, "rms"

    def _process_loop(self):
        speech_buffer = []
        is_speaking = False
        silence_count = 0
        speech_started_at = None
        had_mid_utterance_slice = False
        utterance_vad_backend = self._vad_backend_name

        # Pre-roll: keep a rolling window of the last PRE_ROLL_BLOCKS blocks
        # BEFORE VAD fires.  Prepended to speech_buffer at onset so Whisper
        # receives the low-amplitude opener ("can you...", "what is...") that
        # WebRTC VAD sometimes misses on its first triggered block.
        PRE_ROLL_BLOCKS = 3  # ~200ms at block_ms=200; cheap to keep, high value
        from collections import deque
        pre_roll = deque(maxlen=PRE_ROLL_BLOCKS)
        while self._running:
            try:
                data = self.q.get(timeout=0.5)
            except Exception as e:
                if self._running:
                    logger.debug(f"Audio queue timeout: {e}")
                continue

            rms = float(np.sqrt(np.mean(data**2)))

            # ── Phase 2: Adaptive Ambient Calibration ─────────────────────────
            # Consume the first N blocks as a silent noise floor sample.
            if self._ambient_calib_remaining > 0 and not is_speaking:
                self._ambient_rms_samples.append(rms)
                self._ambient_calib_remaining -= 1
                if self._ambient_calib_remaining == 0 and self._ambient_rms_samples:
                    mean_rms = float(np.mean(self._ambient_rms_samples))
                    std_rms  = float(np.std(self._ambient_rms_samples))
                    # Floor = mean + 2σ ensures even a noisy room doesn't false-trigger
                    self._dynamic_rms_floor = mean_rms + 2.0 * std_rms
                    logger.info(
                        f"[Phase2 VAD] Ambient calibration done — "
                        f"floor={self._dynamic_rms_floor:.5f} "
                        f"(mean={mean_rms:.5f}, std={std_rms:.5f})"
                    )
                continue  # don't process calibration frames as speech

            # Speech detection:
            # - Prefer Silero VAD when available.
            # - Fall back to WebRTC VAD, then RMS.
            # - Also gate on dynamic_rms_floor when calibrated.
            raw_has_speech, backend = self._detect_speech(data, rms)
            floor_ok = (self._dynamic_rms_floor <= 0.0) or (rms > self._dynamic_rms_floor)
            has_speech = raw_has_speech and floor_ok

            if has_speech or is_speaking:
                speech_buffer.append(data)
            elif not is_speaking:
                # Not yet speaking — keep this block as potential pre-roll.
                pre_roll.append(data)
            if has_speech:
                silence_count = 0
                if not is_speaking:
                    # Prepend pre-roll blocks so Whisper sees the onset context.
                    if pre_roll:
                        speech_buffer = list(pre_roll) + speech_buffer
                        pre_roll.clear()
                    speech_started_at = time.time()
                    self._last_interim_at = 0.0
                    self._interim_epoch += 1
                    utterance_vad_backend = backend
                    had_mid_utterance_slice = False
                is_speaking = True

                # ── Phase 2: Hybrid Micro-Pause VAD Chunking ──────────────────
                # Once the buffer exceeds min_chunk_s, start scanning for ANY
                # micro-pause (a single non-speech block) up to max_chunk_s.
                # This slices on a clean word boundary, never mid-syllable.
                if (
                    self._chunking_enabled
                    and speech_started_at is not None
                    and (time.time() - speech_started_at) >= self._chunk_min_s
                ):
                    # We're in the scanning window — if the *next* block is silence
                    # we will catch it in the elif below. Nothing extra needed here.
                    pass

            elif is_speaking:
                silence_count += 1
                required_silence_blocks = self._required_silence_blocks(
                    speech_started_at,
                    had_mid_utterance_slice=had_mid_utterance_slice,
                )

                # ── Phase 2: Hybrid chunk slice on first micro-pause ───────────
                # If we are inside the [min_chunk_s, max_chunk_s] scanning window,
                # ANY silence block is a clean word boundary — slice immediately.
                elapsed = (time.time() - speech_started_at) if speech_started_at else 0.0
                in_scan_window = (
                    self._chunking_enabled
                    and self._chunk_min_s <= elapsed < self._chunk_max_s
                )
                hard_cap_hit = (
                    self._chunking_enabled
                    and elapsed >= self._chunk_max_s
                )

                if in_scan_window and silence_count == 1:
                    # First micro-pause inside the scanning window → clean slice.
                    # IMPORTANT: Keep is_speaking=True so the VAD continues buffering
                    # any speech that immediately follows. The sliced chunk is stored
                    # in the session accumulator; only final silence emits the signal.
                    logger.debug(
                        f"[Phase2 Chunking] Micro-pause slice at {elapsed:.2f}s "
                        f"(window {self._chunk_min_s}-{self._chunk_max_s}s)"
                    )
                    self._interim_epoch += 1
                    self._transcribe_pool.submit(
                        self._transcribe,
                        list(speech_buffer),
                        speech_started_at,
                        False,
                        {
                            "vad_backend": utterance_vad_backend,
                            "chunk_aware_eos": had_mid_utterance_slice,
                        },
                    )
                    speech_buffer = []
                    # Keep silence_count (don't reset) so normal silence_blocks path
                    # is still reachable if user has actually stopped talking.
                    speech_started_at = time.time()  # restart chunk timer
                    had_mid_utterance_slice = True
                    # is_speaking stays True — do NOT set to False here

                elif hard_cap_hit:
                    # Absolute cap reached with no micro-pause → force slice but
                    # keep collecting audio since user is clearly still talking.
                    logger.debug(
                        f"[Phase2 Chunking] Hard-cap slice at {elapsed:.2f}s"
                    )
                    self._interim_epoch += 1
                    self._transcribe_pool.submit(
                        self._transcribe,
                        list(speech_buffer),
                        speech_started_at,
                        False,
                        {
                            "vad_backend": utterance_vad_backend,
                            "chunk_aware_eos": had_mid_utterance_slice,
                        },
                    )
                    speech_buffer = []
                    silence_count = 0
                    speech_started_at = time.time()  # restart timer for next chunk
                    had_mid_utterance_slice = True
                    # is_speaking stays True

                elif silence_count >= required_silence_blocks:
                    # Real end-of-speech — emit the complete joined transcript.
                    self._interim_epoch += 1
                    self._transcribe_pool.submit(
                        self._transcribe,
                        list(speech_buffer),
                        speech_started_at,
                        True,
                        {
                            "vad_backend": utterance_vad_backend,
                            "end_silence_ms": required_silence_blocks * self.block_ms,
                            "chunk_aware_eos": had_mid_utterance_slice,
                        },
                    )
                    speech_buffer = []
                    is_speaking = False
                    silence_count = 0
                    speech_started_at = None
                    had_mid_utterance_slice = False
                    # Reset session accumulator for the NEXT utterance only after
                    # _emit_accumulated has been called (it clears parts itself).
                    # No reset here — emit_accumulated handles it atomically.

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

    def _apply_gain_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Soft gain normalization for quiet audio before Whisper inference.

        Rescues genuine quiet speech (RMS between 0.001 and 0.008) that would
        otherwise yield empty or partial transcripts.  Skips audio that is too
        quiet to be real speech (likely noise) or already at a healthy level.
        A hard 12× ceiling prevents background hiss from being amplified into
        false speech detections.
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.001 or rms >= 0.008:
            return audio  # silence/noise or already healthy — leave untouched
        TARGET_RMS = 0.025
        MAX_GAIN = 12.0
        gain = min(TARGET_RMS / rms, MAX_GAIN)
        logger.debug(f"[GainNorm] rms={rms:.5f} → gain={gain:.1f}× applied")
        return np.clip(audio * gain, -1.0, 1.0)

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

    def _transcribe(self, buffer, speech_started_at=None, is_final: bool = True, vad_meta=None):
        """Route to Groq Cloud or local Faster-Whisper.

        is_final=False → mid-utterance chunk: text stored in session accumulator,
                          no transcription_ready signal emitted yet.
        is_final=True  → end-of-speech: join all accumulated parts + this text,
                          emit one combined transcription_ready signal.
        """
        if self._transcription_provider == "groq":
            self._transcribe_groq(buffer, speech_started_at, is_final, vad_meta=vad_meta)
            return
        self._transcribe_local(buffer, speech_started_at, is_final, vad_meta=vad_meta)

    def _transcribe_local(self, buffer, speech_started_at=None, is_final: bool = True, vad_meta=None):
        """Transcribe using the local Faster-Whisper model.

        For chunked utterances the call sequence is:
          _transcribe(..., is_final=False)  # chunk 1, 2, … appended to session parts
          _transcribe(..., is_final=True)   # final chunk: appends then calls _emit_accumulated

        All calls go through the same single-worker _transcribe_pool, so they are
        strictly ordered by submission time — no race between chunks.

        Edge case handled: if the final speech_buffer has fewer than 5 blocks (user
        stopped talking very quickly after a slice), we still emit whatever was
        accumulated from earlier chunks rather than silently discarding it.
        """
        self._ensure_whisper_loaded()
        if not self.model or len(buffer) < 5:
            if is_final:
                # Short tail — don't transcribe this fragment, but DO flush
                # whatever earlier chunks already accumulated.
                self._emit_accumulated(speech_started_at, provider="local", vad_meta=vad_meta)
            return
        transcribe_started_at = time.time()
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            audio = self._apply_gain_normalization(audio)  # rescue quiet-but-valid speech
            with self._infer_lock:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self._language,
                    beam_size=self._beam_size,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    # Vocabulary bias steers Whisper away from common tech-term
                    # mishearings in isolated coding questions.
                    initial_prompt=self._whisper_initial_prompt or None,
                    # Prevent Whisper from silently skipping the first few low-energy
                    # words of an utterance (e.g. the "can you explain..." leading clause).
                    no_speech_threshold=0.7,
                    suppress_blank=False,
                )
            text = self._filter_segments(segments)

            if not is_final:
                # Accumulate chunk; do not emit yet.
                if text:
                    with self._session_parts_lock:
                        self._session_transcript_parts.append(text)
                return

            # Final path: join all accumulated chunks + this one.
            if text:
                with self._session_parts_lock:
                    self._session_transcript_parts.append(text)
            self._emit_accumulated(
                speech_started_at,
                provider="local",
                transcribe_started_at=transcribe_started_at,
                vad_meta=vad_meta,
            )
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            if is_final:
                self._emit_accumulated(speech_started_at, provider="local", vad_meta=vad_meta)

    def _emit_accumulated(self, speech_started_at, provider="local",
                           transcribe_started_at=None, vad_meta=None):
        """Join all session parts and emit transcription_ready once."""
        with self._session_parts_lock:
            parts = list(self._session_transcript_parts)
            self._session_transcript_parts = []

        # Join with a space; strip each part to prevent "Wouldyou" concatenation.
        cleaned = " ".join(p.strip() for p in parts if p.strip())
        if not cleaned:
            return

        # ── camelCase / tech-term post-processor ────────────────────────────
        # Whisper (small.en) frequently space-splits React hook names and
        # merges common word pairs.  These substitutions run in O(n) on the
        # final joined string — no measurable overhead.
        cleaned = self._fix_tech_terms(cleaned)

        now = time.time()
        audio_duration_ms = 0.0
        if speech_started_at and transcribe_started_at:
            audio_duration_ms = max(
                0.0, (transcribe_started_at - speech_started_at) * 1000.0
            )
        self._last_transcription_metrics = {
            "speech_started_at": speech_started_at,
            "transcribe_started_at": transcribe_started_at or now,
            "transcribe_finished_at": now,
            "speech_to_transcript_ms": (
                (now - speech_started_at) * 1000.0 if speech_started_at else None
            ),
            "transcribe_only_ms": (
                (now - transcribe_started_at) * 1000.0 if transcribe_started_at else 0.0
            ),
            "audio_duration_ms": audio_duration_ms,
            "text_length": len(cleaned),
            "provider": provider,
            "chunks": len(parts),
            "vad_backend": (vad_meta or {}).get("vad_backend", self._vad_backend_name),
            "end_silence_ms": (vad_meta or {}).get("end_silence_ms", self._base_silence_ms),
            "chunk_aware_eos": bool((vad_meta or {}).get("chunk_aware_eos", False)),
        }
        self.transcripts.append(cleaned)
        self.transcription_ready.emit(cleaned)
        logger.debug(
            f"[Transcription] Emitted {len(parts)} chunk(s) as one query: \"{cleaned[:80]}...\""
            if len(cleaned) > 80 else
            f"[Transcription] Emitted {len(parts)} chunk(s): \"{cleaned}\""
        )
        # Update context-aware prompt asynchronously — never blocks the VAD thread.
        try:
            self._transcribe_pool.submit(self._update_prompt_context, cleaned)
        except Exception:
            pass  # pool shutdown or full — prompt stays as-is, not critical


    def _transcribe_groq(self, buffer, speech_started_at=None, is_final: bool = True, vad_meta=None):
        """Phase 2: Transcribe via Groq Cloud Whisper API (whisper-large-v3).

        Writes audio to an in-memory 16 kHz/16-bit/mono WAV and POSTs to Groq.
        Falls back to local Whisper on any error.
        is_final semantics match _transcribe_local.
        """
        import io, wave
        if not buffer or len(buffer) < 2:
            if is_final:
                self._emit_accumulated(speech_started_at, provider="groq", vad_meta=vad_meta)
            return
        transcribe_started_at = time.time()
        try:
            audio = np.concatenate(buffer, axis=0).flatten()
            audio = self._apply_gain_normalization(audio)  # rescue quiet-but-valid speech

            # ── Groq Audio Format Guardrail ─────────────────────────────────
            # Groq Whisper works best with 16 kHz 16-bit mono PCM.
            # Resample if the device captured at a different rate (e.g. 44.1 kHz)
            # to minimise payload size and avoid Groq format quirks.
            TARGET_SR = 16_000
            if self.sr != TARGET_SR:
                target_len = max(1, int(round(len(audio) * TARGET_SR / self.sr)))
                src_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
                dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
                audio = np.interp(dst_x, src_x, audio).astype(np.float32)
            # ──────────────────────────────────────────────────────

            # Build in-memory WAV — 16 kHz / 16-bit / mono (no temp files on disk)
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)           # mono
                wf.setsampwidth(2)           # 16-bit PCM = 2 bytes/sample
                wf.setframerate(TARGET_SR)   # always 16 000 Hz
                pcm = np.clip(audio, -1.0, 1.0)
                wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
            wav_buf.seek(0)

            groq_key = ""
            try:
                from core.config import Config as _Cfg
                groq_key = str(
                    getattr(self, "config", None) and
                    self.config.get_api_key("groq") or ""
                )
            except Exception:
                pass

            if not groq_key:
                logger.warning("[Phase2 Groq] No Groq API key — falling back to local Whisper")
                self._transcribe_local(buffer, speech_started_at, is_final, vad_meta=vad_meta)
                return

            import urllib.request
            boundary = "----OpenAssistBoundary"
            body_parts = []
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3\r\n".encode())
            if self._language:
                body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{self._language}\r\n".encode())
            wav_data = wav_buf.read()
            body_parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
                + wav_data + b"\r\n"
            )
            body_parts.append(f"--{boundary}--\r\n".encode())
            body = b"".join(body_parts)

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                result = json.loads(resp.read().decode())

            text = result.get("text", "").strip()
            if text and not self._is_hall(text):
                if not is_final:
                    with self._session_parts_lock:
                        self._session_transcript_parts.append(text)
                    logger.debug(
                        f"[Phase2 Groq] Chunk stored ({len(text)} chars)"
                    )
                    return
                # Final: accumulate then emit
                with self._session_parts_lock:
                    self._session_transcript_parts.append(text)
                transcribe_finished_at = time.time()
                audio_duration_ms = (len(audio) / float(self.sr)) * 1000.0 if self.sr else 0.0
                self._emit_accumulated(
                    speech_started_at, provider="groq",
                    transcribe_started_at=transcribe_started_at,
                    vad_meta=vad_meta,
                )
                logger.debug(
                    f"[Phase2 Groq] Transcribed {audio_duration_ms:.0f}ms audio in "
                    f"{(transcribe_finished_at - transcribe_started_at)*1000:.0f}ms"
                )
        except Exception as e:
            logger.warning(f"[Phase2 Groq] API error ({e}) — falling back to local Whisper")
            self._transcribe_local(buffer, speech_started_at, is_final, vad_meta=vad_meta)


    # Confidence thresholds for Whisper segment filtering.
    # avg_logprob: how confident Whisper is about the token sequence (lower = less confident).
    # no_speech_prob: how likely the segment contains no real speech.
    # Both are segment-level fields returned by faster-whisper.
    _LOGPROB_THRESHOLD: float = -1.0
    _NO_SPEECH_THRESHOLD: float = 0.6

    # Known Whisper hallucination patterns — text generated when there is silence or
    # very quiet audio.  These appear consistently across model sizes and languages.
    _HALL_PATTERNS = [
        "thank you", "watching", "\u266a", "[music]",
        "subtitles", "transcribed by", "subscribe",
        "www.", ".com", "[silence]", "[blank_audio]",
        "amara.org", "dotsub",
    ]

    def _is_hall(self, t: str) -> bool:
        """Return True if this segment text matches a known hallucination pattern."""
        tl = (t or "").lower().strip()
        if not tl:
            return True
        return any(pat in tl for pat in self._HALL_PATTERNS)

    def _segment_passes_confidence(self, seg) -> bool:
        """Return True if the segment's confidence metrics look like real speech.

        Uses two signals that faster-whisper exposes on every Segment object:
        - avg_logprob: average log probability of generated tokens.  Values below
          -1.0 indicate the model is guessing and the output is likely garbage.
        - no_speech_prob: probability that the segment is silence/non-speech.
          Values above 0.6 indicate the model itself doubts there is speech here.

        Both thresholds are deliberately conservative to avoid dropping real words;
        they only catch the worst-confidence outputs.
        """
        logprob = getattr(seg, "avg_logprob", None)
        no_speech = getattr(seg, "no_speech_prob", None)
        if logprob is not None and logprob < self._LOGPROB_THRESHOLD:
            logger.debug(
                f"[Confidence] Dropping segment (avg_logprob={logprob:.2f} < {self._LOGPROB_THRESHOLD}): "
                f"\"{(seg.text or '').strip()[:40]}\""
            )
            return False
        if no_speech is not None and no_speech > self._NO_SPEECH_THRESHOLD:
            logger.debug(
                f"[Confidence] Dropping segment (no_speech_prob={no_speech:.2f} > {self._NO_SPEECH_THRESHOLD}): "
                f"\"{(seg.text or '').strip()[:40]}\""
            )
            return False
        return True

    # ── Tech-term vocabulary for context extraction ───────────────────────────
    # Used by _update_prompt_context to identify recognized terms in transcripts.
    # Grouped by topic so similarity scoring can work at topic-cluster level.
    _KNOWN_TERMS: dict[str, list[str]] = {
        "react_hooks": [
            "useReducer", "useState", "useEffect", "useCallback", "useMemo",
            "useRef", "useContext", "useLayoutEffect", "useId",
        ],
        "react_core": [
            "React", "component", "props", "state", "render", "reconciliation",
            "Context API", "prop drilling", "re-render", "virtual DOM", "JSX",
            "hook", "hooks", "Redux", "zustand",
        ],
        "javascript": [
            "closure", "prototype", "async", "await", "Promise", "callback",
            "arrow function", "var", "let", "const", "class", "import", "export",
            "difference", "hoisting", "scope", "event loop", "destructuring",
        ],
        "typescript": [
            "TypeScript", "interface", "generic", "type", "decorator", "enum",
        ],
        "backend": [
            "Node.js", "npm", "API", "REST", "GraphQL", "WebSocket",
            "FastAPI", "Flask", "Django", "middleware", "Docker", "Kubernetes",
        ],
        "database": [
            "PostgreSQL", "MongoDB", "SQL", "query", "schema",
        ],
        "tooling": [
            "Tailwind", "Vite", "webpack", "performance", "optimization",
        ],
    }

    def _update_prompt_context(self, transcript: str) -> None:
        """Rebuild the Whisper initial_prompt with context-aware term injection.

        Algorithm
        ---------
        1. Extract tech terms from the new transcript (O(n·m), small).
        2. Push {text, terms, age=0} onto the ring buffer; bump age of old entries.
        3. Identify the TOPIC SET of the new transcript (which topic clusters it hits).
        4. For each entry in the ring buffer, compute Jaccard similarity between
           its topic set and the new transcript's topic set.
        5. Only entries with similarity > TOPIC_THRESHOLD (0.15) AND age <= MAX_AGE
           contribute terms to the injected prefix.
        6. Prepend the deduplicated injected terms to the base prompt.
           Cap at PROMPT_TERM_CAP terms to avoid bloating Whisper's 224-token window.

        Result: useReducer from a previous question is NOT injected when the user
        switches to async/await — but IS re-injected if they pivot back.
        """
        import re

        TOPIC_THRESHOLD = 0.15  # Jaccard sim cutoff; tune up to tighten topic lock

        def _extract_terms(text: str) -> list[str]:
            tl = text.lower()
            found = []
            for terms in self._KNOWN_TERMS.values():
                for t in terms:
                    # word-boundary match, case-insensitive
                    if re.search(r"\b" + re.escape(t.lower()) + r"\b", tl):
                        found.append(t)
            return found

        def _topic_set(terms: list[str]) -> set[str]:
            """Return the set of topic cluster names that the terms belong to."""
            topics = set()
            term_lower = {t.lower() for t in terms}
            for topic, topic_terms in self._KNOWN_TERMS.items():
                if any(tt.lower() in term_lower for tt in topic_terms):
                    topics.add(topic)
            return topics

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        new_terms = _extract_terms(transcript)
        new_topics = _topic_set(new_terms)

        with self._prompt_context_lock:
            # Age existing entries
            aged = []
            for entry in self._recent_transcripts:
                entry["age"] += 1
                if entry["age"] <= self.PROMPT_RING_SIZE:
                    aged.append(entry)
            # Push new entry (age=0 = current turn)
            aged.append({"text": transcript, "terms": new_terms, "topics": new_topics, "age": 0})
            # Trim to ring size
            self._recent_transcripts = aged[-self.PROMPT_RING_SIZE:]

            # Collect injected terms: only from entries on the same topic and within age limit
            injected: list[str] = []
            seen: set[str] = set()
            for entry in reversed(self._recent_transcripts):  # newest first
                if entry["age"] == 0:
                    # Always include terms from the current turn
                    for t in entry["terms"]:
                        if t not in seen:
                            injected.append(t)
                            seen.add(t)
                    continue
                if entry["age"] > self.PROMPT_MAX_AGE:
                    continue
                sim = _jaccard(new_topics, entry.get("topics", set()))
                if sim < TOPIC_THRESHOLD:
                    logger.debug(
                        f"[PromptCtx] Skipping entry age={entry['age']} "
                        f"(jaccard={sim:.2f} < {TOPIC_THRESHOLD}): \"{entry['text'][:40]}\""
                    )
                    continue
                for t in entry["terms"]:
                    if t not in seen:
                        injected.append(t)
                        seen.add(t)

            # Cap and rebuild prompt
            injected = injected[: self.PROMPT_TERM_CAP]
            if injected:
                prefix = ", ".join(injected) + ". "
                self._whisper_initial_prompt = prefix + self._whisper_base_prompt
            else:
                self._whisper_initial_prompt = self._whisper_base_prompt

            logger.debug(
                f"[PromptCtx] Injected {len(injected)} terms: {injected}  "
                f"topics={new_topics}  prompt[:80]=\"{self._whisper_initial_prompt[:80]}\""
            )

    def _filter_segments(self, segments) -> str:
        """Join segments that pass both hallucination and confidence filters."""
        return " ".join(
            s.text.strip()
            for s in segments
            if not self._is_hall(s.text) and self._segment_passes_confidence(s)
        ).strip()


    # Ordered substitution table: (pattern, replacement).
    # Applied left-to-right by _fix_tech_terms after joining session parts.
    # Patterns are case-insensitive but replacements preserve correct casing.
    # ─ React hook space-splits (most common Whisper small.en errors) ────────
    # ─ Merged-token artifacts observed in benchmark sweeps ──────────────────
    _TECH_TERM_SUBS: list[tuple[str, str]] = [
        # React hooks — space-split variants
        (r"\buse effect\b",             "useEffect"),
        (r"\buse state\b",              "useState"),
        (r"\buse reducer\b",            "useReducer"),
        (r"\buse memo\b",               "useMemo"),
        (r"\buse callback\b",           "useCallback"),
        (r"\buse ref\b",                "useRef"),
        (r"\buse context\b",            "useContext"),
        (r"\buse layout effect\b",      "useLayoutEffect"),
        (r"\buse imperative handle\b",  "useImperativeHandle"),
        (r"\buse debug value\b",        "useDebugValue"),
        (r"\buse id\b",                 "useId"),
        # Common JS keywords merged with adjacent words
        (r"\bconstin\b",                "const in"),
        (r"\bbugin\b",                  "bug in"),
        (r"\breacttab\b",               "React app"),
        # "Reducer" without "use" prefix (model drops "use")
        (r"\bReducer is a better\b",    "useReducer is a better"),
        (r"\bReducer a better\b",       "useReducer a better"),
        # "use effect" / "use Effect" capitalisation variants handled by re.IGNORECASE
    ]

    def _fix_tech_terms(self, text: str) -> str:
        """Apply the tech-term substitution table to correct Whisper space-splits.

        Uses compiled regex for performance; compiled once at first call and
        cached on the class.  Case-insensitive matching, exact-case replacement.
        """
        import re
        # Compile and cache patterns on first call.
        if not hasattr(self, "_tech_subs_compiled"):
            self._tech_subs_compiled = [
                (re.compile(pat, re.IGNORECASE), repl)
                for pat, repl in self._TECH_TERM_SUBS
            ]
        for pattern, replacement in self._tech_subs_compiled:
            text = pattern.sub(replacement, text)
        return text


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
            audio = self._apply_gain_normalization(audio)  # rescue quiet-but-valid speech
            if epoch != self._interim_epoch:
                return
            with self._infer_lock:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self._language,
                    beam_size=self._beam_size,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    initial_prompt=self._whisper_initial_prompt or None,
                    no_speech_threshold=0.7,
                    suppress_blank=False,
                )
            if epoch != self._interim_epoch:
                return
            text = self._filter_segments(segments)
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
