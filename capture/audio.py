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
        self._chunking_system_mode_enabled = bool(
            config.get("capture.audio.chunking.system_mode_enabled", False)
        )
        self._chunk_min_s = float(config.get("capture.audio.chunking.min_chunk_s", 3.0))
        self._chunk_max_s = float(config.get("capture.audio.chunking.max_chunk_s", 6.0))

        # ── Phase 2: Transcription Provider ──────────────────────────────────
        self._transcription_provider = str(
            config.get("capture.audio.transcription_provider", "groq")
        ).lower()
        self._groq_stt_model = str(
            config.get("capture.audio.groq_stt_model", "whisper-large-v3-turbo")
            or "whisper-large-v3-turbo"
        )
        self._cloud_stt_unavailable_logged = False

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
        self._system_beam_size = max(
            1,
            min(
                self._beam_size,
                int(config.get("capture.audio.whisper_system_beam_size", 1) or 1),
            ),
        )
        _configured_cpu_threads = int(config.get("capture.audio.whisper_cpu_threads", 0) or 0)
        self._whisper_cpu_threads = (
            _configured_cpu_threads
            if _configured_cpu_threads > 0
            else max(2, min(8, (os.cpu_count() or 4) - 1))
        )
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
        self._session_transcript_part_providers: list = []
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
        self._interim_max_speech_s = float(
            config.get("capture.audio.interim.max_speech_s", 8.0) or 8.0
        )
        self._interim_tail_s = float(
            (config.get("capture.audio.interim.tail_ms", 3500) or 3500) / 1000.0
        )
        self._last_interim_at = 0.0
        self._interim_epoch = 0
        self._interim_inflight = False
        self._trace_session_id = ""
        self._trace_session_started_at = 0.0
        self._trace_raw_audio_logged = False
        self._trace_utterance_counter = 0
        self._whisper_preload_inflight = False
        self._interim_beam_size = max(
            1,
            min(
                self._beam_size,
                int(config.get("capture.audio.interim.beam_size", 1) or 1),
            ),
        )
        self._interim_max_pending_finals = max(
            0,
            int(config.get("capture.audio.interim.max_pending_finals", 1) or 1),
        )
        self._system_noise_gate_enabled = bool(
            config.get("capture.audio.vad.system_noise_gate_enabled", True)
        )
        self._system_start_floor_multiplier = float(
            config.get("capture.audio.vad.system_start_floor_multiplier", 2.0) or 2.0
        )
        self._system_start_min_rms = float(
            config.get("capture.audio.vad.system_start_min_rms", 0.003) or 0.003
        )
        self._system_continue_floor_multiplier = float(
            config.get("capture.audio.vad.system_continue_floor_multiplier", 1.3) or 1.3
        )
        self._system_continue_min_rms = float(
            config.get("capture.audio.vad.system_continue_min_rms", 0.002) or 0.002
        )
        self._system_start_confirm_blocks = max(
            1, int(config.get("capture.audio.vad.system_start_confirm_blocks", 2) or 2)
        )
        self._system_queue_pressure_drop_enabled = bool(
            config.get("capture.audio.vad.system_queue_pressure_drop_enabled", True)
        )
        self._system_queue_pressure_max_pending = max(
            1, int(config.get("capture.audio.vad.system_queue_pressure_max_pending", 3) or 3)
        )
        self._system_queue_pressure_max_speech_s = float(
            (config.get("capture.audio.vad.system_queue_pressure_max_speech_ms", 600) or 600) / 1000.0
        )
        self._system_queue_pressure_max_voiced_blocks = max(
            1, int(config.get("capture.audio.vad.system_queue_pressure_max_voiced_blocks", 3) or 3)
        )
        self._system_queue_pressure_max_peak_rms = float(
            config.get("capture.audio.vad.system_queue_pressure_max_peak_rms", 0.02) or 0.02
        )
        self._system_followup_guard_enabled = bool(
            config.get("capture.audio.vad.system_followup_guard_enabled", True)
        )
        self._system_followup_guard_window_s = float(
            (config.get("capture.audio.vad.system_followup_guard_window_ms", 1800) or 1800) / 1000.0
        )
        self._system_followup_guard_max_speech_s = float(
            (config.get("capture.audio.vad.system_followup_guard_max_speech_ms", 700) or 700) / 1000.0
        )
        self._system_followup_guard_max_peak_rms = float(
            config.get("capture.audio.vad.system_followup_guard_max_peak_rms", 0.03) or 0.03
        )
        self._system_followup_guard_max_voiced_blocks = max(
            1, int(config.get("capture.audio.vad.system_followup_guard_max_voiced_blocks", 18) or 18)
        )
        self._session_start_time = 0.0

        # Short-question tuning: keep end-of-speech tighter for brief prompts.
        self._short_utterance_max_s = float(
            config.get("capture.audio.vad.short_utterance_max_s", 2.8) or 2.8
        )
        self._ultra_short_utterance_max_s = float(
            config.get("capture.audio.vad.ultra_short_utterance_max_s", 0.35) or 0.35
        )
        self._phrase_utterance_max_s = float(
            config.get("capture.audio.vad.phrase_utterance_max_s", 0.9) or 0.9
        )
        self._short_silence_ms = int(
            config.get("capture.audio.vad.short_silence_ms", 500) or 500
        )
        self._short_silence_blocks = max(1, int(self._short_silence_ms / self.block_ms))
        self._system_short_silence_ms = int(
            config.get("capture.audio.vad.system_short_silence_ms", 600) or 600
        )
        self._system_short_silence_blocks = max(
            1, int(self._system_short_silence_ms / self.block_ms)
        )
        self._ultra_short_silence_ms = int(
            config.get("capture.audio.vad.ultra_short_silence_ms", 900) or 900
        )
        self._ultra_short_silence_blocks = max(
            1, int(self._ultra_short_silence_ms / self.block_ms)
        )
        self._phrase_silence_ms = int(
            config.get("capture.audio.vad.phrase_silence_ms", 1200) or 1200
        )
        self._phrase_silence_blocks = max(
            1, int(self._phrase_silence_ms / self.block_ms)
        )
        self._post_chunk_silence_ms = int(
            config.get("capture.audio.vad.post_chunk_silence_ms", 500) or 500
        )
        self._post_chunk_silence_blocks = max(
            1, int(self._post_chunk_silence_ms / self.block_ms)
        )
        self._max_utterance_s = float(
            config.get("capture.audio.vad.max_utterance_s", 30.0) or 30.0
        )
        self._inter_turn_start_silence_ms = int(
            config.get("capture.audio.vad.inter_turn_start_silence_ms", 400) or 400
        )
        self._inter_turn_start_silence_blocks = max(
            1, int(self._inter_turn_start_silence_ms / self.block_ms)
        )
        # Warn when configured ms values floor to a different effective duration.
        # This prevents silent config/behaviour discrepancies during EOS tuning.
        for _cfg_key, _cfg_ms, _blocks in [
            ("ultra_short_silence_ms", self._ultra_short_silence_ms, self._ultra_short_silence_blocks),
            ("phrase_silence_ms", self._phrase_silence_ms, self._phrase_silence_blocks),
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
        # Cloud STT calls must not run on the single transcription worker.  The
        # final transcription job itself runs there, then waits for cloud chunk
        # futures; using the same pool would deadlock until each future timed out.
        self._cloud_stt_pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="cloud-stt"
        )
        # Separate pool for interim ASR so final transcription isn't queued behind interim jobs.
        self._interim_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper-live"
        )

        self._active_streams = []
        self._lock = threading.RLock()
        self._capture_buffer_lock = threading.Lock()
        self._pending_transcription_jobs = 0
        self._standard_transcription_suspended = False
        self._hardware_capture_suspended = False
        self._final_decode_pending = 0
        self._capture_generation = 0
        self._final_submission_generation = 0
        self._last_final_submit_at = 0.0
        self._latest_short_system_final_seq = 0
        self._submitted_final_seq = 0
        self._cloud_stt_session_blocked = False
        self._cloud_stt_failed_key = ""
        self._whisper_device = "cpu"

        if state is None:
            from core.state import AppState

            state = AppState(config)

        self._state = state
        self._state.muted_changed.connect(self._on_state_mute_changed)
        self._muted = self._state.is_muted
        self._paused = self._muted
        self._capture_chunk_buffer = np.empty((0, 1), dtype=np.float32)

    def _on_state_mute_changed(self, muted: bool):
        logger.debug(f"Audio: State sync -> Muted={muted}")
        self._muted = muted
        self._paused = muted
        if not muted:
            self._drain_queue()
            logger.debug("Audio: Buffer flushed on unmute.")

    def _rearm_ambient_calibration(self) -> None:
        self._ambient_calib_remaining = self._ambient_calib_blocks
        self._ambient_rms_samples = []
        self._dynamic_rms_floor = 0.0

    def _session_capture_active(self) -> bool:
        state = getattr(self, "_state", None)
        return bool(getattr(state, "is_capturing", False))

    def _clear_capture_chunk_buffer(self) -> None:
        with self._capture_buffer_lock:
            self._capture_chunk_buffer = np.empty((0, 1), dtype=np.float32)

    def set_trace_context(self, session_id: str = "", session_started_at: float = 0.0) -> None:
        self._capture_generation = int(getattr(self, "_capture_generation", 0)) + 1
        self._final_submission_generation = int(getattr(self, "_final_submission_generation", 0)) + 1
        self._trace_session_id = str(session_id or "")
        self._trace_session_started_at = float(session_started_at or 0.0)
        self._trace_raw_audio_logged = False
        self._trace_utterance_counter = 0
        self._cloud_stt_session_blocked = False
        self._cloud_stt_unavailable_logged = False
        self._last_final_submit_at = 0.0
        self._session_start_time = time.time()
        self._rearm_ambient_calibration()
        self._drain_queue()
        self._clear_capture_chunk_buffer()
        with self._session_parts_lock:
            self._session_transcript_parts = []
            self._session_transcript_part_providers = []
        if self._trace_session_id:
            logger.info("[%s] Audio trace armed", self._trace_session_id)

    def _trace_elapsed_ms(self) -> float:
        if not self._trace_session_started_at:
            return 0.0
        return max(0.0, (time.time() - self._trace_session_started_at) * 1000.0)

    def _next_trace_utterance_id(self) -> str:
        self._trace_utterance_counter += 1
        if self._trace_session_id:
            return f"{self._trace_session_id}:utt_{self._trace_utterance_counter}"
        return f"utt_{self._trace_utterance_counter}"

    def _trace_raw_audio_block(self, source: str, rms: float, frames: int) -> None:
        if self._trace_raw_audio_logged:
            return
        self._trace_raw_audio_logged = True
        if self._trace_session_id:
            logger.info(
                "[%s] RAW AUDIO HEARD | elapsed=%.1fms | source=%s | frames=%d | rms=%.5f",
                self._trace_session_id,
                self._trace_elapsed_ms(),
                source,
                frames,
                rms,
            )
        else:
            logger.info(
                "RAW AUDIO HEARD | source=%s | frames=%d | rms=%.5f",
                source,
                frames,
                rms,
            )

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
                        self._model_name,
                        device=device,
                        compute_type=compute,
                        cpu_threads=self._whisper_cpu_threads if device == "cpu" else 0,
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
                            self._model_name,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=self._whisper_cpu_threads,
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
                self._session_transcript_part_providers = []
            # Reset Groq STT session block so a fresh start can retry the key
            self._cloud_stt_session_blocked = False
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="audio-cap"
            )
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True, name="audio-proc"
            )
            self._capture_thread.start()
            self._process_thread.start()
            # Pre-warm local Whisper in a standalone daemon thread so the first
            # utterance doesn't stall for 20+ seconds on model load. Using a
            # dedicated thread (not _transcribe_pool) ensures incoming utterances
            # never queue behind the model load job.
            threading.Thread(
                target=self._ensure_whisper_loaded,
                daemon=True,
                name="whisper-prewarm",
            ).start()

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
            healthy = self._capture_workers_healthy_locked()
            if was_running and new_mode == self.last_mode and healthy:
                logger.debug(f"🎙️ Audio: Mode '{new_mode}' already active. Skipping restart.")
                return

            if was_running and new_mode == self.last_mode and not healthy:
                logger.warning(
                    "Audio: Restarting unhealthy pipeline in-place for mode '%s'",
                    new_mode,
                )
            else:
                logger.info(f"🎤 Restarting Audio Pipeline: {new_mode}...")
            self._running = False
            self._close_streams()
            self._drain_queue()
            self.capture_mode = new_mode or "system"
            self.last_mode = self.capture_mode

        if was_running:
            time.sleep(0.4)
            self.start()

    def ensure_session_ready(self) -> bool:
        """Best-effort re-arm of capture when a new session begins.

        Returns immediately without blocking to avoid UI freezes.
        Audio warms up asynchronously in background threads.
        """
        if not self.config.get("capture.audio.enabled", True):
            logger.info("Audio: Session requested while capture is disabled")
            return False

        with self._lock:
            self._muted = bool(getattr(self._state, "is_muted", self._muted))
            self._paused = self._muted
            running = self._running
            healthy = self._capture_workers_healthy_locked() if running else False

        if running:
            # Already running — fire async cleanup + fresh start in background
            # to ensure warmth on the very next audio block.
            if healthy:
                self._drain_queue()
                self._clear_capture_chunk_buffer()
                self._ensure_whisper_loaded_async()
            else:
                logger.warning(
                    "Audio: Restarting unhealthy pipeline in-place for mode '%s'",
                    self.capture_mode,
                )
                threading.Thread(target=self.restart, daemon=True).start()
            return True

        # Not running — start immediately, return without waiting
        self.start()
        return True

    def _stream_is_ready(self, stream) -> bool:
        if stream is None:
            return False
        if hasattr(stream, "poll"):
            try:
                return stream.poll() is None
            except Exception:
                return False
        if hasattr(stream, "active"):
            return bool(getattr(stream, "active", False))
        if hasattr(stream, "running"):
            return bool(getattr(stream, "running", False))
        return True

    def _capture_workers_healthy_locked(self) -> bool:
        capture_alive = bool(self._capture_thread and self._capture_thread.is_alive())
        process_alive = bool(self._process_thread and self._process_thread.is_alive())
        streams_ready = (
            True
            if bool(getattr(self, "_hardware_capture_suspended", False))
            else any(self._stream_is_ready(s) for s in self._active_streams)
        )
        return bool(self._running and capture_alive and process_alive and streams_ready)

    def _ensure_whisper_loaded_async(self, *, force: bool = False) -> None:
        if not force and self._effective_transcription_provider(is_final=True) != "local":
            return
        if getattr(self, "_whisper_preload_inflight", False) or self._model_loaded:
            return
        self._whisper_preload_inflight = True

        def _run() -> None:
            try:
                self._ensure_whisper_loaded()
            finally:
                self._whisper_preload_inflight = False

        threading.Thread(target=_run, daemon=True, name="whisper-preload").start()

    def _close_streams(self):
        import subprocess
        for s in list(self._active_streams):
            try:
                if isinstance(s, subprocess.Popen):
                    if s.poll() is None:
                        s.terminate()
                        try:
                            s.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            s.kill()
                else:
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

    def _enqueue_audio_frames(self, frames, source: str = "audio") -> None:
        if frames is None:
            return
        if not self._running or self._paused:
            return

        data = np.asarray(frames, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

        if data.size == 0:
            return

        if data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True).astype(np.float32)
        else:
            data = data.astype(np.float32, copy=False)

        if not bool(getattr(self._state, "is_capturing", False)):
            self._capture_chunk_buffer = np.empty((0, 1), dtype=np.float32)
            return

        if self._capture_chunk_buffer.size:
            data = np.concatenate((self._capture_chunk_buffer, data), axis=0)

        while data.shape[0] >= self.block_size:
            chunk = data[: self.block_size].copy()
            data = data[self.block_size :]
            try:
                self.q.put_nowait(chunk)
                self._current_rms = float(np.sqrt(np.mean(chunk ** 2)))
                self._trace_raw_audio_block(source, self._current_rms, len(chunk))
            except queue.Full:
                logger.debug("Audio queue full; dropping frame")
                break

        self._capture_chunk_buffer = data.copy() if data.size else np.empty((0, 1), dtype=np.float32)

    def _find_system_audio_source(self):
        sources = self._system_audio_sources()
        return sources[0] if sources else (None, "Default", False)

    def _system_audio_sources(self):
        import sounddevice as sd

        candidates = []
        seen = set()

        def add(idx, name, is_loopback):
            if idx is None or idx in seen:
                return
            seen.add(idx)
            candidates.append((idx, name, is_loopback))

        try:
            devices = sd.query_devices()
            apis = sd.query_hostapis()
            wasapi_idx = next(
                (i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None
            )
            if wasapi_idx is not None:
                for i, d in enumerate(devices):
                    if d["hostapi"] == wasapi_idx and d["max_output_channels"] > 0:
                        add(i, d["name"], True)

            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0 and any(
                    x in d["name"].lower()
                    for x in ["cable", "vb-audio", "blackhole", "soundflower"]
                ):
                    add(i, d["name"], False)

            if bool(self.config.get("capture.audio.allow_stereo_mix", True)):
                for i, d in enumerate(devices):
                    if d["max_input_channels"] > 0 and any(
                        x in d["name"].lower()
                        for x in ["stereo mix", "what u hear"]
                    ):
                        add(i, d["name"], False)
        except Exception as e:
            logger.debug(f"Source discovery error: {e}")
        return candidates

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

    def _start_macos_system_audio(self):
        """macOS: Spawn SystemAudioDump and pipe its output to the queue."""
        import subprocess
        import sys
        from utils.platform_utils import PlatformInfo

        # Kill any existing instances first
        try:
            subprocess.run(["pkill", "-f", "SystemAudioDump"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        bin_path = str(PlatformInfo.get_resource_path("assets/SystemAudioDump"))
        if not os.path.exists(bin_path):
            logger.error(f"macOS SystemAudioDump binary not found at {bin_path}")
            return False

        # Ensure executable
        try:
            os.chmod(bin_path, 0o755)
        except Exception as e:
            logger.debug(f"macOS: Failed to chmod SystemAudioDump: {e}")

        logger.info(f"🎤 Binding to macOS SystemAudioDump: {bin_path}")

        # Spawn the process
        proc = subprocess.Popen(
            [bin_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0  # unbuffered
        )
        self._active_streams.append(proc)

        # Start a dedicated thread to read from stdout
        t = threading.Thread(target=self._macos_audio_reader, args=(proc,), daemon=True, name="macos-sysaudio")
        t.start()
        return True

    def _macos_audio_reader(self, proc):
        """Read 24000Hz 16-bit stereo PCM from SystemAudioDump stdout, downsample, and queue."""
        source_rate = 24000
        channels = 2
        bytes_per_sample = 2

        chunk_frames = int(source_rate * (self.block_ms / 1000.0))
        bytes_to_read = chunk_frames * channels * bytes_per_sample

        while self._running and proc.poll() is None:
            try:
                # Read exactly bytes_to_read
                raw_data = proc.stdout.read(bytes_to_read)
                if not raw_data:
                    break

                if (
                    not self._paused
                    and not self._hardware_capture_suspended
                    and self._running
                ):
                    # Convert to numpy int16
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)

                    if len(audio_data) % channels == 0:
                        audio_data = audio_data.reshape(-1, channels)
                    else:
                        frames = len(audio_data) // channels
                        audio_data = audio_data[:frames * channels].reshape(-1, channels)

                    # Convert to float32 [-1.0, 1.0]
                    audio_float = audio_data.astype(np.float32) / 32768.0

                    # Resample to target rate (self.sr)
                    resampled = self._resample_to_target_rate(audio_float, source_rate)

                    # Convert to mono
                    data = np.mean(resampled, axis=1, keepdims=True).astype(np.float32)

                    try:
                        self.q.put_nowait(data)
                        self._current_rms = float(np.sqrt(np.mean(resampled**2)))
                        self._trace_raw_audio_block(
                            "macos-system-audio",
                            self._current_rms,
                            len(data),
                        )
                    except queue.Full:
                        logger.debug("Audio queue full; dropping frame")
            except Exception as e:
                if self._running:
                    logger.debug(f"macOS SystemAudioDump read error: {e}")
                break

        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=1.0)
        except Exception:
            pass

    def _capture_loop(self):
        import sounddevice as sd

        def make_cb(source_rate):
            def cb(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Audio Status: {status}")
                if (
                    not self._paused
                    and not self._hardware_capture_suspended
                    and self._running
                ):
                    try:
                        normalized = self._resample_to_target_rate(indata, source_rate)
                        data = (
                            np.mean(normalized, axis=1, keepdims=True).astype(np.float32)
                            if normalized.shape[1] > 1
                            else normalized.copy().astype(np.float32)
                        )
                        self.q.put_nowait(data)
                        self._current_rms = float(np.sqrt(np.mean(normalized**2)))
                        self._trace_raw_audio_block(
                            f"{self.capture_mode}-audio",
                            self._current_rms,
                            len(data),
                        )
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
                    if self._hardware_capture_suspended:
                        if attempt == 0:
                            logger.info(
                                "Audio hardware capture suspended; using injected audio frames."
                            )
                        while self._running and self._hardware_capture_suspended:
                            self.level.emit(self._current_rms)
                            time.sleep(0.1)
                        if not self._running:
                            return
                        continue

                    if mode in ["mic", "both"]:
                        default_input = sd.query_devices(None, "input")
                        mic_rate = int(default_input.get("default_samplerate", self.sr))
                        s_mic = sd.InputStream(
                            samplerate=mic_rate, channels=1, callback=make_cb(mic_rate)
                        )
                        self._active_streams.append(s_mic)
                        s_mic.start()

                    if mode in ["system", "both"]:
                        import sys
                        if sys.platform == "darwin":
                            self._start_macos_system_audio()
                        else:
                            opened = False
                            last_error = None
                            for idx, name, is_loopback in self._system_audio_sources():
                                try:
                                    d = sd.query_devices(idx)
                                    logger.info(
                                        f"ðŸŽ¤ Binding to System Audio: {name} (Loopback: {is_loopback})"
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
                                    opened = True
                                    break
                                except Exception as source_error:
                                    last_error = source_error
                                    logger.warning(
                                        "System audio source unavailable (%s): %s",
                                        name,
                                        source_error,
                                    )
                            if not opened:
                                fallback_enabled = bool(
                                    self.config.get(
                                        "capture.audio.system_fallback_to_mic",
                                        False,
                                    )
                                )
                                if fallback_enabled:
                                    try:
                                        default_input = sd.query_devices(None, "input")
                                        mic_rate = int(default_input.get("default_samplerate", self.sr))
                                        mic_name = str(default_input.get("name", "default input"))
                                        logger.warning(
                                            "System audio unavailable (%s); falling back to microphone input: %s",
                                            last_error or "no candidate devices",
                                            mic_name,
                                        )
                                        s_sys = sd.InputStream(
                                            samplerate=mic_rate,
                                            channels=1,
                                            callback=make_cb(mic_rate),
                                        )
                                        self._active_streams.append(s_sys)
                                        s_sys.start()
                                        opened = True
                                    except Exception as fallback_error:
                                        last_error = fallback_error
                                if not opened:
                                    raise RuntimeError(
                                        f"No usable system audio source found: {last_error or 'no candidate devices'}"
                                    )
                            if False:
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
            if self._running:
                self._running = False

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
        short_blocks = self._short_silence_blocks
        if str(getattr(self, "capture_mode", "")).lower() == "system":
            short_blocks = max(short_blocks, self._system_short_silence_blocks)
        if not had_mid_utterance_slice and elapsed <= self._ultra_short_utterance_max_s:
            return max(short_blocks, self._ultra_short_silence_blocks)
        if not had_mid_utterance_slice and elapsed <= self._phrase_utterance_max_s:
            return max(short_blocks, self._phrase_silence_blocks)
        if had_mid_utterance_slice and elapsed <= self._short_utterance_max_s:
            # New chunk is still short — use aggressive post-chunk tail.
            return min(self.silence_blocks, self._post_chunk_silence_blocks)
        if elapsed <= self._short_utterance_max_s:
            return min(self.silence_blocks, short_blocks)
        return self.silence_blocks

    def _detect_speech(self, block: np.ndarray, rms: float) -> tuple[bool, str]:
        """Choose the best available local speech detector."""
        if self._vad is not None:
            return self._webrtc_vad_has_speech(block), "webrtc"
        return rms > 0.001, "rms"

    def _effective_final_beam_size(self) -> int:
        if str(getattr(self, "capture_mode", "")).lower() == "system":
            return max(1, min(self._beam_size, self._system_beam_size))
        return self._beam_size

    def _speech_rms_threshold(self, *, is_speaking: bool) -> float:
        threshold = max(0.0, float(self._dynamic_rms_floor or 0.0))
        if str(getattr(self, "capture_mode", "")).lower() != "system" or not self._system_noise_gate_enabled:
            return threshold
        if is_speaking:
            return max(
                threshold * self._system_continue_floor_multiplier,
                self._system_continue_min_rms,
            )
        return max(
            threshold * self._system_start_floor_multiplier,
            self._system_start_min_rms,
        )

    def _passes_speech_rms_gate(self, rms: float, *, is_speaking: bool, peak_rms: float = 0.0) -> bool:
        threshold = self._speech_rms_threshold(is_speaking=is_speaking)
        if threshold > 0.0 and rms < threshold:
            return False
        if is_speaking and peak_rms > 0.01 and rms < (peak_rms * 0.15):
            return False
        return True

    def _required_start_confirm_blocks(self) -> int:
        if str(getattr(self, "capture_mode", "")).lower() == "system" and self._system_noise_gate_enabled:
            return self._system_start_confirm_blocks
        return 1

    def _submit_transcription_job(self, *args) -> bool:
        """Best-effort submit that stays quiet during app shutdown."""
        pool = getattr(self, "_transcribe_pool", None)
        if pool is None:
            return False
        fn = args[0]
        fn_args = args[1:]

        def _tracked_job():
            try:
                fn(*fn_args)
            finally:
                with self._lock:
                    self._pending_transcription_jobs = max(
                        0,
                        int(getattr(self, "_pending_transcription_jobs", 0)) - 1,
                    )

        try:
            with self._lock:
                self._pending_transcription_jobs = int(getattr(self, "_pending_transcription_jobs", 0)) + 1
            pool.submit(_tracked_job)
            return True
        except RuntimeError as e:
            with self._lock:
                self._pending_transcription_jobs = max(
                    0,
                    int(getattr(self, "_pending_transcription_jobs", 0)) - 1,
                )
            logger.warning(f"Audio transcription submit skipped during shutdown: {e}")
            return False

    def has_pending_transcription_jobs(self) -> bool:
        with self._lock:
            return bool(int(getattr(self, "_pending_transcription_jobs", 0)))

    def set_standard_transcription_suspended(self, enabled: bool, reason: str = "") -> None:
        self._standard_transcription_suspended = bool(enabled)
        logger.debug(
            "Audio: standard transcription %s%s",
            "suspended" if enabled else "resumed",
            f" ({reason})" if reason else "",
        )

    def set_hardware_capture_suspended(self, enabled: bool, reason: str = "") -> None:
        self._hardware_capture_suspended = bool(enabled)
        logger.debug(
            "Audio: hardware capture %s%s",
            "suspended" if enabled else "resumed",
            f" ({reason})" if reason else "",
        )

    def _chunking_active(self) -> bool:
        if not bool(getattr(self, "_chunking_enabled", False)):
            return False
        if str(getattr(self, "capture_mode", "")).lower() == "system":
            return bool(getattr(self, "_chunking_system_mode_enabled", False))
        return True

    def _can_run_interim_with_pending_finals(self) -> bool:
        pending = int(getattr(self, "_final_decode_pending", 0) or 0)
        return pending <= self._interim_max_pending_finals

    def _should_drop_queued_final_under_pressure(
        self,
        *,
        speech_duration: float,
        voiced_blocks: int,
        peak_rms: float,
        vad_meta=None,
        buffer_len: int = 0,
    ) -> bool:
        if str(getattr(self, "capture_mode", "")).lower() != "system":
            return False
        if not bool(self.config.get("capture.audio.vad.system_queue_pressure_drop_enabled", True)):
            return False
        if bool((vad_meta or {}).get("chunk_aware_eos", False)):
            return False
        pending = int(getattr(self, "_final_decode_pending", 0))
        max_pending = int(self.config.get("capture.audio.vad.system_queue_pressure_max_pending", 3) or 3)
        max_speech_ms = int(self.config.get("capture.audio.vad.system_queue_pressure_max_speech_ms", 600) or 600)
        max_voiced_blocks = int(self.config.get("capture.audio.vad.system_queue_pressure_max_voiced_blocks", 3) or 3)
        max_peak_rms = float(self.config.get("capture.audio.vad.system_queue_pressure_max_peak_rms", 0.02) or 0.02)
        return (
            pending >= max_pending
            and (speech_duration * 1000.0) <= max_speech_ms
            and voiced_blocks <= max_voiced_blocks
            and peak_rms <= max_peak_rms
        )

    def _should_drop_burst_followup_system_utterance(
        self,
        *,
        speech_duration: float,
        voiced_blocks: int,
        peak_rms: float,
        vad_meta=None,
    ) -> bool:
        if str(getattr(self, "capture_mode", "")).lower() != "system":
            return False
        if not bool(self.config.get("capture.audio.vad.system_followup_guard_enabled", True)):
            return False
        if bool((vad_meta or {}).get("chunk_aware_eos", False)):
            return False
        if int(getattr(self, "_final_decode_pending", 0)) <= 0:
            return False
        window_ms = int(self.config.get("capture.audio.vad.system_followup_guard_window_ms", 1800) or 1800)
        max_speech_ms = int(self.config.get("capture.audio.vad.system_followup_guard_max_speech_ms", 700) or 700)
        max_peak_rms = float(self.config.get("capture.audio.vad.system_followup_guard_max_peak_rms", 0.03) or 0.03)
        max_voiced_blocks = int(self.config.get("capture.audio.vad.system_followup_guard_max_voiced_blocks", 18) or 18)
        age_ms = max(0.0, (time.time() - float(getattr(self, "_last_final_submit_at", 0.0) or 0.0)) * 1000.0)
        return (
            age_ms <= window_ms
            and (speech_duration * 1000.0) <= max_speech_ms
            and peak_rms <= max_peak_rms
            and voiced_blocks <= max_voiced_blocks
        )

    def _should_drop_final_utterance(
        self,
        *,
        speech_duration: float,
        voiced_blocks: int,
        peak_rms: float,
        utterance_id: str = "",
    ) -> bool:
        min_speech_ms = int(self.config.get("capture.audio.vad.min_final_speech_ms", 180) or 180)
        min_voiced_blocks = int(self.config.get("capture.audio.vad.min_final_voiced_blocks", 2) or 2)
        min_peak_rms = float(self.config.get("capture.audio.vad.min_final_peak_rms", 0.003) or 0.003)
        return (
            (speech_duration * 1000.0) < min_speech_ms
            and voiced_blocks < min_voiced_blocks
            and peak_rms < min_peak_rms
        ) or (
            str(getattr(self, "capture_mode", "")).lower() == "system"
            and voiced_blocks <= 0
            and (speech_duration * 1000.0) <= max(min_speech_ms + 20, 220)
        )

    def _effective_transcription_provider(self, *, is_final: bool, speech_started_at=None) -> str:
        provider = str(getattr(self, "_transcription_provider", "local") or "local").lower()
        if provider != "groq":
            return "local"
        if getattr(self, "_cloud_stt_session_blocked", False):
            return "local"
        groq_key = self._groq_api_key()
        failed_key = str(getattr(self, "_cloud_stt_failed_key", "") or "")
        if failed_key and groq_key and failed_key == groq_key:
            return "local"
        if not groq_key:
            if not getattr(self, "_cloud_stt_unavailable_logged", False):
                logger.warning("[Groq STT] No Groq API key configured; using local Whisper.")
                self._cloud_stt_unavailable_logged = True
            return "local"
        return "groq"

    def _groq_api_key_available(self) -> bool:
        return bool(self._groq_api_key())

    def _groq_api_key(self) -> str:
        try:
            return str(self.config.get_api_key("groq") or "").strip()
        except Exception:
            return ""

    def _submit_final_transcription(self, buffer, speech_started_at=None, vad_meta=None) -> bool:
        # Paused capture should not enqueue final ASR work.
        if self._standard_transcription_suspended:
            return False

        meta = dict(vad_meta or {})
        speech_duration = float(meta.get("speech_duration", 0.0) or 0.0)
        voiced_blocks = int(meta.get("voiced_blocks", len(buffer) if buffer is not None else 0) or 0)
        peak_rms = float(meta.get("peak_rms", 0.0) or 0.0)

        have_strength_metrics = any(
            key in meta for key in ("speech_duration", "voiced_blocks", "peak_rms")
        )
        if have_strength_metrics:
            if self._should_drop_final_utterance(
                speech_duration=speech_duration,
                voiced_blocks=voiced_blocks,
                peak_rms=peak_rms,
                utterance_id=str(meta.get("utterance_id", "") or ""),
            ):
                return False
            if self._should_drop_queued_final_under_pressure(
                speech_duration=speech_duration,
                voiced_blocks=voiced_blocks,
                peak_rms=peak_rms,
                vad_meta=meta,
                buffer_len=len(buffer) if buffer is not None else 0,
            ):
                return False
            if self._should_drop_burst_followup_system_utterance(
                speech_duration=speech_duration,
                voiced_blocks=voiced_blocks,
                peak_rms=peak_rms,
                vad_meta=meta,
            ):
                return False

        self._submitted_final_seq = int(getattr(self, "_submitted_final_seq", 0)) + 1
        submission_seq = self._submitted_final_seq
        capture_generation = int(getattr(self, "_capture_generation", 0))
        self._final_submission_generation = capture_generation
        self._final_decode_pending = int(getattr(self, "_final_decode_pending", 0)) + 1
        self._last_final_submit_at = time.time()
        meta.setdefault("speech_finalized_at", self._last_final_submit_at)

        short_system_candidate = (
            str(getattr(self, "capture_mode", "")).lower() == "system"
            and not bool(meta.get("chunk_aware_eos", False))
            and speech_duration > 0.0
            and speech_duration <= 1.0
            and peak_rms <= 0.03
        )
        if short_system_candidate:
            self._latest_short_system_final_seq = submission_seq

        def _run_final():
            try:
                if capture_generation != int(getattr(self, "_capture_generation", 0)):
                    return
                if short_system_candidate and submission_seq != int(getattr(self, "_latest_short_system_final_seq", 0)):
                    return
                self._transcribe(buffer, speech_started_at, True, vad_meta=meta)
            finally:
                self._final_decode_pending = max(0, int(getattr(self, "_final_decode_pending", 0)) - 1)

        queued = self._submit_transcription_job(_run_final)
        if not queued:
            self._final_decode_pending = max(0, int(getattr(self, "_final_decode_pending", 0)) - 1)
        return queued

    def _force_finalize_utterance(
        self,
        speech_buffer,
        speech_started_at,
        utterance_started_at,
        utterance_vad_backend,
        had_mid_utterance_slice,
        reason: str,
        utterance_id: str = "",
    ):
        elapsed = (
            max(0.0, time.time() - utterance_started_at)
            if utterance_started_at
            else 0.0
        )
        logger.warning(
            "[%s] FORCE FINALIZE | elapsed=%.1fms | utterance=%s | speech_elapsed=%.1fs | reason=%s",
            self._trace_session_id or "audio",
            self._trace_elapsed_ms(),
            utterance_id,
            elapsed,
            reason,
        )
        self._interim_epoch += 1
        self._submit_final_transcription(
            list(speech_buffer),
            speech_started_at,
            {
                "utterance_id": utterance_id,
                "vad_backend": utterance_vad_backend,
                "end_silence_ms": 0,
                "chunk_aware_eos": had_mid_utterance_slice,
                "utterance_started_at": utterance_started_at,
                "speech_finalized_at": time.time(),
                "speech_duration": elapsed,
                "voiced_blocks": len(speech_buffer),
            },
        )
        return [], False, 0, None, None, False

    def _process_loop(self):
        speech_buffer = []
        is_speaking = False
        silence_count = 0
        speech_started_at = None
        utterance_started_at = None
        utterance_id = ""
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
            raw_has_speech, backend = self._detect_speech(data, rms)

            # ── Phase 2: Adaptive Ambient Calibration ─────────────────────────
            # Consume the first N blocks as a silent noise floor sample.
            if self._ambient_calib_remaining > 0 and not is_speaking and not raw_has_speech:
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
            floor_ok = self._passes_speech_rms_gate(rms, is_speaking=is_speaking)
            has_speech = raw_has_speech and floor_ok

            if has_speech or is_speaking:
                speech_buffer.append(data)
            elif not is_speaking:
                # Not yet speaking — keep this block as potential pre-roll.
                pre_roll.append(data)
            if has_speech:
                silence_count = 0
                if not is_speaking:
                    utterance_id = self._next_trace_utterance_id()
                    logger.debug(
                        "Audio: Speech started (raw=%s, rms=%.5f, backend=%s)",
                        raw_has_speech,
                        rms,
                        backend,
                    )
                    logger.info(
                        "[%s] SPEECH DETECTED | elapsed=%.1fms | utterance=%s | backend=%s | rms=%.5f",
                        self._trace_session_id or "audio",
                        self._trace_elapsed_ms(),
                        utterance_id,
                        backend,
                        rms,
                    )
                    # Prepend pre-roll blocks so Whisper sees the onset context.
                    if pre_roll:
                        speech_buffer = list(pre_roll) + speech_buffer
                        pre_roll.clear()
                    speech_started_at = time.time()
                    utterance_started_at = speech_started_at
                    self._last_interim_at = 0.0
                    self._interim_epoch += 1
                    utterance_vad_backend = backend
                    had_mid_utterance_slice = False
                is_speaking = True
                if self._max_utterance_exceeded(utterance_started_at):
                    (
                        speech_buffer,
                        is_speaking,
                        silence_count,
                        speech_started_at,
                        utterance_started_at,
                        had_mid_utterance_slice,
                    ) = self._force_finalize_utterance(
                        speech_buffer,
                        speech_started_at,
                        utterance_started_at,
                        utterance_vad_backend,
                        had_mid_utterance_slice,
                        reason="continuous-speech",
                        utterance_id=utterance_id,
                    )
                    utterance_id = ""
                    pre_roll.clear()
                    continue

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
                    self._submit_transcription_job(
                        self._transcribe,
                        list(speech_buffer),
                        speech_started_at,
                        False,
                        {
                            "utterance_id": utterance_id,
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
                    self._submit_transcription_job(
                        self._transcribe,
                        list(speech_buffer),
                        speech_started_at,
                        False,
                        {
                            "utterance_id": utterance_id,
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
                    speech_duration = max(
                        0.0, time.time() - speech_started_at
                    ) if speech_started_at else 0.0
                    logger.debug(
                        "Audio: Speech ended after %.2fs (silence=%d/%d, backend=%s)",
                        speech_duration,
                        silence_count,
                        required_silence_blocks,
                        utterance_vad_backend,
                    )
                    logger.info(
                        "[%s] FINAL TRANSCRIPTION SUBMIT | elapsed=%.1fms | utterance=%s | speech_duration=%.2fs | blocks=%d | backend=%s",
                        self._trace_session_id or "audio",
                        self._trace_elapsed_ms(),
                        utterance_id,
                        speech_duration,
                        len(speech_buffer),
                        utterance_vad_backend,
                    )
                    # Real end-of-speech — emit the complete joined transcript.
                    self._interim_epoch += 1
                    peak_rms = max(
                        [float(np.sqrt(np.mean(np.asarray(block) ** 2))) for block in speech_buffer] or [0.0]
                    )
                    self._submit_final_transcription(
                        list(speech_buffer),
                        speech_started_at,
                        {
                "utterance_id": utterance_id,
                "vad_backend": utterance_vad_backend,
                "end_silence_ms": required_silence_blocks * self.block_ms,
                "chunk_aware_eos": had_mid_utterance_slice,
                "utterance_started_at": utterance_started_at,
                "speech_finalized_at": time.time(),
                "speech_duration": speech_duration,
                "voiced_blocks": len(speech_buffer),
                "peak_rms": peak_rms,
                        },
                    )
                    speech_buffer = []
                    is_speaking = False
                    silence_count = 0
                    speech_started_at = None
                    utterance_started_at = None
                    utterance_id = ""
                    had_mid_utterance_slice = False
                    # Reset session accumulator for the NEXT utterance only after
                    # _emit_accumulated has been called (it clears parts itself).
                    # No reset here — emit_accumulated handles it atomically.

                elif self._max_utterance_exceeded(utterance_started_at):
                    (
                        speech_buffer,
                        is_speaking,
                        silence_count,
                        speech_started_at,
                        utterance_started_at,
                        had_mid_utterance_slice,
                    ) = self._force_finalize_utterance(
                        speech_buffer,
                        speech_started_at,
                        utterance_started_at,
                        utterance_vad_backend,
                        had_mid_utterance_slice,
                        reason="silence-tail",
                        utterance_id=utterance_id,
                    )
                    utterance_id = ""
                    pre_roll.clear()

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
                    self._submit_interim(
                        list(speech_buffer),
                        speech_started_at,
                        self._interim_epoch,
                        utterance_id=utterance_id,
                    )

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
            # Responsive but permissive; downstream buffering prevents false triggers.
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

    def _submit_interim(self, buffer, speech_started_at: float, epoch: int, utterance_id: str = "") -> None:
        if self._interim_inflight:
            return
        if not buffer or len(buffer) < 3:
            return

        speech_duration = max(0.0, time.time() - speech_started_at) if speech_started_at else 0.0
        if self._interim_max_speech_s > 0.0 and speech_duration > self._interim_max_speech_s:
            return
        logger.info(
            "[%s] INTERIM TRANSCRIPTION SUBMIT | elapsed=%.1fms | utterance=%s | speech_duration=%.2fs | blocks=%d | epoch=%d",
            self._trace_session_id or "audio",
            self._trace_elapsed_ms(),
            utterance_id,
            speech_duration,
            len(buffer),
            epoch,
        )

        self._interim_inflight = True

        def _run():
            try:
                self._transcribe_interim(buffer, speech_started_at, epoch, utterance_id=utterance_id)
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
        if self._effective_transcription_provider(is_final=is_final, speech_started_at=speech_started_at) == "groq":
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
                    beam_size=self._effective_final_beam_size(),
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
                    self._append_session_transcript_part(text, provider="local")
                return

            # Final path: join all accumulated chunks + this one.
            if text:
                self._append_session_transcript_part(text, provider="local")
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

    def _append_session_transcript_part(self, text: str, provider: str) -> None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return
        with self._session_parts_lock:
            self._session_transcript_parts.append(cleaned)
            self._session_transcript_part_providers.append(str(provider or "local").lower())

    def _resolve_accumulated_provider(self, fallback: str, part_count: int) -> str:
        providers = [
            str(p or "").strip().lower()
            for p in getattr(self, "_session_transcript_part_providers", [])
            if str(p or "").strip()
        ]
        if not providers:
            return str(fallback or "local").lower()
        if len(providers) < part_count:
            providers.extend([str(fallback or "local").lower()] * (part_count - len(providers)))
        unique = set(providers[:part_count])
        if len(unique) == 1:
            return providers[0]
        return "mixed"

    def _emit_accumulated(self, speech_started_at, provider="local",
                           transcribe_started_at=None, vad_meta=None):
        """Join all session parts and emit transcription_ready once."""
        with self._session_parts_lock:
            parts = list(self._session_transcript_parts)
            effective_provider = self._resolve_accumulated_provider(provider, len(parts))
            self._session_transcript_parts = []
            self._session_transcript_part_providers = []

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
        effective_speech_started_at = (
            (vad_meta or {}).get("utterance_started_at") or speech_started_at
        )
        speech_finalized_at = (
            (vad_meta or {}).get("speech_finalized_at") or transcribe_started_at or now
        )
        audio_duration_ms = 0.0
        if effective_speech_started_at and speech_finalized_at:
            audio_duration_ms = max(
                0.0, (speech_finalized_at - effective_speech_started_at) * 1000.0
            )
        transcribe_started_effective = transcribe_started_at or speech_finalized_at or now
        self._last_transcription_metrics = {
            "speech_started_at": effective_speech_started_at,
            "speech_finalized_at": speech_finalized_at,
            "transcribe_started_at": transcribe_started_effective,
            "transcribe_finished_at": now,
            "speech_to_transcript_ms": (
                (now - effective_speech_started_at) * 1000.0 if effective_speech_started_at else None
            ),
            "transcribe_only_ms": (
                (now - transcribe_started_effective) * 1000.0 if transcribe_started_effective else 0.0
            ),
            "final_queue_wait_ms": (
                max(0.0, (transcribe_started_effective - speech_finalized_at) * 1000.0)
                if transcribe_started_effective and speech_finalized_at
                else 0.0
            ),
            "audio_duration_ms": audio_duration_ms,
            "text_length": len(cleaned),
            "provider": effective_provider,
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
        self._submit_transcription_job(self._update_prompt_context, cleaned)


    def _consume_pending_groq_chunk_futures(
        self,
        speech_started_at=None,
        vad_meta=None,
        timeout_s: float = 13.0,
    ) -> tuple[int, bool]:
        pending = list(getattr(self, "_groq_chunk_futures", []))
        self._groq_chunk_futures = []
        chunk_count = 0
        cloud_failed = False
        for entry in pending:
            fut = entry.get("future") if isinstance(entry, dict) else entry
            entry_buffer = entry.get("buffer", []) if isinstance(entry, dict) else []
            entry_started_at = (
                entry.get("speech_started_at") if isinstance(entry, dict) else speech_started_at
            )
            try:
                text = fut.result(timeout=timeout_s)
                if text and not self._is_hall(text):
                    self._append_session_transcript_part(text, provider="groq")
                    chunk_count += 1
                elif entry_buffer:
                    self._transcribe_local(
                        entry_buffer,
                        speech_started_at=entry_started_at or speech_started_at,
                        is_final=False,
                        vad_meta=vad_meta,
                    )
            except Exception as exc:
                cloud_failed = True
                logger.warning(
                    "[Groq STT] Pending chunk failed (%r); using local fallback for this chunk",
                    exc,
                )
                if entry_buffer:
                    self._transcribe_local(
                        entry_buffer,
                        speech_started_at=entry_started_at or speech_started_at,
                        is_final=False,
                        vad_meta=vad_meta,
                    )
        return chunk_count, cloud_failed

    def _transcribe_groq(self, buffer, speech_started_at=None, is_final: bool = True, vad_meta=None):
        """Phase 2: Transcribe via Groq Cloud Whisper API.

        Multi-chunk parallelism:
          is_final=False → encode audio to WAV, submit Groq call as a Future
                           stored in self._groq_chunk_futures (ordered list).
          is_final=True  → submit this chunk's Groq call, then join all pending
                           futures concurrently, merge results in order and emit.

        For single-chunk utterances the behaviour is identical to before.
        Falls back to local Whisper on any error.

        Session-level 403/401 guard: once Groq returns an auth error for this
        session, all subsequent calls skip straight to local Whisper to avoid
        the 10-second timeout on every utterance.
        """
        import io, wave

        # ── Fast-path: skip Groq if it already failed this session ──────────
        if getattr(self, "_cloud_stt_session_blocked", False):
            self._transcribe_local(buffer, speech_started_at, is_final, vad_meta=vad_meta)
            return

        if not buffer or len(buffer) < 2:
            if is_final:
                transcribe_started_at = time.time()
                chunk_count, cloud_failed = self._consume_pending_groq_chunk_futures(
                    speech_started_at=speech_started_at,
                    vad_meta=vad_meta,
                )
                if cloud_failed:
                    self._cloud_stt_session_blocked = True
                    logger.warning(
                        "[Groq STT] Disabled for this session after pending chunk failure; future utterances will use local Whisper"
                    )
                if chunk_count:
                    logger.info(
                        "[Groq STT] %d pending chunk(s) finalized on short tail | model=%s",
                        chunk_count,
                        self._groq_stt_model,
                    )
                self._emit_accumulated(
                    speech_started_at,
                    provider="groq",
                    transcribe_started_at=transcribe_started_at,
                    vad_meta=vad_meta,
                )
            return

        transcribe_started_at = time.time()

        def _build_wav(buf) -> tuple:
            """Encode buffer to 16 kHz/16-bit/mono WAV bytes. Returns (wav_bytes, n_samples)."""
            audio = np.concatenate(buf, axis=0).flatten()
            audio = self._apply_gain_normalization(audio)
            TARGET_SR = 16_000
            if self.sr != TARGET_SR:
                target_len = max(1, int(round(len(audio) * TARGET_SR / self.sr)))
                src_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
                dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
                audio = np.interp(dst_x, src_x, audio).astype(np.float32)
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SR)
                pcm = np.clip(audio, -1.0, 1.0)
                wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
            return wav_buf.getvalue(), len(audio)

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
            logger.warning("[Groq STT] No Groq API key - falling back to local Whisper")
            self._transcribe_local(buffer, speech_started_at, is_final, vad_meta=vad_meta)
            return

        _groq_stt_model = self._groq_stt_model
        _language = getattr(self, "_language", None)
        _is_hall = self._is_hall

        def _call_groq(wav_bytes: bytes) -> str:
            """POST wav_bytes to Groq and return transcript text (empty on error)."""
            try:
                from groq import Groq
                buf = io.BytesIO(wav_bytes)
                buf.name = "audio.wav"
                client = Groq(api_key=groq_key, max_retries=0, timeout=12.0)
                kwargs = {"file": buf, "model": _groq_stt_model}
                if _language:
                    kwargs["language"] = _language
                result = client.audio.transcriptions.create(**kwargs)
                return (
                    getattr(result, "text", "")
                    or (result.get("text", "") if isinstance(result, dict) else "")
                ).strip()
            except (ImportError, AttributeError, TypeError):
                import urllib.request, json as _json
                boundary = "----OpenAssistBoundary"
                body_parts = [
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{_groq_stt_model}\r\n".encode()
                ]
                if _language:
                    body_parts.append(
                        f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{_language}\r\n".encode()
                    )
                body_parts.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
                    + wav_bytes + b"\r\n"
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
                with urllib.request.urlopen(req, timeout=12) as resp:
                    return _json.loads(resp.read().decode()).get("text", "").strip()

        try:
            wav_bytes, n_samples = _build_wav(buffer)

            if not is_final:
                # ── Non-final chunk: submit concurrently, store future ───────
                if not hasattr(self, "_groq_chunk_futures"):
                    self._groq_chunk_futures = []
                future = self._cloud_stt_pool.submit(_call_groq, wav_bytes)
                self._groq_chunk_futures.append(
                    {
                        "future": future,
                        "buffer": list(buffer),
                        "speech_started_at": speech_started_at,
                    }
                )
                logger.debug(
                    "[Phase2 Groq] Non-final chunk submitted concurrently (%d pending)",
                    len(self._groq_chunk_futures),
                )
                return

            # ── Final chunk: submit concurrently, join all pending ───────────
            final_future = self._cloud_stt_pool.submit(_call_groq, wav_bytes)
            pending = list(getattr(self, "_groq_chunk_futures", []))
            self._groq_chunk_futures = []  # reset before any await

            chunk_texts = []
            cloud_failed = False

            def _fallback_local_part(entry_buffer, entry_started_at) -> None:
                self._transcribe_local(
                    entry_buffer,
                    speech_started_at=entry_started_at or speech_started_at,
                    is_final=False,
                    vad_meta=vad_meta,
                )

            for entry in pending:
                fut = entry.get("future") if isinstance(entry, dict) else entry
                try:
                    t = fut.result(timeout=13.0)
                    if t and not _is_hall(t):
                        chunk_texts.append(t)
                        self._append_session_transcript_part(t, provider="groq")
                    else:
                        _fallback_local_part(
                            entry.get("buffer", []) if isinstance(entry, dict) else [],
                            entry.get("speech_started_at") if isinstance(entry, dict) else speech_started_at,
                        )
                except Exception as exc:
                    cloud_failed = True
                    logger.warning("[Groq STT] Pending chunk failed (%r); using local fallback for this chunk", exc)
                    _fallback_local_part(
                        entry.get("buffer", []) if isinstance(entry, dict) else [],
                        entry.get("speech_started_at") if isinstance(entry, dict) else speech_started_at,
                    )

            try:
                final_text = final_future.result(timeout=13.0)
            except Exception as exc:
                cloud_failed = True
                logger.warning("[Groq STT] Final chunk failed (%r); using local fallback for this chunk", exc)
                final_text = ""

            if final_text and not _is_hall(final_text):
                chunk_texts.append(final_text)
                self._append_session_transcript_part(final_text, provider="groq")

            if not (final_text and not _is_hall(final_text)):
                _fallback_local_part(buffer, speech_started_at)

            if cloud_failed:
                self._cloud_stt_session_blocked = True
                logger.warning(
                    "[Groq STT] Disabled for this session after chunk failure; future utterances will use local Whisper"
                )

            transcribe_finished_at = time.time()
            audio_duration_ms = (n_samples / float(max(self.sr, 1))) * 1000.0
            logger.info(
                "[Groq STT] %d chunk(s) transcribed in %.0fms | model=%s",
                len(chunk_texts),
                (transcribe_finished_at - transcribe_started_at) * 1000.0,
                _groq_stt_model,
            )
            self._emit_accumulated(
                speech_started_at, provider="groq",
                transcribe_started_at=transcribe_started_at,
                vad_meta=vad_meta,
            )

        except Exception as e:
            status = getattr(e, "code", None)
            detail = ""
            if hasattr(e, "read"):
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    detail = ""
            err_str = str(e)
            # Detect auth failures (403/401) and block Groq for the rest of the
            # session to avoid wasting time on every utterance with a dead key.
            if status in {401, 403} or "403" in err_str or "401" in err_str or "Forbidden" in err_str or "Unauthorized" in err_str:
                if not getattr(self, "_cloud_stt_session_blocked", False):
                    logger.warning(
                        "[Groq STT] Auth error (%s)%s - Groq STT disabled for this session; all future utterances will use local Whisper.",
                        e,
                        f" | response={detail}" if detail else "",
                    )
                    self._cloud_stt_session_blocked = True
                    self._cloud_stt_failed_key = groq_key
            else:
                logger.warning(
                    "[Groq STT] API error (%s)%s - Groq STT disabled for this session; falling back to local Whisper",
                    e,
                    f" | response={detail}" if detail else "",
                )
                self._cloud_stt_session_blocked = True
            self._groq_chunk_futures = []
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


    def _transcribe_interim(
        self,
        buffer,
        speech_started_at: float,
        epoch: int,
        utterance_id: str = "",
        capture_generation: int | None = None,
    ) -> None:
        """Best-effort interim transcription while the user is still speaking."""
        if capture_generation is not None and capture_generation != self._capture_generation:
            return
        if epoch != self._interim_epoch:
            return
        if not self._can_run_interim_with_pending_finals():
            return
        if self._effective_transcription_provider(is_final=True) != "local":
            return
        if (
            str(getattr(self, "_transcription_provider", "") or "").lower() == "groq"
            and bool(getattr(self, "_cloud_stt_session_blocked", False))
        ):
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
            acquired = self._infer_lock.acquire(blocking=False)
            if not acquired:
                logger.debug("Interim ASR skipped because Whisper decode is busy")
                return
            try:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self._language,
                    beam_size=self._interim_beam_size,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    initial_prompt=self._whisper_initial_prompt or None,
                    no_speech_threshold=0.7,
                    suppress_blank=False,
                )
            finally:
                self._infer_lock.release()
            if epoch != self._interim_epoch:
                return
            text = self._filter_segments(segments)
            cleaned = (text or "").strip()
            if cleaned and epoch == self._interim_epoch:
                elapsed_since_speech = max(0.0, time.time() - speech_started_at) if speech_started_at else 0.0
                logger.info(
                    f"📝 INTERIM TRANSCRIPTION RESULT | elapsed={elapsed_since_speech:.2f}s | text_len={len(cleaned)} | text='{cleaned[:50]}'"
                )
                self.interim_transcription_ready.emit(cleaned)
        except Exception as e:
            logger.debug(f"Whisper interim transcription error (non-fatal): {e}")

    def get_transcript(self):
        return " ".join(self.transcripts)

    def get_last_transcription_metrics(self):
        return dict(self._last_transcription_metrics)

    def clear(self):
        self.transcripts.clear()
        with self._session_parts_lock:
            self._session_transcript_parts = []
            self._session_transcript_part_providers = []
        with self._prompt_context_lock:
            self._recent_transcripts = []
        self._whisper_initial_prompt = self._whisper_base_prompt
        self._last_transcription_metrics = {}
        self._interim_epoch += 1

    def _max_utterance_exceeded(self, utterance_started_at) -> bool:
        if not utterance_started_at:
            return False
        return max(0.0, time.time() - utterance_started_at) >= max(2.0, self._max_utterance_s)
