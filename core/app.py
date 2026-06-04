"""
OpenAssist AI v1.0.0 — Main Application Controller (Midnight Hardened).
RESTORED: Smooth Gliding (60fps / 5px steps), HUD navigation, and click-through.
FIXED: Standby warmup signal bridge and non-blocking hardware hot-apply.
RESTORATION: Automatic Knowledge Sync (RAG) during warmup.
"""

import sys
import asyncio
import re
import threading
import time
import uuid
import shutil
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, QProcess

from core.config import Config
from core.state import AppState
from core.hotkeys import HotkeyManager
from capture.screen import ScreenCapture
from capture.audio import AudioCapture
from capture.ocr import OCREngine
from ai.engine import AIEngine
from ai.history import ResponseHistory
from ai.rag import RAGEngine
from core.nexus import ContextNexus
from modes import ModeManager
from utils.platform_utils import ProcessUtils, WindowUtils
from ui.overlay import OverlayWindow
from ui.mini_overlay import MiniOverlay
from stealth.anti_detect import StealthManager
from stealth.input_simulator import InputSimulator
from utils.logger import setup_logger
from utils.context_store import get_store as get_context_store
from utils.context_store import get_suggested_preset_for_mode
from core.constants import DB_DIR, CACHE_DIR, LOG_DIR
from ai.memory import LongTermMemory
from ai.context import ContextBuilder, ContextPruner
from ai.prefetch import PredictivePrefetcher
from ai.actions import ActionExecutor
from ai.auto_answer_controller import (
    handle_auto_final_transcription,
    handle_auto_interim_transcription,
)
from ai.auto_mode_runtime import (
    auto_mode_requested,
    init_auto_mode_state,
    reset_auto_mode_turn_state,
    start_auto_mode,
    toggle_auto_mode as toggle_auto_mode_runtime,
)

logger = setup_logger(__name__)


class OpenAssistApp(QObject):
    warmup_status_update = pyqtSignal(str, int, bool)
    run_on_ui_thread = pyqtSignal(object)

    def __init__(self, config: Config, mini_mode: bool = False):
        super().__init__()
        self.config = config
        self.state = AppState(config)
        self.mini_mode = mini_mode
        self.state.is_mini = mini_mode
        self.is_running = True
        self.qt_app = QApplication.instance() or QApplication(sys.argv)

        # Components
        self.history = ResponseHistory()
        self.rag = RAGEngine(config)
        # ModeManager is the single source of truth for the active mode profile
        self.modes = ModeManager(config)
        self.ai = AIEngine(config, self.history, self.rag, mode_manager=self.modes)
        self.ocr = OCREngine(config)
        self.screen = ScreenCapture(config, self.ocr)
        self.audio = AudioCapture(config, state=self.state)
        self.nexus = ContextNexus(config)
        self.stealth = StealthManager(config)
        self.stealth.apply_app_stealth()
        self.simulator = InputSimulator(config)

        # P2.4: Long-term semantic memory (ChromaDB, local, offline)
        self.memory = LongTermMemory(config)
        # P2.1: Incremental context assembly (diff-based screen context)
        self.context_builder = ContextBuilder(config)
        # P3.4: Attention-based context pruner
        self.context_pruner = ContextPruner(config)
        # P3.1: Predictive prefetch engine (instantiate; wired to loop in _run_master_loop)
        self.prefetcher = PredictivePrefetcher(
            config,
            prefetch_fn=self.ai.prefetch_rag,
        )
        # P3.3: Actionable queries executor
        self.actions = ActionExecutor(config)
        self.session_active = False
        self._last_query = ""
        self._current_session_id = None
        self._session_start_time = None
        self._current_response_start_time = None
        self._last_query_time = 0.0
        self._click_through = False
        # M36 FIX: Use itertools.count for atomic increment across threads.
        import itertools
        self._generation_epoch_counter = itertools.count(1)
        self._generation_epoch = 0
        self._screen_analysis_pending = False
        self._pending_request_metadata = None
        self._screen_share_active = False
        self._screen_share_hidden_window = None
        self._pending_incomplete_audio_query = ""
        self._pending_incomplete_audio_at = 0.0
        # Tracks whether the current session_context was auto-suggested by a mode
        # switch (True) or typed/loaded manually (False). Auto-suggested context
        # can be silently replaced when the mode changes; manual context cannot.
        self._context_auto_suggested: bool = False
        self._history_navigation_active = False
        init_auto_mode_state(self)

        # Async Loop
        self.loop = asyncio.new_event_loop()
        self._ai_lock_ready = threading.Event()
        self._async_thread = threading.Thread(target=self._run_master_loop, daemon=True)

        # UI
        self.overlay = OverlayWindow(config, self)
        self.mini_overlay = MiniOverlay(config, self)
        self.hotkeys = HotkeyManager(config, self)
        self.tray = None

        self._wire_signals()
        self.qt_app.aboutToQuit.connect(self.shutdown)

        # Session Context: load the last-used context from disk so it's
        # available immediately when the user opens Settings and starts a session.
        self._context_store = get_context_store()
        last_ctx = self._context_store.get_last_context()
        if last_ctx:
            self.state.session_context = last_ctx
            self.ai.set_session_context(last_ctx)
            logger.debug(f"Session context restored ({len(last_ctx)} chars)")

        # Propagate future context changes from state → AI engine
        self.state.session_context_changed.connect(self.ai.set_session_context)

        # MOVEMENT ENGINE: RESTORED Smooth Glide (60fps)
        self._move_timer = QTimer(self)
        self._move_direction = None
        self._move_timer.timeout.connect(self._do_move)

        # NEXUS ENGINE: Periodic Window Polling (3s)
        self._nexus_timer = QTimer(self)
        self._nexus_timer.timeout.connect(self._poll_nexus_context)
        self._nexus_timer.start(3000)

        # Keep the active HUD window pinned above other apps even after focus churn.
        # P2.10: Timer is started only when a session is ACTIVE to reduce unnecessary Win32 calls.
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self._refresh_topmost_window)
        # Do NOT start here — started in start_session(), stopped in end_session()

        self._screen_share_timer = QTimer(self)
        self._screen_share_timer.timeout.connect(self._check_screen_share_protection)
        self._screen_share_timer.start(5000)

    def _run_master_loop(self):
        asyncio.set_event_loop(self.loop)
        self._ai_lock = asyncio.Lock()
        self._ai_lock_ready.set()
        # P3.1: Wire the prefetcher to the running loop now that it exists
        if hasattr(self, "prefetcher"):
            self.prefetcher.wire(self.loop)
            logger.info("[P3.1 Prefetcher] Wired to async loop")
        self.loop.run_forever()

    def _wire_signals(self):
        self.overlay.user_query.connect(self.generate_response)
        self.mini_overlay.user_query.connect(self.generate_response)
        self.ai.response_chunk.connect(self.overlay.append_response)
        self.ai.response_chunk.connect(self.mini_overlay.append_response)
        if hasattr(self.ai, "background_complete"):
            self.ai.background_complete.connect(self.overlay._on_bg_complete)
            self.ai.background_complete.connect(self.mini_overlay._on_bg_complete)
        self.ai.error_occurred.connect(self.mini_overlay.show_error)
        self.audio.transcription_ready.connect(self._on_transcription)
        if hasattr(self.audio, "interim_transcription_ready"):
            self.audio.interim_transcription_ready.connect(self._on_interim_transcription)
        self.screen.text_captured.connect(self._on_screen_text)
        self.warmup_status_update.connect(self.overlay.update_warmup_status)
        self.warmup_status_update.connect(self.mini_overlay.update_warmup_status)
        self.ai.provider_status.connect(self.overlay.standby_view.set_provider_statuses)
        self.overlay.standby_view.start_clicked.connect(self.start_new_session)
        self.overlay.standby_view.mode_selected.connect(self.switch_mode)
        self.overlay.standby_view.audio_source_changed.connect(
            self._on_audio_source_ui_change
        )
        self.ai.response_complete.connect(self._on_response_complete)
        self.ai.error_occurred.connect(self._on_ai_error)
        self.state.stealth_changed.connect(lambda _: self._apply_ui_only())
        self.run_on_ui_thread.connect(lambda callback: callback())

    def _call_on_ui_thread(self, callback) -> None:
        """Execute a callable on Qt's UI thread."""
        try:
            self.run_on_ui_thread.emit(callback)
        except Exception:
            logger.debug("Failed to schedule UI-thread callback", exc_info=True)

    def _on_response_complete(self, full_text: str):
        """Mark when response generation completes and is ready to display."""
        session_id = getattr(self, "_current_session_id", "unknown_session")
        elapsed_ms = (time.time() - getattr(self, "_session_start_time", time.time())) * 1000
        response_duration = (time.time() - (getattr(self, "_current_response_start_time", None) or time.time())) * 1000

        logger.info(
            f"[{session_id}] ✅ RESPONSE COMPLETE | total_elapsed={elapsed_ms:.1f}ms | response_generation={response_duration:.1f}ms | text_len={len(full_text)}"
        )
        """Update overlay status bar with latency after each response.
        GAP 5: also passes cache_tier + provider to on_complete for the source badge.
        """
        # Ensure we don't accidentally demote a completed task to the background
        if hasattr(self.ai, "_current_gen_kwargs"):
            self.ai._current_gen_kwargs = None

        entries = self.history.get_last(1)
        latency_ms = 0
        provider = None
        metadata = {}
        stage_timings = {}
        req_meta = {}
        providers_tried = []
        race = False
        had_screen = False
        had_audio = False
        had_rag = False
        had_memory = False
        had_action = False
        had_prefetch = False
        cache_tier = 0  # Q14: which cache tier served this response (0=miss/LLM)
        if entries:
            entry = entries[-1]
            latency_ms = getattr(entry, "latency", 0)
            provider = getattr(entry, "provider", None)
            metadata = getattr(entry, "metadata", {}) or {}
            stage_timings = metadata.get("stage_timings", {})
            req_meta = metadata.get("request_metadata", {})
            providers_tried = metadata.get("providers_tried", []) or []
            race = bool(metadata.get("race", False))
            had_screen = bool(metadata.get("had_screen", False))
            had_audio = bool(metadata.get("had_audio", False))
            had_rag = bool(metadata.get("had_rag", False))
            # Q1/Q2/Q3 — P2/P3 feature chip flags from request_metadata
            had_memory = bool(req_meta.get("long_term_memory", ""))
            had_action = bool(req_meta.get("action_output", ""))
            had_prefetch = bool(req_meta.get("prefetch_hit", False))
            cache_tier = int(metadata.get("cache_tier", 0))  # Q14
            if cache_tier:
                logger.info("[Q14 Chip] Response served from cache tier=%d", cache_tier)
            if had_memory:
                logger.info("[Q1 Chip] Long-term memory was injected into this response")
            if had_action:
                logger.info("[Q2 Chip] P3.3 action output was injected into this response")
            if had_prefetch:
                logger.info("[Q3 Chip] P3.1 prefetch hit used for this response")
            if stage_timings:
                summary_parts = []
                speech_to_transcript = req_meta.get("speech_to_transcript_ms")
                if speech_to_transcript is not None:
                    summary_parts.append(
                        f"speech->transcript={speech_to_transcript:.0f}ms"
                    )
                request_to_first = stage_timings.get("request_to_first_token_ms")
                if request_to_first is not None:
                    summary_parts.append(
                        f"request->first_token={request_to_first:.0f}ms"
                    )
                request_to_complete = stage_timings.get("request_to_complete_ms")
                if request_to_complete is not None:
                    summary_parts.append(
                        f"request->complete={request_to_complete:.0f}ms"
                    )
                if summary_parts:
                    logger.info("Latency summary | %s", " | ".join(summary_parts))
            self._log_turn_waterfall_summary(
                provider=provider,
                request_metadata=req_meta,
                stage_timings=stage_timings,
            )

        # Get available providers
        available = []
        if hasattr(self.ai, "_providers"):
            available = [
                p
                for p, prov in self.ai._providers.items()
                if getattr(prov, "is_available", lambda: getattr(prov, "enabled", False))()
            ]

        capture_active = bool(getattr(self.state, "is_capturing", False))
        audio_enabled = bool(self.config.get("capture.audio.enabled", True))
        screen_enabled = bool(self.config.get("capture.screen.enabled", True))

        self.overlay.update_status(
            provider=provider,
            capture_audio=capture_active and audio_enabled,
            capture_screen=capture_active and screen_enabled,
            latency_ms=latency_ms if entries else 0,
            available_providers=available,
            stage_timings=stage_timings,
            request_metadata=req_meta,
            providers_tried=providers_tried,
            race=race,
            had_screen=had_screen,
            had_audio=had_audio,
            had_rag=had_rag,
            had_memory=had_memory,
            had_action=had_action,
            had_prefetch=had_prefetch,
            cache_tier=cache_tier,
        )

        # GAP 5: call on_complete with badge metadata so the source badge renders
        # in the response area header (cache tier or live provider name).
        sanitize_response = getattr(
            self,
            "_sanitize_standard_speech_meta_response",
            None,
        )
        if sanitize_response is None:
            sanitize_response = lambda text, meta: OpenAssistApp._sanitize_standard_speech_meta_response(self, text, meta)
        rendered_text = sanitize_response(
            full_text,
            req_meta,
        )
        if rendered_text != full_text:
            try:
                updater = getattr(self.history, "update_last_response", None)
                if callable(updater) and updater(rendered_text):
                    pass
                elif getattr(self.history, "entries", None):
                    last = self.history.entries[-1]
                    if isinstance(last, dict):
                        last["response"] = rendered_text
                    elif hasattr(last, "response"):
                        last.response = rendered_text
            except Exception:
                pass

        self.overlay.on_complete(
            rendered_text,
            query=self._last_query,
            cache_tier=cache_tier,
            provider=provider or "",
        )
        self.mini_overlay.on_complete(
            rendered_text,
            query=self._last_query,
            cache_tier=cache_tier,
            provider=provider or "",
        )

        # ── Reset transcript label to Listening/Ready ─────────────────────────────
        # Clear the "⏳ Processing..." label that was set when query was submitted.
        if self._screen_analysis_pending:
            via = provider.upper() if provider else "VISION"
            self.overlay.update_transcript(f"Screen captured and analyzed via {via}.")
            if hasattr(self.overlay, "set_analysis_provider_badge"):
                self.overlay.set_analysis_provider_badge(provider=provider)
            self._screen_analysis_pending = False
        else:
            # Reset to listening/ready based on session state
            if getattr(self, "session_active", False):
                latency_str = f" ({latency_ms:.0f}ms)" if latency_ms else ""
                ttfb = stage_timings.get("request_to_first_token_ms") if stage_timings else None
                if ttfb is not None:
                    latency_str = f"{latency_str} ttfb={ttfb:.0f}ms"
                if race:
                    latency_str = f"{latency_str} race"
                self.overlay.update_transcript(
                    f"🌐 Listening{latency_str}...",
                    state="listening",
                )
            else:
                self.overlay.update_transcript("Ready...", state="idle")
        self._reset_turn_local_state("response-complete")

        # P1.2: Sync history UI after response completes to update navigation state
        self._sync_history_ui()

    def _on_ai_error(self, error_text: str):
        if hasattr(self.ai, "_current_gen_kwargs"):
            self.ai._current_gen_kwargs = None
        self._reset_turn_local_state("ai-error")

        if self._screen_analysis_pending:
            self.overlay.update_transcript("Screen captured, but analysis failed.")
            if hasattr(self.overlay, "set_analysis_provider_badge"):
                self.overlay.set_analysis_provider_badge()
            self._screen_analysis_pending = False
        else:
            # Reset transcript label on error
            self.overlay.update_transcript(
                "🟡 Error — Listening..." if getattr(self, "session_active", False) else "Ready...",
                state="error" if getattr(self, "session_active", False) else "idle",
            )

        # P1.10: Surface the error as a visible toast so the user knows what failed
        if hasattr(self.overlay, "show_error_toast"):
            # Trim long stack traces to a readable one-liner
            short = error_text.split("\n")[0][:120]
            self.overlay.show_error_toast(short)
        
        self.overlay.on_complete(f"ERROR: {error_text}")

    def _reset_turn_local_state(self, reason: str = "") -> None:
        """Reset per-turn query/audio state so later utterances start clean."""
        self._last_query = ""
        self._last_query_time = 0.0
        self._pending_request_metadata = None
        detector = getattr(getattr(self, "ai", None), "detector", None)
        if detector and hasattr(detector, "reset_turn_state"):
            detector.reset_turn_state(reason or "turn-reset")
        elif detector and hasattr(detector, "reset_fragment_buffer"):
            detector.reset_fragment_buffer(reason or "turn-reset")

    def _make_turn_id(self, origin: str, request_epoch: int) -> str:
        """Create a unique turn ownership token for the current request."""
        return f"{origin}:{request_epoch}:{uuid.uuid4().hex}"

    def _sanitize_standard_speech_meta_response(self, text: str, request_metadata=None) -> str:
        cleaned = (text or "").strip()
        metadata = request_metadata or {}
        if metadata.get("origin") != "speech":
            return cleaned
        lowered = cleaned.lower()
        prefix = "my approach is to explain it clearly."
        if lowered.startswith(prefix):
            return cleaned[len(prefix):].strip()
        return cleaned

    def _log_turn_waterfall_summary(self, provider=None, request_metadata=None, stage_timings=None) -> None:
        req = request_metadata or {}
        stages = stage_timings or {}
        speech_ms = float(req.get("audio_duration_ms", 0.0) or 0.0)
        asr_ms = float(req.get("transcribe_only_ms", 0.0) or 0.0)
        ttfb_ms = float(stages.get("request_to_first_token_ms", 0.0) or 0.0)
        complete_ms = float(stages.get("request_to_complete_ms", 0.0) or 0.0)
        if not any((speech_ms, asr_ms, ttfb_ms, complete_ms)):
            return
        stream_ms = max(0.0, complete_ms - ttfb_ms)
        total_ms = speech_ms + asr_ms + complete_ms
        logger.info(
            "WATERFALL SUMMARY | speech=%dms | asr=%dms | llm_ttfb=%dms | stream=%dms | total=%dms | utterance=%s | vad=%s | chunks=%s | provider=%s",
            round(speech_ms),
            round(asr_ms),
            round(ttfb_ms),
            round(stream_ms),
            round(total_ms),
            req.get("utterance_id", ""),
            req.get("vad_backend", ""),
            req.get("chunks", ""),
            provider or "",
        )


    def _carry_forward_incomplete_audio_query(self, text: str) -> bool:
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            self._pending_incomplete_audio_query = ""
            self._pending_incomplete_audio_at = 0.0
            return False
        lowered = cleaned.lower()
        if lowered.endswith((",", ":", "-")):
            for prefix in ("what is ", "what are ", "how do ", "how does ", "why is ", "why are ", "can you ", "could you "):
                if lowered.startswith(prefix):
                    remainder = cleaned[len(prefix):].strip(" ,:-?")
                    if remainder:
                        self._pending_incomplete_audio_query = remainder
                        self._pending_incomplete_audio_at = time.time()
                        return True
        return False

    def _combine_with_pending_audio_followup(self, text: str) -> str:
        pending = (getattr(self, "_pending_incomplete_audio_query", "") or "").strip(" ,:-?")
        pending_at = float(getattr(self, "_pending_incomplete_audio_at", 0.0) or 0.0)
        cleaned = " ".join((text or "").split()).strip()
        if not pending or not cleaned:
            return cleaned
        if pending_at > 0.0 and (time.time() - pending_at) > 6.0:
            return cleaned
        lowered = cleaned.lower().rstrip()
        generic_prefixes = ("can you explain", "could you explain", "can you walk me through", "could you walk me through")
        if any(lowered.startswith(prefix) for prefix in generic_prefixes) and lowered.endswith("?"):
            base = cleaned.rstrip("?").strip()
            merged = f"{base} {pending}?".strip()
            self._pending_incomplete_audio_query = ""
            self._pending_incomplete_audio_at = 0.0
            return merged
        return cleaned

    def _auto_mode_requested(self) -> bool:
        return auto_mode_requested(self)

    def _start_auto_mode(self) -> None:
        start_auto_mode(self)

    def toggle_auto_mode(self) -> None:
        toggle_auto_mode_runtime(self)

    @staticmethod
    def _looks_like_acknowledgement(text: str) -> bool:
        cleaned = " ".join((text or "").lower().split()).strip(" .,!?:;-")
        return cleaned in {
            "ok",
            "okay",
            "sure",
            "right",
            "got it",
            "i see",
            "understood",
            "thanks",
            "thank you",
        }

    def _on_audio_source_ui_change(self, source):
        self.state.audio_source = source
        if hasattr(self.config, "save"):
            try:
                self.config.save()
            except Exception as e:
                logger.debug(f"Audio source config save skipped: {e}")
        self.overlay.refresh_standby_state(audio=source)
        if hasattr(self.audio, "restart"):
            self.audio.restart()

    def _sync_state_from_config(self):
        """Pull persisted config back into AppState before async subsystems catch up."""
        self.state.mode = self.config.get("ai.mode", "general")
        self.state.audio_source = self.config.get("capture.audio.mode", "system")
        self.state.is_muted = bool(self.config.get("capture.audio.muted", False))
        self.state.is_stealth = self.config.get("stealth.enabled", True)
        screen_enabled = bool(self.config.get("capture.screen.enabled", True))
        if hasattr(self, "screen") and self.screen and hasattr(self.screen, "set_enabled"):
            self.screen.set_enabled(screen_enabled)
        if hasattr(self, "overlay") and self.overlay and hasattr(self.overlay, "update_vision_state"):
            self.overlay.update_vision_state(reset_to_master=False)

    @staticmethod
    def _looks_like_explicit_screen_request(text: str) -> bool:
        """True when a typed/manual prompt explicitly asks to inspect the screen."""
        q = " ".join((text or "").lower().split())
        if not q:
            return False
        screen_terms = (
            "screen",
            "screenshot",
            "current code",
            "code on the screen",
            "visible code",
            "this code",
            "attached screenshot",
            "shown in the screenshot",
            "present on the screen",
        )
        action_terms = (
            "analyze",
            "analyse",
            "read",
            "look",
            "inspect",
            "explain",
            "output",
            "run",
            "result",
            "what does",
            "what is",
            "give me",
        )
        return any(term in q for term in screen_terms) and any(term in q for term in action_terms)

    def _poll_nexus_context(self):
        """Polls environmental signals for the ContextNexus."""
        if hasattr(self, "screen") and self.screen and not getattr(self.screen, "_enabled", True):
            return
        title = ProcessUtils.get_active_window_title()
        if title:
            # Avoid self-referential capture if we are the active window
            if "OpenAssist" not in title:
                self.nexus.push("window", title)

    # --- 🛠️ FLUID HUD NAVIGATION ---

    def move_up(self):
        self._nudge(0, -5)

    def move_down(self):
        self._nudge(0, 5)

    def move_left(self):
        self._nudge(-5, 0)

    def move_right(self):
        self._nudge(5, 0)

    def _nudge(self, dx, dy):
        v = self.mini_overlay if self.mini_mode else self.overlay
        pos = v.pos()
        v.move(pos.x() + dx, pos.y() + dy)

    def start_move(self, direction):
        if self._move_direction == direction:
            return
        self._move_direction = direction
        self._move_timer.start(16)

    def stop_move(self):
        self._move_timer.stop()
        self._move_direction = None

    def _do_move(self):
        if not self._move_direction:
            return
        d = self._move_direction
        if d == "up":
            self.move_up()
        elif d == "down":
            self.move_down()
        elif d == "left":
            self.move_left()
        elif d == "right":
            self.move_right()

    def toggle_click_through(self):
        view = self._active_view()
        was_visible = view.isVisible() if hasattr(view, "isVisible") else False
        self._click_through = not self._click_through
        self.overlay.set_click_through(self._click_through)
        self.mini_overlay.set_click_through(self._click_through)
        if was_visible and hasattr(view, "show"):
            view.show()
        self._refresh_window_invariants()
        if hasattr(self.overlay, "update_transcript"):
            self.overlay.update_transcript(
                "Click-through enabled. Press Ctrl+M to restore interaction."
                if self._click_through
                else "Click-through disabled."
            )

    # --- 🛠️ CONTROL BRIDGES ---

    def _apply_settings(self):
        """Non-blocking hot-apply for Layer 6 stabilization. Added Master Safety Guard."""
        logger.info("⚙️ Applying Settings (Background Thread)...")

        self._sync_state_from_config()
        # Ensure runtime ModeManager + UI highlights reflect the newly saved mode.
        # Without this, Settings can update config/state but leave the active Mode
        # profile (detector sensitivity, VAD, preferred providers) stale until the
        # user manually switches modes.
        try:
            self.switch_mode(self.state.mode)
        except Exception as e:
            logger.debug(f"Mode apply skipped: {e}")
        self.overlay.refresh_standby_state()

        def _apply():
            try:
                # 1. Hardware Cooldown
                self.audio.restart()
                # M33 FIX: hotkeys interacts with Qt objects; schedule on main thread.
                QTimer.singleShot(0, self.hotkeys.restart)

                # 2. Process Warmup
                # Close old provider network resources (e.g. Ollama aiohttp sessions)
                # before re-initializing providers to avoid "Unclosed client session".
                try:
                    self.ai.close_providers()
                except Exception:
                    pass
                self.ai.warmup(self.loop)

                # 3. UI Synchronization
                QTimer.singleShot(0, self._apply_ui_only)

                # 4. Async Stream Recovery
                if self.loop and self.loop.is_running():
                    self.ai.ensure_health_monitor(self.loop)

                logger.info("✅ Settings Applied Successfully.")
            except Exception as e:
                logger.error(f"❌ Settings Hot-Apply Fault (Handled): {e}")

        # Spawn daemon thread to keep UI interactive
        t = threading.Thread(target=_apply, daemon=True)
        t.start()

    def _apply_ui_only(self):
        self._refresh_window_invariants()

    def _refresh_window_invariants(self, window=None):
        """Reinforce shell/topmost/stealth invariants after lifecycle changes."""
        if window is not None:
            self._apply_window_effects(window)
            return
        self._apply_window_effects(self.overlay)
        self._apply_window_effects(self.mini_overlay)

    def _apply_window_effects(self, window):
        base_opacity = self.config.get("app.opacity", 0.94)
        stealth_opacity = self.config.get("stealth.low_opacity", 0.75)
        is_stealth = bool(getattr(self.state, "is_stealth", False))

        # Don't fight the gaze timer: if gaze fade is enabled and a session
        # is active, the gaze timer owns the opacity — skip overriding it.
        gaze_active = (
            self.config.get("app.gaze_fade.enabled", False)
            and getattr(self, "session_active", False)
        )
        if not gaze_active:
            window.setWindowOpacity(stealth_opacity if is_stealth else base_opacity)

        WindowUtils.hide_from_taskbar(window)
        WindowUtils.ensure_topmost(window)
        self.stealth.apply_to_window(window, is_stealth)

    def _refresh_topmost_window(self):
        for window in (self.overlay, self.mini_overlay):
            try:
                if hasattr(window, "isVisible") and window.isVisible():
                    self._refresh_window_invariants(window)
            except Exception as e:
                logger.debug(f"Topmost refresh skipped: {e}")

    def _check_screen_share_protection(self):
        """Continuously enforce the app's screen-share stealth policy."""
        try:
            sharing = bool(ProcessUtils.is_screen_sharing_active())
        except Exception as e:
            logger.debug("Screen share detection skipped: %s", e)
            return

        if sharing:
            self.ensure_stealth()

        if sharing != self._screen_share_active:
            self._set_screen_share_state(sharing)

    def _set_screen_share_state(self, sharing: bool):
        self._screen_share_active = sharing

        if sharing:
            logger.info("Screen sharing detected - reinforcing stealth protections")
            if getattr(self.stealth, "should_hide_for_screen_share", lambda: False)():
                view = self._active_view()
                if view and hasattr(view, "isVisible") and view.isVisible():
                    self._screen_share_hidden_window = "mini" if self.mini_mode else "overlay"
                    view.hide()
                    logger.info(
                        "Screen sharing active on limited stealth platform - HUD hidden"
                    )
            return

        logger.info("Screen sharing ended - restoring normal stealth posture")
        # Revert stealth to the baseline configuration (defaults to True as an invariant)
        self.state.is_stealth = self.config.get("stealth.enabled", True)
        self._refresh_window_invariants()

        if self._screen_share_hidden_window:
            view = (
                self.mini_overlay
                if self._screen_share_hidden_window == "mini"
                else self.overlay
            )
            self._screen_share_hidden_window = None
            self._present_window(view, focus=False)

    def _active_view(self):
        return self.mini_overlay if self.mini_mode else self.overlay

    def _hud_focus_enabled(self) -> bool:
        return bool(self.config.get("app.focus_on_show", False))

    def _present_window(self, window, focus: bool = False):
        if not window:
            return None

        window.show()
        self._refresh_window_invariants(window)
        if hasattr(window, "raise_"):
            window.raise_()
        if focus and hasattr(window, "activateWindow"):
            window.activateWindow()
        return window

    def _show_active_overlay(self):
        view = self._active_view()
        return self._present_window(view, focus=self._hud_focus_enabled())

    def _restore_standby_surface(self, focus: bool = False):
        """Return to the standby surface for the current HUD mode."""
        self.overlay.show_standby_view()
        self.overlay.refresh_standby_state()

        if self.mini_mode:
            self.overlay.hide()
            return self._present_window(self.mini_overlay, focus=focus)

        return self._present_window(self.overlay, focus=focus)

    def request_close_surface(self, source: str = "overlay"):
        """Handle close requests from the overlay or mini HUD.

        Active sessions are hidden to background only when the user is already on
        the main standby/chat surface. Navigation surfaces always collapse back
        to standby first so the close button behaves like a context-aware dismiss.
        """
        current = getattr(getattr(self.overlay, "stack", None), "currentWidget", lambda: None)()
        standby = getattr(self.overlay, "standby_view", None)
        settings = getattr(self.overlay, "settings_view", None)
        onboarding = getattr(self.overlay, "onboarding_wizard", None)
        history = getattr(self.overlay, "history_feed", None)
        timeline = getattr(self.overlay, "timeline_view", None)
        menu_views = {settings, onboarding, history, timeline}

        if current in menu_views:
            logger.info("Close request from %s -> returning to standby surface", source)
            if getattr(self, "session_active", False):
                if self.mini_mode:
                    self.overlay.hide()
                    return self._present_window(self.mini_overlay, focus=False)
                else:
                    self.overlay.show_chat_view()
                    return self._present_window(self.overlay, focus=False)
            return OpenAssistApp._restore_standby_surface(self, focus=False)

        if getattr(self, "session_active", False):
            active_view = self._active_view()
            if active_view and hasattr(active_view, "hide"):
                logger.info("Close request from %s -> backgrounding active surface", source)
                active_view.hide()
            return "hidden"

        if current is not None and current is not standby:
            logger.info("Close request from %s -> restoring standby surface", source)
            return OpenAssistApp._restore_standby_surface(self, focus=False)

        logger.info("Close request from %s -> quitting application", source)
        if hasattr(self.qt_app, "quit"):
            self.qt_app.quit()
        return "quit"

    def open_settings(self):
        if self.mini_mode:
            self.mini_overlay.hide()
        if hasattr(self.overlay, "stack"):
            self.overlay._prev_stack_index = self.overlay.stack.currentIndex()
            self.overlay._prev_stack_widget = self.overlay.stack.currentWidget()
        show_settings_view = getattr(self.overlay, "show_settings_view", None)
        if callable(show_settings_view):
            show_settings_view()
        else:
            self.overlay.stack.setCurrentWidget(self.overlay.settings_view)
        self._present_window(self.overlay, focus=True)

    def _show_initial_window(self):
        if not self.config.get("onboarding.completed", False):
            self._present_window(self.overlay, focus=True)
            self.overlay.show_onboarding()
            return

        self._show_active_overlay()

    def run(self) -> int:
        self._async_thread.start()
        self.hotkeys.start()

        # Show overlay AFTER Qt event loop starts to ensure window is visible
        QTimer.singleShot(100, self._show_initial_window)

        self._background_warmup()
        return self.qt_app.exec()

    def generate_response(self, q, s="manual", c=None):
        if not q or not self._ai_lock_ready.wait(timeout=2):
            return

        if s == "speech":
            from utils.text_utils import normalize_transcript
            q = normalize_transcript(q) or q

        # P2: Self-Learning Detector Hook
        if hasattr(self.ai, "detector") and hasattr(self.ai.detector, "learn_from_query"):
            self.ai.detector.learn_from_query(q)

        if s == "manual" and c is None and self.session_active and OpenAssistApp._looks_like_explicit_screen_request(q):
            self.analyze_current_screen(query_override=q)
            return

        # Background session-triggered queries should never outlive the session.
        if s in {"speech", "auto"} and not self.session_active:
            return

        # OMEGA DEBOUNCE: Prevent double-triggers (Audio + OCR) for the same query
        now = time.time()
        if (
            s in {"speech", "auto"}
            and q == self._last_query
            and (now - self._last_query_time) < 3.0
        ):
            logger.debug(f"AI: Debouncing duplicate query: {q[:50]}...")
            return

        self._last_query = q
        self._last_query_time = now

        # P1.4: New query should gracefully demote any in-flight generation to background.
        if s in {"manual", "speech", "quick"}:
            if bool(self.config.get("ai.background_generation.enabled", True)) and hasattr(self.ai, "demote_to_background"):
                try:
                    self.ai.demote_to_background()
                except Exception:
                    pass
            elif hasattr(self.ai, "cancel"):
                try:
                    self.ai.cancel()
                except Exception:
                    pass
        # SNAP-LOCK: Capture target window HWND at moment of query
        if s in ["manual", "speech", "quick"]:
            hwnd = self.simulator.get_foreground_window()
            try:
                title = ProcessUtils.get_active_window_title()
                if hwnd and title and "OpenAssist" in title and hasattr(ProcessUtils, "get_window_behind"):
                    hwnd = ProcessUtils.get_window_behind(hwnd) or hwnd
            except Exception:
                pass
            if hwnd:
                self.state.target_window_id = hwnd
                logger.info(f"🔒 Snap-Lock: Target set to HWND {hwnd}")

        # ── Immediate UI feedback ──────────────────────────────────────────────
        # For every query type we do two things instantly (before the async task):
        #   1. Update the transcript bar (small pill at the bottom)
        #   2. For typed / speech / quick queries: show the query text + "Thinking..."
        #      in the full response area so the user always sees what was captured,
        #      even while refinement is running in the background.
        if s in {"manual", "speech", "quick"}:
            label = f"⏳ Processing: {q[:55]}..." if len(q) > 55 else "⏳ Processing..."
            self.overlay.update_transcript(label, state="processing")

            # Show query immediately in response area — do not wait for first token.
            # GAP 2: We set a _pending_query flag instead of writing "Thinking..."
            # immediately.  The first response_chunk (or cache hit) will clear this
            # flag and paint the real content.  If no chunk arrives within 80ms we
            # fall back to showing "Thinking..." via a deferred QTimer so the user
            # never sees a blank screen on slow providers.
            self.overlay.show_chat_view()
            self.overlay._current_query = q
            self.overlay._pending_thinking = True   # signal: waiting for first chunk

            race_hint = ""
            if s in {"manual", "speech"} and bool(self.config.get("ai.text.race_enabled", False)):
                race_hint = " (race mode)"

            # Show QUERY header immediately — this is always useful feedback
            self.overlay.response_area.setHtml(
                f"<div style='color:#64748b;font-size:10px;margin-bottom:5px;'>"
                f"<b>QUERY:</b> {q}</div>"
            )

            # After 80ms: if we still haven't received a chunk, paint "Thinking..."
            # (cache hits return in <20ms so they'll have replaced this already)
            from PyQt6.QtCore import QTimer as _QTimer
            def _show_thinking():
                if getattr(self.overlay, "_pending_thinking", False):
                    from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor
                    cursor = self.overlay.response_area.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    fmt = QTextCharFormat()
                    fmt.setForeground(QColor("#f59e0b"))
                    cursor.setCharFormat(fmt)
                    cursor.insertText(f"⏳ Thinking{race_hint}...")
                    self.overlay.response_area.setTextCursor(cursor)
            _QTimer.singleShot(80, _show_thinking)


        # Start timing for instrumentation
        self._current_request_start = time.time()
        self._current_response_start_time = self._current_request_start  # always stamp so _on_response_complete never sees None
        self._stage_timings = {"start": self._current_request_start}

        request_metadata = dict(self._pending_request_metadata or {})
        request_metadata.setdefault("request_started_at", self._current_request_start)
        request_metadata.setdefault("origin", s)
        request_epoch = self._generation_epoch
        turn_id_factory = getattr(self, "_make_turn_id", None)
        if callable(turn_id_factory):
            turn_id = request_metadata.get("turn_id") or turn_id_factory(s, request_epoch)
        else:
            turn_id = request_metadata.get("turn_id") or f"{s}:{request_epoch}:{uuid.uuid4().hex}"
        request_metadata["turn_id"] = turn_id
        if isinstance(c, dict) and c.get("auto_answer"):
            request_metadata["auto_mode"] = True
            request_metadata["auto_answer"] = True
            if c.get("auto_speculative"):
                request_metadata["auto_speculative"] = True
        # P2: Attach detector hints for routing/logging even in General mode.
        try:
            det = getattr(getattr(self, "ai", None), "detector", None)
            if det and hasattr(det, "detect_language_hint"):
                lang = det.detect_language_hint(q)
                request_metadata.setdefault("detected_language", lang)
                request_metadata.setdefault("is_code", bool(lang))
        except Exception:
            pass
        self._pending_request_metadata = None
        if hasattr(self.ai, "set_foreground_turn_id"):
            try:
                self.ai.set_foreground_turn_id(turn_id)
            except Exception:
                pass

        asyncio.run_coroutine_threadsafe(
            self._process_ai(q, s, c, request_epoch, request_metadata), self.loop
        )

    async def _process_ai(self, q, s, c, request_epoch, request_metadata):
        if request_epoch != self._generation_epoch:
            return
        async with self._ai_lock:
            if request_epoch != self._generation_epoch:
                return
            request_metadata = dict(request_metadata or {})
            isolate_context = bool(
                request_metadata.get("benchmark_isolated")
                or request_metadata.get("suppress_context")
            )
            self._last_query = q
            # Q10: Parallel screen capture + audio transcript when both need live fetch
            if c:
                sc = c.get("screen")
                au = c.get("audio")
                sc_hash = (c.get("screen_hash") or self.screen.last_img_hash or "")
            else:
                # Vision kill-switch short-circuit: if the eye toggle has disabled vision,
                # skip the entire screen capture await immediately — no screenshot, no OCR,
                # no frame hash.  This saves 10-50ms per query versus letting capture_context()
                # run and return empty.  Audio is always fetched regardless.
                vision_on = getattr(self.screen, "_enabled", True)
                if not vision_on:
                    sc = ""
                    sc_hash = ""
                    au = self.audio.get_transcript()
                    request_metadata.setdefault("suppress_screen_context", True)
                    logger.debug("[Vision OFF] Screen capture skipped — AI using audio-only context")
                else:
                    # Fire both concurrently — capture_context is a coroutine,
                    # get_transcript is sync so wrap in an executor to avoid blocking.
                    _t0_parallel = __import__("time").time()
                    async def _get_audio_async():
                        return self.audio.get_transcript()
                    sc, au = await asyncio.gather(
                        self.screen.capture_context(),
                        _get_audio_async(),
                        return_exceptions=True,
                    )
                    # Unwrap exceptions gracefully
                    if isinstance(sc, Exception):
                        logger.warning("[Q10 Parallel] screen capture failed: %s", sc)
                        sc = ""
                    if isinstance(au, Exception):
                        logger.warning("[Q10 Parallel] audio fetch failed: %s", au)
                        au = ""
                    sc_hash = self.screen.last_img_hash or ""
                    logger.debug(
                        "[Q10 Parallel] Screen + audio fetched concurrently in %.0fms",
                        (__import__("time").time() - _t0_parallel) * 1000,
                    )

            if isolate_context:
                sc = ""
                sc_hash = ""
                preserve_session_context = bool(
                    request_metadata.get("preserve_session_context")
                )
                request_metadata.setdefault("suppress_screen_context", True)
                request_metadata.setdefault("suppress_memory_context", True)
                request_metadata.setdefault("suppress_history_context", True)
                request_metadata.setdefault("suppress_rag_context", True)
                request_metadata.setdefault("suppress_response_cache", True)
                request_metadata.setdefault(
                    "suppress_session_context",
                    not preserve_session_context,
                )

            # P1.3: Log the screen_hash being used for cache fingerprinting
            if sc_hash:
                logger.debug(
                    f"[P1.3 Fingerprint] Using screen_hash={sc_hash[:8]}... "
                    f"for cache fingerprint (stable across minor visual noise)"
                )
            else:
                logger.debug(
                    "[P1.3 Fingerprint] No screen_hash available — "
                    "falling back to OCR text for fingerprint"
                )

            # P2.1: Build incremental context (diff from last screen state)
            boost_context: str = ""  # Q16: entity keywords from prior turn
            if sc and hasattr(self, "context_builder"):
                ctx_result = self.context_builder.build(q, sc)
                sc = ctx_result["screen"]
                boost_context = ctx_result.get("entities", "")
                logger.debug(
                    f"[P2.1 ContextBuilder] mode={ctx_result['mode']}, "
                    f"screen_ctx_len={len(sc)}, "
                    f"entities='{boost_context[:60]}'"
                )
                if boost_context:
                    logger.debug("[Q16 Boost] entity context for cache boosting: '%s'", boost_context[:60])

            if request_epoch != self._generation_epoch:
                return
            if sc:
                self.nexus.push("screen", sc)

            window_id = str(getattr(self.state, "target_window_id", "") or "")
            # au is already set above (Q10 parallel block or c.get("audio"))

            # P3.3: Actionable Queries — detect and execute before hitting the AI
            if hasattr(self, "actions"):
                action_output = await self.actions.detect_and_run(q)
                if action_output:
                    logger.info(
                        f"[P3.3 Actions] Action executed — injecting output "
                        f"({len(action_output)} chars) as context"
                    )
                    request_metadata = dict(request_metadata or {})
                    request_metadata["action_output"] = action_output

            # P3.4: Context Pruning — drop irrelevant screen blocks before the prompt
            if sc and q and hasattr(self, "context_pruner"):
                sc = self.context_pruner.prune(sc, q)

            # P2.4: Retrieve relevant long-term memories and attach to request metadata
            memory_ctx: str = ""
            if not isolate_context and hasattr(self, "memory") and self.memory.is_ready():
                memories = self.memory.query(q, mode=str(self.config.get("ai.mode", "general")))
                if memories:
                    memory_ctx = "\n\n".join(memories)
                    logger.info(
                        f"[P2.4 Memory] Injecting {len(memories)} relevant memories "
                        f"({len(memory_ctx)} chars) into prompt"
                    )
                else:
                    logger.debug("[P2.4 Memory] No relevant past memories found for this query")
            if memory_ctx:
                request_metadata = dict(request_metadata or {})
                request_metadata["long_term_memory"] = memory_ctx

            snapshot = {} if isolate_context else self.nexus.get_snapshot()
            if (
                not isolate_context
                and request_metadata.get("suppress_screen_context")
                and isinstance(snapshot, dict)
            ):
                snapshot = dict(snapshot)
                snapshot["latest_ocr"] = ""
                snapshot["active_window"] = "Unknown" 
            if request_epoch != self._generation_epoch:
                return
            await self.ai.generate_response(
                q,
                snapshot,
                screen_context=sc,
                audio_context=au,
                origin=s,
                request_metadata=request_metadata,
                screen_hash=sc_hash,
                window_id=window_id,
                boost_context=boost_context or None,  # Q16
            )

    def toggle_overlay(self):
        v = self.mini_overlay if self.mini_mode else self.overlay
        if not v.isVisible() or v.windowOpacity() <= 0:
            if self._click_through:
                self.toggle_click_through()
            self._present_window(v, focus=self._hud_focus_enabled())
            logger.debug("👁️ HUD Shown via toggle.")
            return

        v.hide()
        logger.debug("👻 HUD Hidden via toggle.")

    def quick_answer(self):
        # P0.2: Never fire outside an active session — no context, no UI target
        if not self.session_active:
            return
        if not self._ai_lock_ready.wait(timeout=2):
            return

        request_epoch = self._generation_epoch
        turn_id_factory = getattr(self, "_make_turn_id", None)
        if callable(turn_id_factory):
            turn_id = turn_id_factory("quick", request_epoch)
        else:
            turn_id = f"quick:{request_epoch}:{uuid.uuid4().hex}"
        if hasattr(self.ai, "set_foreground_turn_id"):
            try:
                self.ai.set_foreground_turn_id(turn_id)
            except Exception:
                pass
        self.overlay.update_transcript("Preparing quick context answer...")

        async def _quick_answer_flow():
            if request_epoch != self._generation_epoch:
                return

            snapshot = self.nexus.get_snapshot()
            audio_text = (
                snapshot.get("recent_audio")
                or self.audio.get_transcript()
                or snapshot.get("full_audio_history", "")
            )
            screen_text = snapshot.get("latest_ocr", "")
            using_cached_context = bool(audio_text or screen_text)

            if using_cached_context:
                context_parts = []
                if audio_text:
                    context_parts.append("cached audio")
                if screen_text:
                    context_parts.append("cached screen")
                self.overlay.update_transcript(
                    f"Quick answer using {' + '.join(context_parts)}..."
                )

            if not audio_text and not screen_text:
                self.overlay.update_transcript(
                    "Quick answer missing cached context. Refreshing screen once..."
                )
                screen_text = await self.screen.capture_context()
                if screen_text and request_epoch == self._generation_epoch:
                    self.nexus.push("screen", screen_text)
                    snapshot = self.nexus.get_snapshot()

            if request_epoch != self._generation_epoch:
                return

            await self.ai.generate_quick_response(
                snapshot,
                screen_context=screen_text,
                audio_context=audio_text,
                turn_id=turn_id,
            )

        asyncio.run_coroutine_threadsafe(_quick_answer_flow(), self.loop)

    def cancel_generation(self):
        """Abort the current AI stream (best-effort)."""
        if hasattr(self.ai, "cancel"):
            try:
                self.ai.cancel()
            except Exception:
                pass
        self._reset_turn_local_state("cancel-generation")
        if getattr(self, "session_active", False):
            self.overlay.update_transcript("Cancelled — Listening...", state="listening")
        else:
            self.overlay.update_transcript("Cancelled.", state="idle")
        if hasattr(self.mini_overlay, "set_ready"):
            self.mini_overlay.set_ready()

    def paste_as_context(self):
        """Q12: Clipboard-as-context shortcut.

        Reads the current clipboard text and injects it directly as screen context
        for the next AI query.  Zero OCR latency — ideal for interview demos where
        the candidate highlights a LeetCode / coding problem and hits Ctrl+Shift+X.
        """
        try:
            from PyQt6.QtWidgets import QApplication as _QApp
            clipboard = _QApp.clipboard()
            text = (clipboard.text() or "").strip()
        except Exception as exc:
            logger.warning("[Q12 Clipboard] Failed to read clipboard: %s", exc)
            text = ""

        if not text:
            logger.info("[Q12 Clipboard] Clipboard is empty — nothing to inject")
            if hasattr(self.overlay, "show_error_toast"):
                self.overlay.show_error_toast("Clipboard is empty. Copy some text first.")
            return

        # Truncate to a sensible context size
        max_chars = int(self.config.get("context.max_screen_chars", 8000))
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated to context limit ...]"

        # Inject into Nexus as screen context
        self.nexus.push("screen", text)
        logger.info(
            "[Q12 Clipboard] Injected %d chars of clipboard text as screen context",
            len(text),
        )

        # Q13: Record clipboard use
        try:
            from utils.telemetry import telemetry as _tel
            _tel.record_clipboard_use()
        except Exception:
            pass

        # UI feedback
        preview = text[:60].replace("\n", " ")
        if hasattr(self.overlay, "update_transcript"):
            self.overlay.update_transcript(
                f"Clipboard context ready: \"{preview}...\"",
                state="listening",
            )
        if hasattr(self.overlay, "show_error_toast"):
            # Use a green-ish info toast (reuse the toast widget with a positive message)
            pass  # update_transcript is sufficient

    def analyze_current_screen(self, query_override: str = ""):
        if not self.session_active:
            self.start_new_session()
            return

        if not getattr(self, "ai", None) or not getattr(self, "screen", None):
            return

        self.overlay.update_transcript("Screen captured. Analyzing current screen...")
        if hasattr(self.overlay, "set_analysis_provider_badge"):
            self.overlay.set_analysis_provider_badge(pending=True)
        self._screen_analysis_pending = True
        request_epoch = self._generation_epoch
        query_override_text = " ".join((query_override or "").split()).strip()
        logger.info("[AnalyzeScreen] Manual screen analysis requested")

        async def _capture_and_analyze():
            if request_epoch != self._generation_epoch:
                return
            image_bytes = await self.screen.capture_image_bytes(for_analysis=True)
            if request_epoch != self._generation_epoch:
                return
            if not image_bytes:
                self._screen_analysis_pending = False
                OpenAssistApp._call_on_ui_thread(
                    self,
                    lambda: (
                        self.overlay.update_transcript("Screen capture failed."),
                        self.overlay.set_analysis_provider_badge()
                        if hasattr(self.overlay, "set_analysis_provider_badge")
                        else None,
                    ),
                )
                return
            logger.info("[AnalyzeScreen] Captured screenshot bytes=%d", len(image_bytes))

            audio_text = self.audio.get_transcript()
            snapshot = self.nexus.get_snapshot()
            ocr_task = asyncio.create_task(self.screen.extract_text_from_image_bytes(image_bytes))
            fallback_query = query_override_text or (
                "Analyze the current screen and answer using the visible content "
                "and live session context."
            )
            query = fallback_query
            self._last_query = fallback_query

            def _surface_analysis_failure(message: str, detail: str = "") -> None:
                self._screen_analysis_pending = False
                response = message if not detail else f"{message}\n\n{detail}"

                def _ui_failure():
                    if hasattr(self.overlay, "set_analysis_provider_badge"):
                        self.overlay.set_analysis_provider_badge()
                    if hasattr(self.overlay, "show_error_toast"):
                        self.overlay.show_error_toast(message)
                    self.overlay.update_transcript(message, state="error")
                    if hasattr(self.overlay, "on_complete"):
                        self.overlay.on_complete(response, query=fallback_query, provider="vision")
                    if hasattr(self.mini_overlay, "on_complete"):
                        self.mini_overlay.on_complete(response, query=fallback_query, provider="vision")

                OpenAssistApp._call_on_ui_thread(self, _ui_failure)

            try:
                # P0: Start vision immediately (don't block on OCR).
                # We only use OCR if it's ready fast; otherwise we proceed with
                # snapshot OCR and let OCR finish in the background for fallback.
                ocr_text = ""
                try:
                    ocr_text = await asyncio.wait_for(ocr_task, timeout=0.35)
                except Exception:
                    ocr_text = ""

                screen_text = ocr_text or snapshot.get("latest_ocr", "") or ""

                # Task-first analysis: extract the most actionable on-screen prompt
                # (ignore chrome UI, URLs, weather, etc.) and answer that directly.
                def _extract_task(text: str) -> str:
                    import re

                    if not text:
                        return ""
                    lines = [l.strip() for l in text.splitlines() if l and l.strip()]
                    # Drop obvious noise
                    cleaned = []
                    for l in lines:
                        ll = l.lower()
                        if "http" in ll or "www." in ll:
                            continue
                        if ll.startswith(("chrome", "edge", "firefox")):
                            continue
                        cleaned.append(l)
                    if not cleaned:
                        cleaned = lines

                    # Prefer explicit "build/implement" style prompts
                    for l in cleaned:
                        if re.search(r"\b(build|implement|create|design|write)\b", l.lower()):
                            return l

                    # Next: a numbered title followed by a description
                    for i, l in enumerate(cleaned[:-1]):
                        if re.match(r"^\d+\.\s+\S+", l):
                            nxt = cleaned[i + 1]
                            if len(nxt.split()) >= 5:
                                return nxt

                    # Fallback: the longest non-noise line (often the prompt)
                    return max(cleaned, key=lambda s: len(s), default="")

                task = "" if query_override_text else _extract_task(screen_text)
                if query_override_text:
                    query = (
                        f"{query_override_text}\n\n"
                        "Use the provided screenshot as the source of truth. "
                        "If code is visible, reason from the visible code and provide the output/result."
                    )
                elif task:
                    query = (
                        "Complete the on-screen task.\n\n"
                        f"Task: {task}\n\n"
                        "If this is a coding/UI task, include a minimal working implementation (code) and brief notes."
                    )
                else:
                    query = (
                        "Identify the single most important task from the screenshot "
                        "(ignore URLs, browser chrome, and unrelated UI), then complete it. "
                        "If it is a coding/UI task, include code."
                    )

                try:
                    await self.ai.analyze_image_response(
                        query,
                        image_bytes,
                        snapshot,
                        screen_context=screen_text,
                        audio_context=audio_text,
                    )
                    # Best-effort: if OCR wasn't ready earlier, try to grab it quickly
                    # after vision completes so Nexus stays up-to-date.
                    latest_ocr = ocr_text
                    if not latest_ocr:
                        try:
                            latest_ocr = await asyncio.wait_for(ocr_task, timeout=0.25)
                        except Exception:
                            latest_ocr = ""
                    if latest_ocr and request_epoch == self._generation_epoch:
                        self.nexus.push("screen", latest_ocr)
                    return
                except Exception as exc:
                    if request_epoch != self._generation_epoch:
                        return
                    self._screen_analysis_pending = False
                    logger.warning(f"Vision analysis exhausted providers: {exc}")
                    def _ui_fallback_notice():
                        if hasattr(self.overlay, "set_analysis_provider_badge"):
                            self.overlay.set_analysis_provider_badge()
                        if hasattr(self.overlay, "show_error_toast"):
                            self.overlay.show_error_toast(
                                "Image analysis failed - using OCR/text fallback."
                            )
                        self.overlay.update_transcript(
                            "Screen captured, but image analysis failed. Using OCR/text fallback...",
                            state="error",
                        )

                    OpenAssistApp._call_on_ui_thread(self, _ui_fallback_notice)
                    # Fall back to OCR/text routing (best-effort).
                    try:
                        screen_text_fb = await asyncio.wait_for(ocr_task, timeout=2.0)
                    except Exception:
                        screen_text_fb = ""
                    if screen_text_fb:
                        self.nexus.push("screen", screen_text_fb)
                    if not screen_text_fb.strip():
                        _surface_analysis_failure(
                            "Screen captured, but analysis could not read it.",
                            "No vision provider completed successfully, and OCR is unavailable or returned no text. Check your vision provider settings or install the missing WinRT OCR runtime dependency.",
                        )
                        return
                    OpenAssistApp._call_on_ui_thread(
                        self,
                        lambda q=query, text=screen_text_fb, audio=audio_text: self.generate_response(
                            q,
                            "screen_analysis",
                            {"screen": text, "audio": audio},
                        ),
                    )
                    return
            except Exception:
                if request_epoch != self._generation_epoch:
                    return
                # Never await OCR unbounded here — EasyOCR on CPU can take minutes.
                try:
                    screen_text = await asyncio.wait_for(ocr_task, timeout=2.0)
                except Exception:
                    screen_text = ""
                if screen_text:
                    self.nexus.push("screen", screen_text)
                if not screen_text.strip():
                    _surface_analysis_failure(
                        "Screen captured, but analysis could not read it.",
                        "The screenshot was captured, but both image analysis and OCR fallback failed before producing usable screen text.",
                    )
                    return
                OpenAssistApp._call_on_ui_thread(
                    self,
                    lambda q=query, text=screen_text, audio=audio_text: self.generate_response(
                        q,
                        "screen_analysis",
                        {"screen": text, "audio": audio},
                    ),
                )

        asyncio.run_coroutine_threadsafe(_capture_and_analyze(), self.loop)

    def switch_mode(self, mode: str = None):
        if not mode:
            modes = ["general", "interview", "coding", "meeting", "exam", "writing"]
            # P0.3: Safe cycler — fall back to 'general' if config holds an unknown mode
            current = self.config.get("ai.mode", "general")
            current = current if current in modes else "general"
            mode = modes[(modes.index(current) + 1) % len(modes)]
        self.state.mode = mode
        self.overlay.update_mode(mode)
        self.mini_overlay.update_mode(mode)

        # Switch ModeManager — get the full profile back
        profile = self.modes.switch(mode)

        # Propagate detector sensitivity from the mode profile
        ai = self.__dict__.get("ai")
        if ai and hasattr(ai, "detector"):
            ai.detector.set_mode(mode)
            if hasattr(ai.detector, "set_sensitivity"):
                ai.detector.set_sensitivity(profile.detector_sensitivity)

        # Propagate VAD silence window from the mode profile (or config override) — live, no restart
        custom_stop = self.config.get("capture.audio.vad.stop_silence_ms")
        effective_vad = custom_stop if custom_stop is not None else profile.vad_silence_ms
        if hasattr(self.audio, "set_vad_silence_ms"):
            self.audio.set_vad_silence_ms(effective_vad)

        logger.info(
            f"🔄 Mode Switched: {mode.upper()} | "
            f"ollama_hint={profile.ollama_model_hint} | "
            f"sensitivity={profile.detector_sensitivity:.2f} | "
            f"vad={effective_vad}ms"
        )

        # ── Context auto-suggest ─────────────────────────────────────────────
        # If context is currently empty or was auto-suggested by a previous mode
        # switch, silently replace it with the best-fit preset for the new mode.
        # If the user typed their own context, never overwrite it — but still
        # show the suggestion chip so they know what the preset would be.
        preset_name, preset_text = get_suggested_preset_for_mode(mode)
        standby = getattr(getattr(self, "overlay", None), "standby_view", None)

        ctx_is_auto_or_empty = (
            not self.state.session_context
            or self._context_auto_suggested
        )

        if preset_name and ctx_is_auto_or_empty and preset_text:
            # Auto-load: context was empty or previously auto-suggested
            self.state.session_context = preset_text
            self.ai.set_session_context(preset_text)
            self._context_store.set_last_context(preset_text)
            self._context_auto_suggested = True
            logger.debug(f"Context auto-suggested: '{preset_name}' for mode '{mode}'")
            if standby and hasattr(standby, "show_context_chip"):
                standby.show_context_chip(preset_name, applied=True)
        elif preset_name:
            # User has manual context — just surface the suggestion without loading
            if standby and hasattr(standby, "show_context_chip"):
                standby.show_context_chip(preset_name, applied=False)
        else:
            # No suggestion for this mode (e.g., General) — clear the chip
            if standby and hasattr(standby, "show_context_chip"):
                standby.show_context_chip(None)

        auto_requested = bool(self._auto_mode_requested())
        if getattr(self, "session_active", False) and auto_requested:
            self._start_auto_mode()

    def toggle_audio(self):
        config = getattr(self, "config", None)
        muted = self.audio.toggle()
        self.state.is_muted = muted
        if config is not None:
            self.config.set("capture.audio.muted", muted)
        if config is not None and hasattr(self.config, "save"):
            try:
                self.config.save()
            except Exception as e:
                logger.debug(f"Audio mute config save skipped: {e}")

    def scroll_up(self):
        if (
            hasattr(self.overlay, "stack")
            and self.overlay.stack.currentWidget() is getattr(self.overlay, "settings_view", None)
            and hasattr(self.overlay.settings_view, "scroll_up")
        ):
            self.overlay.settings_view.scroll_up()
            return
        self.overlay.scroll_up()
        self.mini_overlay.scroll_up()

    def scroll_down(self):
        if (
            hasattr(self.overlay, "stack")
            and self.overlay.stack.currentWidget() is getattr(self.overlay, "settings_view", None)
            and hasattr(self.overlay.settings_view, "scroll_down")
        ):
            self.overlay.settings_view.scroll_down()
            return
        self.overlay.scroll_down()
        self.mini_overlay.scroll_down()

    def history_prev(self):
        if (
            hasattr(self.overlay, "stack")
            and self.overlay.stack.currentWidget() is getattr(self.overlay, "settings_view", None)
            and hasattr(self.overlay.settings_view, "select_prev_tab")
        ):
            self.overlay.settings_view.select_prev_tab()
            return
        self.history.move_prev()
        self._sync_history_ui()

    def history_next(self):
        if (
            hasattr(self.overlay, "stack")
            and self.overlay.stack.currentWidget() is getattr(self.overlay, "settings_view", None)
            and hasattr(self.overlay.settings_view, "select_next_tab")
        ):
            self.overlay.settings_view.select_next_tab()
            return
        self.history.move_next()
        self._sync_history_ui()

    def _sync_history_ui(self):
        """Sync history navigation state to both HUDs.

        Guards:
        - Only pushes state when session is active so preloaded prior-session
          entries (GAP4) don't bleed into the UI on restart or when on standby.
        - Only calls mini_overlay.on_complete() when the user has navigated to a
          non-latest entry; for the latest entry _on_response_complete already
          called on_complete directly, so calling it again would be redundant.
        """
        if not getattr(self, "session_active", False):
            return

        st = self.history.get_state()
        idx, total, entry = st
        at_latest = bool(total > 0 and idx == total - 1)
        if at_latest:
            restored_overlay = bool(
                hasattr(self.overlay, "restore_active_response_view")
                and self.overlay.restore_active_response_view()
            )
            restored_mini = bool(
                hasattr(self.mini_overlay, "restore_active_response_view")
                and self.mini_overlay.restore_active_response_view()
            )
            if restored_overlay or restored_mini:
                self._history_navigation_active = False
                return

        self.overlay.update_history_state(*st)
        self.mini_overlay.update_history_state(*st)

        # Auto-expand mini HUD when navigating to a previous history entry.
        # For the *latest* entry (idx == total - 1) _on_response_complete already
        # called mini_overlay.on_complete() — skip to avoid a double render.
        if self.mini_mode and entry:
            if not at_latest:
                self.mini_overlay.on_complete(entry["response"], entry["query"])
        self._history_navigation_active = not at_latest

    def emergency_erase(self):
        """Action: Nukes all data and kills all hardware loops immediately."""
        logger.warning("☣️ EMERGENCY ERASE TRIGGERED")
        self.history.clear()
        self.overlay.hide()
        self.mini_overlay.hide()
        self._stop_runtime_for_reset()

        # Force quit after brief delay to allow cleanup
        QTimer.singleShot(800, self.qt_app.quit)

    def _stop_runtime_for_reset(self):
        """Stop active capture, hotkeys, and background tasks before a hard reset."""
        self.is_running = False
        self.session_active = False
        self._screen_analysis_pending = False
        if hasattr(self, "_nexus_timer"):
            self._nexus_timer.stop()
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.stop()
        if hasattr(self, "_move_timer"):
            self._move_timer.stop()

        if hasattr(self.ai, "cancel"):
            self.ai.cancel()
        if hasattr(self.audio, "stop"):
            self.audio.stop()
        if hasattr(self.audio, "clear"):
            self.audio.clear()
        if hasattr(self.hotkeys, "stop"):
            self.hotkeys.stop()
        if hasattr(self.hotkeys, "reset_state"):
            self.hotkeys.reset_state()
        if hasattr(self.screen, "stop"):
            self.screen.stop()

        self.state.target_window_id = None
        self.state.is_capturing = False
        self._stop_background_tasks()

    def _clear_factory_reset_artifacts(self):
        """Wipe persisted user state so the next launch is treated as first-run."""
        self.history.clear()
        self.nexus.clear()

        if hasattr(self.ai, "_rag_cache"):
            self.ai._rag_cache.clear()
        if hasattr(self.rag, "_cache"):
            self.rag._cache.clear()
        if hasattr(self.rag, "stop"):
            self.rag.stop()

        self.config.reset_all()

        for path_str in [DB_DIR, CACHE_DIR, LOG_DIR]:
            shutil.rmtree(Path(path_str), ignore_errors=True)

    def _restart_app(self) -> bool:
        """Best-effort detached restart for script and packaged app runs."""
        try:
            if getattr(sys, "frozen", False):
                # M35 FIX: PyQt6 startDetached() returns (success, pid) tuple.
                success, _pid = QProcess.startDetached(sys.executable, sys.argv[1:])
                return success
            success, _pid = QProcess.startDetached(sys.executable, sys.argv)
            return success
        except Exception as e:
            logger.error(f"Factory reset restart failed: {e}")
            return False

    def _show_onboarding_after_reset(self):
        """Fallback when restart is unavailable: return the current process to first-run UI."""
        self.is_running = True
        self._sync_state_from_config()
        if hasattr(self, "_nexus_timer"):
            self._nexus_timer.start(3000)
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.start(1500)
        if hasattr(self.hotkeys, "start"):
            self.hotkeys.start()
        self._background_warmup()
        self._present_window(self.overlay, focus=True)
        self.overlay.show_onboarding()

    def factory_reset(self, restart: bool = True):
        """Full first-run reset: clears persisted state, caches, and hardware state."""
        logger.warning("🧹 FACTORY RESET TRIGGERED")
        self.overlay.hide()
        self.mini_overlay.hide()
        self._stop_runtime_for_reset()
        self._clear_factory_reset_artifacts()

        if restart and self._restart_app():
            QTimer.singleShot(150, self.qt_app.quit)
            return

        self._show_onboarding_after_reset()

    def start_new_session(self):
        """Start a clean session: wipe all previous content and begin fresh."""
        session_start_time = time.time()
        self._generation_epoch = next(self._generation_epoch_counter)
        self._session_start_time = session_start_time
        session_id = f"session_{self._generation_epoch}_{int(session_start_time*1000)%100000}"
        self._current_session_id = session_id

        logger.info(
            f"[{session_id}] 🚀 SESSION START | epoch={self._generation_epoch} | time={session_start_time:.3f}"
        )

        self.ai.cancel()
        if hasattr(self.ai, "reset_session_failures"):
            try:
                self.ai.reset_session_failures()
            except Exception:
                pass
        self.nexus.clear()
        self.history.start_new_session()
        audio_enabled = bool(self.config.get("capture.audio.enabled", True))
        self.state.is_muted = bool(self.config.get("capture.audio.muted", False))
        self._screen_analysis_pending = False
        self.audio.clear()
        if hasattr(self.audio, "set_trace_context"):
            self.audio.set_trace_context(session_id, session_start_time)
        audio_ready = True
        if not audio_enabled:
            logger.info("[%s] Audio capture is disabled; starting session with mic off.", session_id)
        elif hasattr(self.audio, "ensure_session_ready"):
            audio_start = time.time()
            audio_ready = bool(self.audio.ensure_session_ready())
            audio_elapsed = (time.time() - audio_start) * 1000
            logger.info(
                f"[{session_id}] 🎤 Audio ready check | ready={audio_ready} | elapsed={audio_elapsed:.1f}ms"
            )
        elif hasattr(self.audio, "start"):
            self.audio.start()
            audio_ready = bool(getattr(self.audio, "running", True))
        if not audio_ready:
            logger.error("[%s] Session start aborted because audio capture is not ready.", session_id)
            self.overlay.update_transcript("Audio capture unavailable. Check the selected input/source.", state="error")
            self._reset_turn_local_state("audio-not-ready")
            self.session_active = False
            self.state.is_capturing = False
            return
        self._reset_turn_local_state("start-session")
        self.session_active = True
        self.state.is_capturing = True

        # P2.1: Reset ContextBuilder so entity tracking starts fresh
        if hasattr(self, "context_builder"):
            self.context_builder.reset()
            logger.info(f"[{session_id}] [P2.1] ContextBuilder reset for new session")

        # P2.4: Log memory store size at session start
        if hasattr(self, "memory") and self.memory.is_ready():
            logger.info(f"[P2.4] LongTermMemory: {self.memory.count()} memories available")

        # P3.1: Reset prefetcher so stale symbol cache from last session is cleared
        if hasattr(self, "prefetcher"):
            self.prefetcher.reset()
            logger.info("[P3.1] Prefetcher reset for new session")

        clear_overlay = getattr(self.overlay, "clear_session_response", None)
        if callable(clear_overlay):
            clear_overlay()
        else:
            self.overlay._current_query = ""
            self.overlay._raw_buffer = ""
            self.overlay._is_streaming = False
            self.overlay.response_area.clear()
        clear_mini = getattr(self.mini_overlay, "clear_session_response", None)
        if callable(clear_mini):
            clear_mini()
        else:
            self.mini_overlay.on_complete("", "")
        start_session_ui = getattr(self.overlay, "start_session_ui", None)
        if callable(start_session_ui):
            start_session_ui()
        else:
            self.overlay.stack.setCurrentWidget(self.overlay.chat_view)

        if audio_ready:
            self.overlay.update_transcript("Listening for context...")
        else:
            self.overlay.update_transcript(
                "Audio capture unavailable. Check audio source/settings.",
                state="error",
            )
        if hasattr(self.overlay, "set_analysis_provider_badge"):
            self.overlay.set_analysis_provider_badge()
        self.mini_overlay.set_ready()
        m = self.config.get("ai.mode", "general")
        self.overlay.update_mode(m)
        self.mini_overlay.update_mode(m)
        auto_requested = bool(self._auto_mode_requested())
        if hasattr(self.overlay, "update_auto_mode_state"):
            self.overlay.update_auto_mode_state(
                auto_requested,
                False,
                reconnecting=False,
                fallback=False,
            )
        if auto_requested:
            self._start_auto_mode()
        # P2.10: Pin-to-top is only needed during an active session
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.start(1500)
        logger.info("🚀 New Session Started — slate is clean.")

    def end_session(self):
        """Fully terminate the active session: cancel AI, wipe content, archive history."""
        self._generation_epoch = next(self._generation_epoch_counter)
        self.ai.cancel()
        self._screen_analysis_pending = False
        self.session_active = False
        self.state.is_capturing = False
        if hasattr(self.audio, "stop"):
            self.audio.stop()
        self.state.target_window_id = None
        self.nexus.clear()
        self._reset_turn_local_state("end-session")

        # P0.5: Reset auto-suggest flag so next session can freely pick presets
        self._context_auto_suggested = False

        # Archive the current session so next start_new_session() begins with a blank slate.
        # This also clears in-memory entries so the history block injected into prompts
        # doesn't carry over previous session context.
        # P2.4: Persist the last AI exchange to long-term memory before the
        # history archive clears active entries.
        if hasattr(self, "memory"):
            last_entries = self.history.get_last(1)
            if last_entries:
                e = last_entries[0]
                q = getattr(e, "query", "") or ""
                r = getattr(e, "response", "") or ""
                metadata = getattr(e, "metadata", {}) or {}
                req_meta = metadata.get("request_metadata", {}) or {}
                mode = str(self.config.get("ai.mode", "general"))
                should_skip_memory = bool(
                    req_meta.get("benchmark_isolated")
                    or req_meta.get("suppress_memory_context")
                )
                if q and r and not should_skip_memory:
                    session_id = str(int(time.time()))
                    logger.info(
                        f"[P2.4] Archiving session memory — mode={mode}, "
                        f"query_preview='{q[:60]}'"
                    )
                    self.memory.store(session_id, q, r, mode=mode)

        # Archive the current session so next start_new_session() begins with a blank slate.
        # This also clears in-memory entries so the history block injected into prompts
        # doesn't carry over previous session context.
        self.history.start_new_session()

        # Clear AI caches so the next session has no stale context
        if hasattr(self.ai, "clear_rag_prefetch"):
            self.ai.clear_rag_prefetch()
        if hasattr(self.ai, "_complexity_cache"):
            self.ai._complexity_cache.clear()

        # Persist the current context to disk before clearing from state,
        # so it reloads on next app launch (user doesn't retype every time).
        if hasattr(self, "_context_store"):
            self._context_store.set_last_context(self.state.session_context)
        # Clear from live state — next session starts clean unless user re-selects
        self.state.session_context = ""
        self.ai.set_session_context("")

        # Hard-wipe the visible response area — user should see a blank screen
        # when they return to standby, and again when next session starts.
        clear_overlay = getattr(self.overlay, "clear_session_response", None)
        if callable(clear_overlay):
            clear_overlay()
        else:
            self.overlay._current_query = ""
            self.overlay._raw_buffer = ""
            self.overlay._is_streaming = False
            self.overlay.response_area.clear()
        clear_mini = getattr(self.mini_overlay, "clear_session_response", None)
        if callable(clear_mini):
            clear_mini()
        else:
            self.mini_overlay.on_complete("", "")
        end_session_ui = getattr(self.overlay, "end_session_ui", None)
        if callable(end_session_ui):
            end_session_ui()
        else:
            self.overlay.stack.setCurrentWidget(self.overlay.standby_view)
        self.overlay.update_transcript("Ready...")
        auto_requested = bool(self._auto_mode_requested())
        if hasattr(self.overlay, "update_auto_mode_state"):
            self.overlay.update_auto_mode_state(
                auto_requested,
                False,
                reconnecting=False,
                fallback=False,
            )
        if hasattr(self.overlay, "set_analysis_provider_badge"):
            self.overlay.set_analysis_provider_badge()

        # P0.6: Clear the context chip — stale chip from previous session would be misleading
        standby = getattr(self.overlay, "standby_view", None)
        if standby and hasattr(standby, "show_context_chip"):
            standby.show_context_chip(None)

        self.mini_overlay.set_ready()
        # P2.10: Stop pinning to top while on standby — no need to fight the OS scheduler
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.stop()
        logger.info("🛑 Session Ended — history archived, slate wiped.")

    def reset_benchmark_fixture_runtime(self):
        """Reset transient per-fixture state without ending the active benchmark session."""
        self.ai.cancel()
        if hasattr(self.ai, "set_session_context"):
            self.ai.set_session_context("")
        self._screen_analysis_pending = False
        self._reset_turn_local_state("benchmark-fixture-reset")
        reset_auto_mode_turn_state(self)

        clear_overlay = getattr(self.overlay, "clear_session_response", None)
        if callable(clear_overlay):
            clear_overlay()
        else:
            self.overlay._current_query = ""
            self.overlay._raw_buffer = ""
            self.overlay._is_streaming = False
            self.overlay.response_area.clear()
        clear_mini = getattr(self.mini_overlay, "clear_session_response", None)
        if callable(clear_mini):
            clear_mini()
        else:
            self.mini_overlay.on_complete("", "")

        if self._auto_mode_requested():
            self.overlay.update_transcript("Auto Mode listening...", state="listening")
        else:
            self.overlay.update_transcript("Listening for context...", state="listening")

    def set_current_mode(self, name):
        """Update the active AI mode by name (e.g. 'general', 'coding', 'interview').

        Persists the selection to config and notifies the UI overlays.
        Previously referenced undefined self.mode_buttons — fixed to use
        the config + overlay update path that the rest of the app uses.
        """
        self.config.set("ai.mode", name)
        if hasattr(self, "overlay"):
            self.overlay.update_mode(name)
        if hasattr(self, "mini_overlay"):
            self.mini_overlay.update_mode(name)

    def toggle_mini_mode(self):
        self.mini_mode = not self.mini_mode
        self.state.is_mini = self.mini_mode
        # M34 FIX: Actually show/hide the correct overlay views when toggling.
        if self.mini_mode:
            self.overlay.hide()
            self.mini_overlay.show()
            self.mini_overlay.raise_()
        else:
            self.mini_overlay.hide()
            self.overlay.show()
            self.overlay.raise_()
        self._refresh_window_invariants()
        # P3.1: Sync history UI when switching modes to preserve navigation state
        self._sync_history_ui()

    def ensure_stealth(self):
        self.state.is_stealth = True
        self._apply_ui_only()

    def toggle_stealth_mode(self):
        """Compatibility alias for older callers; stealth is always enforced."""
        self.ensure_stealth()

    def type_last_response(self):
        """Action: Type the latest AI response into the Snap-Locked window."""
        entries = self.history.get_last(1)
        if not entries:
            return

        response = entries[-1].response
        target = self.state.target_window_id

        if target:
            self.simulator.type_text(response, target)
        else:
            logger.warning("Sim: No Snap-Locked target window available.")

    def type_response(self):
        """Compatibility alias for UI code that expects app.type_response()."""
        return self.type_last_response()



    def _on_transcription(self, t):
        if not self.session_active or not t:
            return

        session_id = getattr(self, "_current_session_id", "unknown_session")
        elapsed_ms = (time.time() - getattr(self, "_session_start_time", time.time())) * 1000

        # Phase 1: Smart Transcription Repair
        from utils.text_utils import is_question_complete, normalize_transcript
        t = normalize_transcript(t)
        if not t:
            return

        capture_mode = str(getattr(self.audio, "capture_mode", "") or "").lower()
        last_final_text = (getattr(self, "_last_final_transcript_text", "") or "").strip()
        last_final_at = float(getattr(self, "_last_final_transcript_at", 0.0) or 0.0)
        now = time.time()
        if capture_mode == "system" and t == last_final_text and last_final_at > 0.0 and (now - last_final_at) <= 2.0:
            logger.info("[%s] Ignoring duplicate system-audio final transcript", session_id)
            return
        self._last_final_transcript_text = t
        self._last_final_transcript_at = now

        logger.info(
            "[%s] \U0001f4dd FINAL TRANSCRIPT RECEIVED | elapsed=%.1fms | text='%s'",
            session_id, elapsed_ms, t[:60],
        )

        transcription_metrics = self.audio.get_last_transcription_metrics()
        if transcription_metrics:
            vad_backend = transcription_metrics.get("vad_backend", "unknown")
            audio_duration = transcription_metrics.get("audio_duration_ms", 0)
            final_queue_wait = transcription_metrics.get("final_queue_wait_ms", 0)
            transcribe_only = transcription_metrics.get("transcribe_only_ms", 0)
            logger.info(
                "[%s] \U0001f3a4  Audio metrics | vad=%s | duration=%.0fms | final_queue=%.0fms | transcribe=%.0fms",
                session_id, vad_backend, audio_duration, final_queue_wait or 0, transcribe_only or 0,
            )
        request_metadata = None
        if transcription_metrics:
            speech_to_transcript_ms = transcription_metrics.get("speech_to_transcript_ms")
            transcribe_only_ms = transcription_metrics.get("transcribe_only_ms")
            if speech_to_transcript_ms is not None:
                logger.info(
                    "[%s] \u23f1\ufe0f  Transcription latency | speech\u2192transcript=%.0fms | transcribe_only=%.0fms",
                    session_id, speech_to_transcript_ms, transcribe_only_ms or 0,
                )
            request_metadata = {
                **transcription_metrics,
                "transcript_received_at": time.time(),
            }

        if self._should_ignore_final_transcript(t):
            logger.info("[%s] \U0001f6ab Ignoring short final transcript fragment: %r", session_id, t)
            detector = getattr(getattr(self, "ai", None), "detector", None)
            if detector and hasattr(detector, "reset_turn_state"):
                detector.reset_turn_state("ignored-final-fragment")
            elif detector and hasattr(detector, "reset_fragment_buffer"):
                detector.reset_fragment_buffer("ignored-final-fragment")
            return

        self.nexus.push("audio", t)

        if handle_auto_final_transcription(self, t, session_id, request_metadata=request_metadata):
            return

        # ── Standard (non-live) transcription path ─────────────────────────────
        self.overlay.update_transcript(t)
        # M37 FIX: Reuse request_metadata built above (L2085-2097) instead of
        # rebuilding it and double-logging transcription latency.

        # ── Background RAG Prefetch ────────────────────────────────────────────
        if self.loop and self.loop.is_running() and hasattr(self.ai, "prefetch_rag"):
            snapshot = self.nexus.get_snapshot()
            asyncio.run_coroutine_threadsafe(
                self.ai.prefetch_rag(
                    screen_text=snapshot.get("latest_ocr", ""),
                    audio_text=t,
                ),
                self.loop,
            )

        q = self.ai.detector.detect_with_confidence(t, source="audio")
        logger.info("[%s] \U0001f3a4 Transcribed: '%s'", session_id, t)
        if q.triggered:
            can_auto = q.should_auto_respond() and is_question_complete(q.detected_text)
            logger.info(
                "[%s] \u26a1 QUESTION DETECTED | auto_respond=%s | question='%s'",
                session_id, can_auto, q.detected_text[:40],
            )
            if can_auto:
                self._pending_request_metadata = dict(request_metadata or {})
                self._current_response_start_time = time.time()
                logger.info(
                    "[%s] \U0001f916 RESPONSE GENERATION START | elapsed=%.1fms",
                    session_id, elapsed_ms,
                )
                combine_followup = getattr(self, "_combine_with_pending_audio_followup", None)
                if combine_followup is None:
                    combine_followup = lambda query: OpenAssistApp._combine_with_pending_audio_followup(self, query)
                final_query = combine_followup(q.detected_text)
                self.generate_response(final_query, "speech", {"audio": final_query})
            else:
                carry_forward = getattr(self, "_carry_forward_incomplete_audio_query", None)
                if carry_forward is None:
                    carry_forward = lambda text: OpenAssistApp._carry_forward_incomplete_audio_query(self, text)
                carry_forward(t)
                self.overlay.update_transcript(
                    "\U0001f914 Possible question detected (tap \u238a to answer)",
                    state="processing",
                )

        elif self._looks_question_like_transcript(t) or "difference between" in t.lower():
            combine_followup = getattr(self, "_combine_with_pending_audio_followup", None)
            if combine_followup is None:
                combine_followup = lambda query: OpenAssistApp._combine_with_pending_audio_followup(self, query)
            final_query = combine_followup(t)
            self._pending_request_metadata = dict(request_metadata or {})
            self._current_response_start_time = time.time()
            logger.info(
                "[%s] \U0001f916 RESPONSE GENERATION START | elapsed=%.1fms",
                session_id, elapsed_ms,
            )
            self.generate_response(final_query, "speech", {"audio": final_query})


    def _should_ignore_final_transcript(self, text: str) -> bool:
        """Drop tiny non-question final ASR scraps before they enter context/history.

        Auto Mode keeps setup/context fragments so it can answer the full spoken
        prompt, so we skip this filter when Auto Mode is enabled.
        """
        from utils.text_utils import is_likely_fragment

        cleaned = (text or "").strip()
        if not cleaned:
            return True

        # Auto Mode keeps setup/context fragments so it can answer the full prompt.
        if self._auto_mode_requested():
            return False

        if is_likely_fragment(cleaned):
            return True

        words = cleaned.replace("?", " ").split()
        min_chars = int(self.config.get("detection.final_min_chars", 6) or 6)
        min_words = int(self.config.get("detection.final_min_words", 2) or 2)
        requires_signal = bool(
            self.config.get("detection.final_requires_question_signal", True)
        )

        if len(cleaned) >= min_chars and len(words) >= min_words:
            return False
        if not requires_signal:
            return False
        return not self._looks_question_like_transcript(cleaned)

    def _looks_question_like_transcript(self, text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        lower = cleaned.lower()
        if "?" in cleaned:
            return True
        try:
            from ai.auto_query_utils import looks_like_setup_statement
            if looks_like_setup_statement(cleaned):
                return False
        except Exception:
            pass

        detector = getattr(getattr(self, "ai", None), "detector", None)
        prefixes = getattr(detector, "question_prefixes", []) or []
        patterns = getattr(detector, "question_patterns", []) or []
        return any(lower.startswith(prefix) for prefix in prefixes) or any(
            pattern in lower for pattern in patterns
        )

    def _on_interim_transcription(self, t: str):
        """Display live (partial) ASR updates while the user is still speaking.

        Shows interim text in the overlay for real-time feedback, and also
        checks for early question detection.
        """
        if not self.session_active or not t:
            return
        from utils.text_utils import normalize_transcript

        t = normalize_transcript(t)
        if not t:
            return

        session_id = getattr(self, "_current_session_id", "unknown_session")
        elapsed_ms = (time.time() - getattr(self, "_session_start_time", time.time())) * 1000
        logger.info(
            f"[{session_id}] ⏱️  INTERIM TEXT | elapsed={elapsed_ms:.1f}ms | text='{t[:50]}'"
        )

        # Show interim text in the overlay for live feedback
        try:
            self.overlay.update_transcript(t, state="interim")
        except Exception as e:
            logger.debug(f"Interim overlay update failed: {e}")

        if self._auto_mode_requested():
            try:
                handle_auto_interim_transcription(self, t, session_id)
            except Exception as e:
                logger.debug(f"Auto Mode interim speculation skipped: {e}")
            return

        # Check for early question detection
        try:
            det = getattr(self.ai, "detector", None)
            if not det or not hasattr(det, "detect_interim_with_guardrails"):
                return
            candidate = det.detect_interim_with_guardrails(t)
            if candidate:
                logger.info(
                    f"[{session_id}] ⚡ QUESTION DETECTED (interim) | candidate='{candidate[:40]}'"
                )
                self.generate_response(candidate, "speech", {"audio": t})
        except Exception as e:
            logger.debug(f"Interim transcription detector error (non-fatal): {e}")

    def _on_screen_text(self, t):
        if not self.session_active:
            return

        # ── Background RAG Prefetch ───────────────────────────────────────────
        # Pre-warm the RAG cache with what's visible on screen so when the user
        # asks a question the RAG result is already ready.
        if self.loop and self.loop.is_running() and hasattr(self.ai, "prefetch_rag"):
            asyncio.run_coroutine_threadsafe(
                self.ai.prefetch_rag(
                    screen_text=t,
                    audio_text=self.audio.get_transcript() if hasattr(self.audio, "get_transcript") else "",
                ),
                self.loop,
            )

        # P3.1: Predictive Prefetch — fire when we detect IDE symbols on screen
        if hasattr(self, "prefetcher"):
            window_title = str(
                getattr(self.state, "active_window_title", "")
                or (self.nexus.get_snapshot() or {}).get("active_window", "")
            )
            self.prefetcher.analyze(t, window_title=window_title)

        q = self.ai.detector.detect_with_confidence(t, source="screen")
        if q.triggered:
            if q.should_auto_respond():
                self.generate_response(q.detected_text, "auto", {"screen": t})
            else:
                # P2.1: Low-confidence screen detection — show hint, do not auto-answer
                self.overlay.update_transcript(
                    "🤔 Possible cue on screen (tap ⎨ to answer)",
                    state="processing",
                )

    def _background_warmup(self):
        """Warm components before the session button unlocks.

        Start Session must not become clickable until the first real prompt can
        be captured and answered.  RAG indexing can still finish in the
        background, but AI, vision, and Whisper are part of the interactive
        readiness gate.
        """

        def _warm_vision():
            try:
                if not bool(self.config.get("capture.screen.enabled", True)):
                    logger.info("Vision warmup skipped: capture.screen.enabled=false")
                    return
                self.screen.initialize()
                self.warmup_status_update.emit("👁️ Vision Ready", 40, False)
            except Exception as e:
                logger.error(f"Vision Warmup Fault: {e}")

        def _warm_brain():
            try:
                self.ai.warmup(self.loop)
                try:
                    from ai.auto_answer_controller import _get_intent_classifier

                    classifier = _get_intent_classifier()
                    if classifier is not None:
                        classifier.classify("What is the warmup question?")
                except Exception as e:
                    logger.debug("Intent classifier warmup skipped: %s", e)
                self.ai.ensure_health_monitor(self.loop)
                # Opportunistically pre-warm Groq's TCP/TLS connection so the
                # first real user prompt may hit an already-open socket. READY
                # does not wait for this. Periodic keepalive is config-gated
                # because each ping is still a Groq API request.
                if hasattr(self.ai, "warm_groq_connection"):
                    self.ai.warm_groq_connection(self.loop)
                if hasattr(self.ai, "start_groq_keepalive"):
                    self.ai.start_groq_keepalive(self.loop)
                asyncio.run_coroutine_threadsafe(
                    self._continuous_capture_loop(), self.loop
                )
                self.warmup_status_update.emit("🧠 Brain Ready", 80, False)
            except Exception as e:
                logger.error(f"Brain Warmup Fault: {e}")

        def _warm_audio():
            """Warm STT before the Start Session button unlocks.

            Groq is the primary STT path in Auto Mode, but local Whisper is the
            safety net for provider stalls, auth errors, and offline use.  The
            standby READY state means the first real prompt can be handled, so
            this deliberately blocks on local Whisper preload even when cloud
            STT is selected.  It does not start hardware capture.
            """
            try:
                if not bool(self.config.get("capture.audio.enabled", True)):
                    logger.info("Audio warmup skipped: capture.audio.enabled=false")
                    return
                provider = "local"
                if hasattr(self.audio, "_effective_transcription_provider"):
                    provider = self.audio._effective_transcription_provider(is_final=True)
                if provider == "groq":
                    if hasattr(self.audio, "_ensure_whisper_loaded_async"):
                        self.audio._ensure_whisper_loaded_async(force=False)
                    self.warmup_status_update.emit("Cloud STT Ready", 90, False)
                    return
                if hasattr(self.audio, "_ensure_whisper_loaded_async"):
                    self.audio._ensure_whisper_loaded_async(force=True)
                self.warmup_status_update.emit("Local STT Preloading...", 90, False)
            except Exception as e:
                logger.error(f"Audio Warmup Fault: {e}")

        def _warm_knowledge_deferred():
            """Index knowledge base — handles PDFs, Q&A files, and text/code docs.

            Drop any file into knowledge/documents/ and it is indexed here:
              .pdf              → PyMuPDF text extraction → chunked → indexed
              .json/.txt (Q&A)  → Q/A pair parser → 'Q:..\nA:..' chunks → indexed
              .txt/.md/.py etc  → add_directory as before
            Does NOT block the READY signal — runs fully in background.
            """
            try:
                if not bool(self.config.get("rag.enabled", True)):
                    logger.info("RAG warmup skipped: rag.enabled=false")
                    return
                from knowledge.ingest import ingest_all
                from pathlib import Path
                from core.constants import DOCS_DIR
                ingest_all(self.rag, Path(DOCS_DIR))
                self.warmup_status_update.emit("📚 Knowledge Ready", 95, False)
            except Exception as e:
                logger.error(f"RAG Warmup Fault: {e}")
                try:
                    self.rag.add_directory("./knowledge")
                except Exception:
                    pass

        critical = [
            threading.Thread(target=_warm_brain, daemon=True, name="warmup-brain"),
            threading.Thread(target=_warm_vision, daemon=True, name="warmup-vision"),
            threading.Thread(target=_warm_audio, daemon=True, name="warmup-whisper"),
        ]
        deferred = [
            threading.Thread(target=_warm_knowledge_deferred, daemon=True, name="warmup-rag"),
        ]
        for t in critical + deferred:
            t.start()

        def _monitor():
            try:
                for t in critical:  # only wait for critical tasks
                    t.join()
                self.warmup_status_update.emit("✅ READY", 100, True)
                logger.info("🚀 All systems online - Initialization Complete")
            except (RuntimeError, AttributeError):
                logger.warning("Monitor: UI context lost during warmup. Skipping signal.")

        threading.Thread(target=_monitor, daemon=True, name="warmup-monitor").start()



    async def _continuous_capture_loop(self):
        """Asynchronous vision loop: periodically updates screen context for the Nexus."""
        logger.info("👁️ Continuous Vision Loop Started.")

        # Phase 2 async tuning: use a wall-clock ticker so that OCR extraction time
        # doesn't drift the capture interval.
        interval = self.config.get("capture.screen.interval_ms", 3000) / 1000.0
        next_tick = asyncio.get_event_loop().time()

        while self.is_running:
            try:
                # Screen context remains active during a session even if audio is muted.
                if self.session_active:
                    # Capture current screen state and push to context
                    text = await self.screen.capture_context()
                    if text:
                        # self._on_screen_text(text)  # Optional: logic for auto-trigger
                        self.nexus.push("screen", text)

                # Update interval in case it changed in settings
                interval = self.config.get("capture.screen.interval_ms", 3000) / 1000.0
                next_tick += interval

                now = asyncio.get_event_loop().time()
                sleep_time = next_tick - now
                if sleep_time <= 0:
                    # We fell behind (e.g. OCR took longer than interval). Skip ticks to catch up.
                    next_tick = now
                    sleep_time = 0

                await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Vision Loop Error: {e}")
                await asyncio.sleep(5)
                next_tick = asyncio.get_event_loop().time()

    def _stop_background_tasks(self):
        """Stops all background AI monitoring and capture tasks."""
        self.is_running = False
        self._nexus_timer.stop()
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.stop()
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.ai.stop_health_monitor(), self.loop)

    def shutdown(self):
        """Proper shutdown: stops health monitor, event loop, and joins thread."""
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        logger.info("🛑 Shutting down OpenAssist...")

        # P4: If a session is active (especially in mini_mode), end it gracefully
        if getattr(self, "session_active", False):
            logger.info("🛑 Session active during shutdown, ending it now...")
            self.end_session()

        self.is_running = False  # Signal all loops to break immediately

        self._stop_background_tasks()

        # Stop the event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        # Join the async thread
        if self._async_thread.is_alive():
            self._async_thread.join(timeout=3)

        # Stop other components
        self.audio.stop()
        self.hotkeys.stop()
        self.rag.stop()

        # Save history
        self.history.save()
        logger.info("✅ OpenAssist shutdown complete")
