"""
OpenAssist AI v4.1 — Main Application Controller (Midnight Hardened).
RESTORED: Smooth Gliding (60fps / 5px steps), HUD navigation, and click-through.
FIXED: Standby warmup signal bridge and non-blocking hardware hot-apply.
RESTORATION: Automatic Knowledge Sync (RAG) during warmup.
"""

import sys
import asyncio
import threading
import time
import shutil
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon
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
from core.tray import SystemTray
from stealth.anti_detect import StealthManager
from stealth.input_simulator import InputSimulator
from utils.logger import setup_logger
from utils.context_store import get_store as get_context_store
from utils.context_store import get_suggested_preset_for_mode
from core.constants import DB_DIR, CACHE_DIR, LOG_DIR

logger = setup_logger(__name__)


class OpenAssistApp(QObject):
    warmup_status_update = pyqtSignal(str, int, bool)

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
        self.simulator = InputSimulator(config)

        self.session_active = False
        self._last_query = ""
        self._last_query_time = 0.0
        self._click_through = False
        self._generation_epoch = 0
        self._screen_analysis_pending = False
        self._pending_request_metadata = None
        # Tracks whether the current session_context was auto-suggested by a mode
        # switch (True) or typed/loaded manually (False). Auto-suggested context
        # can be silently replaced when the mode changes; manual context cannot.
        self._context_auto_suggested: bool = False

        # Async Loop
        self.loop = asyncio.new_event_loop()
        self._ai_lock_ready = threading.Event()
        self._async_thread = threading.Thread(target=self._run_master_loop, daemon=True)

        # UI
        self.overlay = OverlayWindow(config, self)
        self.mini_overlay = MiniOverlay(config, self)
        self.hotkeys = HotkeyManager(config, self)
        self.tray = self._create_system_tray()

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

    def _run_master_loop(self):
        asyncio.set_event_loop(self.loop)
        self._ai_lock = asyncio.Lock()
        self._ai_lock_ready.set()
        self.loop.run_forever()

    def _wire_signals(self):
        self.overlay.user_query.connect(self.generate_response)
        self.mini_overlay.user_query.connect(self.generate_response)
        self.ai.response_chunk.connect(lambda c: self.overlay.append_response(c))
        self.ai.response_chunk.connect(lambda c: self.mini_overlay.append_response(c))
        self.ai.response_complete.connect(
            lambda t: self.overlay.on_complete(t, self._last_query)
        )
        self.ai.response_complete.connect(
            lambda t: self.mini_overlay.on_complete(t, self._last_query)
        )
        self.ai.error_occurred.connect(
            lambda e: self.overlay.on_complete(f"ERROR: {e}")
        )
        self.ai.error_occurred.connect(lambda e: self.mini_overlay.show_error(e))
        self.audio.transcription_ready.connect(self._on_transcription)
        self.screen.text_captured.connect(self._on_screen_text)
        self.warmup_status_update.connect(self.overlay.update_warmup_status)
        self.ai.provider_status.connect(self.overlay.standby_view.set_provider_statuses)
        self.overlay.standby_view.start_clicked.connect(self.start_new_session)
        self.overlay.standby_view.mode_selected.connect(self.switch_mode)
        self.overlay.standby_view.audio_source_changed.connect(
            self._on_audio_source_ui_change
        )
        self.ai.response_complete.connect(self._on_response_complete)
        self.ai.error_occurred.connect(self._on_ai_error)
        self.state.stealth_changed.connect(lambda _: self._apply_ui_only())

    def _on_response_complete(self, full_text: str):
        """Update overlay status bar with latency after each response."""
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

        # Get available providers
        available = []
        if hasattr(self.ai, "_providers"):
            available = [
                p
                for p, prov in self.ai._providers.items()
                if hasattr(prov, "enabled") and prov.enabled
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

    def _on_ai_error(self, error_text: str):
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
        self.state.is_stealth = self.config.get("stealth.enabled", True)

    def _poll_nexus_context(self):
        """Polls environmental signals for the ContextNexus."""
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
                self.hotkeys.restart()

                # 2. Process Warmup
                # Close old provider network resources (e.g. Ollama aiohttp sessions)
                # before re-initializing providers to avoid "Unclosed client session".
                try:
                    self.ai.close_providers()
                except Exception:
                    pass
                self.ai.warmup()

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
        self._apply_window_effects(self.overlay)
        self._apply_window_effects(self.mini_overlay)

    def _apply_window_effects(self, window):
        base_opacity = self.config.get("app.opacity", 0.94)
        stealth_opacity = self.config.get("stealth.low_opacity", 0.75)
        is_stealth = bool(getattr(self.state, "is_stealth", False))

        window.setWindowOpacity(stealth_opacity if is_stealth else base_opacity)
        WindowUtils.hide_from_taskbar(window)
        WindowUtils.ensure_topmost(window)
        self.stealth.apply_to_window(window, is_stealth)

    def _refresh_topmost_window(self):
        for window in (self.overlay, self.mini_overlay):
            try:
                if hasattr(window, "isVisible") and window.isVisible():
                    WindowUtils.ensure_topmost(window)
            except Exception as e:
                logger.debug(f"Topmost refresh skipped: {e}")

    def _create_system_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.info("System tray unavailable; tray controls disabled.")
            return None
        try:
            return SystemTray(self)
        except Exception as e:
            logger.warning(f"System tray initialization failed: {e}")
            return None

    def _active_view(self):
        return self.mini_overlay if self.mini_mode else self.overlay

    def _hud_focus_enabled(self) -> bool:
        return bool(self.config.get("app.focus_on_show", False))

    def _present_window(self, window, focus: bool = False):
        if not window:
            return None

        window.show()
        WindowUtils.ensure_topmost(window)
        if hasattr(window, "raise_"):
            window.raise_()
        if focus and hasattr(window, "activateWindow"):
            window.activateWindow()
        return window

    def _show_active_overlay(self):
        view = self._active_view()
        return self._present_window(view, focus=self._hud_focus_enabled())

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

        if self.config.get("app.start_minimized", False):
            self.overlay.hide()
            self.mini_overlay.hide()
            logger.info("Startup minimized to system tray.")
            return

        self._show_active_overlay()

    def run(self) -> int:
        self._async_thread.start()
        self.audio.start()
        self.hotkeys.start()

        # Show overlay AFTER Qt event loop starts to ensure window is visible
        QTimer.singleShot(100, self._show_initial_window)

        self._background_warmup()
        return self.qt_app.exec()

    def generate_response(self, q, s="manual", c=None):
        if not q or not self._ai_lock_ready.wait(timeout=2):
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
        # SNAP-LOCK: Capture target window HWND at moment of query
        if s in ["manual", "speech", "quick"]:
            hwnd = self.simulator.get_foreground_window()
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

            # Show query immediately in response area — do not wait for first token
            self.overlay.show_chat_view()
            self.overlay._current_query = q
            self.overlay.response_area.clear()
            race_hint = ""
            if s in {"manual", "speech"} and bool(self.config.get("ai.text.race_enabled", False)):
                race_hint = " (race mode — no streaming)"
            self.overlay.response_area.setHtml(
                f"<div style='color:#64748b;font-size:10px;margin-bottom:5px;'>"
                f"<b>QUERY:</b> {q}</div>"
                f"<div style='color:#f59e0b;font-size:11px;font-style:italic;'>⏳ Thinking{race_hint}...</div>"
            )

        # Start timing for instrumentation
        self._current_request_start = time.time()
        self._stage_timings = {"start": self._current_request_start}
        request_metadata = dict(self._pending_request_metadata or {})
        request_metadata.setdefault("request_started_at", self._current_request_start)
        request_metadata.setdefault("origin", s)
        self._pending_request_metadata = None
        request_epoch = self._generation_epoch

        asyncio.run_coroutine_threadsafe(
            self._process_ai(q, s, c, request_epoch, request_metadata), self.loop
        )

    async def _process_ai(self, q, s, c, request_epoch, request_metadata):
        if request_epoch != self._generation_epoch:
            return
        async with self._ai_lock:
            if request_epoch != self._generation_epoch:
                return
            self._last_query = q
            sc = c.get("screen") if c else await self.screen.capture_context()
            if request_epoch != self._generation_epoch:
                return
            if sc:
                self.nexus.push("screen", sc)

            au = c.get("audio") if c else self.audio.get_transcript()
            # Push last transcript fragment specifically if available

            snapshot = self.nexus.get_snapshot()
            if request_epoch != self._generation_epoch:
                return
            await self.ai.generate_response(
                q,
                snapshot,
                screen_context=sc,
                audio_context=au,
                origin=s,
                request_metadata=request_metadata,
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
            )

        asyncio.run_coroutine_threadsafe(_quick_answer_flow(), self.loop)

    def analyze_current_screen(self):
        if not self.session_active:
            return

        self.overlay.update_transcript("Screen captured. Analyzing current screen...")
        if hasattr(self.overlay, "set_analysis_provider_badge"):
            self.overlay.set_analysis_provider_badge(pending=True)
        self._screen_analysis_pending = True
        request_epoch = self._generation_epoch

        async def _capture_and_analyze():
            if request_epoch != self._generation_epoch:
                return
            image_bytes = await self.screen.capture_image_bytes(for_analysis=True)
            if request_epoch != self._generation_epoch:
                return
            if not image_bytes:
                self._screen_analysis_pending = False
                self.overlay.update_transcript("Screen capture failed.")
                if hasattr(self.overlay, "set_analysis_provider_badge"):
                    self.overlay.set_analysis_provider_badge()
                return

            audio_text = self.audio.get_transcript()
            snapshot = self.nexus.get_snapshot()
            ocr_task = asyncio.create_task(self.screen.extract_text_from_image_bytes(image_bytes))
            try:
                # Prefer OCR extracted from the *same* screenshot so "Analyze Screen"
                # is task-directed even when the periodic OCR loop is idle.
                try:
                    ocr_text = await asyncio.wait_for(ocr_task, timeout=3.0)
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

                task = _extract_task(screen_text)
                if task:
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
                    latest_ocr = ocr_text
                    if latest_ocr and request_epoch == self._generation_epoch:
                        self.nexus.push("screen", latest_ocr)
                    return
                except Exception as exc:
                    if request_epoch != self._generation_epoch:
                        return
                    self._screen_analysis_pending = False
                    if hasattr(self.overlay, "set_analysis_provider_badge"):
                        self.overlay.set_analysis_provider_badge()
                    logger.warning(f"Vision analysis exhausted providers: {exc}")
                    if hasattr(self.overlay, "show_error_toast"):
                        self.overlay.show_error_toast(
                            "Image analysis failed — using OCR/text fallback."
                        )
                    self.overlay.update_transcript(
                        "Screen captured, but image analysis failed. Using OCR/text fallback...",
                        state="error",
                    )

                    # Fall back to OCR/text routing (best-effort).
                    try:
                        screen_text_fb = await asyncio.wait_for(ocr_task, timeout=2.0)
                    except Exception:
                        screen_text_fb = ""
                    if screen_text_fb:
                        self.nexus.push("screen", screen_text_fb)
                    self.generate_response(
                        query,
                        "screen_analysis",
                        {"screen": screen_text_fb, "audio": audio_text},
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
                self.generate_response(
                    query if "query" in locals() else "Analyze the current screen and answer using the visible content and live session context.",
                    "screen_analysis",
                    {"screen": screen_text, "audio": audio_text},
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

        # Propagate VAD silence window from the mode profile — live, no restart
        if hasattr(self.audio, "set_vad_silence_ms"):
            self.audio.set_vad_silence_ms(profile.vad_silence_ms)

        logger.info(
            f"🔄 Mode Switched: {mode.upper()} | "
            f"ollama_hint={profile.ollama_model_hint} | "
            f"sensitivity={profile.detector_sensitivity:.2f} | "
            f"vad={profile.vad_silence_ms}ms"
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

    def toggle_audio(self):
        muted = self.audio.toggle()
        self.state.is_muted = muted

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
        st = self.history.get_state()
        self.overlay.update_history_state(*st)
        self.mini_overlay.update_history_state(*st)
        # Auto-expand mini HUD when navigating history
        if self.mini_mode and st[2]:
            self.mini_overlay.on_complete(st[2]["response"], st[2]["query"])

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
                return QProcess.startDetached(sys.executable, sys.argv[1:])
            return QProcess.startDetached(sys.executable, sys.argv)
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
        if hasattr(self.audio, "start"):
            self.audio.start()
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
        self._generation_epoch += 1
        self.ai.cancel()
        self.nexus.clear()
        self.history.start_new_session()
        self.state.is_muted = False
        self._screen_analysis_pending = False
        self.audio.clear()
        self._last_query = ""
        self.session_active = True
        self.state.is_capturing = True

        # Hard-clear the response area so no previous Q&A is visible
        self.overlay._current_query = ""
        self.overlay._raw_buffer = ""
        self.overlay._is_streaming = False
        self.overlay.response_area.clear()

        show_chat_view = getattr(self.overlay, "show_chat_view", None)
        if callable(show_chat_view):
            show_chat_view()
        else:
            self.overlay.stack.setCurrentWidget(self.overlay.chat_view)

        self.overlay.on_complete("", "")
        self.overlay.update_transcript("Listening for context...")
        if hasattr(self.overlay, "set_analysis_provider_badge"):
            self.overlay.set_analysis_provider_badge()
        start_session_ui = getattr(self.overlay, "start_session_ui", None)
        if callable(start_session_ui):
            start_session_ui()
        self.mini_overlay.on_complete("", "")
        self.mini_overlay.set_ready()
        m = self.config.get("ai.mode", "general")
        self.overlay.update_mode(m)
        self.mini_overlay.update_mode(m)
        # P2.10: Pin-to-top is only needed during an active session
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.start(1500)
        logger.info("🚀 New Session Started — slate is clean.")

    def end_session(self):
        """Fully terminate the active session: cancel AI, wipe content, archive history."""
        self._generation_epoch += 1
        self.ai.cancel()
        self._screen_analysis_pending = False
        self.session_active = False
        self.state.is_capturing = False
        self.state.target_window_id = None
        self.nexus.clear()
        self.audio.clear()
        self._last_query = ""

        # P0.5: Reset auto-suggest flag so next session can freely pick presets
        self._context_auto_suggested = False

        # Archive the current session so next start_new_session() begins with a blank slate.
        # This also clears in-memory entries so the history block injected into prompts
        # doesn't carry over previous session context.
        self.history.start_new_session()

        # Clear AI caches so the next session has no stale context
        if hasattr(self.ai, "clear_rag_prefetch"):
            self.ai.clear_rag_prefetch()
        if hasattr(self.ai, "_complexity_cached"):
            self.ai._complexity_cached.cache_clear()

        # Persist the current context to disk before clearing from state,
        # so it reloads on next app launch (user doesn't retype every time).
        if hasattr(self, "_context_store"):
            self._context_store.set_last_context(self.state.session_context)
        # Clear from live state — next session starts clean unless user re-selects
        self.state.session_context = ""
        self.ai.set_session_context("")

        # Hard-wipe the visible response area — user should see a blank screen
        # when they return to standby, and again when next session starts.
        self.overlay._current_query = ""
        self.overlay._raw_buffer = ""
        self.overlay._is_streaming = False
        self.overlay.response_area.clear()

        show_standby_view = getattr(self.overlay, "show_standby_view", None)
        if callable(show_standby_view):
            show_standby_view()
        else:
            self.overlay.stack.setCurrentWidget(self.overlay.standby_view)
        end_session_ui = getattr(self.overlay, "end_session_ui", None)
        if callable(end_session_ui):
            end_session_ui()
        self.overlay.update_transcript("Ready...")
        self.overlay.on_complete("", "")
        if hasattr(self.overlay, "set_analysis_provider_badge"):
            self.overlay.set_analysis_provider_badge()

        # P0.6: Clear the context chip — stale chip from previous session would be misleading
        standby = getattr(self.overlay, "standby_view", None)
        if standby and hasattr(standby, "show_context_chip"):
            standby.show_context_chip(None)

        self.mini_overlay.on_complete("", "")
        self.mini_overlay.set_ready()
        # P2.10: Stop pinning to top while on standby — no need to fight the OS scheduler
        if hasattr(self, "_topmost_timer"):
            self._topmost_timer.stop()
        logger.info("🛑 Session Ended — history archived, slate wiped.")

    def set_current_mode(self, name):
        for m_name, btn in self.mode_buttons.items():
            is_active = m_name == name
            btn.setChecked(is_active)
            btn.setStyleSheet(self.ACTIVE_STYLE if is_active else self.NORMAL_STYLE)

    def toggle_mini_mode(self):
        self.mini_mode = not self.mini_mode
        self.state.is_mini = self.mini_mode
        if self.mini_mode:
            self._sync_history_ui()

    def toggle_stealth_mode(self):
        self.state.is_stealth = True
        if hasattr(self.config, "save"):
            try:
                self.config.save()
            except Exception as e:
                logger.debug(f"Stealth config save skipped: {e}")
        self._apply_ui_only()

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
        if not self.session_active:
            return
        self.nexus.push("audio", t)
        self.overlay.update_transcript(t)
        transcription_metrics = self.audio.get_last_transcription_metrics()
        if transcription_metrics:
            speech_to_transcript_ms = transcription_metrics.get("speech_to_transcript_ms")
            transcribe_only_ms = transcription_metrics.get("transcribe_only_ms")
            if speech_to_transcript_ms is not None:
                logger.info(
                    "Latency audio->transcript | speech->transcript=%.0fms | transcribe=%.0fms | audio=%.0fms",
                    speech_to_transcript_ms,
                    transcribe_only_ms or 0.0,
                    transcription_metrics.get("audio_duration_ms") or 0.0,
                )
            self._pending_request_metadata = {
                **transcription_metrics,
                "transcript_received_at": time.time(),
            }

        # ── Background RAG Prefetch ───────────────────────────────────────────
        # Fire-and-forget: build RAG context from latest audio while user is
        # still speaking/listening — ready before they submit their query.
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
        if q.triggered:
            if q.should_auto_respond():
                self.generate_response(q.detected_text, "speech", {"audio": t})
            else:
                # P2.1: Low-confidence — surface hint without auto-firing
                self.overlay.update_transcript(
                    f"🤔 Possible question detected (tap ⎨ to answer)",
                    state="processing",
                )

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
        """Two-tier parallelized warmup — fast startup by separating critical from deferred.

        Tier 1 — CRITICAL (joined before emitting 'All systems online'):
            • Vision/OCR  — needed for first screen capture
            • Brain/AI    — needed before any query can be answered

        Tier 2 — DEFERRED (start immediately, do NOT block the READY signal):
            • Whisper     — lazy-loaded on first audio; pre-warming adds 20s+ for no gain
            • RAG          — indexes knowledge base quietly in the background

        Result: startup goes from ~30s → ~8s.
        """

        def _warm_vision():
            try:
                self.screen.initialize()
                self.warmup_status_update.emit("👁️ Vision Ready", 40, False)
            except Exception as e:
                logger.error(f"Vision Warmup Fault: {e}")

        def _warm_brain():
            try:
                self.ai.warmup()
                self.ai.ensure_health_monitor(self.loop)
                asyncio.run_coroutine_threadsafe(
                    self._continuous_capture_loop(), self.loop
                )
                self.warmup_status_update.emit("🧠 Brain Ready", 80, False)
            except Exception as e:
                logger.error(f"Brain Warmup Fault: {e}")

        def _warm_audio_deferred():
            """Pre-warm Whisper silently — does NOT block READY signal."""
            try:
                self.audio._ensure_whisper_loaded()
                self.warmup_status_update.emit("🎙️ Audio Ready", 90, False)
            except Exception as e:
                logger.error(f"Audio Warmup Fault: {e}")

        def _warm_knowledge_deferred():
            """Index knowledge base quietly — does NOT block READY signal."""
            try:
                self.rag.add_directory("./knowledge")
                self.warmup_status_update.emit("📚 Knowledge Ready", 95, False)
            except Exception as e:
                logger.error(f"RAG Warmup Fault: {e}")

        # Tier 1: critical — only these are joined before emitting READY
        # Both complete in ~2-3s (Ollama health + basic AI setup)
        critical = [
            threading.Thread(target=_warm_brain,  daemon=True, name="warmup-brain"),
        ]
        # Tier 2: deferred — run in background, don't delay READY
        # EasyOCR takes ~8s, Whisper takes ~8s — both start immediately
        # but the user can already interact before they finish.
        deferred = [
            threading.Thread(target=_warm_vision,             daemon=True, name="warmup-vision"),
            threading.Thread(target=_warm_audio_deferred,     daemon=True, name="warmup-whisper"),
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
        while self.is_running:
            try:
                # Screen context remains active during a session even if audio is muted.
                if self.session_active:
                    # Capture current screen state and push to context
                    text = await self.screen.capture_context()
                    if text:
                        # self._on_screen_text(text)  # Optional: logic for auto-trigger
                        self.nexus.push("screen", text)

                # Dynamic interval based on performance settings (Default: 3s)
                interval = self.config.get("capture.screen.interval_ms", 3000) / 1000.0
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Vision Loop Error: {e}")
                await asyncio.sleep(5)

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
        logger.info("🛑 Shutting down OpenAssist...")
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
