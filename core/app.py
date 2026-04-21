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
from utils.platform_utils import ProcessUtils, WindowUtils
from ui.overlay import OverlayWindow
from ui.mini_overlay import MiniOverlay
from core.tray import SystemTray
from stealth.anti_detect import StealthManager
from stealth.input_simulator import InputSimulator
from utils.logger import setup_logger
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
        self.ai = AIEngine(config, self.history, self.rag)
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

        # MOVEMENT ENGINE: RESTORED Smooth Glide (60fps)
        self._move_timer = QTimer(self)
        self._move_direction = None
        self._move_timer.timeout.connect(self._do_move)

        # NEXUS ENGINE: Periodic Window Polling (3s)
        self._nexus_timer = QTimer(self)
        self._nexus_timer.timeout.connect(self._poll_nexus_context)
        self._nexus_timer.start(3000)

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
        if entries:
            entry = entries[-1]
            latency_ms = getattr(entry, "latency", 0)
            provider = getattr(entry, "provider", None)

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
        )
        if self._screen_analysis_pending:
            via = provider.upper() if provider else "VISION"
            self.overlay.update_transcript(f"Screen captured and analyzed via {via}.")
            if hasattr(self.overlay, "set_analysis_provider_badge"):
                self.overlay.set_analysis_provider_badge(provider=provider)
            self._screen_analysis_pending = False

    def _on_ai_error(self, error_text: str):
        if self._screen_analysis_pending:
            self.overlay.update_transcript("Screen captured, but analysis failed.")
            if hasattr(self.overlay, "set_analysis_provider_badge"):
                self.overlay.set_analysis_provider_badge()
            self._screen_analysis_pending = False

    def _on_audio_source_ui_change(self, source):
        self.state.audio_source = source
        self._apply_settings()

    def _sync_state_from_config(self):
        """Pull persisted config back into AppState before async subsystems catch up."""
        self.state.mode = self.config.get("ai.mode", "general")
        self.state.audio_source = self.config.get("capture.audio.mode", "system")
        self.state.is_stealth = self.config.get("stealth.enabled", False)

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
        self.overlay.refresh_standby_state()

        def _apply():
            try:
                # 1. Hardware Cooldown
                self.audio.restart()
                self.hotkeys.restart()

                # 2. Process Warmup
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
        self.stealth.apply_to_window(window, is_stealth)

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

    def _show_active_overlay(self):
        view = self._active_view()
        view.show()
        if hasattr(view, "raise_"):
            view.raise_()
        if hasattr(view, "activateWindow"):
            view.activateWindow()
        return view

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
        self.overlay.show()
        self.overlay.raise_()
        self.overlay.activateWindow()

    def _show_initial_window(self):
        if not self.config.get("onboarding.completed", False):
            self.overlay.show()
            self.overlay.raise_()
            self.overlay.activateWindow()
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
        # We only do this for manual/speech queries, not background auto-captures
        if s in ["manual", "speech", "quick"]:
            hwnd = self.simulator.get_foreground_window()
            # Safety: Don't lock to our own overlay
            if hwnd:
                self.state.target_window_id = hwnd
                logger.info(f"🔒 Snap-Lock: Target set to HWND {hwnd}")

        # Start timing for instrumentation
        self._current_request_start = time.time()
        self._stage_timings = {"start": self._current_request_start}
        request_epoch = self._generation_epoch

        asyncio.run_coroutine_threadsafe(
            self._process_ai(q, s, c, request_epoch), self.loop
        )

    async def _process_ai(self, q, s, c, request_epoch):
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
            )

    def toggle_overlay(self):
        v = self.mini_overlay if self.mini_mode else self.overlay
        if not v.isVisible() or v.windowOpacity() <= 0:
            if self._click_through:
                self.toggle_click_through()
            v.show()
            v.raise_()
            v.activateWindow()
            logger.debug("👁️ HUD Shown via toggle.")
            return

        if not self._click_through:
            self.toggle_click_through()
            logger.debug("🖱️ HUD switched to click-through mode.")
            return

        v.hide()
        logger.debug("👻 HUD Hidden via toggle.")

    def quick_answer(self):
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
            image_bytes = await self.screen.capture_image_bytes()
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
            ocr_task = asyncio.create_task(
                self.screen.extract_text_from_image_bytes(image_bytes)
            )
            try:
                screen_text = snapshot.get("latest_ocr", "")
                await self.ai.analyze_image_response(
                    "Analyze the current screen and answer using the visible content, recent audio, and live session context.",
                    image_bytes,
                    snapshot,
                    screen_context=screen_text,
                    audio_context=audio_text,
                )
                latest_ocr = await ocr_task
                if latest_ocr and request_epoch == self._generation_epoch:
                    self.nexus.push("screen", latest_ocr)
            except Exception:
                if request_epoch != self._generation_epoch:
                    return
                screen_text = await ocr_task
                if screen_text:
                    self.nexus.push("screen", screen_text)
                self.generate_response(
                    "Analyze the current screen and answer using the visible content, recent audio, and live session context.",
                    "screen_analysis",
                    {"screen": screen_text, "audio": audio_text},
                )

        asyncio.run_coroutine_threadsafe(_capture_and_analyze(), self.loop)

    def switch_mode(self, mode: str = None):
        if not mode:
            modes = ["general", "interview", "coding", "meeting", "exam", "writing"]
            mode = modes[
                (modes.index(self.config.get("ai.mode", "general")) + 1) % len(modes)
            ]
        self.state.mode = mode
        self.overlay.update_mode(mode)
        self.mini_overlay.update_mode(mode)
        
        # Propagate to detector for mode-aware sensitivity
        ai = self.__dict__.get("ai")
        if ai and hasattr(ai, "detector"):
            ai.detector.set_mode(mode)
            
        logger.info(f"🔄 Mode Switched: {mode.upper()}")

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
        if hasattr(self.audio, "start"):
            self.audio.start()
        if hasattr(self.hotkeys, "start"):
            self.hotkeys.start()
        self._background_warmup()
        self.overlay.show()
        self.overlay.raise_()
        self.overlay.activateWindow()
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
        logger.info("🚀 Session Started.")

    def end_session(self):
        self._generation_epoch += 1
        self.ai.cancel()
        self._screen_analysis_pending = False
        self.session_active = False
        self.state.is_capturing = False
        self.state.target_window_id = None
        self.nexus.clear()
        self.audio.clear()
        self._last_query = ""
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
        self.mini_overlay.on_complete("", "")
        self.mini_overlay.set_ready()

    def set_current_mode(self, name):
        for m_name, btn in self.mode_buttons.items():
            is_active = m_name == name
            btn.setChecked(is_active)
            btn.setStyleSheet(self.ACTIVE_STYLE if is_active else self.NORMAL_STYLE)

    def toggle_mini_mode(self):
        self.mini_mode = not self.mini_mode
        self.state.is_mini = self.mini_mode

    def toggle_stealth_mode(self):
        self.state.is_stealth = not self.state.is_stealth
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

    def _on_transcription(self, t):
        if not self.session_active:
            return
        self.nexus.push("audio", t)
        self.overlay.update_transcript(t)
        q = self.ai.detector.detect(t)
        if q:
            self.generate_response(q, "speech", {"audio": t})

    def _on_screen_text(self, t):
        if not self.session_active:
            return
        q = self.ai.detector.detect(t)
        if q:
            self.generate_response(q, "auto", {"screen": t})

    def _background_warmup(self):
        """Parallelized component warmup — v5.1 (Speed Focus)."""

        def _warm_audio():
            try:
                self.audio._ensure_whisper_loaded()
                self.warmup_status_update.emit("🎙️ Audio Ready", 25, False)
            except Exception as e:
                logger.error(f"Audio Warmup Fault: {e}")

        def _warm_vision():
            try:
                self.screen.initialize()
                self.warmup_status_update.emit("👁️ Vision Ready", 50, False)
            except Exception as e:
                logger.error(f"Vision Warmup Fault: {e}")

        def _warm_knowledge():
            try:
                self.rag.add_directory("./knowledge")
                self.warmup_status_update.emit("📚 Knowledge Ready", 75, False)
            except Exception as e:
                logger.error(f"RAG Warmup Fault: {e}")

        def _warm_brain():
            try:
                self.ai.warmup()
                self.ai.ensure_health_monitor(self.loop)
                # Kicks off continuous capture once brain is ready
                asyncio.run_coroutine_threadsafe(
                    self._continuous_capture_loop(), self.loop
                )
            except Exception as e:
                logger.error(f"Brain Warmup Fault: {e}")

        # Dispatch all in parallel
        threads = [
            threading.Thread(target=_warm_audio, daemon=True),
            threading.Thread(target=_warm_vision, daemon=True),
            threading.Thread(target=_warm_knowledge, daemon=True),
            threading.Thread(target=_warm_brain, daemon=True),
        ]
        for t in threads:
            t.start()

        # Monitoring thread for final Ready signal
        def _monitor():
            try:
                for t in threads:
                    t.join()
                # Layer 6 Safety: Check if 'self' still exists in Qt land
                self.warmup_status_update.emit("✅ READY", 100, True)
                logger.info("🚀 All systems online - Initialization Complete.")
            except (RuntimeError, AttributeError):
                logger.warning("Monitor: UI context lost during warmup. Skipping signal.")

        threading.Thread(target=_monitor, daemon=True).start()

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
