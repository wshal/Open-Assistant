"""
OpenAssist AI v4.1 — Main Application Controller (Midnight Hardened).
RESTORED: Smooth Gliding (60fps / 5px steps), HUD navigation, and click-through.
FIXED: Standby warmup signal bridge and non-blocking hardware hot-apply.
RESTORATION: Automatic Knowledge Sync (RAG) during warmup.
"""

import sys
import asyncio
import threading
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.config import Config
from core.hotkeys import HotkeyManager
from capture.screen import ScreenCapture
from capture.audio import AudioCapture
from capture.ocr import OCREngine
from ai.engine import AIEngine
from ai.history import ResponseHistory
from ai.rag import RAGEngine
from ui.overlay import OverlayWindow
from ui.mini_overlay import MiniOverlay
from stealth.anti_detect import StealthManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAssistApp(QObject):
    warmup_status_update = pyqtSignal(str, int, bool)

    def __init__(self, config: Config, mini_mode: bool = False):
        super().__init__()
        self.config = config
        self.mini_mode = mini_mode
        self.is_running = True
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        
        # Components
        self.history = ResponseHistory()
        self.rag = RAGEngine(config)
        self.ai = AIEngine(config, self.history, self.rag)
        self.ocr = OCREngine(config)
        self.screen = ScreenCapture(config, self.ocr)
        self.audio = AudioCapture(config)
        self.stealth = StealthManager(config)
        
        self.session_active = False
        self._last_query = ""
        self._click_through = False
        
        # Async Loop
        self.loop = asyncio.new_event_loop()
        self._ai_lock_ready = threading.Event()
        self._async_thread = threading.Thread(target=self._run_master_loop, daemon=True)
        
        # UI
        self.overlay = OverlayWindow(config, self)
        self.mini_overlay = MiniOverlay(config, self)
        self.hotkeys = HotkeyManager(config, self)

        self._wire_signals()
        
        # MOVEMENT ENGINE: RESTORED Smooth Glide (60fps)
        self._move_timer = QTimer(self)
        self._move_direction = None
        self._move_timer.timeout.connect(self._do_move)

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
        self.ai.response_complete.connect(lambda t: self.overlay.on_complete(t, self._last_query))
        self.ai.response_complete.connect(lambda t: self.mini_overlay.on_complete(t, self._last_query))
        self.ai.error_occurred.connect(lambda e: self.overlay.on_complete(f"ERROR: {e}"))
        self.ai.error_occurred.connect(lambda e: self.mini_overlay.show_error(e))
        self.audio.transcription_ready.connect(self._on_transcription)
        self.screen.text_captured.connect(self._on_screen_text)
        self.warmup_status_update.connect(self.overlay.update_warmup_status)
        self.ai.provider_status.connect(self.overlay.standby_view.set_provider_statuses)
        self.overlay.standby_view.start_clicked.connect(self.start_new_session)
        self.overlay.standby_view.mode_selected.connect(self.switch_mode)
        self.overlay.standby_view.audio_source_changed.connect(self._on_audio_source_ui_change)

    def _on_audio_source_ui_change(self, source):
        self.config.set("capture.audio.mode", source)
        self._apply_settings()

    # --- 🛠️ FLUID HUD NAVIGATION ---

    def move_up(self): self._nudge(0, -5)
    def move_down(self): self._nudge(0, 5)
    def move_left(self): self._nudge(-5, 0)
    def move_right(self): self._nudge(5, 0)

    def _nudge(self, dx, dy):
        v = self.mini_overlay if self.mini_mode else self.overlay
        pos = v.pos()
        v.move(pos.x() + dx, pos.y() + dy)

    def start_move(self, direction):
        if self._move_direction == direction: return
        self._move_direction = direction
        self._move_timer.start(16) 

    def stop_move(self):
        self._move_timer.stop()
        self._move_direction = None

    def _do_move(self):
        if not self._move_direction: return
        d = self._move_direction
        if d == "up": self.move_up()
        elif d == "down": self.move_down()
        elif d == "left": self.move_left()
        elif d == "right": self.move_right()

    def toggle_click_through(self):
        self._click_through = not self._click_through
        self.overlay.set_click_through(self._click_through)
        self.mini_overlay.set_click_through(self._click_through)

    # --- 🛠️ CONTROL BRIDGES ---

    def _apply_settings(self):
        """Non-blocking hot-apply for Layer 6 stabilization. Added Master Safety Guard."""
        logger.info("⚙️ Applying Settings (Background Thread)...")
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
                    asyncio.run_coroutine_threadsafe(self.ai.poll_provider_health_loop(), self.loop)
                
                logger.info("✅ Settings Applied Successfully.")
            except Exception as e:
                logger.error(f"❌ Settings Hot-Apply Fault (Handled): {e}")
        
        # Spawn daemon thread to keep UI interactive
        t = threading.Thread(target=_apply, daemon=True)
        t.start()

    def _apply_ui_only(self):
        opa = self.config.get("app.opacity", 0.94)
        self.overlay.setWindowOpacity(opa)
        self.mini_overlay.setWindowOpacity(opa)
        is_stealth = self.config.get("stealth.enabled", False)
        self.stealth.apply_to_window(self.overlay, is_stealth)
        self.stealth.apply_to_window(self.mini_overlay, is_stealth)
        self.overlay.update_audio_state(self.audio._muted)
        self.mini_overlay.update_audio_state(self.audio._muted)

    def run(self) -> int:
        self._async_thread.start()
        self.audio.start()
        self.hotkeys.start()
        self.overlay.show()
        self._background_warmup()
        return self.qt_app.exec()

    def generate_response(self, q, s="manual", c=None):
        if not q or not self._ai_lock_ready.wait(timeout=2): return
        asyncio.run_coroutine_threadsafe(self._process_ai(q, s, c), self.loop)

    async def _process_ai(self, q, s, c):
        async with self._ai_lock:
            self._last_query = q
            sc = c.get("screen") if c else self.screen.capture_context()
            au = c.get("audio") if c else self.audio.get_transcript()
            await self.ai.generate_response(q, sc, au)

    def toggle_overlay(self):
        v = self.mini_overlay if self.mini_mode else self.overlay
        if v.isVisible(): v.hide()
        else: v.show(); v.raise_(); v.activateWindow()

    def quick_answer(self): self.generate_response("Summarize current context.", "quick")
    
    def switch_mode(self, mode: str = None):
        if not mode:
            modes = ["general", "interview", "coding", "meeting", "exam", "writing"]
            mode = modes[(modes.index(self.config.get("ai.mode", "general")) + 1) % len(modes)]
        self.config.set("ai.mode", mode)
        self.overlay.update_mode(mode)
        self.mini_overlay.update_mode(mode)
        logger.info(f"🔄 Mode Switched: {mode.upper()}")

    def toggle_audio(self): 
        muted = self.audio.toggle()
        self.overlay.update_audio_state(muted)
        self.mini_overlay.update_audio_state(muted)

    def scroll_up(self): 
        self.overlay.response_area.verticalScrollBar().setValue(self.overlay.response_area.verticalScrollBar().value() - 60)
        self.mini_overlay.scroll_up()

    def scroll_down(self): 
        self.overlay.response_area.verticalScrollBar().setValue(self.overlay.response_area.verticalScrollBar().value() + 60)
        self.mini_overlay.scroll_down()

    def history_prev(self): self.history.move_prev(); self._sync_history_ui()
    def history_next(self): self.history.move_next(); self._sync_history_ui()
    def _sync_history_ui(self):
        st = self.history.get_state()
        self.overlay.update_history_state(*st)
        self.mini_overlay.update_history_state(*st)
        # Auto-expand mini HUD when navigating history
        if self.mini_mode and st[2]:
            self.mini_overlay.on_complete(st[2].response, st[2].query)

    def emergency_erase(self): 
        self.history.clear(); self.overlay.hide()
        self.audio.stop()
        self.hotkeys.stop()
        QTimer.singleShot(500, self.qt_app.quit)

    def start_new_session(self): 
        self.session_active = True
        self.overlay.stack.setCurrentIndex(1)
        m = self.config.get("ai.mode", "general")
        self.overlay.update_mode(m)
        self.mini_overlay.update_mode(m)
        logger.info("🚀 Session Started.")

    def end_session(self): self.session_active = False; self.overlay.stack.setCurrentIndex(0)
    def toggle_mini_mode(self):
        if self.mini_mode: self.mini_overlay.hide(); self.overlay.show()
        else: self.overlay.hide(); self.mini_overlay.show()
        self.mini_mode = not self.mini_mode

    def toggle_stealth_mode(self):
        self.config.set("stealth.enabled", not self.config.get("stealth.enabled", False))
        self._apply_settings()

    def _on_transcription(self, t):
        self.overlay.update_transcript(t)
        q = self.ai.detector.detect(t)
        if q: self.generate_response(q, "speech", {"audio": t})

    def _on_screen_text(self, t):
        q = self.ai.detector.detect(t)
        if q: self.generate_response(q, "auto", {"screen": t})

    def _background_warmup(self):
        def _warm():
            try:
                self.warmup_status_update.emit("🎙️ Audio...", 10, False)
                self.audio._ensure_whisper_loaded()
                
                self.warmup_status_update.emit("📚 Knowledge...", 30, False)
                self.rag.add_directory("./knowledge")
                
                self.warmup_status_update.emit("👁️ Vision...", 50, False)
                self.screen.initialize()
                
                self.warmup_status_update.emit("🧠 Brain...", 80, False)
                self.ai.warmup()
                asyncio.run_coroutine_threadsafe(self.ai.poll_provider_health_loop(), self.loop)
                
                self.warmup_status_update.emit("✅ READY", 100, True)
            except Exception as e:
                logger.error(f"Warmup Failed: {e}")
        threading.Thread(target=_warm, daemon=True).start()
