"""
Main overlay window — v4.1 (Layer 4 Hardened).
RESTORED: Markdown Render Debounce (150ms) and Manual Scroll Lock.
FIXED: Connection of transcript and audio-status bridges.
"""

import time
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QFrame,
    QApplication,
    QStackedWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QTextCursor
from ui.markdown_renderer import MarkdownRenderer
from ui.standby_view import StandbyView
from ui.settings_view import SettingsView
from ui.history_feed import HistoryFeedView
from ui.onboarding_wizard import OnboardingWizard
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OverlayWindow(QMainWindow):
    user_query = pyqtSignal(str)

    def __init__(self, config, app):
        super().__init__()
        self.config = config
        self.app = app
        self._drag = False
        self._drag_pos = QPoint()
        self._current_query = ""
        self._is_streaming = False
        self._raw_buffer = ""
        self._user_is_scrolling = False
        self._prev_stack_index = 0
        self._prev_stack_widget = None

        self.md = MarkdownRenderer()
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(150)
        self._render_timer.timeout.connect(self._render_markdown_now)

        # NEURAL UX: Gaze-based transparency
        self._gaze_timer = QTimer(self)
        self._gaze_timer.timeout.connect(self._check_gaze)
        self._gaze_timer.start(100)

        self._build()
        self._connect_state()

        # Session timer
        self._session_start_time = None
        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._update_session_timer)

    def _update_session_timer(self):
        if self._session_start_time:
            elapsed = int(time.time() - self._session_start_time)
            mins, secs = divmod(elapsed, 60)
            self.session_timer.setText(f"{mins:02d}:{secs:02d}")

    def start_session_ui(self):
        """Called when session starts - show timer and end button"""
        self._session_start_time = time.time()
        self.session_timer.setVisible(True)
        self.audio_status.setVisible(True)
        self.btn_end_session.setVisible(True)
        self.btn_history.setVisible(False)
        self.btn_settings.setVisible(False)
        self.set_analysis_provider_badge()
        self.update_audio_state(self.app.state.is_muted)
        self._session_timer.start(1000)

    def end_session_ui(self):
        """Called when session ends - hide timer and end button"""
        self._session_start_time = None
        self.session_timer.setVisible(False)
        self.audio_status.setVisible(False)
        self.btn_end_session.setVisible(False)
        self.btn_history.setVisible(True)
        self.btn_settings.setVisible(True)
        self._session_timer.stop()
        self.session_timer.setText("00:00")
        self.set_analysis_provider_badge()

    def _connect_state(self):
        self.app.state.muted_changed.connect(self.update_audio_state)
        self.app.state.mode_changed.connect(self.update_mode)
        self.app.state.audio_source_changed.connect(self.update_audio_source)
        self.app.state.hud_mode_changed.connect(self._on_hud_mode_changed)
        self.app.state.capturing_changed.connect(self.update_capture_state)

    def _on_hud_mode_changed(self, is_mini):
        if is_mini:
            self.hide()
        else:
            self.show()

    def _build(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(self.config.get("app.opacity", 0.94))

        # Get screen dimensions for full-height window
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            # Width: 400px, Height: from top to taskbar
            # Use setFixedSize to prevent layout-driven expansion beyond screen bounds
            self.setFixedSize(400, geom.height())
            # Position at right edge of screen, respecting taskbar/top offset
            self.move(geom.right() - 400, geom.top())

        self.container = QWidget()
        self.setCentralWidget(self.container)
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.box = QFrame()
        self.box.setObjectName("box")
        self.box.setStyleSheet(
            "#box { background: rgba(12, 12, 25, 250); border: 1px solid rgba(80, 85, 255, 30); border-radius: 14px; }"
        )
        box_layout = QVBoxLayout(self.box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(0)
        layout.addWidget(self.box)

        # Global Style: Hide scrollbars but keep them functional
        self.setStyleSheet("""
            QScrollBar:vertical { width: 0px; background: transparent; }
            QScrollBar:horizontal { height: 0px; background: transparent; }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: transparent; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        """)

        # Header
        self.header = QFrame()
        self.header.setFixedHeight(40)
        self.header.setStyleSheet(
            "background: rgba(25, 25, 45, 255); border-bottom: 1px solid rgba(255,255,255,10);"
        )
        hl = QHBoxLayout(self.header)

        self.title_lbl = QLabel("OPENASSIST AI")
        self.title_lbl.setStyleSheet(
            "color: #a0a0cc; font-weight: bold; font-size: 11px;"
        )
        hl.addWidget(self.title_lbl)

        # Session timer label
        self.session_timer = QLabel("00:00")
        self.session_timer.setStyleSheet(
            "color: #64748b; font-size: 10px; font-family: monospace; background: transparent;"
        )
        self.session_timer.setToolTip("Session Duration")
        self.session_timer.setVisible(False)
        hl.addWidget(self.session_timer)

        hl.addStretch()

        # RESTORATION: Interactive Audio Status Pill in Header
        self.audio_status = QPushButton("🎙️")
        self.audio_status.setToolTip("Mute or unmute the selected session audio capture")
        self.audio_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audio_status.setFixedSize(24, 24)
        self.audio_status.setStyleSheet("""
            QPushButton { 
                color: #4ade80; font-size: 14px; background: transparent; border: none; border-radius: 12px;
            }
            QPushButton:hover { 
                background: rgba(255, 255, 255, 0.05); 
            }
        """)
        self.audio_status.clicked.connect(self.app.toggle_audio)
        self.audio_status.setVisible(False)
        hl.addWidget(self.audio_status)

        self.btn_end_session = QPushButton("⏹")
        self.btn_end_session.setToolTip("End Session")
        self.btn_end_session.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_end_session.setFixedSize(24, 24)
        self.btn_end_session.setStyleSheet("""
            QPushButton { 
                color: #f59e0b; font-size: 12px; background: transparent; border: none; border-radius: 12px;
            }
            QPushButton:hover { 
                background: rgba(255, 255, 255, 0.05); 
            }
        """)
        self.btn_end_session.clicked.connect(self.app.end_session)
        self.btn_end_session.setVisible(False)
        hl.addWidget(self.btn_end_session)

        self.btn_history = QPushButton("📜")
        self.btn_history.setToolTip("History")
        self.btn_history.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_history.setFixedSize(24, 24)
        self.btn_history.setStyleSheet("""
            QPushButton { 
                color: #818cf8; font-size: 12px; background: transparent; border: none; border-radius: 12px;
            }
            QPushButton:hover { 
                background: rgba(255, 255, 255, 0.05); 
            }
        """)
        self.btn_history.clicked.connect(self._show_history)
        hl.addWidget(self.btn_history)

        self._status_snapshot = ""

        btn_set = QPushButton("⚙️")
        btn_set.setToolTip("Settings")
        btn_set.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_set.setStyleSheet(
            "color: #667; border: none; font-size: 14px; background: transparent;"
        )
        btn_set.clicked.connect(self._show_settings)
        self.btn_settings = btn_set
        hl.addWidget(btn_set)

        btn_close = QPushButton("✕")
        btn_close.setStyleSheet(
            "color: #667; border: none; font-size: 14px; background: transparent;"
        )
        btn_close.clicked.connect(self.hide)
        hl.addWidget(btn_close)
        box_layout.addWidget(self.header)

        self.stack = QStackedWidget()
        box_layout.addWidget(self.stack)

        self.standby_view = StandbyView(self)
        self.stack.addWidget(self.standby_view)

        # History Feed (Index 3)
        self.history_feed = HistoryFeedView(self.app.history, self)
        self.stack.addWidget(self.history_feed)

        self.chat_view = QWidget()
        cv_layout = QVBoxLayout(self.chat_view)

        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setStyleSheet(
            "background: transparent; color: #d0d0e8; border: none; font-size: 13px;"
        )
        self.response_area.verticalScrollBar().valueChanged.connect(
            self._on_scroll_changed
        )
        cv_layout.addWidget(self.response_area)

        # RESTORATION: Transcription floating bar
        self.transcript_bar = QFrame()
        self.transcript_bar.setFixedHeight(30)
        self.transcript_bar.setStyleSheet(
            "background: rgba(30,30,50,150); border-radius: 8px; margin: 0 5px;"
        )
        tl = QHBoxLayout(self.transcript_bar)
        self.transcript_lbl = QLabel("Ready...")
        self.transcript_lbl.setStyleSheet(
            "color: #64748b; font-size: 10px; font-style: italic;"
        )
        tl.addWidget(self.transcript_lbl)
        cv_layout.addWidget(self.transcript_bar)

        self.analysis_badge = QLabel("")
        self.analysis_badge.setVisible(False)
        self.analysis_badge.setStyleSheet(
            "background: rgba(56,189,248,0.12); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.30); border-radius: 10px; padding: 5px 9px; margin: 0 5px 2px 5px; font-size: 10px; font-weight: 700;"
        )
        cv_layout.addWidget(self.analysis_badge)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask about the live session or use Analyze Screen...")
        self.input.returnPressed.connect(self._send)
        self.input.setStyleSheet(
            "background: transparent; color: white; border: none; padding: 0 4px; font-size: 12px;"
        )

        self.input_bar = QFrame()
        self.input_bar.setStyleSheet(
            "background: rgba(18,20,38,220); border: 1px solid rgba(99,102,241,0.18); border-radius: 14px; margin: 6px 5px 8px 5px;"
        )
        input_layout = QHBoxLayout(self.input_bar)
        input_layout.setContentsMargins(12, 8, 8, 8)
        input_layout.setSpacing(8)
        input_layout.addWidget(self.input, 1)

        self.btn_analyze_screen = QPushButton("ANALYZE SCREEN")
        self.btn_analyze_screen.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_analyze_screen.setToolTip("Capture the current screen and analyze it with live session context")
        self.btn_analyze_screen.setStyleSheet("""
            QPushButton {
                background: rgba(56, 189, 248, 0.12);
                color: #7dd3fc;
                border: 1px solid rgba(56, 189, 248, 0.28);
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: rgba(56, 189, 248, 0.18);
                color: white;
            }
        """)
        self.btn_analyze_screen.clicked.connect(self._analyze_screen)
        input_layout.addWidget(self.btn_analyze_screen)

        self.btn_send = QPushButton("ASK")
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.setToolTip("Ask about the current live session")
        self.btn_send.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5b4cf1, stop:1 #8b5cf6);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                font-size: 10px;
                font-weight: 900;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: #7367ff;
            }
        """)
        self.btn_send.clicked.connect(self._send)
        input_layout.addWidget(self.btn_send)

        cv_layout.addWidget(self.input_bar)

        self.stack.addWidget(self.chat_view)

        self.settings_view = SettingsView(self.config, self.app)
        self.settings_view.mode_changed.connect(self.update_mode)
        self.settings_view.audio_source_changed.connect(self.update_audio_source)
        self.settings_view.closed.connect(self._on_settings_closed)
        self.stack.addWidget(self.settings_view)

        # Onboarding Wizard (Index 4)
        self.onboarding_wizard = OnboardingWizard(self.config, app=self.app, parent=self)
        self.onboarding_wizard.finished.connect(self._on_onboarding_finished)
        self.onboarding_wizard.skipped.connect(self._on_onboarding_skipped)
        self.stack.addWidget(self.onboarding_wizard)

    def _on_scroll_changed(self, value):
        sb = self.response_area.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 50
        self._user_is_scrolling = not at_bottom

    def _render_markdown_now(self):
        content = self._raw_buffer

        # PERFORMANCE GUARD:
        # Skip render if content hasn't changed enough to justify a full reflow,
        # unless first/last tokens or long delay.
        if hasattr(self, "_last_rendered_content"):
            delta = len(content) - len(self._last_rendered_content)
            if delta < 30 and self._is_streaming:
                # Still start timer for next check
                self._render_timer.start()
                return

        self._last_rendered_content = content

        q_html = (
            f"<div style='color: #64748b; font-size: 10px; margin-bottom: 5px;'><b>QUERY:</b> {self._current_query}</div>"
            if self._current_query
            else ""
        )

        # CPU-Heavy Operation: Markdown -> HTML
        rendered = self.md.render(content)

        # DOM-Heavy Operation: HTML -> Widget
        self.response_area.setHtml(q_html + rendered)

        if not self._user_is_scrolling:
            self.response_area.moveCursor(QTextCursor.MoveOperation.End)

    def append_response(self, text: str):
        if not self._is_streaming:
            self._is_streaming = True
            self._raw_buffer = ""
        self._raw_buffer += text
        self._render_timer.start()

    def on_complete(self, full_text: str, query: str = None):
        self._is_streaming = False
        if query:
            self._current_query = query
        self._raw_buffer = full_text
        self._render_markdown_now()

    def update_warmup_status(self, m, p, r):
        self.standby_view.set_warmup_status(m, p, r)

    def update_status(
        self,
        provider=None,
        capture_audio=False,
        capture_screen=False,
        latency_ms=0,
        available_providers=None,
    ):
        """Update the status bar with current state."""
        parts = []

        # Show current mode
        current_mode = self.app.state.mode if hasattr(self.app, "state") else "general"
        mode_icons = {
            "general": "🧠",
            "interview": "🎯",
            "coding": "💻",
            "meeting": "🤝",
            "exam": "🎓",
            "writing": "✍️",
        }
        parts.append(f"{mode_icons.get(current_mode, '🧠')} {current_mode.upper()}")

        # Show available providers (from app.router or config)
        if available_providers and len(available_providers) > 0:
            parts.append(f"📡 [{', '.join(available_providers)}]")

        if provider:
            parts.append(f"🧠 {provider}")
        if capture_audio:
            parts.append("🎙️")
        if capture_screen:
            parts.append("👁️")
        if latency_ms > 0:
            parts.append(f"{latency_ms / 1000:.1f}s")
        else:
            parts.append("⚡ Ready")

        self._status_snapshot = " | ".join(parts)

    # --- RESTORED BRIDGES ---
    def update_transcript(self, text):
        self.transcript_lbl.setText(text[:80] + ("..." if len(text) > 80 else ""))
        self.transcript_lbl.setStyleSheet(
            "color: #4ade80; font-size: 10px; font-style: normal;"
        )

    def set_analysis_provider_badge(self, provider: str = None, pending: bool = False):
        if pending:
            self.analysis_badge.setText("Analyzing screen...")
            self.analysis_badge.setVisible(True)
            return

        if provider:
            self.analysis_badge.setText(f"Analyzed via {provider.capitalize()}")
            self.analysis_badge.setVisible(True)
            return

        self.analysis_badge.clear()
        self.analysis_badge.setVisible(False)

    def update_audio_state(self, muted):
        self.audio_status.setText("🔇" if muted else "🎙️")
        self.audio_status.setStyleSheet(
            f"color: {'#ef4444' if muted else '#4ade80'}; font-size: 12px;"
        )

        self.refresh_standby_state()

    def update_mode(self, mode):
        self.standby_view.refresh_highlights(mode=mode)

    def update_audio_source(self, source):
        self.standby_view.refresh_highlights(audio=source)

    def update_capture_state(self, capturing):
        """Update UI to reflect if active capture is running."""
        # Simple status text update
        status = "Active" if capturing else "Idle"
        self.update_status(capture_screen=capturing)

    def update_history_state(self, i, t, e=None):
        if not e:
            return
        self._current_query = e.get("query", "")
        self._raw_buffer = e.get("response", "")
        self._is_streaming = False
        self._render_markdown_now()

    def show_standby_view(self):
        self.stack.setCurrentWidget(self.standby_view)

    def show_chat_view(self):
        self.stack.setCurrentWidget(self.chat_view)

    def show_history_view(self):
        self.history_feed.refresh()
        self.stack.setCurrentWidget(self.history_feed)

    def show_settings_view(self):
        self.stack.setCurrentWidget(self.settings_view)

    def _show_history(self):
        """Show full history feed."""
        if self.stack.currentWidget() is self.history_feed:
            self.show_standby_view()
            return
        self._prev_stack_index = self.stack.currentIndex()
        self._prev_stack_widget = self.stack.currentWidget()
        self.show_history_view()

    def _show_settings(self):
        """Show settings and remember where we came from."""
        if self.stack.currentWidget() is self.settings_view:
            self.show_standby_view()
            return
        self._prev_stack_index = self.stack.currentIndex()
        self._prev_stack_widget = self.stack.currentWidget()
        self.show_settings_view()

    def _on_settings_closed(self):
        """Return to previous view."""
        self.refresh_standby_state()
        if self._prev_stack_widget is not None:
            self.stack.setCurrentWidget(self._prev_stack_widget)
        else:
            self.stack.setCurrentIndex(self._prev_stack_index)

    def _on_onboarding_finished(self):
        """Handle onboarding completion - go to standby."""
        self.show_standby_view()
        self.refresh_standby_state()

    def _on_onboarding_skipped(self):
        """Handle onboarding skip - go to standby."""
        self.show_standby_view()
        self.refresh_standby_state()

    def show_onboarding(self):
        """Show the onboarding wizard, resetting it to step 0."""
        self._prev_stack_index = 0
        self._prev_stack_widget = self.standby_view
        if hasattr(self, "onboarding_wizard"):
            self.onboarding_wizard.reset()
        self.stack.setCurrentWidget(self.onboarding_wizard)

    def refresh_standby_state(self, mode=None, audio=None):
        self.standby_view.refresh_highlights(mode=mode, audio=audio)

    def _send(self):
        q = self.input.text().strip()
        if q:
            self.input.clear()
            self._user_is_scrolling = False
            self.user_query.emit(q)

    def _analyze_screen(self):
        if hasattr(self.app, "analyze_current_screen"):
            self._user_is_scrolling = False
            self.app.analyze_current_screen()

    def set_click_through(self, enabled: bool):
        """Toggle mouse interaction transparency."""
        if enabled:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
            self.box.setStyleSheet(
                "#box { background: rgba(12, 12, 25, 100); border: 1px solid rgba(80, 85, 255, 10); border-radius: 14px; }"
            )
            self.response_area.setStyleSheet(
                "background: transparent; color: rgba(208, 208, 232, 150); border: none; font-size: 13px;"
            )
        else:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)
            self.box.setStyleSheet(
                "#box { background: rgba(12, 12, 25, 250); border: 1px solid rgba(80, 85, 255, 30); border-radius: 14px; }"
            )
            self.response_area.setStyleSheet(
                "background: transparent; color: #d0d0e8; border: none; font-size: 13px;"
            )

        # Windows requires a hide/show to refresh flags immediately
        # Use app.mini_mode check to prevent showing the hidden "Max" window when in mini mode
        if not self.app.mini_mode and self.isVisible():
            self.show()

    def scroll_up(self):
        sb = self.response_area.verticalScrollBar()
        sb.setValue(sb.value() - 60)

    def scroll_down(self):
        sb = self.response_area.verticalScrollBar()
        sb.setValue(sb.value() + 60)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(e.position().toPoint())
            if child in [self.box, self.header] or child is None:
                self._drag = True
                self._drag_pos = (
                    e.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )

    def mouseMoveEvent(self, e):
        if self._drag:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag = False

    def showEvent(self, e):
        super().showEvent(e)
        if hasattr(self.app, "_apply_window_effects"):
            self.app._apply_window_effects(self)
        if hasattr(self.app, "hotkeys"):
            self.app.hotkeys.reset_state()

    def hideEvent(self, e):
        super().hideEvent(e)
        if hasattr(self.app, "hotkeys"):
            self.app.hotkeys.reset_state()

    def _check_gaze(self):
        """Dynamic Gaze: Fades the window if the mouse is close to allow viewing content underneath.

        Only active during an active session - not on standby/settings screens.
        Can be disabled via config 'app.gaze_fade.enabled'.
        """
        # Check if gaze fade is enabled in config
        if not self.config.get("app.gaze_fade.enabled", False):
            return

        if getattr(self.app.state, "is_stealth", False):
            return

        # Only fade during active session - not on standby/settings screens
        if not self.isVisible() or self.app.mini_mode:
            return

        # Check if session is active - only fade when user is in a running session
        if not getattr(self.app, "session_active", False):
            return

        cursor_pos = self.mapFromGlobal(self.cursor().pos())
        dist_x = min(abs(cursor_pos.x()), abs(cursor_pos.x() - self.width()))
        dist_y = min(abs(cursor_pos.y()), abs(cursor_pos.y() - self.height()))

        inside = self.rect().contains(cursor_pos)
        margin = self.config.get("app.gaze_fade.margin", 60)
        target_opa = self.config.get("app.opacity", 0.94)

        if inside or (dist_x < margin and dist_y < margin):
            target_opa = self.config.get("app.gaze_fade.target_opacity", 0.12)

        current_opa = self.windowOpacity()
        if abs(current_opa - target_opa) > 0.01:
            # Smooth transition
            new_opa = current_opa + (target_opa - current_opa) * 0.3
            self.setWindowOpacity(new_opa)
