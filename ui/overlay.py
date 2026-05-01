"""
Main overlay window - v4.1 (Layer 4 Hardened).
RESTORED: Markdown Render Debounce (150ms) and Manual Scroll Lock.
FIXED: Connection of transcript and audio-status bridges.
P0.1 FIX: Removed duplicate CRLF class definition artifact.
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
from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor
from ui.markdown_renderer import MarkdownRenderer
from ui.standby_view import StandbyView
from ui.settings_view import SettingsView
from ui.history_feed import HistoryFeedView
from ui.onboarding_wizard import OnboardingWizard
from ui.nexus_timeline import NexusTimelineView  # P2.9
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

        # ── Streaming render strategy ──────────────────────────────────────
        # During streaming we append raw text directly to QTextEdit for
        # zero-latency display. A periodic 300ms timer re-renders the
        # accumulated buffer as Markdown without blocking chunk display.
        # On completion we do a final full markdown render.
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(300)          # periodic during stream
        self._render_timer.timeout.connect(self._render_markdown_now)
        self._analyze_timer = QTimer(self)
        self._analyze_timer.setInterval(220)
        self._analyze_timer.timeout.connect(self._tick_analyze_button)
        self._analyze_success_timer = QTimer(self)
        self._analyze_success_timer.setSingleShot(True)
        self._analyze_success_timer.setInterval(1800)
        self._analyze_success_timer.timeout.connect(self._reset_analyze_button_idle)
        self._analyze_frames = [
            "ANALYZE SCREEN",
            "ANALYZING SCREEN.",
            "ANALYZING SCREEN..",
            "ANALYZING SCREEN...",
        ]
        self._analyze_frame_index = 0

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
        self.vision_status.setVisible(True)
        self.update_vision_state(reset_to_master=True)
        self.btn_end_session.setVisible(True)
        self.btn_history.setVisible(False)
        self.btn_settings.setVisible(False)
        # Reduce header clutter during an active session.
        if hasattr(self, "btn_timeline"):
            self.btn_timeline.setVisible(False)
        self.set_analysis_provider_badge()
        self.update_audio_state(self.app.state.is_muted)
        self._session_timer.start(1000)

    def end_session_ui(self):
        """Called when session ends - hide timer, reset standby view to ready state."""
        self._session_start_time = None
        self.session_timer.setVisible(False)
        self.audio_status.setVisible(False)
        self.vision_status.setVisible(False)
        self.btn_end_session.setVisible(False)
        self.btn_history.setVisible(True)
        self.btn_settings.setVisible(True)
        if hasattr(self, "btn_timeline"):
            self.btn_timeline.setVisible(True)
        self._session_timer.stop()
        self.session_timer.setText("00:00")
        self.set_analysis_provider_badge()

        # Reset standby subtitle so user knows the slate is clean and ready
        if hasattr(self, "standby_view"):
            sv = self.standby_view
            sv.subtitle.setText("SESSION ENDED — READY TO START")
            sv.start_btn.setText("START NEW SESSION")
            sv.start_btn.setEnabled(True)
            sv.start_btn.setStyleSheet(sv.START_BUTTON_READY_STYLE)
            sv.progress_bar.setValue(100)

    def _connect_state(self):
        self.app.state.muted_changed.connect(self.update_audio_state)
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
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(self.config.get("app.opacity", 0.94))

        # Get screen dimensions for full-height window
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            # Width: 400px, Height: from top to taskbar
            # Use setFixedSize to prevent layout-driven expansion beyond screen bounds
            self.setFixedSize(400, geom.height())
            # Restore saved position (Q9); fall back to right edge of screen
            saved_x = self.config.get("app.overlay_x", None)
            saved_y = self.config.get("app.overlay_y", None)
            if saved_x is not None and saved_y is not None:
                # Clamp to ensure it stays on screen after monitor changes
                safe_x = max(geom.left(), min(int(saved_x), geom.right() - 400))
                safe_y = max(geom.top(), min(int(saved_y), geom.bottom() - 100))
                self.move(safe_x, safe_y)
                logger.debug("[Q9 Position] Restored overlay to (%d, %d)", safe_x, safe_y)
            else:
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

        # PHASE 1: Interactive Vision Kill Switch (Placed BEFORE mic)
        self.vision_status = QPushButton("👁️")
        self.vision_status.setToolTip("Toggle Vision (Screen Capture + OCR) — when OFF the AI answers from audio only, no screenshots are taken")
        self.vision_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.vision_status.setFixedSize(24, 24)
        self.vision_status.setStyleSheet("""
            QPushButton { 
                color: #4ade80; font-size: 14px; background: transparent; border: none; border-radius: 12px;
            }
            QPushButton:hover { 
                background: rgba(255,255,255,12); 
            }
        """)
        self.vision_status.clicked.connect(self._toggle_vision)
        self.vision_status.setVisible(False)
        hl.addWidget(self.vision_status)

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
                background: rgba(255,255,255,12); 
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
                background: rgba(255,255,255,12); 
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
                background: rgba(255,255,255,12); 
            }
        """)
        self.btn_history.clicked.connect(self._show_history)

        # P2.9: Timeline button
        self.btn_timeline = QPushButton("⏱")
        self.btn_timeline.setToolTip("Context Timeline — see what the AI saw")
        self.btn_timeline.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_timeline.setFixedSize(24, 24)
        self.btn_timeline.setStyleSheet("""
            QPushButton { background: rgba(255,255,255,12); color: #aaa; border: none; border-radius: 12px; font-size: 13px; }
            QPushButton:hover { background: rgba(255,255,255,25); color: #c0c0ff; }
        """)
        self.btn_timeline.clicked.connect(self._show_timeline)
        hl.addWidget(self.btn_timeline)
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
        self.standby_view.mode_selected.connect(self._on_standby_mode_selected)
        self.standby_view.audio_source_changed.connect(self._on_standby_audio_source_changed)
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
            "background: rgba(56,189,248,30); color: #7dd3fc; border: 1px solid rgba(56,189,248,76); border-radius: 10px; padding: 5px 9px; margin: 0 5px 2px 5px; font-size: 10px; font-weight: 700;"
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
            "background: rgba(18,20,38,220); border: 1px solid rgba(99,102,241,45); border-radius: 14px; margin: 6px 5px 8px 5px;"
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
                background: rgba(56,189,248,30);
                color: #7dd3fc;
                border: 1px solid rgba(56,189,248,71);
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 10px;
                font-weight: 800;
            }
            QPushButton:hover {
                background: rgba(56,189,248,45);
                color: white;
            }
            QPushButton:disabled {
                background: rgba(56,189,248,25);
                color: rgba(125,211,252,183);
                border: 1px solid rgba(56,189,248,45);
            }
        """)
        self.btn_analyze_screen.clicked.connect(self._analyze_screen)
        input_layout.addWidget(self.btn_analyze_screen)

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

        # P2.9: Context Timeline (last in stack)
        self.timeline_view = NexusTimelineView(self.app.nexus, self)
        self.stack.addWidget(self.timeline_view)

    def _on_scroll_changed(self, value):
        sb = self.response_area.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 50
        self._user_is_scrolling = not at_bottom

    def _render_markdown_now(self):
        """Full markdown re-render of the accumulated buffer."""
        content = self._raw_buffer
        if not content:
            return

        # Avoid full HTML rebuild during streaming when fenced code blocks are present.
        # We'll do the final markdown render on completion instead.
        if self._is_streaming and "```" in content:
            return

        self._last_rendered_content = content

        # GAP 5: Inject source badge into query header (set by on_complete).
        # Only shown after completion — empty string during streaming.
        badge = getattr(self, "_source_badge", "")
        q_html = (
            f"<div style='color: #64748b; font-size: 10px; margin-bottom: 5px;'>"
            f"<b>QUERY:</b> {self._current_query}{badge}</div>"
            if self._current_query
            else ""
        )
        rendered = self.md.render(content)
        self.response_area.setHtml(q_html + rendered)

        if not self._user_is_scrolling:
            self.response_area.moveCursor(QTextCursor.MoveOperation.End)

    def append_response(self, text: str):
        """Called on every streaming chunk from AIEngine.response_chunk.

        Strategy:
        - Append the raw chunk directly to the QTextEdit for zero-latency display.
          The user sees every token the moment it arrives from the provider.
        - Accumulate in _raw_buffer for the periodic Markdown re-render (300ms).
        - On the very first chunk, clear the widget and start the periodic timer.
        """
        if not self._is_streaming:
            # First chunk: switch into streaming mode
            self._is_streaming = True
            self._raw_buffer = ""
            # GAP 2: Clear the deferred-Thinking flag so the 80ms QTimer
            # singleShot won't overwrite our content with "Thinking...".
            self._pending_thinking = False
            self.response_area.clear()
            # Show query header immediately so user sees their question reflected
            if self._current_query:
                self.response_area.setHtml(
                    f"<div style='color: #64748b; font-size: 10px; margin-bottom: 5px;'>"
                    f"<b>QUERY:</b> {self._current_query}</div>"
                )
            # Start periodic markdown re-render every 300ms
            self._render_timer.start()

        self._raw_buffer += text

        # Stop periodic markdown re-renders once a code fence appears to prevent
        # layout wobble while code blocks stream in.
        if self._render_timer.isActive() and "```" in self._raw_buffer:
            self._render_timer.stop()

        # Immediate plaintext append — character-level, zero lag.
        # Use QTextCharFormat to keep text in the correct colour (#d0d0e8)
        # even after a prior setHtml() call switched the document to rich-text mode.
        cursor = self.response_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#d0d0e8"))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.response_area.setTextCursor(cursor)

        if not self._user_is_scrolling:
            self.response_area.moveCursor(QTextCursor.MoveOperation.End)

    def on_complete(self, full_text: str, query: str = None, cache_tier: int = 0, provider: str = ""):
        """Called when streaming finishes. Does a final full Markdown render.

        GAP 5: Prepends a source badge to the query header so the user always
        knows whether the answer came from cache (and which tier) or a live
        provider call.
        """
        self._render_timer.stop()    # stop the periodic timer
        self._is_streaming = False
        self._pending_thinking = False  # GAP 2: always clear on completion
        if query:
            self._current_query = query
        self._raw_buffer = full_text

        # GAP 5: Build source badge for the response header
        if cache_tier and cache_tier > 0:
            tier_labels = {1: ("T1", "#4ade80", "⚡"), 2: ("T2", "#818cf8", "🔮"),
                           3: ("T3", "#a78bfa", "🧠"), 4: ("T4", "#94a3b8", "≈")}
            label, colour, icon = tier_labels.get(cache_tier, ("T?", "#94a3b8", "≈"))
            badge = (
                f"<span style='background:rgba(74,222,128,0.12);color:{colour};"
                f"font-size:9px;font-weight:700;padding:1px 5px;border-radius:4px;"
                f"margin-left:6px;'>{icon} Cache {label}</span>"
            )
        elif provider:
            safe_p = provider[:12]
            badge = (
                f"<span style='background:rgba(99,102,241,0.12);color:#818cf8;"
                f"font-size:9px;font-weight:700;padding:1px 5px;border-radius:4px;"
                f"margin-left:6px;'>🌐 {safe_p}</span>"
            )
        else:
            badge = ""

        self._source_badge = badge   # stored so _render_markdown_now can inject it
        self._render_markdown_now()  # final full render with proper markdown

    def update_warmup_status(self, m, p, r):
        self.standby_view.set_warmup_status(m, p, r)

    def update_status(
        self,
        provider=None,
        capture_audio=False,
        capture_screen=False,
        latency_ms=0,
        available_providers=None,
        stage_timings=None,
        request_metadata=None,
        providers_tried=None,
        race: bool = False,
        had_screen: bool = False,
        had_audio: bool = False,
        had_rag: bool = False,
        # Q1/Q2/Q3 — P2/P3 feature chips
        had_memory: bool = False,
        had_action: bool = False,
        had_prefetch: bool = False,
        # Q14: cache tier badge (0 = LLM response, 1-4 = cache tier)
        cache_tier: int = 0,
    ):
        """Update the status bar with current state."""
        parts = []

        # Show current mode
        current_mode = self.app.state.mode if hasattr(self.app, "state") else "general"
        mode_icons = {
            "general": "\U0001f9e0",
            "interview": "\U0001f3af",
            "coding": "\U0001f4bb",
            "meeting": "\U0001f91d",
            "exam": "\U0001f393",
            "writing": "\u270d\ufe0f",
        }
        parts.append(f"{mode_icons.get(current_mode, chr(0x1F9E0))} {current_mode.upper()}")

        # Show available providers (from app.router or config)
        if available_providers and len(available_providers) > 0:
            parts.append(f"\U0001f4e1 [{', '.join(available_providers)}]")

        if provider:
            parts.append(f"\U0001f9e0 {provider}")
        if capture_audio:
            parts.append("\U0001f399\ufe0f")
        if capture_screen:
            parts.append("\U0001f441\ufe0f")
        if latency_ms > 0:
            parts.append(f"{latency_ms / 1000:.1f}s")
        else:
            parts.append("\u26a1 Ready")

        # Q1: Long-term memory chip
        if had_memory:
            parts.append("\U0001f4be Memory")
            logger.debug("[UI Chip] Memory chip shown — long-term memory was injected")

        # Q2: Actionable query chip
        if had_action:
            parts.append("\u26a1 Action")
            logger.debug("[UI Chip] Action chip shown — P3.3 command was executed")

        # Q3: Predictive prefetch chip
        if had_prefetch:
            parts.append("\U0001f52e Prefetched")
            logger.debug("[UI Chip] Prefetch chip shown — RAG was pre-warmed")

        # Q14: Cache tier badge
        if cache_tier > 0:
            tier_labels = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}
            parts.append(f"\u2248 {tier_labels.get(cache_tier, 'T?')} cached")
            logger.debug("[UI Chip] Cache tier=%d badge shown", cache_tier)

        self._status_snapshot = " | ".join(parts)

        # Surface stage timings as a tooltip to aid latency debugging without UI clutter.
        try:
            st = stage_timings or {}
            rm = request_metadata or {}
            tip_parts = []
            if race:
                tip_parts.append("Text race mode: fastest success (no streaming)")
            if providers_tried:
                tip_parts.append(f"Providers tried: {', '.join(providers_tried)}")
            ctx_parts = []
            if had_screen:
                ctx_parts.append("Screen")
            if had_audio:
                ctx_parts.append("Audio")
            if had_rag:
                ctx_parts.append("RAG")
            if ctx_parts:
                tip_parts.append(f"Context: {', '.join(ctx_parts)}")
            speech_to_transcript = rm.get("speech_to_transcript_ms")
            if speech_to_transcript is not None:
                tip_parts.append(f"Speech->Transcript: {speech_to_transcript:.0f}ms")
            ttfb = st.get("request_to_first_token_ms")
            if ttfb is not None:
                tip_parts.append(f"TTFB: {ttfb:.0f}ms")
            total = st.get("request_to_complete_ms")
            if total is not None:
                tip_parts.append(f"Total: {total:.0f}ms")
            rag_refine = st.get("rag_refine_parallel_ms")
            if rag_refine is not None:
                tip_parts.append(f"RAG+Refine: {rag_refine:.0f}ms")
            tip = " | ".join(tip_parts).strip()
            if tip:
                self.transcript_lbl.setToolTip(tip)
        except Exception:
            pass

    # --- RESTORED BRIDGES ---
    def update_transcript(self, text: str, state: str = "auto"):
        """Update the transcript/status label.

        state:
          'auto'       → infer colour from text content
          'listening'  → green
          'processing' → amber
          'error'      → red
          'idle'       → grey
        """
        display = text[:80] + ("..." if len(text) > 80 else "")
        self.transcript_lbl.setText(display)

        if state == "listening" or (state == "auto" and ("🌐" in text or "Listening" in text)):
            colour = "#4ade80"   # green
            italic = False
        elif state == "processing" or (state == "auto" and ("⏳" in text or "Processing" in text)):
            colour = "#f59e0b"   # amber
            italic = True
        elif state == "error" or (state == "auto" and ("🟡" in text or "Error" in text or "failed" in text.lower())):
            colour = "#ef4444"   # red
            italic = False
        else:
            colour = "#64748b"   # default grey
            italic = True

        self.transcript_lbl.setStyleSheet(
            f"color: {colour}; font-size: 10px; font-style: {'italic' if italic else 'normal'};"
        )

    def show_error_toast(self, message: str, duration_ms: int = 6000):
        """P1.10: Surface AI/provider errors as a dismissible red toast above the transcript.

        Auto-dismisses after `duration_ms`. If called again before dismiss,
        the timer resets so the latest error always gets full visibility.
        """
        if not hasattr(self, "_error_toast"):
            # Create the toast label once and reuse it
            self._error_toast = QLabel(self.container)
            self._error_toast.setWordWrap(True)
            self._error_toast.setStyleSheet(
                "background: rgba(239,68,68,45); color: #ef4444;"
                " border: 1px solid rgba(239,68,68,127); border-radius: 8px;"
                " padding: 6px 10px; font-size: 10px;"
            )
            self._error_toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._error_toast.hide()
            self._error_toast_timer = QTimer(self)
            self._error_toast_timer.setSingleShot(True)
            self._error_toast_timer.timeout.connect(self._error_toast.hide)

        self._error_toast.setText(f"⚠ {message}")
        # Position above the transcript area — top of the chat view
        self._error_toast.setFixedWidth(self.width() - 24)
        self._error_toast.move(12, 48)
        self._error_toast.raise_()
        self._error_toast.show()
        self._error_toast_timer.start(duration_ms)

    def set_analysis_provider_badge(self, provider: str = None, pending: bool = False):
        if pending:
            self._start_analyze_button_animation()
            self.analysis_badge.setText("Analyzing screen...")
            self.analysis_badge.setVisible(True)
            return

        if provider:
            self._show_analyze_button_success()
            self.analysis_badge.setText(f"Analyzed via {provider.capitalize()}")
            self.analysis_badge.setVisible(True)
            return

        self._stop_analyze_button_animation()
        self.analysis_badge.clear()
        self.analysis_badge.setVisible(False)

    def update_audio_state(self, muted):
        self.audio_status.setText("🔇" if muted else "🎙️")
        self.audio_status.setStyleSheet(
            f"color: {'#ef4444' if muted else '#4ade80'}; font-size: 12px;"
        )

        self.refresh_standby_state()

    def _toggle_vision(self):
        """Full vision kill switch — toggles screen capture + OCR as a unit.

        When OFF:
          - ScreenCapture._enabled = False  → no screenshots, no OCR, no frame hashing
          - _process_ai short-circuits the screen branch (saves 10-50ms per query)
          - ANALYZE SCREEN button is disabled (no point capturing when vision is off)
          - Config is persisted so the state survives session restarts
        When ON:
          - All of the above restored
          - ANALYZE SCREEN button re-enabled
        """
        current = self.app.screen._enabled if hasattr(self.app, "screen") and self.app.screen else False
        new_state = not current

        # 1. Apply to the live ScreenCapture module
        if hasattr(self.app, "screen") and self.app.screen:
            self.app.screen.set_enabled(new_state)

        # 2. Persist to config so it survives session restarts
        try:
            self.config.set("capture.screen.enabled", new_state)
            if hasattr(self.config, "save"):
                self.config.save()
        except Exception:
            pass

        # 3. Update UI
        self.update_vision_state(reset_to_master=False)

        # 4. Transcript feedback so user knows what happened
        if getattr(self.app, "session_active", False):
            msg = "🌐 Listening..." if new_state else "👁️ Vision OFF — answering from audio only"
            state = "listening" if new_state else "idle"
            self.update_transcript(msg, state=state)

    def update_vision_state(self, reset_to_master=False):
        """Sync eye icon + ANALYZE SCREEN button to current ScreenCapture._enabled state.

        If reset_to_master=True, reads the canonical value from config and applies it
        to the live module (called at session start to snap back to the saved setting).
        """
        if reset_to_master:
            master_enabled = self.config.get("capture.screen.enabled", True)
            if hasattr(self.app, "screen") and self.app.screen:
                self.app.screen.set_enabled(master_enabled)

        # Read true live state from module (not from config — they may differ mid-session)
        enabled = self.app.screen._enabled if hasattr(self.app, "screen") and self.app.screen else False

        # Eye icon: 👁️ = on (green), 🙈 = off (red)
        # 🙈 is the standard single-emoji "hidden" symbol — renders reliably on Windows
        self.vision_status.setText("👁️" if enabled else "🙈")
        self.vision_status.setStyleSheet(
            f"color: {'#4ade80' if enabled else '#ef4444'}; font-size: 14px; background: transparent; border: none;"
        )

        # Disable ANALYZE SCREEN when vision is fully off — no point capturing
        if hasattr(self, "btn_analyze_screen"):
            self.btn_analyze_screen.setEnabled(enabled)
            self.btn_analyze_screen.setToolTip(
                "Capture the current screen and analyze it with live session context"
                if enabled
                else "Vision is OFF — enable vision (👁️) to use screen analysis"
            )

    def _on_standby_mode_selected(self, mode):
        """Update app state when user clicks a mode button on standby screen."""
        if hasattr(self.app, "state"):
            self.app.state.mode = mode
            self.app._apply_settings()

    def _on_standby_audio_source_changed(self, source):
        """Update app state when user clicks an audio button on standby screen."""
        if hasattr(self.app, "state"):
            self.app.state.audio_source = source
            self.app._apply_settings()

    def update_mode(self, mode):
        # Kept for compatibility if called externally, but StandbyView is now self-connected
        self.standby_view.set_current_mode(mode)

    def update_audio_source(self, source):
        # Kept for compatibility if called externally, but StandbyView is now self-connected
        self.standby_view.set_current_audio_source(source)

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

    def _show_timeline(self):
        """P2.9: Show/hide the context timeline view."""
        if self.stack.currentWidget() is self.timeline_view:
            # Navigating away — pause the timer to save CPU
            self.timeline_view.deactivate()
            self.show_standby_view()
            return
        self._prev_stack_index = self.stack.currentIndex()
        self._prev_stack_widget = self.stack.currentWidget()
        self.timeline_view.activate()
        self.timeline_view.resume()
        self.stack.setCurrentWidget(self.timeline_view)

    def _show_settings(self):
        """Show settings and remember where we came from."""
        if self.stack.currentWidget() is self.settings_view:
            self.show_standby_view()
            return
        self._prev_stack_index = self.stack.currentIndex()
        self._prev_stack_widget = self.stack.currentWidget()
        self.show_settings_view()

    def _on_settings_closed(self):
        """Return to previous view and push any live config changes to running subsystems."""
        # P1.7: Push new screen capture interval to running ScreenCapture live
        if hasattr(self.app, "screen") and hasattr(self.app.screen, "_debounce"):
            new_interval = self.app.config.get("capture.screen.interval_ms", 500) / 1000.0
            self.app.screen._debounce = new_interval
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
            # ── Immediately reflect the query in the response area ──────────────
            # The user typed this — show it instantly without waiting for the
            # first streaming chunk or on_complete to set _current_query.
            self._current_query = q
            self.response_area.clear()
            self.response_area.setHtml(
                f"<div style='color: #64748b; font-size: 10px; margin-bottom: 5px;'>"
                f"<b>QUERY:</b> {q}</div>"
                f"<div style='color: #4ade80; font-size: 11px; font-style: italic;'>⏳ Thinking...</div>"
            )
            self.show_chat_view()
            self.user_query.emit(q)

    def _analyze_screen(self):
        if hasattr(self.app, "analyze_current_screen"):
            self._user_is_scrolling = False
            self.app.analyze_current_screen()

    def _tick_analyze_button(self):
        self._analyze_frame_index = (self._analyze_frame_index + 1) % len(self._analyze_frames)
        self.btn_analyze_screen.setText(self._analyze_frames[self._analyze_frame_index])

    def _start_analyze_button_animation(self):
        if self._analyze_success_timer.isActive():
            self._analyze_success_timer.stop()
        self._analyze_frame_index = 0
        self.btn_analyze_screen.setEnabled(False)
        self.btn_analyze_screen.setText(self._analyze_frames[self._analyze_frame_index])
        if not self._analyze_timer.isActive():
            self._analyze_timer.start()

    def _stop_analyze_button_animation(self):
        if self._analyze_timer.isActive():
            self._analyze_timer.stop()
        if self._analyze_success_timer.isActive():
            self._analyze_success_timer.stop()
        self._reset_analyze_button_idle()

    def _show_analyze_button_success(self):
        if self._analyze_timer.isActive():
            self._analyze_timer.stop()
        self.btn_analyze_screen.setEnabled(False)
        self.btn_analyze_screen.setText("SCREEN ANALYZED")
        self._analyze_success_timer.start()

    def _reset_analyze_button_idle(self):
        self.btn_analyze_screen.setEnabled(True)
        self.btn_analyze_screen.setText("ANALYZE SCREEN")

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
        # Q9: Persist overlay position so it survives restarts
        pos = self.pos()
        try:
            self.config.set("app.overlay_x", pos.x())
            self.config.set("app.overlay_y", pos.y())
            logger.debug("[Q9 Position] Saved overlay position: (%d, %d)", pos.x(), pos.y())
        except Exception as _e:
            logger.debug("[Q9 Position] Could not save position: %s", _e)

    def showEvent(self, e):
        super().showEvent(e)
        if hasattr(self.app, "_apply_window_effects"):
            self.app._apply_window_effects(self)
        if hasattr(self.app, "hotkeys"):
            self.app.hotkeys.reset_state()
        if hasattr(self, "_gaze_timer") and not self._gaze_timer.isActive():
            self._gaze_timer.start(100)

    def hideEvent(self, e):
        if hasattr(self, "_gaze_timer") and self._gaze_timer.isActive():
            self._gaze_timer.stop()
        super().hideEvent(e)
        if hasattr(self.app, "hotkeys"):
            self.app.hotkeys.reset_state()

    def _check_gaze(self):
        """Dynamic Gaze: Fades the window if the mouse is near/over it.

        Only active during an active session. Uses the same stylesheet technique
        as set_click_through() which is proven to work with WA_TranslucentBackground.
        Works alongside Stealth Mode — stealth anti-capture remains active.
        """
        if not self.isVisible() or self.app.mini_mode:
            return

        gaze_enabled = self.config.get("app.gaze_fade.enabled", False)
        session_active = getattr(self.app, "session_active", False)

        # Determine if cursor is near/over the window
        near = False
        if gaze_enabled and session_active:
            cursor_pos = self.mapFromGlobal(self.cursor().pos())
            x = cursor_pos.x()
            y = cursor_pos.y()
            margin = self.config.get("app.gaze_fade.margin", 80)
            if -margin <= x <= self.width() + margin and -margin <= y <= self.height() + margin:
                near = True

        # Track previous state to avoid redundant stylesheet writes
        was_near = getattr(self, "_gaze_near", None)
        if near == was_near:
            return
        self._gaze_near = near

        if near:
            # Same style as set_click_through(True) — proven to work visually
            # Hook up config target_opacity (0.0-1.0) to alpha (0-255)
            t_opa = self.config.get("app.gaze_fade.target_opacity", 0.12)
            alpha = int(t_opa * 255)
            
            self.box.setStyleSheet(
                f"#box {{ background: rgba(12, 12, 25, {alpha}); border: 1px solid rgba(80, 85, 255, 10); border-radius: 14px; }}"
            )
            self.response_area.setStyleSheet(
                f"background: transparent; color: rgba(208, 208, 232, {min(255, max(40, alpha+40))}); border: none; font-size: 13px;"
            )
        else:
            # Restore to fully opaque
            self.box.setStyleSheet(
                "#box { background: rgba(12, 12, 25, 250); border: 1px solid rgba(80, 85, 255, 30); border-radius: 14px; }"
            )
            self.response_area.setStyleSheet(
                "background: transparent; color: #d0d0e8; border: none; font-size: 13px;"
            )

