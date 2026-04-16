"""
Main overlay window — v4.1 (Layer 4 Hardened).
RESTORED: Markdown Render Debounce (150ms) and Manual Scroll Lock.
FIXED: Connection of transcript and audio-status bridges.
"""

import pyperclip
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel,
    QFrame, QComboBox, QApplication, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPoint, QSize
from PyQt6.QtGui import QFont, QTextCursor, QShortcut, QKeySequence
from ui.markdown_renderer import MarkdownRenderer
from ui.standby_view import StandbyView
from ui.settings_view import SettingsView
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

        self.md = MarkdownRenderer()
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(150)
        self._render_timer.timeout.connect(self._render_markdown_now)

        self._build()

    def _build(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(self.config.get("app.opacity", 0.94))
        self.resize(400, 660)

        self.container = QWidget()
        self.setCentralWidget(self.container)
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.box = QFrame()
        self.box.setObjectName("box")
        self.box.setStyleSheet("#box { background: rgba(12, 12, 25, 250); border: 1px solid rgba(80, 85, 255, 30); border-radius: 14px; }")
        box_layout = QVBoxLayout(self.box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.box)

        # Header
        self.header = QFrame()
        self.header.setFixedHeight(40)
        self.header.setStyleSheet("background: rgba(25, 25, 45, 255); border-bottom: 1px solid rgba(255,255,255,10);")
        hl = QHBoxLayout(self.header)
        
        self.title_lbl = QLabel("OPENASSIST AI")
        self.title_lbl.setStyleSheet("color: #a0a0cc; font-weight: bold; font-size: 11px;")
        hl.addWidget(self.title_lbl)
        
        hl.addStretch()
        
        # RESTORATION: Interactive Audio Status Pill in Header
        self.audio_status = QPushButton("🎙️")
        self.audio_status.setToolTip("Toggle AI Audio Capture")
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
        hl.addWidget(self.audio_status)
        
        btn_set = QPushButton("⚙️")
        btn_set.setToolTip("Settings")
        btn_set.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_set.setStyleSheet("color: #667; border: none; font-size: 14px; background: transparent;")
        btn_set.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        hl.addWidget(btn_set)
        
        btn_close = QPushButton("✕")
        btn_close.setStyleSheet("color: #667; border: none; font-size: 14px; background: transparent;")
        btn_close.clicked.connect(self.hide)
        hl.addWidget(btn_close)
        box_layout.addWidget(self.header)

        self.stack = QStackedWidget()
        box_layout.addWidget(self.stack)

        self.standby_view = StandbyView(self)
        self.stack.addWidget(self.standby_view)

        self.chat_view = QWidget()
        cv_layout = QVBoxLayout(self.chat_view)
        
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setStyleSheet("background: transparent; color: #d0d0e8; border: none; font-size: 13px;")
        self.response_area.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)
        cv_layout.addWidget(self.response_area)

        # RESTORATION: Transcription floating bar
        self.transcript_bar = QFrame()
        self.transcript_bar.setFixedHeight(30)
        self.transcript_bar.setStyleSheet("background: rgba(30,30,50,150); border-radius: 8px; margin: 0 5px;")
        tl = QHBoxLayout(self.transcript_bar)
        self.transcript_lbl = QLabel("Ready...")
        self.transcript_lbl.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic;")
        tl.addWidget(self.transcript_lbl)
        cv_layout.addWidget(self.transcript_bar)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask anything...")
        self.input.returnPressed.connect(self._send)
        self.input.setStyleSheet("background: rgba(25,25,50,200); color: white; border: 1px solid rgba(255,255,255,10); border-radius: 10px; padding: 10px; margin: 5px;")
        cv_layout.addWidget(self.input)

        self.stack.addWidget(self.chat_view)
        
        self.settings_view = SettingsView(self.config, self.app)
        self.settings_view.closed.connect(lambda: self.stack.setCurrentIndex(0))
        self.stack.addWidget(self.settings_view)

    def _on_scroll_changed(self, value):
        sb = self.response_area.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 50
        self._user_is_scrolling = not at_bottom

    def _render_markdown_now(self):
        content = self._raw_buffer
        q_html = f"<div style='color: #64748b; font-size: 10px; margin-bottom: 5px;'><b>QUERY:</b> {self._current_query}</div>" if self._current_query else ""
        self.response_area.setHtml(q_html + self.md.render(content))
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
        if query: self._current_query = query
        self._raw_buffer = full_text
        self._render_markdown_now()

    def update_warmup_status(self, m, p, r): self.standby_view.set_warmup_status(m, p, r)
    
    # --- RESTORED BRIDGES ---
    def update_transcript(self, text):
        self.transcript_lbl.setText(text[:80] + ("..." if len(text) > 80 else ""))
        self.transcript_lbl.setStyleSheet("color: #4ade80; font-size: 10px; font-style: normal;")

    def update_audio_state(self, muted):
        self.audio_status.setText("🔇" if muted else "🎙️")
        self.audio_status.setStyleSheet(f"color: {'#ef4444' if muted else '#4ade80'}; font-size: 12px;")

    def update_mode(self, mode):
        self.standby_view.set_current_mode(mode)

    def update_history_state(self, i, t, e=None): pass

    def _send(self):
        q = self.input.text().strip()
        if q:
            self.input.clear()
            self._user_is_scrolling = False
            self.user_query.emit(q)

    def set_click_through(self, enabled: bool):
        """Toggle mouse interaction transparency."""
        if enabled:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
            self.box.setStyleSheet("#box { background: rgba(12, 12, 25, 100); border: 1px solid rgba(80, 85, 255, 10); border-radius: 14px; }")
            self.response_area.setStyleSheet("background: transparent; color: rgba(208, 208, 232, 150); border: none; font-size: 13px;")
        else:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)
            self.box.setStyleSheet("#box { background: rgba(12, 12, 25, 250); border: 1px solid rgba(80, 85, 255, 30); border-radius: 14px; }")
            self.response_area.setStyleSheet("background: transparent; color: #d0d0e8; border: none; font-size: 13px;")
        
        # Windows requires a hide/show to refresh flags immediately
        if self.isVisible():
            self.show()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            child = self.childAt(e.position().toPoint())
            if child in [self.box, self.header] or child is None:
                self._drag = True
                self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag: self.move(e.globalPosition().toPoint() - self._drag_pos)
    def mouseReleaseEvent(self, e): self._drag = False

    def showEvent(self, e):
        super().showEvent(e)
        if hasattr(self.app, "hotkeys"):
            self.app.hotkeys.reset_state()

    def hideEvent(self, e):
        super().hideEvent(e)
        if hasattr(self.app, "hotkeys"):
            self.app.hotkeys.reset_state()
