"""
Compact floating mini-overlay — v4.1 (Cleaned & Hardened).
FIXED: Mode icon resolution and AI response bridging.
"""

import pyperclip
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFrame, QApplication, 
    QScrollArea, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QFont
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MiniOverlay(QMainWindow):
    user_query = pyqtSignal(str)

    def __init__(self, config, app):
        super().__init__()
        self.config = config
        self.app = app
        self._response = ""
        self._raw_buffer = ""
        self._drag = False
        self._drag_pos = QPoint()
        self._expanded = False
        self._build()

    def _build(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.95)
        self.setFixedWidth(280)
        self.setFixedHeight(48)

        c = QWidget(); self.setCentralWidget(c)
        self.ml = QVBoxLayout(c); self.ml.setContentsMargins(0, 0, 0, 0); self.ml.setSpacing(4)

        self.bar = QFrame()
        self.bar.setStyleSheet("background: rgba(20,20,35,250); border: 1px solid rgba(80,80,150,80); border-radius: 24px;")
        bl = QHBoxLayout(self.bar); bl.setContentsMargins(12, 6, 12, 6); bl.setSpacing(6)

        self.mode_icon = QLabel("🧠")
        self.mode_icon.setStyleSheet("font-size: 15px;")
        bl.addWidget(self.mode_icon)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask anything...")
        self.input.returnPressed.connect(self._send)
        self.input.setStyleSheet("background: transparent; color: #c0c0dd; border: none; font-size: 12px; padding: 2px;")
        bl.addWidget(self.input, 1)

        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #4ade80; font-size: 10px;")
        bl.addWidget(self.dot)

        self.expand_btn = QPushButton("▲")
        self.expand_btn.setFixedSize(22, 22)
        self.expand_btn.setStyleSheet("background: rgba(80,80,255,0.1); color: #8888bb; border: none; border-radius: 11px; font-size: 9px;")
        self.expand_btn.clicked.connect(self._toggle_expand)
        bl.addWidget(self.expand_btn)
        self.ml.addWidget(self.bar)

        from ui.markdown_renderer import MarkdownRenderer
        self.md = MarkdownRenderer()
        
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setVisible(False)
        self.response_area.setStyleSheet("background: rgba(15,15,30,240); border: 1px solid rgba(80,85,255,30); border-radius: 12px; color: #d0d0e8; font-size: 11px; padding: 8px;")
        self.ml.addWidget(self.response_area)

    def _send(self):
        q = self.input.text().strip()
        if q:
            self.input.clear()
            self.set_thinking()
            self.user_query.emit(q)

    def set_response(self, text: str):
        self._raw_buffer = text
        html = self.md.render(text or "Waiting...")
        self.response_area.setHtml(html)
        if text and not self._expanded:
            self._toggle_expand(True)
        self._adjust_height()

    def append_response(self, text: str):
        self._raw_buffer += text
        self.set_response(self._raw_buffer)

    def on_complete(self, full_text: str, query: str = None):
        self._raw_buffer = full_text or ""
        self.set_response(self._raw_buffer)
        self.set_ready()

    def show_error(self, err: str):
        self.set_error()
        self.response_area.setPlainText(f"❌ {err}")
        self._toggle_expand(True)

    def _adjust_height(self):
        """ADAPTIVE: Calculate required height up to 300px max."""
        if not self._expanded:
            self.setFixedHeight(48)
            return
        
        doc = self.response_area.document()
        height = int(doc.size().height()) + 20 
        new_h = min(max(height + 60, 100), 320)
            
        self.setFixedHeight(new_h)
        self.response_area.verticalScrollBar().setValue(self.response_area.verticalScrollBar().maximum())

    def _toggle_expand(self, force=None):
        self._expanded = force if force is not None else not self._expanded
        self.response_area.setVisible(self._expanded)
        self.expand_btn.setText("▼" if self._expanded else "▲")
        self._adjust_height()

    def update_warmup_status(self, m, p, r): self.setToolTip(f"{m} ({p}%)")
    def update_mode(self, mode):
        icons = {"general": "🧠", "interview": "🎯", "coding": "💻", "meeting": "🤝", "exam": "🎓", "writing": "✍️"}
        self.mode_icon.setText(icons.get(mode.lower(), "🤖"))

    def update_audio_state(self, muted: bool):
        self.dot.setStyleSheet(f"color: {'#ef4444' if muted else '#4ade80'}; font-size: 10px;")

    def set_thinking(self): self.dot.setStyleSheet("color: #f59e0b; font-size: 10px;")
    def set_ready(self): self.dot.setStyleSheet("color: #4ade80; font-size: 10px;")
    def set_error(self): self.dot.setStyleSheet("color: #ef4444; font-size: 10px;")

    def set_click_through(self, enabled: bool):
        if enabled: self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        else: self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)
        
        if self.isVisible():
            self.show()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True; self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag: self.move(e.globalPosition().toPoint() - self._drag_pos)
    def mouseReleaseEvent(self, e): self._drag = False
