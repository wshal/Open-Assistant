"""
Compact floating mini-overlay — v4.1 (Cleaned & Hardened).
FIXED: Mode icon resolution and AI response bridging.
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFrame,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QTextCursor, QTextCharFormat, QColor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MiniOverlay(QMainWindow):
    COLLAPSED_HEIGHT = 48
    HEADER_HEIGHT = 48
    MAX_WINDOW_HEIGHT = 360
    MIN_VISIBLE_RESPONSE_ROWS = 10
    # P2.6: Nano mode dimensions
    NANO_WIDTH = 200
    NANO_HEIGHT = 36
    FULL_WIDTH = 280

    user_query = pyqtSignal(str)

    def __init__(self, config, app):
        super().__init__()
        self.config = config
        self.app = app
        self._response = ""
        self._raw_buffer = ""
        self._drag = False
        self._expanded = False
        self._nano_mode = False  # P2.6: ultra-compact nano state

        # NEURAL UX: Gaze-based transparency
        self._gaze_timer = QTimer(self)
        self._gaze_timer.timeout.connect(self._check_gaze)
        self._gaze_timer.start(100)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(300)  # periodic markdown re-render during streaming
        self._render_timer.timeout.connect(self._render_markdown_now)

        self._build()
        self._connect_state()

    def _connect_state(self):
        self.app.state.muted_changed.connect(self.update_audio_state)
        self.app.state.mode_changed.connect(self.update_mode)
        self.app.state.hud_mode_changed.connect(self._on_hud_mode_changed)

    def _on_hud_mode_changed(self, is_mini):
        if is_mini:
            self._sync_existing_response()
            self.show()
            self.raise_()
        else:
            self.hide()

    def _build(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(0.95)
        self.setFixedWidth(280)
        self.setFixedHeight(self.COLLAPSED_HEIGHT)

        c = QWidget()
        c.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        c.setStyleSheet("background: transparent; border: none;")
        self.setCentralWidget(c)
        self.ml = QVBoxLayout(c)
        self.ml.setContentsMargins(0, 0, 0, 0)
        self.ml.setSpacing(4)

        self.bar = QFrame()
        self.bar.setStyleSheet(
            "background: rgba(20,20,35,250); border: 1px solid rgba(80,80,150,80); border-radius: 24px;"
        )
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(12, 6, 12, 6)
        bl.setSpacing(6)

        self.mode_icon = QLabel("🧠")
        self.mode_icon.setStyleSheet("font-size: 15px;")
        self.mode_icon.setToolTip("Double-click for nano/compact mode")
        self.mode_icon.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_icon.mouseDoubleClickEvent = lambda e: self._toggle_nano_mode()
        bl.addWidget(self.mode_icon)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask anything...")
        self.input.returnPressed.connect(self._send)
        self.input.setStyleSheet(
            "background: transparent; color: #c0c0dd; border: none; font-size: 12px; padding: 2px;"
        )
        bl.addWidget(self.input, 1)

        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #4ade80; font-size: 10px;")
        bl.addWidget(self.dot)

        # P1.1: Type response button — injects the last AI answer into the focused window
        self.type_btn = QPushButton("⌨")
        self.type_btn.setFixedSize(22, 22)
        self.type_btn.setToolTip("Type last response into active window")
        self.type_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.type_btn.setStyleSheet(
            "background: rgba(80,200,120,0.12); color: #4ade80; border: none;"
            " border-radius: 11px; font-size: 11px;"
            " QPushButton:hover { background: rgba(80,200,120,0.25); }"
        )
        self.type_btn.clicked.connect(self._type_response)
        self.type_btn.setVisible(False)  # only shown once a response exists
        bl.addWidget(self.type_btn)

        self.expand_btn = QPushButton("▲")
        self.expand_btn.setFixedSize(22, 22)
        self.expand_btn.setStyleSheet(
            "background: rgba(80,80,255,0.1); color: #8888bb; border: none; border-radius: 11px; font-size: 9px;"
        )
        self.expand_btn.clicked.connect(self._toggle_expand)
        bl.addWidget(self.expand_btn)
        self.ml.addWidget(self.bar)

        from ui.markdown_renderer import MarkdownRenderer

        self.md = MarkdownRenderer()

        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setVisible(False)
        self.response_area.setStyleSheet(
            "background: rgba(15,15,30,240); border: none; border-radius: 12px; color: #d0d0e8; font-size: 11px; padding: 8px;"
        )
        self.response_area.setFrameShape(QFrame.Shape.NoFrame)
        self.ml.addWidget(self.response_area)

    def _sync_existing_response(self):
        if self._raw_buffer.strip():
            self._toggle_expand(True)
            self._render_markdown_now()
            return

        if not hasattr(self.app, "history") or not hasattr(self.app.history, "get_state"):
            return

        _, _, entry = self.app.history.get_state()
        if entry and entry.get("response"):
            self.on_complete(entry.get("response", ""), entry.get("query", ""))

    def _send(self):
        q = self.input.text().strip()
        if q:
            self.input.clear()
            # Immediately show query + thinking state before any AI response
            self._raw_buffer = ""
            self.response_area.clear()
            race_hint = ""
            if bool(self.config.get("ai.text.race_enabled", False)):
                race_hint = " (race mode — no streaming)"
            self.response_area.setHtml(
                f"<div style='color:#64748b;font-size:10px;'><b>Q:</b> {q}</div>"
                f"<div style='color:#f59e0b;font-size:11px;font-style:italic;'>⏳ Thinking{race_hint}...</div>"
            )
            self._toggle_expand(True)
            self.set_thinking()
            self.user_query.emit(q)

    def _render_markdown_now(self):
        text = self._raw_buffer
        if hasattr(self, "_last_rendered") and self._last_rendered == text:
            return
        self._last_rendered = text

        # Avoid full HTML rebuild during streaming when fenced code blocks are present.
        # We'll do the final markdown render on completion instead.
        if self._render_timer.isActive() and "```" in (text or ""):
            return

        html = self.md.render(text or "Waiting...")
        self.response_area.setHtml(html)
        if text and not self._expanded:
            self._toggle_expand(True)
        self._adjust_height()

    def append_response(self, text: str):
        """Called on every streaming chunk — immediate display + periodic markdown re-render."""
        if not self._raw_buffer:
            # First chunk: expand panel, clear area, start periodic timer
            self._toggle_expand(True)
            self.response_area.clear()
            self.set_thinking()
            self._render_timer.start()

        self._raw_buffer += text

        # Stop periodic markdown re-renders once a code fence appears to prevent
        # layout wobble while code blocks stream in.
        if self._render_timer.isActive() and "```" in self._raw_buffer:
            self._render_timer.stop()

        # Immediate colour-correct text append
        cursor = self.response_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#d0d0e8"))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.response_area.setTextCursor(cursor)
        # Auto-scroll to bottom
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )

    def on_complete(self, full_text: str, query: str = None):
        """Streaming finished — stop timer, do final markdown render, update dot."""
        self._render_timer.stop()
        self._raw_buffer = full_text or ""
        self._render_markdown_now()
        self.set_ready()
        # P1.1: Show type button once there's a response to inject
        if full_text and full_text.strip():
            self.type_btn.setVisible(True)
        else:
            self.type_btn.setVisible(False)

    def set_response(self, text: str):
        self._raw_buffer = text or ""
        self._render_markdown_now()

    def show_error(self, err: str):
        self.set_error()
        self.response_area.setPlainText(f"❌ {err}")
        self._toggle_expand(True)

    def _adjust_height(self):
        """ADAPTIVE: Calculate required height up to 300px max."""
        if not self._expanded:
            self.setFixedHeight(self.COLLAPSED_HEIGHT)
            return

        doc = self.response_area.document()
        content_height = int(doc.size().height()) + 20
        line_height = max(14, self.response_area.fontMetrics().lineSpacing())
        min_response_height = (line_height * self.MIN_VISIBLE_RESPONSE_ROWS) + 20
        response_height = max(content_height, min_response_height)
        max_response_height = self.MAX_WINDOW_HEIGHT - self.HEADER_HEIGHT
        response_height = min(response_height, max_response_height)
        new_h = min(self.HEADER_HEIGHT + response_height, self.MAX_WINDOW_HEIGHT)

        self.setFixedHeight(new_h)
        self.response_area.setMinimumHeight(min_response_height)
        self.response_area.setMaximumHeight(max_response_height)
        self.response_area.verticalScrollBar().setValue(
            self.response_area.verticalScrollBar().maximum()
        )

    def _toggle_expand(self, force=None):
        self._expanded = force if force is not None else not self._expanded
        self.response_area.setVisible(self._expanded)
        self.expand_btn.setText("▼" if self._expanded else "▲")
        self._adjust_height()

    def update_warmup_status(self, m, p, r):
        self.setToolTip(f"{m} ({p}%)")

    def update_mode(self, mode):
        icons = {
            "general": "🧠",
            "interview": "🎯",
            "coding": "💻",
            "meeting": "🤝",
            "exam": "🎓",
            "writing": "✍️",
        }
        self.mode_icon.setText(icons.get(mode.lower(), "🤖"))

    def update_audio_state(self, muted: bool):
        self.dot.setStyleSheet(
            f"color: {'#ef4444' if muted else '#4ade80'}; font-size: 10px;"
        )

    def update_history_state(self, i, t, e=None):
        if e and self._expanded:
            self.set_response(e.get("response", ""))

    def set_thinking(self):
        self.dot.setStyleSheet("color: #f59e0b; font-size: 10px;")

    def set_ready(self):
        self.dot.setStyleSheet("color: #4ade80; font-size: 10px;")

    def set_error(self):
        self.dot.setStyleSheet("color: #ef4444; font-size: 10px;")

    def set_click_through(self, enabled: bool):
        if enabled:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        else:
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)

        # Only refresh if we are the currently active HUD mode to prevent hidden window popups
        if self.app.mini_mode and self.isVisible():
            self.show()

    def showEvent(self, e):
        super().showEvent(e)
        if hasattr(self.app, "_apply_window_effects"):
            self.app._apply_window_effects(self)
        # P1.6: Resume gaze timer when the overlay becomes visible
        self._gaze_timer.start(100)

    def hideEvent(self, e):
        # P1.6: Pause gaze polling while hidden — avoids 100ms busy-loop when invisible
        self._gaze_timer.stop()
        super().hideEvent(e)

    def _type_response(self):
        """P1.1: Inject the last AI response into the focused window via the app simulator."""
        if hasattr(self.app, "type_response") and callable(self.app.type_response):
            self.app.type_response()
        else:
            logger.warning("MiniOverlay: app.type_response() not available")

    def _toggle_nano_mode(self):
        """P2.6: Toggle ultra-compact nano mode (200×36) ↔ full mini mode (280×48)."""
        self._nano_mode = not self._nano_mode
        if self._nano_mode:
            # Collapse to nano — hide most controls, show icon + response preview
            self.input.hide()
            self.type_btn.hide()
            self.expand_btn.hide()
            self.dot.hide()
            if self._expanded:
                self._toggle_expand(False)
            # Show a compact text preview in the mode icon slot
            preview = (self._raw_buffer[:35] + "…") if self._raw_buffer else "Ready"
            self.mode_icon.setToolTip(self._raw_buffer)
            self.bar.setStyleSheet(
                "background: rgba(20,20,35,220); border: 1px solid rgba(80,80,150,60);"
                " border-radius: 18px;"
            )
            self.setFixedSize(self.NANO_WIDTH, self.NANO_HEIGHT)
        else:
            # Restore full mini mode
            self.input.show()
            self.expand_btn.show()
            self.dot.show()
            if self._raw_buffer:
                self.type_btn.show()
            self.mode_icon.setToolTip("Double-click for nano/compact mode")
            self.bar.setStyleSheet(
                "background: rgba(20,20,35,250); border: 1px solid rgba(80,80,150,80);"
                " border-radius: 24px;"
            )
            self.setFixedWidth(self.FULL_WIDTH)
            self.setFixedHeight(self.COLLAPSED_HEIGHT)

    def show_error(self, err: str):
        """P1.10: Surface provider errors as a visible red toast in the mini overlay."""
        self.set_error()
        self._raw_buffer = ""
        self.response_area.setHtml(
            f"<div style='color:#ef4444; font-size:11px; padding:4px;'>❌ {err}</div>"
        )
        self._toggle_expand(True)
        # Auto-dismiss after 6 seconds
        QTimer.singleShot(6000, lambda: (
            self.response_area.clear() if not self._raw_buffer else None
        ))

    def scroll_up(self):
        sb = self.response_area.verticalScrollBar()
        sb.setValue(sb.value() - 40)

    def scroll_down(self):
        sb = self.response_area.verticalScrollBar()
        sb.setValue(sb.value() + 40)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._drag_pos = (
                e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, e):
        if self._drag:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag = False

    def _check_gaze(self):
        """Neural UX: Fades the mini-overlay if mouse is nearby to allow viewing background content.

        Only active during an active session - not on standby/settings screens.
        Can be disabled via config 'app.gaze_fade.enabled'.
        """
        # Check if gaze fade is enabled in config
        if not self.config.get("app.gaze_fade.enabled", True):  # P1.2: default ON
            return

        if getattr(self.app.state, "is_stealth", False):
            return

        # Only fade during active session - not on standby/settings screens
        if not self.isVisible() or not self.app.state.is_mini:
            return

        # Check if session is active - only fade when user is in a running session
        if not getattr(self.app, "session_active", False):
            return

        cursor_pos = self.mapFromGlobal(self.cursor().pos())
        inside = self.rect().contains(cursor_pos)

        # Use config values with sensible defaults for mini mode
        margin = self.config.get("app.gaze_fade.margin", 30)
        dist_x = min(abs(cursor_pos.x()), abs(cursor_pos.x() - self.width()))
        dist_y = min(abs(cursor_pos.y()), abs(cursor_pos.y() - self.height()))

        target_opa = 0.95
        if inside or (dist_x < margin and dist_y < margin):
            target_opa = self.config.get("app.gaze_fade.target_opacity", 0.15)

        current_opa = self.windowOpacity()
        if abs(current_opa - target_opa) > 0.01:
            # Smooth interpolation
            self.setWindowOpacity(current_opa + (target_opa - current_opa) * 0.3)
