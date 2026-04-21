"""
Premium Standby View - OpenAssist AI.

Refined for:
- safer icon handling via Unicode escape sequences
- cleaner, centralized UI constants
- more responsive footer/button sizing
- clearer provider dashboard empty state
"""

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from utils.logger import setup_logger

logger = setup_logger(__name__)


class StandbyView(QWidget):
    start_clicked = pyqtSignal()
    mode_selected = pyqtSignal(str)
    audio_source_changed = pyqtSignal(str)

    HERO_ICON = "\U0001F9E0"
    MODE_OPTIONS = [
        [("\U0001F9E0", "GENERAL"), ("\U0001F3AF", "INTERVIEW")],
        [("\U0001F4BB", "CODING"), ("\U0001F91D", "MEETING")],
        [("\U0001F393", "EXAM"), ("\u270D\uFE0F", "WRITING")],
    ]
    AUDIO_OPTIONS = [
        ("\U0001F399\uFE0F MIC", "mic"),
        ("\U0001F50A SYSTEM", "system"),
        ("\U0001F310 BOTH", "both"),
    ]
    STATUS_ITEMS = [
        ("\U0001F399\uFE0F", "STABLE"),
        ("\u2728", "SYNCED"),
        ("\U0001F6E1\uFE0F", "ACTIVE"),
    ]

    STYLE_INACTIVE = """
        QPushButton {
            color: #94a3b8;
            background: rgba(28, 30, 43, 210);
            border: 1px solid rgba(99, 102, 241, 18);
            border-radius: 12px;
            padding: 8px 14px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.2px;
        }
        QPushButton:hover {
            background: rgba(42, 45, 68, 240);
            border: 1px solid rgba(129, 140, 248, 42);
            color: white;
        }
    """

    STYLE_ACTIVE = """
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #5b4cf1,
                stop:0.55 #6d4cf6,
                stop:1 #8b5cf6
            );
            color: white;
            border: 1px solid rgba(255, 255, 255, 72);
            border-radius: 12px;
            padding: 8px 14px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1.2px;
        }
    """

    START_BUTTON_STYLE = """
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #5b4cf1,
                stop:0.55 #7047f5,
                stop:1 #8b5cf6
            );
            color: white;
            border-radius: 26px;
            font-weight: 900;
            font-size: 13px;
            letter-spacing: 3px;
            border: 1px solid rgba(255, 255, 255, 18);
        }
        QPushButton:hover:enabled {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #6a68ff,
                stop:1 #9b6bff
            );
            border: 1px solid rgba(255, 255, 255, 40);
        }
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    """

    START_BUTTON_READY_STYLE = """
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #10b981,
                stop:1 #059669
            );
            color: white;
            border-radius: 26px;
            font-weight: 900;
            font-size: 13px;
            letter-spacing: 3px;
            border: 1px solid rgba(255, 255, 255, 18);
        }
        QPushButton:hover:enabled {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #34d399,
                stop:1 #10b981
            );
            border: 1px solid rgba(255, 255, 255, 40);
        }
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_status_widgets = {}
        self.mode_buttons = {}
        self.audio_btns = {}
        self._boot_sync_scheduled = False
        self._boot_sync_logged = False

        self._init_ui()
        self._connect_state()
        self._schedule_boot_sync()

    def _connect_state(self):
        parent = self.parent()
        app = getattr(parent, "app", None)
        state = getattr(app, "state", None)
        if state is not None:
            state.mode_changed.connect(self.set_current_mode)
            state.audio_source_changed.connect(self.set_current_audio_source)

    def showEvent(self, event):
        """Final UI sync trigger on window mapping."""
        super().showEvent(event)
        self._schedule_boot_sync()

    def _init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(25, 18, 25, 30)
        self.main_layout.setSpacing(0)

        layout = self.main_layout

        self.hero_container = QFrame()
        self.hero_container.setFixedHeight(138)
        self.hero_container.setStyleSheet(
            """
            QFrame {
                background: qradialgradient(
                    cx:0.5, cy:0.42, radius:0.72,
                    fx:0.5, fy:0.35,
                    stop:0 rgba(167, 139, 250, 72),
                    stop:0.35 rgba(99, 102, 241, 26),
                    stop:1 rgba(0, 0, 0, 0)
                );
                border: none;
                border-radius: 28px;
            }
            """
        )
        hero_layout = QVBoxLayout(self.hero_container)
        hero_layout.setContentsMargins(0, 0, 0, 16)
        hero_layout.setSpacing(0)

        self.hero_label = QLabel(self.HERO_ICON)
        self.hero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_label.setStyleSheet(
            """
            font-size: 80px;
            background: transparent;
            margin-bottom: 0px;
            color: #f3e8ff;
            """
        )
        self.hero_label.setMinimumHeight(88)
        hero_layout.addWidget(self.hero_label, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        hero_layout.addStretch(1)
        layout.addWidget(self.hero_container)

        layout.addSpacing(30)

        self.subtitle = QLabel("NEURAL ENGINE INITIALIZING...")
        self.subtitle.setFixedHeight(18)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet(
            """
            font-size: 9px;
            color: #9aa5ff;
            letter-spacing: 3px;
            font-weight: 900;
            background: transparent;
            """
        )
        layout.addWidget(self.subtitle)

        layout.addSpacing(18)

        status_layout = QHBoxLayout()
        status_layout.setSpacing(10)
        status_widgets = [
            self._create_status_pill(icon, label) for icon, label in self.STATUS_ITEMS
        ]
        self.mic_pill, self.ai_pill, self.stealth_pill = status_widgets
        for pill in status_widgets:
            status_layout.addWidget(pill)
        layout.addLayout(status_layout)

        layout.addSpacing(25)

        lbl_m = QLabel("AI MODES")
        lbl_m.setStyleSheet(
            "font-size: 10px; color: #64748b; font-weight: 900; letter-spacing: 3px;"
        )
        lbl_m.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_m)

        layout.addSpacing(14)

        for row_data in self.MODE_OPTIONS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(12)
            for icon, name in row_data:
                btn = QPushButton(f"{icon} {name}")
                btn.setCheckable(True)
                btn.setFixedHeight(40)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(
                    lambda checked=False, n=name.lower(): self._on_mode_btn_clicked(n)
                )
                btn.setStyleSheet(self.STYLE_INACTIVE)
                row_layout.addWidget(btn)
                self.mode_buttons[name.lower()] = btn
            layout.addLayout(row_layout)
            layout.addSpacing(14)

        layout.addSpacing(12)

        lbl_a = QLabel("CAPTURE SOURCE")
        lbl_a.setStyleSheet(
            "font-size: 10px; color: #64748b; font-weight: 900; letter-spacing: 2.5px;"
        )
        lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_a)

        layout.addSpacing(14)

        audio_layout = QHBoxLayout()
        audio_layout.setSpacing(10)
        for label, name in self.AUDIO_OPTIONS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self.STYLE_INACTIVE)
            btn.clicked.connect(
                lambda checked=False, n=name: self._on_audio_btn_clicked(n)
            )
            audio_layout.addWidget(btn)
            self.audio_btns[name] = btn
        layout.addLayout(audio_layout)

        layout.addSpacing(25)

        self.model_bar = QFrame()
        self.model_bar.setFixedHeight(34)
        self.model_bar.setStyleSheet(
            """
            QFrame {
                background: rgba(14, 16, 28, 205);
                border-radius: 17px;
                border: 1px solid rgba(99, 102, 241, 24);
            }
            """
        )
        self.model_bar_layout = QHBoxLayout(self.model_bar)
        self.model_bar_layout.setContentsMargins(15, 0, 15, 0)
        self.model_bar_layout.setSpacing(15)
        self.model_bar_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_bar)
        self.set_provider_statuses({})

        layout.addStretch(1)

        footer_container = QVBoxLayout()
        footer_container.setSpacing(15)
        footer_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background: rgba(255, 255, 255, 0.035);
                border-radius: 2px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6a68ff,
                    stop:1 #c15cff
                );
                border-radius: 2px;
            }
            """
        )
        footer_container.addWidget(self.progress_bar)

        self.start_btn = QPushButton("START SESSION")
        self.start_btn.setMinimumHeight(52)
        self.start_btn.setMinimumWidth(260)
        self.start_btn.setMaximumWidth(420)
        self.start_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(self.START_BUTTON_STYLE)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        footer_container.addWidget(self.start_btn, 0, Qt.AlignmentFlag.AlignCenter)

        footer_container.addSpacing(10)
        layout.addLayout(footer_container)

    def set_provider_statuses(self, statuses: dict):
        while self.model_bar_layout.count():
            item = self.model_bar_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not statuses:
            badge = QLabel("WAITING FOR PROVIDERS")
            badge.setStyleSheet(
                "color: #64748b; font-size: 9px; font-weight: 900; letter-spacing: 1.8px;"
            )
            self.model_bar_layout.addWidget(badge)
            return

        for pid, info in statuses.items():
            state = info.get("state", "unknown")
            color = (
                "#4ade80"
                if state == "active"
                else "#f59e0b"
                if state == "cooldown"
                else "#ef4444"
            )
            badge = QLabel(pid.upper())
            badge.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: 900; letter-spacing: 1.5px;"
            )
            self.model_bar_layout.addWidget(badge)

    def _on_mode_btn_clicked(self, name):
        self.set_current_mode(name)
        self.mode_selected.emit(name)

    def set_current_mode(self, name):
        """Force the active mode button to the highlighted style."""
        if not name:
            return

        target = str(name).strip().lower()
        logger.debug("Injecting Mode Highlight: '%s'", target)

        for mode_name, btn in self.mode_buttons.items():
            active = mode_name == target
            btn.setChecked(active)
            btn.setStyleSheet(self.STYLE_ACTIVE if active else self.STYLE_INACTIVE)

    def _on_audio_btn_clicked(self, name):
        self.set_current_audio_source(name)
        self.audio_source_changed.emit(name)

    def set_current_audio_source(self, name):
        """Force the active audio source button to the highlighted style."""
        if not name:
            return

        target = str(name).strip().lower()
        logger.debug("Injecting Audio Highlight: '%s'", target)

        for source_name, btn in self.audio_btns.items():
            active = source_name == target
            btn.setChecked(active)
            btn.setStyleSheet(self.STYLE_ACTIVE if active else self.STYLE_INACTIVE)

    def set_warmup_status(self, message: str, progress: int = 0, ready: bool = False):
        self.subtitle.setText(message.upper())
        self.progress_bar.setValue(progress)
        self.start_btn.setEnabled(ready)
        if ready:
            self.start_btn.setText("SESSION READY")
            self.start_btn.setStyleSheet(self.START_BUTTON_READY_STYLE)
        else:
            self.start_btn.setText("START SESSION")
            self.start_btn.setStyleSheet(self.START_BUTTON_STYLE)

    def _create_status_pill(self, label, status):
        frame = QFrame()
        frame.setFixedHeight(30)
        frame.setStyleSheet(
            "background: rgba(255,255,255,0.055); border: 1px solid rgba(255,255,255,0.03); border-radius: 15px;"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(15, 0, 15, 0)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel(label)
        text_label = QLabel(status)
        icon_label.setStyleSheet("font-size: 11px;")
        text_label.setStyleSheet(
            "font-size: 10px; color: #8f9bff; font-weight: 900; letter-spacing: 1.2px;"
        )
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return frame

    def _apply_initial_highlights(self):
        """Hardened boot sync with forced defaults."""
        mode, audio = self._resolve_initial_selection()
        if not self._boot_sync_logged:
            logger.info("Boot sync: mode='%s', audio='%s'", mode, audio)
            self._boot_sync_logged = True
        self.set_current_mode(mode)
        self.set_current_audio_source(audio)

    def refresh_highlights(self, mode=None, audio=None):
        """Force refresh selection states from explicit values or resolved state."""
        resolved_mode, resolved_audio = self._resolve_initial_selection()
        self.set_current_mode(mode or resolved_mode)
        self.set_current_audio_source(audio or resolved_audio)

    def _resolve_initial_selection(self):
        mode = "general"
        audio = "system"

        parent = self.parent()
        app = getattr(parent, "app", None)

        if app and hasattr(app, "state"):
            state = app.state
            mode = (getattr(state, "mode", None) or mode).strip().lower()
            audio = (getattr(state, "audio_source", None) or audio).strip().lower()

        if app and hasattr(app, "config"):
            mode = (app.config.get("ai.mode", mode) or mode).strip().lower()
            audio = (app.config.get("capture.audio.mode", audio) or audio).strip().lower()

        return mode or "general", audio or "system"

    def _schedule_boot_sync(self):
        if self._boot_sync_scheduled:
            return
        self._boot_sync_scheduled = True
        for delay_ms in (100, 500, 1500):
            QTimer.singleShot(delay_ms, self._apply_initial_highlights)
