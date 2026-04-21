"""
Premium Standby View - OpenAssist AI v4.1 (Midnight Hardened).
RESTORATION: High-density horizontal grid, Active Model Dashboard.
ENHANCEMENT: Borderless status pills for a cleaner 'flat' aesthetic.
FIXED: Removed square borders from audio/stealth indicators.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QProgressBar,
    QSpacerItem,
    QSizePolicy,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StandbyView(QWidget):
    start_clicked = pyqtSignal()
    mode_selected = pyqtSignal(str)
    audio_source_changed = pyqtSignal(str)

    # MASTER SELECTION STYLES (Explicit Injection)
    STYLE_INACTIVE = """
        QPushButton {
            color: #94a3b8; 
            background: rgba(30,32,45,180); 
            border: 1px solid rgba(80,85,255,15); 
            border-radius: 8px; 
            padding: 8px 12px; 
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        QPushButton:hover {
            background: rgba(45,50,90,240);
            border: 1px solid rgba(120,130,255,40);
            color: white;
        }
    """

    STYLE_ACTIVE = """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
            color: white; 
            border: 1px solid rgba(255,255,255,60);
            border-radius: 8px; 
            padding: 8px 12px; 
            font-size: 10px;
            font-weight: 800; 
            letter-spacing: 1px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_status_widgets = {}
        self.mode_buttons = {}
        self.audio_btns = {}
        
        self._init_ui()
        self._connect_state()
        
        # MASTER SYNC SEQUENCE: Multiple intervals with forced defaults
        self._schedule_boot_sync()

    def _connect_state(self):
        if hasattr(self.parent(), "app"):
            self.parent().app.state.mode_changed.connect(self.set_current_mode)
            self.parent().app.state.audio_source_changed.connect(
                self.set_current_audio_source
            )

    def showEvent(self, event):
        """Final UI sync trigger on window mapping."""
        super().showEvent(event)
        self._schedule_boot_sync()

    def _init_ui(self):
        # MASTER LAYOUT: Global spacing 0; total control via hard spacers
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 45, 30, 45)
        self.main_layout.setSpacing(0) 

        layout = self.main_layout

        # 1. Hero
        self.hero_label = QLabel("🧠")
        self.hero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_label.setStyleSheet("font-size: 70px; background: transparent; margin-bottom: 2px;")
        layout.addWidget(self.hero_label)

        layout.addSpacing(10)

        self.subtitle = QLabel("NEURAL ENGINE INITIALIZING...")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet(
            "font-size: 9px; color: #6366f1; letter-spacing: 3px; font-weight: 900;"
        )
        layout.addWidget(self.subtitle)

        layout.addSpacing(25)

        # 2. Status Row
        status_layout = QHBoxLayout()
        status_layout.setSpacing(12)
        self.mic_pill = self._create_status_pill("🎙️", "STABLE")
        self.ai_pill = self._create_status_pill("✨", "SYNCED")
        self.stealth_pill = self._create_status_pill("🛡️", "ACTIVE")
        status_layout.addWidget(self.mic_pill)
        status_layout.addWidget(self.ai_pill)
        status_layout.addWidget(self.stealth_pill)
        layout.addLayout(status_layout)

        layout.addSpacing(30)

        # 3. Mode Selection - FORCED SPACING Rows
        lbl_m = QLabel("AI MODES")
        lbl_m.setStyleSheet(
            "font-size: 9px; color: #475569; font-weight: 900; letter-spacing: 2.5px;"
        )
        lbl_m.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_m)

        layout.addSpacing(15)

        modes = [
            [("🧠", "GENERAL"), ("🎯", "INTERVIEW")],
            [("💻", "CODING"), ("🤝", "MEETING")],
            [("🎓", "EXAM"), ("✍️", "WRITING")]
        ]
        
        for row_data in modes:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(15)
            for icon, name in row_data:
                btn = QPushButton(f"{icon} {name}")
                btn.setCheckable(True)
                btn.setFixedHeight(38)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda checked, n=name.lower(): self._on_mode_btn_clicked(n))
                # Set initial inactive state
                btn.setStyleSheet(self.STYLE_INACTIVE)
                row_layout.addWidget(btn)
                # Map lowercase cleaned key
                self.mode_buttons[name.strip().lower()] = btn
            layout.addLayout(row_layout)
            layout.addSpacing(15) # EXPLICIT: Hardcoded vertical gap

        layout.addSpacing(10)

        # 4. Audio Selection
        lbl_a = QLabel("CAPTURE SOURCE")
        lbl_a.setStyleSheet(
            "font-size: 9px; color: #475569; font-weight: 900; letter-spacing: 2px;"
        )
        lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_a)

        layout.addSpacing(15)

        audio_layout = QHBoxLayout()
        audio_layout.setSpacing(12)
        sources = [
            ("🎙️ MIC", "mic"),
            ("🔊 SYSTEM", "system"),
            ("🌐 BOTH", "both"),
        ]
        for label, name in sources:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.setStyleSheet(self.STYLE_INACTIVE)
            btn.clicked.connect(lambda checked, n=name: self._on_audio_btn_clicked(n))
            audio_layout.addWidget(btn)
            self.audio_btns[name.strip().lower()] = btn
        layout.addLayout(audio_layout)

        layout.addSpacing(25)

        # 5. Active Models Dashboard
        self.model_bar = QFrame()
        self.model_bar.setFixedHeight(32)
        self.model_bar.setStyleSheet(
            "background: rgba(0,0,0,0.4); border-radius: 16px; border: 1px solid rgba(255,255,255,5);"
        )
        self.model_bar_layout = QHBoxLayout(self.model_bar)
        self.model_bar_layout.setContentsMargins(15, 0, 15, 0)
        self.model_bar_layout.setSpacing(12)
        self.model_bar_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_bar)

        # BOTTOM STRETCH: Enforces that all items Above stay compact and spaced correctly
        layout.addStretch(1) 

        footer_container = QVBoxLayout()
        footer_container.setSpacing(12)
        footer_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background: rgba(255,255,255,0.03); border-radius: 3px; border: none; } 
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed); border-radius: 3px; }
        """)
        footer_container.addWidget(self.progress_bar)

        self.start_btn = QPushButton("SESSION READY")
        self.start_btn.setFixedSize(320, 56)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 28px; font-weight: 900; font-size: 12px; letter-spacing: 3px;
                border: 1px solid rgba(255,255,255,10);
            }
            QPushButton:hover:enabled { background: #6366f1; border: 1px solid rgba(255,255,255,40); }
            QPushButton:disabled { background: rgba(255,255,255,0.02); color: rgba(255, 255, 255, 0.1); border: none; }
        """)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        footer_container.addWidget(self.start_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        footer_container.addSpacing(10)
        layout.addLayout(footer_container)

    def set_provider_statuses(self, statuses: dict):
        while self.model_bar_layout.count():
            item = self.model_bar_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

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
        """EXPLICIT STYLE INJECTION: Force-sets Indigo Glow on selection."""
        if not name: return
        target = str(name).strip().lower()
        logger.debug(f"Injecting Mode Highlight: '{target}'")
        
        for m_name, btn in self.mode_buttons.items():
            active = (m_name == target)
            btn.setChecked(active)
            # DIRECT INJECTION - Deterministic and Bulletproof
            btn.setStyleSheet(self.STYLE_ACTIVE if active else self.STYLE_INACTIVE)

    def _on_audio_btn_clicked(self, name):
        self.set_current_audio_source(name)
        self.audio_source_changed.emit(name)

    def set_current_audio_source(self, name):
        """EXPLICIT STYLE INJECTION: Force-sets Indigo Glow on selection."""
        if not name: return
        target = str(name).strip().lower()
        logger.debug(f"Injecting Audio Highlight: '{target}'")
        
        for n, btn in self.audio_btns.items():
            active = (n == target)
            btn.setChecked(active)
            btn.setStyleSheet(self.STYLE_ACTIVE if active else self.STYLE_INACTIVE)

    def set_warmup_status(self, message: str, progress: int = 0, ready: bool = False):
        self.subtitle.setText(message.upper())
        self.progress_bar.setValue(progress)
        self.start_btn.setEnabled(ready)
        if ready:
            self.start_btn.setText("SESSION READY")
            curr_ss = self.start_btn.styleSheet()
            new_ss = curr_ss.replace("#4f46e5", "#10b981").replace("#7c3aed", "#059669")
            self.start_btn.setStyleSheet(new_ss)
        else:
            self.start_btn.setText("START SESSION")
            curr_ss = self.start_btn.styleSheet()
            new_ss = curr_ss.replace("#10b981", "#4f46e5").replace("#059669", "#7c3aed")
            self.start_btn.setStyleSheet(new_ss)

    def _create_status_pill(self, label, status):
        f = QFrame()
        f.setFixedHeight(30)
        f.setStyleSheet(
            "background: rgba(255,255,255,0.05); border: none; border-radius: 15px;"
        )
        l = QHBoxLayout(f)
        l.setContentsMargins(15, 0, 15, 0)
        l.setSpacing(5)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t1 = QLabel(label)
        t2 = QLabel(status)
        t1.setStyleSheet("font-size: 11px;")
        t2.setStyleSheet("font-size: 10px; color: #818cf8; font-weight: 900;")
        l.addWidget(t1)
        l.addWidget(t2)
        f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return f

    def _apply_initial_highlights(self):
        """Hardened Boot Sync with forced defaults."""
        mode, audio = self._resolve_initial_selection()
        logger.info(f"BOOT SYNC: Mode='{mode}', Audio='{audio}'")
        self.set_current_mode(mode)
        self.set_current_audio_source(audio)

    def refresh_highlights(self, mode=None, audio=None):
        """Force refresh selection pills from explicit values or resolved state."""
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
            audio = (
                app.config.get("capture.audio.mode", audio) or audio
            ).strip().lower()

        return mode or "general", audio or "system"

    def _schedule_boot_sync(self):
        for delay_ms in (100, 500, 1500):
            QTimer.singleShot(delay_ms, self._apply_initial_highlights)
