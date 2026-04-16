"""
Premium Standby View - OpenAssist AI v4.1 (Midnight Hardened).
RESTORATION: High-density horizontal grid, Active Model Dashboard.
ENHANCEMENT: Borderless status pills for a cleaner 'flat' aesthetic.
FIXED: Removed square borders from audio/stealth indicators.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QProgressBar, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont

class StandbyView(QWidget):
    start_clicked = pyqtSignal()
    mode_selected = pyqtSignal(str)
    audio_source_changed = pyqtSignal(str)

    # PREMIUM DARK PALETTE
    NORMAL_STYLE = """
        QPushButton {
            color: #94a3b8; 
            background: rgba(30,32,50,220); 
            border: 1px solid rgba(80,85,255,20); 
            border-radius: 10px; 
            padding: 8px; 
            font-size: 11px;
            font-weight: 500;
        }
        QPushButton:hover {
            background: rgba(45,50,90,240);
            border: 1px solid rgba(120,130,255,40);
            color: white;
        }
    """
    ACTIVE_STYLE = """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
            color: white; 
            border: 1px solid rgba(255,255,255,25);
            font-weight: bold; 
            border-radius: 10px; 
            padding: 8px; 
            font-size: 11px;
        }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_status_widgets = {}
        self.mode_buttons = {}
        self.audio_btns = {}
        self._init_ui()
        QTimer.singleShot(100, self._apply_initial_highlights)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(30, 5, 30, 50)
        layout.setSpacing(18) 

        # 1. Hero
        self.hero_label = QLabel("🧠")
        self.hero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_label.setStyleSheet("font-size: 86px; background: transparent; margin-bottom: 2px;")
        layout.addWidget(self.hero_label)

        self.subtitle = QLabel("INITIALIZING ENGINES...")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet("font-size: 10px; color: #64748b; letter-spacing: 5px; font-weight: bold;")
        layout.addWidget(self.subtitle)

        # 2. Status Row (Borderless Pills)
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        self.mic_pill = self._create_status_pill("🎙️", "STABLE")
        self.ai_pill = self._create_status_pill("✨", "SYNCED")
        self.stealth_pill = self._create_status_pill("🛡️", "ACTIVE")
        status_layout.addWidget(self.mic_pill)
        status_layout.addWidget(self.ai_pill)
        status_layout.addWidget(self.stealth_pill)
        layout.addLayout(status_layout)

        # 3. Mode Selection Grid (MOVED UP)
        lbl_m = QLabel("INTERACTION MODES")
        lbl_m.setStyleSheet("font-size: 10px; color: #475569; font-weight: 800; letter-spacing: 2px;")
        lbl_m.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_m)

        self.mode_grid = QGridLayout()
        self.mode_grid.setSpacing(10)
        modes = [
            ("🧠 General", "general", 0, 0), ("🎯 Interview", "interview", 0, 1), ("💻 Coding", "coding", 0, 2),
            ("🤝 Meeting", "meeting", 1, 0), ("🎓 Exam", "exam", 1, 1), ("✍️ Writing", "writing", 1, 2)
        ]
        for label, name, r, c in modes:
            btn = QPushButton(label)
            btn.setCheckable(True); btn.setFixedHeight(36)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self.NORMAL_STYLE)
            btn.clicked.connect(lambda checked, n=name: self._on_mode_btn_clicked(n))
            self.mode_grid.addWidget(btn, r, c)
            self.mode_buttons[name] = btn
        layout.addLayout(self.mode_grid)

        # 4. Audio Selection (MOVED UP)
        lbl_a = QLabel("AUDIO CAPTURE ENGINE")
        lbl_a.setStyleSheet("font-size: 10px; color: #475569; font-weight: 800; letter-spacing: 2px;")
        lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_a)

        audio_layout = QHBoxLayout(); audio_layout.setSpacing(10)
        sources = [("🎙️ Microphone", "mic"), ("🔊 System Audio", "system"), ("🌐 Full Access", "both")]
        for label, name in sources:
            btn = QPushButton(label)
            btn.setCheckable(True); btn.setFixedHeight(36)
            btn.setStyleSheet(self.NORMAL_STYLE)
            btn.clicked.connect(lambda checked, n=name: self._on_audio_btn_clicked(n))
            audio_layout.addWidget(btn)
            self.audio_btns[name] = btn
        layout.addLayout(audio_layout)

        # 5. Active Models Bar (MOVED DOWN)
        self.model_bar = QFrame()
        self.model_bar.setFixedHeight(30)
        self.model_bar.setStyleSheet("background: rgba(0,0,0,0.3); border-radius: 15px; border: none;")
        self.model_bar_layout = QHBoxLayout(self.model_bar)
        self.model_bar_layout.setContentsMargins(15, 0, 15, 0)
        self.model_bar_layout.setSpacing(10)
        self.model_bar_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_bar)

        layout.addSpacing(30) # Maintain space between content and footer

        footer_container = QVBoxLayout()
        footer_container.setSpacing(8) # REDUCED: Tighter grouping
        footer_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background: rgba(255,255,255,0.05); border-radius: 4px; border: none; } 
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed); border-radius: 4px; }
        """)
        footer_container.addWidget(self.progress_bar)

        self.start_btn = QPushButton("START SESSION")
        self.start_btn.setFixedSize(300, 50) # SLIGHTLY REDUCED: Better proportion
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 25px; font-weight: 900; font-size: 11px; letter-spacing: 3px;
                border: 1px solid rgba(255,255,255,10);
            }
            QPushButton:hover:enabled { background: #6366f1; border: 1px solid rgba(255,255,255,40); }
            QPushButton:disabled { background: rgba(255,255,255,0.02); color: rgba(255, 255, 255, 0.1); border: none; }
        """)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        footer_container.addWidget(self.start_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(footer_container)
        layout.addStretch(1) # Puhses everything UP

    def set_provider_statuses(self, statuses: dict):
        while self.model_bar_layout.count():
            item = self.model_bar_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        
        for pid, info in statuses.items():
            state = info.get("state", "unknown")
            color = "#4ade80" if state == "active" else "#f59e0b" if state == "cooldown" else "#ef4444"
            badge = QLabel(pid.upper())
            badge.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: 900; letter-spacing: 1.5px;")
            self.model_bar_layout.addWidget(badge)

    def _on_mode_btn_clicked(self, name):
        self.set_current_mode(name)
        self.mode_selected.emit(name)

    def set_current_mode(self, name):
        for m_name, btn in self.mode_buttons.items():
            is_active = (m_name == name)
            btn.setCheckedUnsafe = True if is_active else False # Logic only
            btn.setChecked(is_active)
            btn.setStyleSheet(self.ACTIVE_STYLE if is_active else self.NORMAL_STYLE)

    def _on_audio_btn_clicked(self, name):
        self.set_current_audio_source(name)
        self.audio_source_changed.emit(name)

    def set_current_audio_source(self, name):
        for n, btn in self.audio_btns.items():
            is_active = (n == name)
            btn.setChecked(is_active)
            btn.setStyleSheet(self.ACTIVE_STYLE if is_active else self.NORMAL_STYLE)

    def set_warmup_status(self, message: str, progress: int = 0, ready: bool = False):
        self.subtitle.setText(message.upper())
        self.progress_bar.setValue(progress)
        self.start_btn.setEnabled(ready)
        if ready:
            self.start_btn.setText("SESSION READY")
            # Smooth transition to emerald success state
            curr_ss = self.start_btn.styleSheet()
            new_ss = curr_ss.replace("#4f46e5", "#10b981").replace("#7c3aed", "#059669")
            self.start_btn.setStyleSheet(new_ss)
        else:
            self.start_btn.setText("START SESSION")
            # Restore indigo preparation state
            curr_ss = self.start_btn.styleSheet()
            new_ss = curr_ss.replace("#10b981", "#4f46e5").replace("#059669", "#7c3aed")
            self.start_btn.setStyleSheet(new_ss)

    def _create_status_pill(self, label, status):
        # Clean Borderless Pill
        f = QFrame(); f.setFixedHeight(30)
        f.setStyleSheet("background: rgba(255,255,255,0.05); border: none; border-radius: 15px;")
        l = QHBoxLayout(f); l.setContentsMargins(15, 0, 15, 0); l.setSpacing(5)
        l.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        t1 = QLabel(label); t1.setStyleSheet("font-size: 11px;")
        t2 = QLabel(status); t2.setStyleSheet("font-size: 10px; color: #818cf8; font-weight: 900;")
        l.addWidget(t1); l.addWidget(t2)
        f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return f

    def _apply_initial_highlights(self):
        self.set_current_mode("general")
        self.set_current_audio_source("system")
