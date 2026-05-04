"""
Onboarding Wizard - OpenAssist AI v4.1
First-run setup wizard to guide users through initial configuration.
"""

import os
import traceback
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QScrollArea,
    QProgressBar,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from ui.custom_widgets import PremiumCheckBox
from PyQt6.QtGui import QFont, QDesktopServices
from PyQt6.QtCore import QUrl
from core.constants import PROVIDERS
from ui.settings_view import ProviderTestWorker
from utils.logger import setup_logger

logger = setup_logger(__name__)

STYLE_CARD = """
    QFrame {
        background: rgba(255,255,255,7);
        border-radius: 12px;
        border: 1px solid rgba(80,85,255,25);
        padding: 20px;
    }
"""

STYLE_BTN_PRIMARY = """
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed);
        color: white;
        border-radius: 12px;
        font-weight: 800;
        font-size: 12px;
        padding: 15px 30px;
        border: none;
    }
    QPushButton:hover {
        background: #6366f1;
        border: 1px solid rgba(255,255,255,20);
    }
"""

STYLE_BTN_SECONDARY = """
    QPushButton {
        background: rgba(255,255,255,12);
        color: #94a3b8;
        border-radius: 12px;
        font-weight: 600;
        font-size: 11px;
        padding: 12px 24px;
        border: 1px solid rgba(255,255,255,25);
    }
    QPushButton:hover {
        background: rgba(255,255,255,20);
        color: white;
    }
"""

STYLE_LINK_BUTTON = """
    QPushButton {
        background: rgba(56,189,248,30);
        color: #bae6fd;
        border-radius: 10px;
        font-weight: 700;
        font-size: 11px;
        padding: 0 16px;
        border: 1px solid rgba(56,189,248,71);
    }
    QPushButton:hover {
        background: rgba(56,189,248,45);
        border: 1px solid rgba(125,211,252,127);
        color: white;
    }
"""

STYLE_TEST_BUTTON = """
    QPushButton {
        background: rgba(80,85,255,45);
        color: #f8fafc;
        border-radius: 8px;
        font-size: 10px;
        font-weight: 800;
        padding: 0 14px;
        border: 1px solid rgba(129,140,248,140);
    }
    QPushButton:hover:enabled {
        background: rgba(80,85,255,89);
        color: white;
        border: 1px solid rgba(165,180,252,216);
    }
    QPushButton:pressed {
        background: rgba(67,56,202,140);
    }
    QPushButton:disabled {
        background: rgba(255,255,255,7);
        color: rgba(255,255,255,51);
        border: 1px solid rgba(255,255,255,12);
    }
"""

STYLE_BTN_SKIP = """
    QPushButton {
        background: transparent;
        color: #64748b;
        border: none;
        font-size: 10px;
        padding: 8px;
    }
    QPushButton:hover {
        color: #94a3b8;
    }
"""

STYLE_INPUT = """
    QLineEdit, QComboBox {
        background: rgba(0,0,0,76);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,25);
        border-radius: 8px;
        padding: 12px;
        font-size: 12px;
    }
    QLineEdit:focus, QComboBox:focus {
        border: 1px solid #6366f1;
    }
"""


TEXT_PRIMARY = "color: #f1f5f9; font-size: 14px; background: transparent;"
TEXT_SECONDARY = "color: #94a3b8; font-size: 12px; background: transparent;"
TEXT_MUTED = "color: #64748b; font-size: 10px; background: transparent;"


class OnboardingWizard(QWidget):
    """Multi-step onboarding wizard for first-time users."""

    finished = pyqtSignal()
    skipped = pyqtSignal()

    def __init__(self, config, app=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.app = app
        self.reset_state()
        self._setup_ui()

    def reset_state(self):
        """Reset internal wizard progress and re-sync with core config."""
        self._current_step = 0
        self._total_steps = 4
        self._provider_test_cooldowns = {}
        self._provider_test_worker = None
        self._provider_preview_statuses = {}
        
        # In-memory state for "Save-at-End" logic
        self.wizard_state = {
            "provider": self.config.get("ai.fixed_provider", "groq"),
            "api_keys": {
                "groq": self.config.get_api_key("groq") or "",
                "cerebras": self.config.get_api_key("cerebras") or "",
                "gemini": self.config.get_api_key("gemini") or "",
                "ollama": "",
            },
            "provider_test_results": {},
            "ai_mode": self.config.get("ai.mode", "general"),
            "audio_mode": self.config.get("capture.audio.mode", "system"),
            "gaze_enabled": self.config.get("app.gaze_fade.enabled", False),
        }

    def reset(self):
        """Public method to restart the wizard from Screen 0."""
        self.reset_state()
        self._show_step(0)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 15)
        layout.setSpacing(8)

        # Header with progress
        header = QFrame()
        header.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)

        lbl_title = QLabel("🧠 OPENASSIST SETUP")
        lbl_title.setStyleSheet(
            "color: #f1f5f9; font-size: 14px; font-weight: 800; background: transparent;"
        )
        hl.addWidget(lbl_title)

        hl.addStretch()

        self.step_indicator = QLabel("1 / 4")
        self.step_indicator.setStyleSheet(
            "color: #6366f1; font-size: 11px; font-weight: 700; background: transparent;"
        )
        hl.addWidget(self.step_indicator)

        layout.addWidget(header)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(25)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet("""
            QProgressBar { background: rgba(255,255,255,12); border-radius: 2px; border: none; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed); border-radius: 2px; }
        """)
        layout.addWidget(self.progress)

        # Content area (Scrollable)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Flicker Mitigation: Identify viewport as translucent to prevent Windows compositor stutters
        self.scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.scroll.viewport().setAutoFillBackground(False)

        self.content_area = QFrame()
        self.content_area.setStyleSheet("background: transparent; border: none;")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 10, 12, 10)
        self.content_layout.setSpacing(0) # Controlled spacing only
        
        self.scroll.setWidget(self.content_area)
        layout.addWidget(self.scroll)

        # Truncate stretch to ensure content sits tighter next to nav buttons
        layout.addSpacing(4)

        # Navigation buttons
        nav = QHBoxLayout()
        nav.setSpacing(15)

        self.btn_skip = QPushButton("Skip Setup")
        self.btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip.setStyleSheet(STYLE_BTN_SKIP)
        self.btn_skip.clicked.connect(self._on_skip)
        nav.addWidget(self.btn_skip)

        nav.addStretch()

        self.btn_back = QPushButton("← Back")
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_back.setStyleSheet(STYLE_BTN_SECONDARY)
        self.btn_back.clicked.connect(self._on_back)
        self.btn_back.setVisible(False)
        nav.addWidget(self.btn_back)

        self.btn_next = QPushButton("Continue →")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.setStyleSheet(STYLE_BTN_PRIMARY)
        self.btn_next.clicked.connect(self._on_next)
        nav.addWidget(self.btn_next)

        layout.addLayout(nav)

        # Show first step
        self._show_step(0)

    def _show_step(self, step):
        """Display the given step content."""
        self._current_step = step

        # Update progress
        self.progress.setValue(int((step + 1) / self._total_steps * 100))
        self.step_indicator.setText(f"{step + 1} / {self._total_steps}")

        # Recursively clear previous content (including nested layouts)
        def clear_layout(layout):
            if layout is None: return
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    clear_layout(item.layout())

        clear_layout(self.content_layout)

        # Show/hide back button
        self.btn_back.setVisible(step > 0)

        # Update button text
        if step == self._total_steps - 1:
            self.btn_next.setText("Get Started →")
        else:
            self.btn_next.setText("Continue →")

        # Show step content
        if step == 0:
            self._step_welcome()
        elif step == 1:
            self._step_ai_provider()
        elif step == 2:
            self._step_audio()
        elif step == 3:
            self._step_complete()

    def _step_welcome(self):
        """Welcome step - introduce the app."""
        self.content_layout.addStretch(1) # Top Balance

        icon = QLabel("🧠")
        icon.setStyleSheet("font-size: 80px; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(icon)

        title = QLabel("Welcome to OpenAssist AI")
        title.setStyleSheet(
            "color: #f1f5f9; font-size: 16px; font-weight: 800; background: transparent;"
        )
        title.setWordWrap(True)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(title)

        desc = QLabel(
            "Your free, privacy-focused AI assistant that works directly on your screen. "
            "Quick setup will take under a minute."
        )
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; text-align: center; line-height: 1.2;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.content_layout.addSpacing(10) # Gap after icon
        self.content_layout.addWidget(title)
        self.content_layout.addSpacing(10)  # Tiny gap for tagline
        self.content_layout.addWidget(desc)

        # Features list (Left-Aligned, Spacing-Optimized)
        features = [
            ("🧠", "Smart AI Context Awareness"),
            ("🎙️", "Real-time Audio Transcription"),
            ("👁️", "Screen Understanding & OCR"),
            ("🔒", "Privacy-first: Data stays local"),
        ]
        
        # Feature Block Centering Wrap
        center_wrap = QHBoxLayout()
        center_wrap.addStretch()
        
        feature_container = QWidget()
        feature_container.setStyleSheet("background: transparent;")
        f_vbox = QVBoxLayout(feature_container)
        f_vbox.setSpacing(10)
        f_vbox.setContentsMargins(0, 0, 0, 0)

        for icon_text, text in features:
            row = QHBoxLayout()
            row.setSpacing(15) # Tighter internal grouping
            
            lbl = QLabel(icon_text)
            lbl.setStyleSheet("font-size: 16px; background: transparent;")
            row.addWidget(lbl)
            
            t = QLabel(text)
            t.setStyleSheet("color: #e2e8f0; font-size: 12px; font-weight: 500;")
            row.addWidget(t)
            
            f_vbox.addLayout(row)
            
        center_wrap.addWidget(feature_container)
        center_wrap.addStretch()
            
        self.content_layout.addSpacing(40) # Gap before list
        self.content_layout.addLayout(center_wrap)
        
        # Balanced Spacing: Push everything slightly up from center
        self.content_layout.addStretch(2)

    def _step_ai_provider(self):
        """AI provider selection step."""
        self.content_layout.addStretch(1) # Top Balance

        title = QLabel("Choose Your AI Engine")
        title.setStyleSheet(
            "color: #f1f5f9; font-size: 18px; font-weight: 800; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(title)

        desc = QLabel(
            "Select an AI provider. Groq is fast. Ollama is for privacy."
        )
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; text-align: center; line-height: 1.2;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.content_layout.addSpacing(5) # Group desc tightly with title
        self.content_layout.addWidget(desc)
        
        self.content_layout.addSpacing(50) # Clear gap before inputs

        # Provider selection
        self.provider_combo = QComboBox()
        providers = ["groq", "cerebras", "gemini", "ollama"]
        self.provider_combo.addItems(
            [
                "🚀 Groq Cloud (Fastest)",
                "⚡ Cerebras AI (High Speed)",
                "🧠 Google Gemini (Quality)",
                "🏠 Ollama (Local/Offline)",
            ]
        )
        # Restore selection
        current_p = self.wizard_state.get("provider", "groq")
        if current_p in providers:
            self.provider_combo.setCurrentIndex(providers.index(current_p))
            
        self.provider_combo.setStyleSheet(STYLE_INPUT)
        self.provider_combo.setFixedWidth(280)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.content_layout.addWidget(self.provider_combo, 0, Qt.AlignmentFlag.AlignCenter)

        self.content_layout.addSpacing(20)

        self.provider_card = QFrame()
        self.provider_card.setFixedWidth(300)
        self.provider_card.setStyleSheet(
            "QFrame {"
            "background: rgba(255,255,255,5);"
            "border-radius: 12px;"
            "border: 1px solid rgba(255,255,255,10);"
            "padding: 10px;"
            "}"
        )
        card_layout = QVBoxLayout(self.provider_card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(8)

        card_top = QHBoxLayout()
        card_top.setContentsMargins(0, 0, 0, 0)
        card_top.setSpacing(10)

        self.provider_card_label = QLabel("Connection")
        self.provider_card_label.setStyleSheet(
            "color: #a0a0cc; font-weight: 700; font-size: 11px; background: transparent;"
        )
        card_top.addWidget(self.provider_card_label)
        card_top.addStretch()

        self.provider_status_icon = QLabel("\u26aa")
        self.provider_status_icon.setStyleSheet(
            "background: transparent; font-size: 11px; color: #64748b;"
        )
        card_top.addWidget(self.provider_status_icon)

        self.provider_test_btn = QPushButton("TEST")
        self.provider_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.provider_test_btn.setStyleSheet(STYLE_TEST_BUTTON)
        self.provider_test_btn.setFixedSize(68, 28)
        self.provider_test_btn.clicked.connect(self._test_selected_provider)
        card_top.addWidget(self.provider_test_btn)
        card_layout.addLayout(card_top)

        self.provider_link_btn = QPushButton("Get API key")
        self.provider_link_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.provider_link_btn.setStyleSheet(STYLE_LINK_BUTTON)
        self.provider_link_btn.setFixedHeight(30)
        self.provider_link_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.provider_link_btn.clicked.connect(self._open_provider_link)
        card_layout.addWidget(self.provider_link_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self.lbl_key = QLabel("API Key")
        self.lbl_key.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        card_layout.addWidget(self.lbl_key)

        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.setSpacing(8)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter your API key...")
        self.api_key_input.setStyleSheet(STYLE_INPUT)
        self.api_key_input.setFixedHeight(42)
        self.api_key_input.setText(self.wizard_state["api_keys"].get(current_p, ""))
        self.api_key_input.textChanged.connect(self._on_provider_key_changed)
        key_row.addWidget(self.api_key_input)

        lock_lbl = QLabel("\U0001f512")
        lock_lbl.setStyleSheet("background: transparent; font-size: 12px;")
        lock_lbl.setToolTip(
            "Your API key is stored encrypted (AES-128 Fernet) on this machine only. "
            "Never sent to any server."
        )
        key_row.addWidget(lock_lbl)
        card_layout.addLayout(key_row)

        self.provider_detail = QLabel("Not tested yet")
        self.provider_detail.setWordWrap(True)
        self.provider_detail.setStyleSheet("color: #64748b; font-size: 10px; background: transparent;")
        card_layout.addWidget(self.provider_detail)

        self.input_container = self.provider_card
        self.content_layout.addWidget(self.input_container, 0, Qt.AlignmentFlag.AlignCenter)

        self._refresh_provider_link(current_p)
        self._apply_provider_test_state(current_p)

        self.content_layout.addSpacing(10)

        hint = QLabel(
            "💡 Get free API keys from provider websites."
        )
        hint.setStyleSheet(f"{TEXT_MUTED}; font-size: 9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(hint)
        
        self.content_layout.addStretch(2) # Balanced Spacing

    def _step_audio(self):
        """Audio configuration step."""
        self.content_layout.addStretch(1) # Top Balance

        title = QLabel("Audio Capture Setup")
        title.setStyleSheet(
            "color: #f1f5f9; font-size: 18px; font-weight: 800; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(title)

        desc = QLabel(
            "Configure how OpenAssist listens to audio."
        )
        desc.setStyleSheet("color: #94a3b8; font-size: 11px; text-align: center; line-height: 1.2;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.content_layout.addSpacing(2) # Group tightly with title
        self.content_layout.addWidget(desc)
        
        self.content_layout.addSpacing(30) # Clear gap before inputs

        lbl_ai_mode = QLabel("AI Mode")
        lbl_ai_mode.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        lbl_ai_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(lbl_ai_mode)

        self.content_layout.addSpacing(4)

        self.ai_mode = QComboBox()
        self.ai_mode.addItems(
            [
                "General",
                "Interview",
                "Coding",
                "Meeting",
                "Exam",
                "Writing",
            ]
        )
        self.ai_mode.setStyleSheet(STYLE_INPUT)
        self.ai_mode.setFixedWidth(280)
        ai_modes = ["general", "interview", "coding", "meeting", "exam", "writing"]
        current_ai_mode = self.wizard_state.get("ai_mode", "general")
        if current_ai_mode in ai_modes:
            self.ai_mode.setCurrentIndex(ai_modes.index(current_ai_mode))
        self.content_layout.addWidget(self.ai_mode, 0, Qt.AlignmentFlag.AlignCenter)

        self.content_layout.addSpacing(20)

        # Audio source grouping
        lbl_mode = QLabel("Audio Source")
        lbl_mode.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
        lbl_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(lbl_mode)
        
        self.content_layout.addSpacing(4)

        self.audio_mode = QComboBox()
        self.audio_mode.addItems(
            [
                "🖥️ System Audio (Recommended)",
                "🎙️ Microphone Only",
                "🌐 Both System & Microphone",
            ]
        )
        self.audio_mode.setStyleSheet(STYLE_INPUT)
        self.audio_mode.setFixedWidth(280)
        
        # Restore mode
        modes = ["system", "mic", "both"]
        current_m = self.wizard_state.get("audio_mode", "system")
        if current_m in modes:
            self.audio_mode.setCurrentIndex(modes.index(current_m))
            
        self.content_layout.addWidget(self.audio_mode, 0, Qt.AlignmentFlag.AlignCenter)

        self.content_layout.addSpacing(12)

        live_note = QLabel(
            "Live Mode is fastest with Gemini; standard mode is best for the usual transcript-first flow."
        )
        live_note.setWordWrap(True)
        live_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_note.setToolTip(
            "Live Mode keeps one Gemini Live session open for low-latency audio replies.\n"
            "Standard mode waits for the transcript, then sends it through the normal provider pipeline."
        )
        live_note.setStyleSheet(
            "color: #94a3b8; font-size: 10px; background: rgba(255,255,255,6); "
            "border: 1px solid rgba(255,255,255,12); border-radius: 10px; padding: 10px 12px;"
        )
        self.content_layout.addWidget(live_note)

        self.content_layout.addSpacing(20)

        # Gaze detection toggle (Custom Painted Widget)
        self.chk_gaze = PremiumCheckBox("Enable gaze-based window fading")
        self.chk_gaze.setChecked(self.wizard_state.get("gaze_enabled", False))
        self.content_layout.addWidget(self.chk_gaze, 0, Qt.AlignmentFlag.AlignCenter)

        self.content_layout.addSpacing(2)

        desc_gaze = QLabel(
            "Window fades when your mouse is near."
        )
        desc_gaze.setStyleSheet("color: #64748b; font-size: 9px; text-align: center;")
        desc_gaze.setWordWrap(True)
        desc_gaze.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(desc_gaze)

        self.content_layout.addSpacing(18)

        self.content_layout.addStretch(2) # Balanced Spacing


    def _step_complete(self):
        """Dynamic completion summary."""
        icon = QLabel("✨")
        icon.setStyleSheet("font-size: 40px; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(icon)

        title = QLabel("SYSTEM READY")
        title.setStyleSheet("color: #f1f5f9; font-size: 18px; font-weight: 900;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(title)

        self.content_layout.addSpacing(5)

        # Advanced Summary Dashboard
        dash = QFrame()
        dash.setStyleSheet("background: rgba(255,255,255,7); border-radius: 12px; border: 1px solid rgba(255,255,255,12);")
        dl = QVBoxLayout(dash)
        dl.setContentsMargins(20, 20, 20, 20)
        dl.setSpacing(12)

        # Rows
        def create_summary_row(icon, label_ref_name, initial_val):
            row = QHBoxLayout()
            i_lbl = QLabel(icon)
            i_lbl.setFixedWidth(25)
            i_lbl.setStyleSheet("font-size: 14px; background: transparent;")
            row.addWidget(i_lbl)
            
            v_lbl = QLabel(initial_val)
            v_lbl.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: 600; background: transparent;")
            setattr(self, label_ref_name, v_lbl)
            row.addWidget(v_lbl)
            row.addStretch()
            
            status = QLabel("ACTIVE")
            status.setStyleSheet("color: #10b981; font-size: 9px; font-weight: 800; background: rgba(16,185,129,25); padding: 2px 6px; border-radius: 4px;")
            row.addWidget(status)
            return row

        dl.addLayout(create_summary_row("🧠", "summary_provider", "AI Provider"))
        dl.addLayout(create_summary_row("🎙️", "summary_audio", "Audio Source"))
        dl.addLayout(create_summary_row("👁️", "summary_gaze", "Gaze Sync"))
        
        self.content_layout.addWidget(dash)
        
        # Immediate prompt
        hint = QLabel("Settings will be applied instantly.")
        hint.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(hint)
        
        self._update_summary() 
    def _on_next(self):
        """Handle next button click."""
        if self._current_step < self._total_steps - 1:
            # Save current step data to in-memory state
            self._save_step_data()
            self._show_step(self._current_step + 1)
            return

        try:
            # Final Commitment Screen
            self._save_step_data()

            # --- FINAL COMMIT TO DISK ---
            # 1. AI Provider & Keys
            p = self.wizard_state["provider"]
            self.config.set("ai.fixed_provider", p)
            for prov, k in self.wizard_state["api_keys"].items():
                if k:
                    self.config.set_api_key(prov, k)

            # 2. Audio & UI
            self.config.set("ai.mode", self.wizard_state["ai_mode"])
            self.config.set("capture.audio.mode", self.wizard_state["audio_mode"])
            self.config.set("app.gaze_fade.enabled", self.wizard_state["gaze_enabled"])

            # 3. Mark Completed
            self.config.set("onboarding.completed", True)
            self.config.save()
            logger.info(
                "[Onboarding] Saved provider=%s mode=%s audio=%s",
                self.wizard_state["provider"],
                self.wizard_state["ai_mode"],
                self.wizard_state["audio_mode"],
            )

            # Apply settings immediately to engine
            if self.app:
                self.app.state.mode = self.wizard_state["ai_mode"]
                self.app.state.audio_source = self.wizard_state["audio_mode"]
                logger.info("[Onboarding] Applying settings to running app")
                self.app._apply_settings()
                if hasattr(self.app, "overlay") and hasattr(
                    self.app.overlay, "standby_view"
                ):
                    self.app.overlay.standby_view.refresh_highlights(
                        mode=self.wizard_state["ai_mode"],
                        audio=self.wizard_state["audio_mode"],
                    )
            logger.info("[Onboarding] Complete setup finished; emitting finished signal")
            self.finished.emit()
        except Exception as e:
            logger.error("[Onboarding] Finalize failed: %s", e)
            logger.error(traceback.format_exc())
            self.btn_next.setEnabled(True)
            self.btn_next.setText("Get Started ->")

    def _on_back(self):
        """Handle back button click."""
        if self._current_step > 0:
            # Save current work before going back
            self._save_step_data()
            self._show_step(self._current_step - 1)

    def _on_skip(self):
        """Handle skip button click."""
        # Even on skip, we mark as completed to stop showing the wizard
        self.config.set("onboarding.completed", True)
        self.config.save()
        self.skipped.emit()

    def _save_step_data(self):
        """Update in-memory wizard_state from visible widgets."""
        if self._current_step == 1:
            # AI Provider step
            if hasattr(self, "provider_combo"):
                idx = self.provider_combo.currentIndex()
                providers = ["groq", "cerebras", "gemini", "ollama"]
                selected_provider = providers[idx]
                previous_provider = self.wizard_state.get("provider", selected_provider)

                if hasattr(self, "api_key_input"):
                    self.wizard_state["api_keys"][previous_provider] = (
                        self.api_key_input.text().strip()
                    )
                    if previous_provider != selected_provider:
                        self.wizard_state["api_keys"][selected_provider] = (
                            self.wizard_state["api_keys"].get(selected_provider, "")
                        )

                # Update key for just THIS provider in-memory
                self.wizard_state["provider"] = selected_provider

        elif self._current_step == 2:
            # Audio step
            if hasattr(self, "ai_mode"):
                idx = self.ai_mode.currentIndex()
                ai_modes = ["general", "interview", "coding", "meeting", "exam", "writing"]
                self.wizard_state["ai_mode"] = ai_modes[idx]

            if hasattr(self, "audio_mode"):
                idx = self.audio_mode.currentIndex()
                modes = ["system", "mic", "both"]
                self.wizard_state["audio_mode"] = modes[idx]

            if hasattr(self, "chk_gaze"):
                self.wizard_state["gaze_enabled"] = self.chk_gaze.isChecked()

        # Update summary UI if we are entering or in the summary step
        self._update_summary()

    def _on_provider_changed(self, idx):
        """Update API key input when provider changes in Screen 2. Sync with in-memory state."""
        providers = ["groq", "cerebras", "gemini", "ollama"]
        p = providers[idx]

        prev_provider = self.wizard_state.get("provider", p)
        if hasattr(self, "api_key_input"):
            self.wizard_state["api_keys"][prev_provider] = self.api_key_input.text().strip()
        self.wizard_state["provider"] = p
        
        # In-memory restoration
        self.api_key_input.setText(self.wizard_state["api_keys"].get(p, ""))
        
        self._refresh_provider_link(p)
        self._apply_provider_test_state(p)

    def _refresh_provider_link(self, provider_id: str):
        provider_meta = PROVIDERS.get(provider_id, {})
        url = provider_meta.get("url", "")
        if not url:
            self.provider_link_btn.setVisible(False)
            self.provider_link_btn.setToolTip("")
            self.provider_link_btn.setProperty("provider_url", "")
            return

        link_text = "Install Ollama" if provider_id == "ollama" else "Get API key"
        self.provider_link_btn.setText(link_text)
        self.provider_link_btn.setVisible(True)
        self.provider_link_btn.setProperty("provider_url", url)
        self.provider_card_label.setText(
            "Local engine" if provider_id == "ollama" else "Connection"
        )
        self.lbl_key.setText("Endpoint" if provider_id == "ollama" else "API Key")
        self.api_key_input.setPlaceholderText(
            "Optional Ollama endpoint (default: http://localhost:11434)"
            if provider_id == "ollama"
            else "Enter your API key..."
        )
        self.provider_link_btn.setToolTip(
            f"Open {provider_meta.get('name', provider_id)} to "
            f"{'install the local engine' if provider_id == 'ollama' else 'create or copy an API key'}."
        )

    def _open_provider_link(self):
        url = self.provider_link_btn.property("provider_url") or ""
        if url:
            QDesktopServices.openUrl(QUrl(str(url)))

    def _update_summary(self):
        """Refresh summary labels from in-memory wizard_state with safety guards."""
        try:
            if not hasattr(self, "summary_provider") or not self.summary_provider: 
                return

            # 1. AI Provider & Key mask
            prov_id = self.wizard_state["provider"]
            prov_names = {"groq": "Groq Cloud", "cerebras": "Cerebras AI", "gemini": "Google Gemini", "ollama": "Ollama (Local)"}
            key = self.wizard_state["api_keys"].get(prov_id, "")
            
            # Mask key: first 4 and last 4
            test_result = self.wizard_state.get("provider_test_results", {}).get(prov_id, {})
            verified = bool(test_result.get("success"))
            key_masked = f"{key[:4]}...{key[-4:]}" if key and len(key) > 8 else "Active" if key else "No Key Found"
            if prov_id == "ollama":
                key_masked = "Local Engine Ready" if verified else "Local Engine Not Tested"
            elif verified:
                key_masked = f"{key_masked}  |  Verified"
            
            self.summary_provider.setText(f"{prov_names.get(prov_id, prov_id)}: {key_masked}")

            # 2. Audio Mode
            ai_mode = self.wizard_state["ai_mode"]
            ai_mode_names = {
                "general": "General",
                "interview": "Interview",
                "coding": "Coding",
                "meeting": "Meeting",
                "exam": "Exam",
                "writing": "Writing",
            }
            mode = self.wizard_state["audio_mode"]
            mode_names = {"system": "Desktop Speakers", "mic": "Microphone", "both": "Hybrid (System+Mic)"}
            self.summary_audio.setText(
                f"{ai_mode_names.get(ai_mode, ai_mode)} | {mode_names.get(mode, mode)}"
            )

            # 3. Gaze Fade
            gaze = self.wizard_state["gaze_enabled"]
            self.summary_gaze.setText(f"Gaze Tracking: {'ENABLED' if gaze else 'DISABLED'}")
            
        except (RuntimeError, AttributeError):
            # Widget might be deleted during step transitions, safe to ignore
            pass

    def _on_provider_key_changed(self, text: str):
        provider_id = self.wizard_state.get("provider", "groq")
        self.wizard_state["api_keys"][provider_id] = (text or "").strip()
        if provider_id != "ollama":
            self.provider_test_btn.setEnabled(bool((text or "").strip()))
        self._set_provider_test_idle(provider_id, clear_previous_result=True)

    def _set_provider_test_idle(self, provider_id: str, clear_previous_result: bool = False):
        if not hasattr(self, "provider_test_btn"):
            return
        if provider_id == "ollama":
            self.provider_test_btn.setEnabled(True)
        else:
            self.provider_test_btn.setEnabled(bool(self.api_key_input.text().strip()))
        self.provider_test_btn.setText("TEST")
        if clear_previous_result:
            self.provider_status_icon.setText("\u26aa")
            self.provider_status_icon.setStyleSheet(
                "background: transparent; font-size: 11px; color: #64748b;"
            )
            if provider_id in self.wizard_state.get("provider_test_results", {}):
                self.wizard_state["provider_test_results"].pop(provider_id, None)
                self._provider_preview_statuses.pop(provider_id, None)
                self._push_provider_preview_statuses()
            if hasattr(self, "provider_detail"):
                self.provider_detail.setStyleSheet(
                    "color: #64748b; font-size: 10px; background: transparent;"
                )
                self.provider_detail.setText(
                    "Test the connection before continuing for a more reliable setup."
                    if provider_id != "ollama"
                    else "Check the local Ollama server and discover installed models."
                )

    def _apply_provider_test_state(self, provider_id: str):
        if not hasattr(self, "provider_detail"):
            return
        result = self.wizard_state.get("provider_test_results", {}).get(provider_id)
        if not result:
            self._set_provider_test_idle(provider_id, clear_previous_result=False)
            return

        success = bool(result.get("success"))
        detail = str(result.get("message", "Not tested"))
        self.provider_status_icon.setText("\u2705" if success else "\u274c")
        self.provider_status_icon.setStyleSheet(
            f"background: transparent; font-size: 11px; color: {'#4ade80' if success else '#fda4af'};"
        )
        self.provider_detail.setStyleSheet(
            f"color: {'#4ade80' if success else '#fda4af'}; font-size: 10px; background: transparent;"
        )
        self.provider_detail.setText(detail)
        self.provider_test_btn.setText("RETEST")
        self.provider_test_btn.setEnabled(True if provider_id == "ollama" else bool(self.api_key_input.text().strip()))

    def _test_selected_provider(self):
        provider_id = self.wizard_state.get("provider", "groq")
        if self._provider_test_cooldowns.get(provider_id, False):
            return

        key_override = ""
        if provider_id != "ollama":
            key_override = self.api_key_input.text().strip()
            if not key_override:
                self._set_provider_test_idle(provider_id, clear_previous_result=False)
                self.provider_detail.setStyleSheet(
                    "color: #fda4af; font-size: 10px; background: transparent;"
                )
                self.provider_detail.setText("Enter an API key before testing.")
                self.provider_status_icon.setText("\u274c")
                self.provider_status_icon.setStyleSheet(
                    "background: transparent; font-size: 11px; color: #fda4af;"
                )
                return

        self._provider_test_cooldowns[provider_id] = True
        QTimer.singleShot(
            3000,
            lambda p=provider_id: self._provider_test_cooldowns.update({p: False}),
        )

        self.provider_status_icon.setText("\u23f3")
        self.provider_status_icon.setStyleSheet(
            "background: transparent; font-size: 11px; color: #7dd3fc;"
        )
        self.provider_detail.setStyleSheet(
            "color: #7dd3fc; font-size: 10px; background: transparent;"
        )
        self.provider_detail.setText(
            "Checking local Ollama server..."
            if provider_id == "ollama"
            else "Checking API key and provider connection..."
        )
        self.provider_test_btn.setEnabled(False)
        self.provider_test_btn.setText("TESTING")

        self._provider_test_worker = ProviderTestWorker(
            provider_id,
            self.config,
            self,
            key_override=key_override,
        )
        self._provider_test_worker.result_ready.connect(self._on_provider_test_result)
        self._provider_test_worker.finished.connect(self._clear_provider_test_worker)
        self._provider_test_worker.start()

    def _clear_provider_test_worker(self):
        self._provider_test_worker = None

    def _on_provider_test_result(self, provider_id: str, success: bool, message: str, details):
        provider_meta = PROVIDERS.get(provider_id, {})
        label = provider_meta.get("name", provider_id)
        detail_text = message

        if provider_id == "ollama" and isinstance(details, dict):
            models = details.get("models", []) or []
            if success and models:
                preview = ", ".join(models[:2])
                extra = "" if len(models) <= 2 else f" +{len(models) - 2} more"
                detail_text = f"{message} | {preview}{extra}"

        self.wizard_state.setdefault("provider_test_results", {})[provider_id] = {
            "success": success,
            "message": detail_text,
            "details": details,
        }

        if self.wizard_state.get("provider") == provider_id:
            self._apply_provider_test_state(provider_id)

        if success:
            self._provider_preview_statuses[provider_id] = {
                "state": "active",
                "selected": True,
                "usable": True,
                "label": label,
            }
        else:
            self._provider_preview_statuses.pop(provider_id, None)

        self._push_provider_preview_statuses()
        self._update_summary()

    def _push_provider_preview_statuses(self):
        if not self.app or not hasattr(self.app, "overlay") or not hasattr(self.app.overlay, "standby_view"):
            return
        statuses = {}
        selected_provider = self.wizard_state.get("provider", "")
        for pid, info in self._provider_preview_statuses.items():
            statuses[pid] = {
                "state": info.get("state", "active"),
                "selected": pid == selected_provider,
                "usable": bool(info.get("usable", True)),
            }
        self.app.overlay.standby_view.set_provider_statuses(statuses)
