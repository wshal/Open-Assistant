from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTabWidget, QCheckBox, QComboBox, QSlider,
    QScrollArea, QFrame, QGridLayout, QSizePolicy, QMessageBox,
    QTextEdit, QInputDialog
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from core.constants import PROVIDERS
from ui.custom_widgets import PremiumCheckBox
from ui.settings.constants import (
    BG_DARK, TEXT_PRIMARY, TEXT_MUTED, SS_INPUT, STYLE_BTN_PRIMARY, STYLE_BTN_SECONDARY
)


class UiTabMixin:


    def _tab_ui(self):
        """Display settings including gaze fade."""
        w = QScrollArea()
        w.setWidgetResizable(True)
        w.setStyleSheet("background: transparent; border: none;")
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(20, 20, 20, 20)
        l.setSpacing(20)

        lbl_opacity_main = QLabel("STEALTH VISIBILITY")
        lbl_opacity_main.setStyleSheet(
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; background: transparent;"
        )
        l.addWidget(lbl_opacity_main)

        stealth_opacity_row = QHBoxLayout()
        self.stealth_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.stealth_opacity_slider.setRange(70, 100)
        self.stealth_opacity_slider.setSingleStep(1)
        self.stealth_opacity_slider.setPageStep(5)
        self.stealth_opacity_slider.setValue(
            self._slider_percent(self.config.get("stealth.low_opacity", 0.75))
        )
        self.stealth_opacity_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                background: rgba(255,255,255,20);
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #10b981;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            """
        )
        self.stealth_opacity_value = QLabel()
        self.stealth_opacity_value.setStyleSheet(
            f"{TEXT_PRIMARY} font-size: 10px; font-weight: 700; background: transparent;"
        )
        self._set_opacity_label(
            self.stealth_opacity_value, self.stealth_opacity_slider.value()
        )
        self.stealth_opacity_slider.valueChanged.connect(
            lambda value: self._set_opacity_label(self.stealth_opacity_value, value)
        )
        self.stealth_opacity_slider.valueChanged.connect(
            lambda _: self._preview_window_opacity()
        )
        stealth_opacity_row.addWidget(self.stealth_opacity_slider, 1)
        stealth_opacity_row.addWidget(self.stealth_opacity_value)
        l.addLayout(stealth_opacity_row)

        desc_hud_opacity = QLabel(
            "Primary control for how visible the HUD remains while stealth mode is enabled."
        )
        desc_hud_opacity.setWordWrap(True)
        desc_hud_opacity.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_hud_opacity)

        lbl_normal_opacity = QLabel("NORMAL HUD VISIBILITY")
        lbl_normal_opacity.setStyleSheet(
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; background: transparent;"
        )
        l.addWidget(lbl_normal_opacity)

        hud_opacity_row = QHBoxLayout()
        self.hud_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.hud_opacity_slider.setRange(70, 100)
        self.hud_opacity_slider.setSingleStep(1)
        self.hud_opacity_slider.setPageStep(5)
        self.hud_opacity_slider.setValue(
            self._slider_percent(self.config.get("app.opacity", 0.94))
        )
        self.hud_opacity_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                background: rgba(255,255,255,20);
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #6366f1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            """
        )
        self.hud_opacity_value = QLabel()
        self.hud_opacity_value.setStyleSheet(
            f"{TEXT_PRIMARY} font-size: 10px; font-weight: 700; background: transparent;"
        )
        self._set_opacity_label(self.hud_opacity_value, self.hud_opacity_slider.value())
        self.hud_opacity_slider.valueChanged.connect(
            lambda value: self._set_opacity_label(self.hud_opacity_value, value)
        )
        self.hud_opacity_slider.valueChanged.connect(
            lambda _: self._preview_window_opacity()
        )
        hud_opacity_row.addWidget(self.hud_opacity_slider, 1)
        hud_opacity_row.addWidget(self.hud_opacity_value)
        l.addLayout(hud_opacity_row)

        desc_stealth_opacity = QLabel(
            "Optional fallback for non-stealth use, setup, or when you want the regular HUD more readable."
        )
        desc_stealth_opacity.setWordWrap(True)
        desc_stealth_opacity.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_stealth_opacity)

        # Gaze Fade Section
        lbl_gaze = QLabel("NEURAL GAZE DETECTION")
        lbl_gaze.setStyleSheet(
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; background: transparent;"
        )
        l.addWidget(lbl_gaze)

        self.chk_gaze = PremiumCheckBox("Enable gaze-based window fading")
        self.chk_gaze.setChecked(self.config.get("app.gaze_fade.enabled", False))
        l.addWidget(self.chk_gaze)
        desc_gaze = QLabel(
            "When enabled, the window fades to low opacity when your mouse is near it. "
            "Only active during active sessions — not on standby or settings screens. "
            "Works alongside Stealth Mode; stealth anti-capture remains active."
        )
        desc_gaze.setWordWrap(True)
        desc_gaze.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_gaze)

        self.chk_start_minimized = PremiumCheckBox("Start minimized to system tray")
        self.chk_start_minimized.setChecked(
            self.config.get("app.start_minimized", False)
        )
        l.addWidget(self.chk_start_minimized)
        desc_tray = QLabel(
            "Keep OpenAssist running in the tray on launch instead of opening the HUD immediately."
        )
        desc_tray.setWordWrap(True)
        desc_tray.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_tray)

        self.chk_focus_on_show = PremiumCheckBox("Focus HUD when showing it")
        self.chk_focus_on_show.setChecked(
            self.config.get("app.focus_on_show", False)
        )
        l.addWidget(self.chk_focus_on_show)
        desc_focus_on_show = QLabel(
            "When enabled, showing the HUD brings it to the front and gives it keyboard focus. "
            "When disabled, the HUD stays floating on top without interrupting the app underneath."
        )
        desc_focus_on_show.setWordWrap(True)
        desc_focus_on_show.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_focus_on_show)

        # Detection Margin
        margin_layout = QHBoxLayout()
        lbl_margin = QLabel("Detection margin:")
        lbl_margin.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        margin_layout.addWidget(lbl_margin)

        self.margin_slider = QComboBox()
        self.margin_slider.addItems(
            [
                "20px (sensitive)",
                "30px (mini default)",
                "40px",
                "50px",
                "60px (default)",
                "80px",
            ]
        )
        current_margin = self.config.get("app.gaze_fade.margin", 60)
        margin_map = {20: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5}
        self.margin_slider.setCurrentIndex(margin_map.get(current_margin, 4))
        self.margin_slider.setStyleSheet(SS_INPUT)
        margin_layout.addWidget(self.margin_slider)
        margin_layout.addStretch()
        l.addLayout(margin_layout)

        # Target Opacity
        opacity_layout = QHBoxLayout()
        lbl_opacity = QLabel("Faded opacity:")
        lbl_opacity.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        opacity_layout.addWidget(lbl_opacity)

        self.opacity_slider = QComboBox()
        self.opacity_slider.addItems(["5%", "10% (subtle)", "15%", "20%", "25%"])
        current_opacity = int(
            self.config.get("app.gaze_fade.target_opacity", 0.12) * 100
        )
        opacity_map = {5: 0, 10: 1, 15: 2, 20: 3, 25: 4}
        self.opacity_slider.setCurrentIndex(opacity_map.get(current_opacity, 1))
        self.opacity_slider.setStyleSheet(SS_INPUT)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addStretch()
        l.addLayout(opacity_layout)

        # Reset section (Factory Reset below performs a full first-run reset)
        sep = QFrame()
        sep.setStyleSheet(
            "background: rgba(255,255,255,12); height: 1px; margin: 20px 0;"
        )
        l.addWidget(sep)

        btn_factory_reset = QPushButton("FACTORY RESET")
        btn_factory_reset.setStyleSheet(
            """
            QPushButton {
                background: rgba(220,38,38,35);
                color: #fecaca;
                border-radius: 12px;
                font-weight: 800;
                font-size: 11px;
                padding: 12px 24px;
                border: 1px solid rgba(248,113,113,89);
            }
            QPushButton:hover {
                background: rgba(220,38,38,56);
                color: white;
            }
            """
        )
        btn_factory_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_factory_reset.clicked.connect(self._factory_reset)
        l.addWidget(btn_factory_reset)

        desc_factory_reset = QLabel(
            "Wipes settings, encrypted API keys, history, caches, logs, and sends the app back to first-run onboarding."
        )
        desc_factory_reset.setWordWrap(True)
        desc_factory_reset.setStyleSheet(
            "color: #fca5a5; font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_factory_reset)

        l.addStretch()
        c.setLayout(l)
        w.setWidget(c)
        return w


