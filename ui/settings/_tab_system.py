from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea, QFrame
from PyQt6.QtCore import Qt
from ui.custom_widgets import PremiumCheckBox
from ui.settings.constants import TEXT_MUTED


class SystemTabMixin:
    def _tab_system(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(14, 16, 14, 16)
        l.setSpacing(20)
        w.setStyleSheet("background: transparent;")

        lbl_launch = self._make_section_label("LAUNCH BEHAVIOR")
        l.addWidget(lbl_launch)

        self.chk_start_minimized = PremiumCheckBox("Launch in background (system tray)")
        self.chk_start_minimized.setChecked(
            self.config.get("app.start_minimized", False)
        )
        l.addWidget(self.chk_start_minimized)

        desc_tray = QLabel(
            "Keeps OpenAssist in the system tray on launch instead of opening the HUD immediately."
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

        sep_tools = QFrame()
        sep_tools.setFixedHeight(1)
        sep_tools.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep_tools)

        lbl_tools = self._make_section_label("MAINTENANCE")
        l.addWidget(lbl_tools)

        tools_desc = QLabel(
            "Reserved for future repair, debug, cache, and recovery actions so system-level controls stay grouped in one place."
        )
        tools_desc.setWordWrap(True)
        tools_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(tools_desc)

        sep_reset = QFrame()
        sep_reset.setFixedHeight(1)
        sep_reset.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep_reset)

        lbl_reset = self._make_section_label("DANGER ZONE")
        l.addWidget(lbl_reset)

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
        scroll.setWidget(w)
        return scroll
