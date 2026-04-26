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


class StealthTabMixin:


    def _tab_stealth(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(20, 20, 20, 20)
        l.setSpacing(15)
        w.setStyleSheet("background: transparent;")
        lbl = QLabel("GHOST ANTI-RECORDING")
        lbl.setStyleSheet("background: transparent;")
        l.addWidget(lbl)
        self.chk_ghost = PremiumCheckBox("Mask overlay from Screen Recorders (Always On)")
        self.chk_ghost.setChecked(True)
        self.chk_ghost.setEnabled(False)
        l.addWidget(self.chk_ghost)
        desc = QLabel(
            "Stealth protection is enforced by default so Zoom, Teams, Meet, and OBS should not capture the overlay."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(desc)
        l.addStretch()
        return w


