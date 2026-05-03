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


class HotkeysTabMixin:


    def _tab_hotkeys(self):
        w = QScrollArea()
        w.setWidgetResizable(True)
        w.setStyleSheet("background: transparent; border: none;")
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(2)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(10, 0, 10, 5)
        h1 = QLabel("COMMAND")
        h1.setStyleSheet(
            f"{TEXT_MUTED} font-size: 9px; font-weight: 800; border: none; background: transparent;"
        )
        h2 = QLabel("SHORTCUT KEY")
        h2.setStyleSheet(
            f"{TEXT_MUTED} font-size: 9px; font-weight: 800; border: none; background: transparent;"
        )
        hdr.addWidget(h1, 1)
        hdr.addWidget(h2, 1)
        l.addLayout(hdr)
        hk_labels = {
            "toggle": "Show/Hide HUD (Single Press)",
            "quick_answer": "Quick Context Answer",
            "analyze_screen": "Analyze Current Screen",
            "history_prev": "Previous History Entry",
            "history_next": "Next History Entry",
            "scroll_up": "Scroll Response Up",
            "scroll_down": "Scroll Response Down",
            "switch_mode": "Rotate AI Modes",
            "toggle_audio": "Mute/Unmute Mic",
            "mini_mode": "Switch to Mini-HUD",
            "toggle_click_through": "Focus Click-Through",
            "emergency_erase": "Emergency System Wipe",
            "move_up": "Glide HUD Up",
            "move_down": "Glide HUD Down",
            "move_left": "Glide HUD Left",
            "move_right": "Glide HUD Right",
        }
        keys_cfg = self.config.get("hotkeys", {})
        for action in sorted(hk_labels.keys()):
            row_frame = QFrame()
            row_frame.setStyleSheet(
                "QFrame { background: rgba(255,255,255,5); border-bottom: 1px solid rgba(255,255,255,7); }"
            )
            rl = QHBoxLayout(row_frame)
            rl.setContentsMargins(10, 8, 10, 8)
            lbl = QLabel(hk_labels[action])
            lbl.setStyleSheet(
                "background: transparent; color: #94a3b8; font-size: 11px; border: none;"
            )
            rl.addWidget(lbl, 1)
            inp = QLineEdit()
            inp.setFixedWidth(140)
            inp.setStyleSheet(
                SS_INPUT
                + "QLineEdit { border: none; background: rgba(0,0,0,51); text-align: center; }"
            )
            inp.setText(keys_cfg.get(action, ""))
            self.hotkey_inputs[action] = inp
            rl.addWidget(inp)
            l.addWidget(row_frame)
        l.addStretch()
        c.setLayout(l)
        w.setWidget(c)
        return w


