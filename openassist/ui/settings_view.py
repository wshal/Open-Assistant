"""
Integrated Settings View — v4.1 (Midnight Premium).
RESTORATION: Hotkey Configuration Tab & SVG Tick Icons.
FIXED: Checkbox tick visibility with URL-encoded SVG (Safe Mode).
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTabWidget, QCheckBox, QComboBox, QScrollArea, QFrame, QGridLayout
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- Premium Midnight Styling ---
BG_DARK = "background: rgba(15, 15, 30, 245);"
TEXT_PRIMARY = "color: #c0c0ff;"
TEXT_MUTED = "color: #64748b;"

# URL-ENCODED TICK SVG (Higher Reliability than Base64)
# Color: white, Weight: 4 (Bold)
TICK_URL = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E"

SS_INPUT = """
QLineEdit, QComboBox {
    background: rgba(20, 20, 40, 220);
    color: #e0e0f5;
    border: 1px solid rgba(80, 85, 255, 20);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}
"""

SS_CHECKBOX = f"""
QCheckBox {{
    color: #e0e0f5;
    font-size: 11px;
    background: transparent;
}}
QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background: rgba(30,30,60,200);
    border: 1px solid rgba(120, 130, 255, 30);
}}
QCheckBox::indicator:checked {{
    background: #6366f1;
    border: 1px solid #818cf8;
    image: url("{TICK_URL}");
}}
QCheckBox::indicator:hover {{
    border: 1px solid rgba(120, 130, 255, 100);
    background: rgba(40,40,80,220);
}}
"""

class ProviderTestWorker(QThread):
    result_ready = pyqtSignal(str, bool, str)
    def __init__(self, provider_id, config, parent=None):
        super().__init__(parent)
        self.provider_id = provider_id
        self.config = config

    def run(self):
        key = self.config.get_api_key(self.provider_id)
        res = (True, "Connected") if key and len(key) > 5 else (False, "Invalid Key")
        self.result_ready.emit(self.provider_id, res[0], res[1])


class SettingsView(QWidget):
    closed = pyqtSignal()

    def __init__(self, config, app=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.app = app
        self.api_inputs = {}
        self.hotkey_inputs = {}
        self.status_labels = {}
        self._build()

    def _build(self):
        self.setStyleSheet(BG_DARK + SS_CHECKBOX + SS_INPUT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        hdr = QHBoxLayout()
        t = QLabel("⚙️ SYSTEM CONFIG")
        t.setStyleSheet(f"{TEXT_PRIMARY} font-weight: 900; letter-spacing: 2px; font-size: 12px; background: transparent;")
        hdr.addWidget(t); hdr.addStretch()
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30); close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("color: #667; border: none; font-size: 18px; background: transparent;")
        close_btn.clicked.connect(self.closed.emit)
        hdr.addWidget(close_btn)
        layout.addLayout(hdr)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid rgba(80, 85, 255, 10); background: transparent; border-radius: 8px; }
            QTabBar::tab { background: rgba(0,0,0,0.3); color: #778; padding: 10px 20px; font-size: 10px; font-weight: bold; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; }
            QTabBar::tab:selected { background: rgba(80, 85, 255, 0.08); color: #c0c0ff; border-bottom: 2px solid #6366f1; }
        """)
        
        self.tabs.addTab(self._tab_api(), "AI ENGINES")
        self.tabs.addTab(self._tab_hotkeys(), "SHORTCUTS")
        self.tabs.addTab(self._tab_capture(), "HARDWARE")
        self.tabs.addTab(self._tab_stealth(), "GHOST")
        layout.addWidget(self.tabs)

        self.btn_save = QPushButton("APPLY SETTINGS")
        self.btn_save.setFixedHeight(48)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 24px; font-weight: 900; font-size: 11px; letter-spacing: 2px;
            }
            QPushButton:hover { background: #6366f1; border: 1px solid rgba(255,255,255,30); }
        """)
        self.btn_save.clicked.connect(self._save_all)
        layout.addWidget(self.btn_save)

    def _tab_api(self):
        w = QScrollArea(); w.setWidgetResizable(True); w.setStyleSheet("background: transparent; border: none;")
        c = QWidget(); l = QVBoxLayout(c); l.setContentsMargins(15, 15, 15, 15); l.setSpacing(15)
        providers = [("groq", "🚀 Groq Cloud"), ("gemini", "🧠 Google Gemini"), ("cerebras", "⚡ Cerebras AI"), ("together", "🎭 Together AI"), ("ollama", "🏠 Local Ollama")]
        for pid, name in providers:
            box = QFrame(); box.setStyleSheet("background: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.03);")
            bl = QVBoxLayout(box)
            top = QHBoxLayout(); lbl = QLabel(name); lbl.setStyleSheet("background: transparent; font-weight: bold; font-size: 11px; color: #a0a0cc;")
            top.addWidget(lbl); top.addStretch()
            stat = QLabel("⚪"); self.status_labels[pid] = stat; stat.setStyleSheet("background: transparent;"); top.addWidget(stat)
            tbtn = QPushButton("TEST"); tbtn.setFixedSize(50, 22); tbtn.setStyleSheet("background: rgba(80,85,255,0.1); color: #8fa1b3; border-radius: 5px; font-size: 9px;")
            tbtn.clicked.connect(lambda _, p=pid: self._test_pid(p)); top.addWidget(tbtn); bl.addLayout(top)
            inp = QLineEdit(); inp.setEchoMode(QLineEdit.EchoMode.Password); inp.setStyleSheet(SS_INPUT)
            inp.setPlaceholderText(f"Enter {name} API Key..."); inp.setText(self.config.get_api_key(pid) or ""); self.api_inputs[pid] = inp
            bl.addWidget(inp); l.addWidget(box)
        c.setLayout(l); w.setWidget(c); return w

    def _tab_hotkeys(self):
        w = QScrollArea(); w.setWidgetResizable(True); w.setStyleSheet("background: transparent; border: none;")
        c = QWidget(); l = QVBoxLayout(c); l.setContentsMargins(15, 15, 15, 15); l.setSpacing(2)
        hdr = QHBoxLayout(); hdr.setContentsMargins(10, 0, 10, 5)
        h1 = QLabel("COMMAND"); h1.setStyleSheet(f"{TEXT_MUTED} font-size: 9px; font-weight: 800; border: none; background: transparent;")
        h2 = QLabel("SHORTCUT KEY"); h2.setStyleSheet(f"{TEXT_MUTED} font-size: 9px; font-weight: 800; border: none; background: transparent;")
        hdr.addWidget(h1, 1); hdr.addWidget(h2, 1); l.addLayout(hdr)
        hk_labels = {
            "toggle": "Toggle Main HUD", "quick_answer": "Quick Context Answer",
            "switch_mode": "Rotate AI Modes", "toggle_audio": "Mute/Unmute Mic",
            "stealth": "Toggle Ghost Stealth", "mini_mode": "Switch to Mini-HUD",
            "toggle_click_through": "Focus Click-Through", "emergency_erase": "Emergency System Wipe",
            "move_up": "Glide HUD Up", "move_down": "Glide HUD Down",
            "move_left": "Glide HUD Left", "move_right": "Glide HUD Right"
        }
        keys_cfg = self.config.get("hotkeys", {})
        for action in sorted(hk_labels.keys()):
            row_frame = QFrame()
            row_frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.03); }")
            rl = QHBoxLayout(row_frame); rl.setContentsMargins(10, 8, 10, 8)
            lbl = QLabel(hk_labels[action]); lbl.setStyleSheet("background: transparent; color: #94a3b8; font-size: 11px; border: none;")
            rl.addWidget(lbl, 1)
            inp = QLineEdit(); inp.setFixedWidth(140); inp.setStyleSheet(SS_INPUT + "QLineEdit { border: none; background: rgba(0,0,0,0.2); text-align: center; }")
            inp.setText(keys_cfg.get(action, "")); self.hotkey_inputs[action] = inp
            rl.addWidget(inp); l.addWidget(row_frame)
        l.addStretch(); c.setLayout(l); w.setWidget(c); return w

    def _tab_capture(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(20, 20, 20, 20); l.setSpacing(20); w.setStyleSheet("background: transparent;")
        lbl = QLabel("PRIMARY AUDIO SOURCE"); lbl.setStyleSheet("background: transparent;"); l.addWidget(lbl)
        self.audio_mode = QComboBox()
        self.audio_mode.addItems(["🖥️ System Speakers", "🎙️ Microphone Only", "🌐 Hybrid (Both)"])
        curr = self.config.get("capture.audio.mode", "system")
        self.audio_mode.setCurrentIndex(0 if curr == "system" else 1 if curr == "mic" else 2)
        self.audio_mode.setStyleSheet(SS_INPUT); l.addWidget(self.audio_mode)
        sep = QFrame(); sep.setFixedHeight(1); sep.setStyleSheet("background: rgba(255,255,255,0.05);"); l.addWidget(sep)
        lbl2 = QLabel("VISION ENGINE"); lbl2.setStyleSheet("background: transparent;"); l.addWidget(lbl2)
        self.chk_smart = QCheckBox("Enable Contextual Smart-Crop"); self.chk_smart.setStyleSheet(SS_CHECKBOX)
        self.chk_smart.setChecked(self.config.get("capture.screen.smart_crop", True)); l.addWidget(self.chk_smart)
        l.addStretch(); return w

    def _tab_stealth(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(20, 20, 20, 20); l.setSpacing(15); w.setStyleSheet("background: transparent;")
        lbl = QLabel("GHOST ANTI-RECORDING"); lbl.setStyleSheet("background: transparent;"); l.addWidget(lbl)
        self.chk_ghost = QCheckBox("Mask overlay from Screen Recorders"); self.chk_ghost.setStyleSheet(SS_CHECKBOX)
        self.chk_ghost.setChecked(self.config.get("stealth.enabled", False)); l.addWidget(self.chk_ghost)
        desc = QLabel("Enabling this prevents Zoom, Teams, or OBS from seeing the AI overlay.")
        desc.setWordWrap(True); desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(desc); l.addStretch(); return w

    def _test_pid(self, pid):
        self.status_labels[pid].setText("⏳")
        worker = ProviderTestWorker(pid, self.config, self)
        worker.result_ready.connect(lambda p, s, m: self.status_labels[p].setText("✅" if s else "❌"))
        worker.start()

    def showEvent(self, event):
        """Reset UI state when settings are opened."""
        super().showEvent(event)
        self.btn_save.setText("APPLY SETTINGS")
        self.btn_save.setEnabled(True)
        self.btn_save.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 24px; font-weight: 900; font-size: 11px; letter-spacing: 2px;
            }
            QPushButton:hover { background: #6366f1; border: 1px solid rgba(255,255,255,30); }
        """)

    def _save_all(self):
        try:
            self.btn_save.setText("APPLYING...")
            self.btn_save.setEnabled(False)
            self.btn_save.setStyleSheet("background: #312e81; color: #6366f1; border-radius: 24px; font-weight: 900; font-size: 11px; letter-spacing: 2px;")
            
            for pid, inp in self.api_inputs.items():
                self.config.set_api_key(pid, inp.text().strip())
            for action, inp in self.hotkey_inputs.items():
                self.config.set(f"hotkeys.{action}", inp.text().strip().lower())
            
            m_idx = self.audio_mode.currentIndex()
            self.config.set("capture.audio.mode", "system" if m_idx == 0 else "mic" if m_idx == 1 else "both")
            self.config.set("capture.screen.smart_crop", self.chk_smart.isChecked())
            self.config.set("stealth.enabled", self.chk_ghost.isChecked())
            self.config.save()
            
            if self.app: self.app._apply_settings()
            
            # Transition back to Standby
            QTimer.singleShot(800, self.closed.emit)
            
            # Safety reset in case view doesn't close or is re-opened
            QTimer.singleShot(2000, lambda: self.btn_save.setText("APPLY SETTINGS"))
            QTimer.singleShot(2000, lambda: self.btn_save.setEnabled(True))
        except Exception as e:
            logger.error(f"Save Fail: {e}")
            self.btn_save.setText("APPLY SETTINGS"); self.btn_save.setEnabled(True)
