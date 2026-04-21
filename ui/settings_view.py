"""
Integrated Settings View — v4.1 (Midnight Premium).
RESTORATION: Hotkey Configuration Tab & SVG Tick Icons.
FIXED: Checkbox tick visibility with URL-encoded SVG (Safe Mode).
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
import os
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTabWidget,
    QCheckBox,
    QComboBox,
    QSlider,
    QScrollArea,
    QFrame,
    QGridLayout,
    QSizePolicy,
    QMessageBox,
)
from PyQt6.QtGui import QDesktopServices
from core.constants import PROVIDERS
from ui.custom_widgets import PremiumCheckBox
from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- Premium Midnight Styling ---
BG_DARK = "background: rgba(15, 15, 30, 245);"
TEXT_PRIMARY = "color: #c0c0ff;"
TEXT_MUTED = "color: #64748b;"
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

STYLE_BTN_PRIMARY = """
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed);
    color: white;
    border-radius: 12px;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 15px 30px;
    border: none;
}}
QPushButton:hover {{
    background: #6366f1;
    border: 1px solid rgba(255,255,255,20);
}}
"""

STYLE_BTN_SECONDARY = """
QPushButton {{
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    border-radius: 12px;
    font-weight: 600;
    font-size: 11px;
    padding: 12px 24px;
    border: 1px solid rgba(255,255,255,0.1);
}}
QPushButton:hover {{
    background: rgba(255,255,255,0.08);
    color: white;
}}
"""


class ProviderTestWorker(QThread):
    """Real API test worker - sends actual requests to providers."""

    result_ready = pyqtSignal(str, bool, str)

    def __init__(self, provider_id, config, parent=None):
        super().__init__(parent)
        self.provider_id = provider_id
        self.config = config

    def run(self):
        import asyncio
        import aiohttp

        key = self.config.get_api_key(self.provider_id)

        # Check key format first
        if not key or len(key) < 10:
            self.result_ready.emit(self.provider_id, False, "Invalid Key")
            return

        # Test endpoints for different providers
        ollama_model = self.config.get("ai.providers.ollama.model") or "llama3.1:8b"
        endpoints = {
            "groq": (
                "https://api.groq.com/openai/v1/chat/completions",
                {
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            ),
            "gemini": (
                f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={key}",
                {"contents": [{"parts": [{"text": "Hi"}]}]},
            ),
            "cerebras": (
                "https://api.cerebras.ai/v1/chat/completions",
                {
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            ),
            "together": (
                "https://api.together.xyz/v1/chat/completions",
                {
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            ),
            "ollama": (
                "http://localhost:11434/api/generate",
                {"model": ollama_model, "prompt": "Hi", "stream": False},
            ),
        }

        if self.provider_id not in endpoints:
            self.result_ready.emit(self.provider_id, False, "Provider not supported")
            return

        url, payload = endpoints[self.provider_id]
        headers = {"Content-Type": "application/json"}

        # Add auth header for most providers
        if self.provider_id != "ollama":
            headers["Authorization"] = f"Bearer {key}"

        try:
            if self.provider_id == "gemini":

                async def test_gemini():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            return await resp.json()

                result = asyncio.run(test_gemini())
                if "error" in result:
                    self.result_ready.emit(
                        self.provider_id,
                        False,
                        result["error"].get("message", "API Error"),
                    )
                else:
                    self.result_ready.emit(self.provider_id, True, "Connected")
            elif self.provider_id == "ollama":

                async def test_ollama():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url, json=payload, timeout=aiohttp.ClientTimeout(total=5)
                        ) as resp:
                            return await resp.json()

                result = asyncio.run(test_ollama())
                if "error" in result:
                    self.result_ready.emit(
                        self.provider_id,
                        False,
                        result.get("error", "Connection failed"),
                    )
                else:
                    self.result_ready.emit(self.provider_id, True, "Connected")
            else:

                async def test_api():
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            return await resp.json()

                result = asyncio.run(test_api())
                if "error" in result:
                    self.result_ready.emit(
                        self.provider_id,
                        False,
                        result["error"].get("message", "API Error"),
                    )
                elif "choices" in result or "output" in result:
                    self.result_ready.emit(self.provider_id, True, "Connected")
                else:
                    self.result_ready.emit(
                        self.provider_id, False, "Unexpected response"
                    )
        except asyncio.TimeoutError:
            self.result_ready.emit(self.provider_id, False, "Timeout")
        except Exception as e:
            self.result_ready.emit(self.provider_id, False, str(e)[:30])


class SettingsView(QWidget):
    closed = pyqtSignal()
    mode_changed = pyqtSignal(str)
    audio_source_changed = pyqtSignal(str)

    def __init__(self, config, app=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.app = app
        self.api_inputs = {}
        self.hotkey_inputs = {}
        self.status_labels = {}
        self._build()

    @staticmethod
    def _slider_percent(opacity: float) -> int:
        return int(round(opacity * 100))

    def _set_opacity_label(self, label: QLabel, value: int):
        label.setText(f"{value}%")

    def _preview_window_opacity(self):
        if not self.app or not hasattr(self.app, "_apply_ui_only"):
            return

        if hasattr(self, "hud_opacity_slider"):
            self.config.set(
                "app.opacity",
                self.hud_opacity_slider.value() / 100.0,
            )
        if hasattr(self, "stealth_opacity_slider"):
            self.config.set(
                "stealth.low_opacity",
                self.stealth_opacity_slider.value() / 100.0,
            )

        self.app._apply_ui_only()

    def _make_section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; font-weight: 800; letter-spacing: 1px; background: transparent;"
        )
        label.setWordWrap(True)
        return label

    def _style_combo(self, combo: QComboBox):
        combo.setStyleSheet(SS_INPUT)
        combo.setMinimumHeight(36)
        combo.setMinimumContentsLength(18)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _build(self):
        self.setStyleSheet(BG_DARK + SS_INPUT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        hdr = QHBoxLayout()
        t = QLabel("⚙️ SYSTEM CONFIG")
        t.setStyleSheet(
            f"{TEXT_PRIMARY} font-weight: 900; letter-spacing: 2px; font-size: 12px; background: transparent;"
        )
        hdr.addWidget(t)
        hdr.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            "color: #667; border: none; font-size: 18px; background: transparent;"
        )
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
        self.tabs.addTab(self._tab_ui(), "DISPLAY")
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
        w = QScrollArea()
        w.setWidgetResizable(True)
        w.setStyleSheet("background: transparent; border: none;")
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(15)
        providers = [
            ("groq", "🚀 Groq Cloud"),
            ("gemini", "🧠 Google Gemini"),
            ("cerebras", "⚡ Cerebras AI"),
            ("together", "🎭 Together AI"),
            ("ollama", "🏠 Local Ollama"),
        ]
        for pid, name in providers:
            provider_meta = PROVIDERS.get(pid, {})
            box = QFrame()
            box.setStyleSheet(
                "background: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.03);"
            )
            bl = QVBoxLayout(box)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 5, 0)
            top.setSpacing(10)
            
            lbl = QLabel(name)
            lbl.setStyleSheet(
                "background: transparent; font-weight: bold; font-size: 11px; color: #a0a0cc;"
            )
            top.addWidget(lbl)
            top.addStretch()
            
            stat = QLabel("⚪")
            self.status_labels[pid] = stat
            stat.setStyleSheet("background: transparent; font-size: 11px;")
            top.addWidget(stat)
            
            tbtn = QPushButton("TEST")
            tbtn.setFixedSize(60, 24)
            tbtn.setCursor(Qt.CursorShape.PointingHandCursor)
            tbtn.setToolTip(f"Test {name} connection")
            tbtn.setStyleSheet(
                "QPushButton { background: rgba(80,85,255,0.18); color: #f8fafc; border-radius: 4px; font-size: 10px; font-weight: 800; border: 1px solid rgba(129,140,248,0.55); padding: 0 8px; text-align: center; }"
                "QPushButton:hover { background: rgba(80,85,255,0.35); color: white; border: 1px solid rgba(165,180,252,0.85); }"
                "QPushButton:pressed { background: rgba(67,56,202,0.55); }"
            )
            tbtn.clicked.connect(lambda _, p=pid: self._test_pid(p))
            top.addWidget(tbtn)
            bl.addLayout(top)

            link_text = "Install Ollama" if pid == "ollama" else "Get API key"
            link = QLabel(f'<a href="{provider_meta.get("url", "")}">{link_text}</a>')
            link.setOpenExternalLinks(False)
            link.linkActivated.connect(
                lambda url, _pid=pid: QDesktopServices.openUrl(QUrl(url))
            )
            link.setToolTip(
                f"Open {provider_meta.get('name', name)} to "
                f"{'install the local engine' if pid == 'ollama' else 'create or copy an API key'}."
            )
            link.setStyleSheet(
                """
                QLabel {
                    color: #bae6fd;
                    font-size: 10px;
                    font-weight: 700;
                    background: rgba(56, 189, 248, 0.12);
                    border: 1px solid rgba(56, 189, 248, 0.28);
                    border-radius: 10px;
                    padding: 4px 10px;
                    margin: 0 0 4px 0;
                }
                QLabel:hover {
                    background: rgba(56, 189, 248, 0.18);
                    border: 1px solid rgba(125, 211, 252, 0.5);
                }
                """
            )
            bl.addWidget(link)
             
            # Local providers like Ollama don't need API keys
            if pid != "ollama":
                inp = QLineEdit()
                inp.setEchoMode(QLineEdit.EchoMode.Password)
                inp.setStyleSheet(SS_INPUT)
                inp.setPlaceholderText(f"Enter {name} API Key...")
                inp.setText(self.config.get_api_key(pid) or "")
                self.api_inputs[pid] = inp
                bl.addWidget(inp)
            else:
                desc = QLabel("Local Engine (No API Key Required)")
                desc.setStyleSheet(f"color: #64748b; font-size: 10px; font-style: italic; background: transparent;")
                bl.addWidget(desc)
                
            l.addWidget(box)

        # Parallel Inference Toggle
        sep = QFrame()
        sep.setStyleSheet(
            "background: rgba(255,255,255,0.05); height: 1px; margin: 15px 0;"
        )
        l.addWidget(sep)

        lbl_parallel = QLabel("⚡ PARALLEL INFERENCE")
        lbl_parallel.setStyleSheet(
            "font-size: 10px; color: #475569; font-weight: 800; letter-spacing: 2px;"
        )
        l.addWidget(lbl_parallel)

        self.chk_parallel = PremiumCheckBox("Enable parallel multi-provider inference")
        self.chk_parallel.setChecked(self.config.get("ai.parallel.enabled", False))
        l.addWidget(self.chk_parallel)

        desc_parallel = QLabel(
            "When enabled, queries are sent to multiple providers simultaneously and the fastest response is used."
        )
        desc_parallel.setWordWrap(True)
        desc_parallel.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_parallel)

        c.setLayout(l)
        w.setWidget(c)
        return w

    def _tab_hotkeys(self):
        w = QScrollArea()
        w.setWidgetResizable(True)
        w.setStyleSheet("background: transparent; border: none;")
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(15, 15, 15, 15)
        l.setSpacing(2)

        hint = QLabel("`Ctrl+\\` hides or shows the HUD once per press. `Ctrl+M` controls click-through.")
        hint.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; border: none; background: transparent;"
        )
        l.addWidget(hint)

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
            "stealth": "Toggle Ghost Stealth",
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
                "QFrame { background: rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.03); }"
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
                + "QLineEdit { border: none; background: rgba(0,0,0,0.2); text-align: center; }"
            )
            inp.setText(keys_cfg.get(action, ""))
            self.hotkey_inputs[action] = inp
            rl.addWidget(inp)
            l.addWidget(row_frame)
        l.addStretch()
        c.setLayout(l)
        w.setWidget(c)
        return w

    def _tab_capture(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(20, 20, 20, 20)
        l.setSpacing(20)
        w.setStyleSheet("background: transparent;")

        lbl_mode = self._make_section_label("ACTIVE AI MODE")
        l.addWidget(lbl_mode)
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
        current_mode = self.config.get("ai.mode", "general")
        mode_map = {
            "general": 0,
            "interview": 1,
            "coding": 2,
            "meeting": 3,
            "exam": 4,
            "writing": 5,
        }
        self.ai_mode.setCurrentIndex(mode_map.get(current_mode, 0))
        self._style_combo(self.ai_mode)
        l.addWidget(self.ai_mode)

        sep_mode = QFrame()
        sep_mode.setFixedHeight(1)
        sep_mode.setStyleSheet("background: rgba(255,255,255,0.05);")
        l.addWidget(sep_mode)

        lbl = self._make_section_label("PRIMARY AUDIO SOURCE")
        l.addWidget(lbl)
        self.audio_mode = QComboBox()
        self.audio_mode.addItems(
            ["🖥️ System Speakers", "🎙️ Microphone Only", "🌐 Hybrid (Both)"]
        )
        curr = self.config.get("capture.audio.mode", "system")
        self.audio_mode.setCurrentIndex(
            0 if curr == "system" else 1 if curr == "mic" else 2
        )
        self._style_combo(self.audio_mode)
        l.addWidget(self.audio_mode)
        audio_desc = QLabel(
            "Choose which source the assistant should listen to while collecting audio context."
        )
        audio_desc.setWordWrap(True)
        audio_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(audio_desc)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(255,255,255,0.05);")
        l.addWidget(sep)
        lbl2 = self._make_section_label("VISION ENGINE")
        l.addWidget(lbl2)
        self.chk_smart = PremiumCheckBox("Enable Contextual Smart-Crop")
        self.chk_smart.setChecked(self.config.get("capture.screen.smart_crop", True))
        l.addWidget(self.chk_smart)
        smart_desc = QLabel(
            "Keeps OCR focused on the active region so the vision pipeline stays relevant and efficient."
        )
        smart_desc.setWordWrap(True)
        smart_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(smart_desc)

        self.chk_paid_vision_fallback = PremiumCheckBox(
            "Allow paid vision fallback providers"
        )
        self.chk_paid_vision_fallback.setChecked(
            self.config.get("ai.vision.allow_paid_fallback", False)
        )
        l.addWidget(self.chk_paid_vision_fallback)

        paid_fallback_desc = QLabel(
            "When disabled, screenshot analysis stays on free-capable vision providers like Gemini and compatible local Ollama models."
        )
        paid_fallback_desc.setWordWrap(True)
        paid_fallback_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(paid_fallback_desc)
        # Screenshot Interval
        lbl_interval = self._make_section_label("SCREEN CAPTURE INTERVAL")
        l.addWidget(lbl_interval)

        self.screenshot_interval = QComboBox()
        self.screenshot_interval.addItems(
            [
                "500ms (Real-time)",
                "1s (Fast)",
                "2s (Normal)",
                "3s (Eco)",
                "5s (Battery Saver)",
            ]
        )
        current_interval = self.config.get("capture.screen.interval_ms", 500)
        interval_map = {500: 0, 1000: 1, 2000: 2, 3000: 3, 5000: 4}
        self.screenshot_interval.setCurrentIndex(interval_map.get(current_interval, 0))
        self._style_combo(self.screenshot_interval)
        l.addWidget(self.screenshot_interval)

        # Image Quality
        lbl_quality = self._make_section_label("IMAGE QUALITY")
        l.addWidget(lbl_quality)

        self.image_quality = QComboBox()
        self.image_quality.addItems(
            ["Low (Faster)", "Medium (Balanced)", "High (Best)"]
        )
        current_quality = self.config.get("capture.screen.quality", "medium")
        quality_map = {"low": 0, "medium": 1, "high": 2}
        self.image_quality.setCurrentIndex(quality_map.get(current_quality, 1))
        self._style_combo(self.image_quality)
        l.addWidget(self.image_quality)

        l.addStretch()
        scroll.setWidget(w)
        return scroll

    def _tab_stealth(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(20, 20, 20, 20)
        l.setSpacing(15)
        w.setStyleSheet("background: transparent;")
        lbl = QLabel("GHOST ANTI-RECORDING")
        lbl.setStyleSheet("background: transparent;")
        l.addWidget(lbl)
        self.chk_ghost = PremiumCheckBox("Mask overlay from Screen Recorders")
        self.chk_ghost.setChecked(self.config.get("stealth.enabled", False))
        l.addWidget(self.chk_ghost)
        desc = QLabel(
            "Enabling this prevents Zoom, Teams, or OBS from seeing the AI overlay."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(desc)
        l.addStretch()
        return w

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
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; letter-spacing: 1px; background: transparent;"
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
                background: rgba(255,255,255,0.08);
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
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; letter-spacing: 1px; background: transparent;"
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
                background: rgba(255,255,255,0.08);
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
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; letter-spacing: 1px; background: transparent;"
        )
        l.addWidget(lbl_gaze)

        self.chk_gaze = PremiumCheckBox("Enable gaze-based window fading")
        self.chk_gaze.setChecked(self.config.get("app.gaze_fade.enabled", False))
        l.addWidget(self.chk_gaze)
        desc_gaze = QLabel(
            "When enabled, the window will fade to low opacity when your mouse is near it. "
            "Only active during active sessions - not on standby or settings screens."
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

        # Reset Onboarding
        sep = QFrame()
        sep.setStyleSheet(
            "background: rgba(255,255,255,0.05); height: 1px; margin: 20px 0;"
        )
        l.addWidget(sep)

        btn_reset_onboard = QPushButton("🔄 Reset Setup Wizard")
        btn_reset_onboard.setStyleSheet(STYLE_BTN_SECONDARY)
        btn_reset_onboard.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_onboard.clicked.connect(self._reset_onboarding)
        l.addWidget(btn_reset_onboard)

        desc_reset = QLabel("Run the setup wizard again to reconfigure your settings.")
        desc_reset.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(desc_reset)

        btn_factory_reset = QPushButton("FACTORY RESET")
        btn_factory_reset.setStyleSheet(
            """
            QPushButton {
                background: rgba(220, 38, 38, 0.14);
                color: #fecaca;
                border-radius: 12px;
                font-weight: 800;
                font-size: 11px;
                padding: 12px 24px;
                border: 1px solid rgba(248, 113, 113, 0.35);
            }
            QPushButton:hover {
                background: rgba(220, 38, 38, 0.22);
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

    def _reset_onboarding(self):
        """Reset onboarding flag and show wizard."""
        self.config.set("onboarding.completed", False)
        self.config.save()
        self.closed.emit()  # Close settings
        # Show onboarding
        if hasattr(self, "app") and hasattr(self.app, "overlay"):
            self.app.overlay.show_onboarding()

    def _factory_reset(self):
        """Run a full first-run reset after explicit confirmation."""
        if not self.app or not hasattr(self.app, "factory_reset"):
            return

        result = QMessageBox.question(
            self,
            "Factory Reset",
            (
                "Factory reset OpenAssist?\n\n"
                "This will remove settings, encrypted API keys, history, caches, and logs, "
                "then send the app back to onboarding."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        self.app.factory_reset()

    def _test_pid(self, pid):
        self.status_labels[pid].setText("⏳")
        worker = ProviderTestWorker(pid, self.config, self)
        worker.result_ready.connect(
            lambda p, s, m: self.status_labels[p].setText("✅" if s else "❌")
        )
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
            self.btn_save.setStyleSheet(
                "background: #312e81; color: #6366f1; border-radius: 24px; font-weight: 900; font-size: 11px; letter-spacing: 2px;"
            )

            for pid, inp in self.api_inputs.items():
                self.config.set_api_key(pid, inp.text().strip())
            for action, inp in self.hotkey_inputs.items():
                self.config.set(f"hotkeys.{action}", inp.text().strip().lower())

            mode_idx = self.ai_mode.currentIndex() if hasattr(self, "ai_mode") else 0
            selected_mode = [
                "general",
                "interview",
                "coding",
                "meeting",
                "exam",
                "writing",
            ][mode_idx]
            self.config.set("ai.mode", selected_mode)

            m_idx = self.audio_mode.currentIndex()
            selected_audio = "system" if m_idx == 0 else "mic" if m_idx == 1 else "both"
            self.config.set(
                "capture.audio.mode",
                selected_audio,
            )
            self.config.set("capture.screen.smart_crop", self.chk_smart.isChecked())
            if hasattr(self, "chk_paid_vision_fallback"):
                self.config.set(
                    "ai.vision.allow_paid_fallback",
                    self.chk_paid_vision_fallback.isChecked(),
                )
            self.config.set("stealth.enabled", self.chk_ghost.isChecked())

            # Save screenshot interval
            if hasattr(self, "screenshot_interval"):
                interval_values = [500, 1000, 2000, 3000, 5000]
                self.config.set(
                    "capture.screen.interval_ms",
                    interval_values[self.screenshot_interval.currentIndex()],
                )

            # Save image quality
            if hasattr(self, "image_quality"):
                quality_values = ["low", "medium", "high"]
                self.config.set(
                    "capture.screen.quality",
                    quality_values[self.image_quality.currentIndex()],
                )

            # Save parallel inference setting
            if hasattr(self, "chk_parallel"):
                self.config.set("ai.parallel.enabled", self.chk_parallel.isChecked())

            if hasattr(self, "hud_opacity_slider"):
                self.config.set(
                    "app.opacity",
                    self.hud_opacity_slider.value() / 100.0,
                )
            if hasattr(self, "stealth_opacity_slider"):
                self.config.set(
                    "stealth.low_opacity",
                    self.stealth_opacity_slider.value() / 100.0,
                )

            # Save gaze fade settings
            if hasattr(self, "chk_gaze"):
                self.config.set("app.gaze_fade.enabled", self.chk_gaze.isChecked())
            if hasattr(self, "chk_start_minimized"):
                self.config.set(
                    "app.start_minimized",
                    self.chk_start_minimized.isChecked(),
                )
            if hasattr(self, "margin_slider"):
                margin_values = [20, 30, 40, 50, 60, 80]
                self.config.set(
                    "app.gaze_fade.margin",
                    margin_values[self.margin_slider.currentIndex()],
                )
            if hasattr(self, "opacity_slider"):
                opacity_values = [0.05, 0.10, 0.15, 0.20, 0.25]
                self.config.set(
                    "app.gaze_fade.target_opacity",
                    opacity_values[self.opacity_slider.currentIndex()],
                )

            self.config.save()

            if self.app:
                self.app.state.mode = selected_mode
                self.app.state.audio_source = selected_audio
                self.app._apply_settings()
                self.mode_changed.emit(selected_mode)
                self.audio_source_changed.emit(selected_audio)

            # Transition back to Standby
            QTimer.singleShot(800, self.closed.emit)

            # Safety reset in case view doesn't close or is re-opened
            QTimer.singleShot(2000, lambda: self.btn_save.setText("APPLY SETTINGS"))
            QTimer.singleShot(2000, lambda: self.btn_save.setEnabled(True))
        except Exception as e:
            logger.error(f"Save Fail: {e}")
            self.btn_save.setText("APPLY SETTINGS")
            self.btn_save.setEnabled(True)

    def _current_tab_scroll_area(self):
        if not hasattr(self, "tabs"):
            return None
        current = self.tabs.currentWidget()
        return current if isinstance(current, QScrollArea) else None

    def scroll_up(self):
        area = self._current_tab_scroll_area()
        if area:
            sb = area.verticalScrollBar()
            sb.setValue(sb.value() - 80)

    def scroll_down(self):
        area = self._current_tab_scroll_area()
        if area:
            sb = area.verticalScrollBar()
            sb.setValue(sb.value() + 80)

    def select_prev_tab(self):
        if hasattr(self, "tabs"):
            count = self.tabs.count()
            if count:
                self.tabs.setCurrentIndex((self.tabs.currentIndex() - 1) % count)

    def select_next_tab(self):
        if hasattr(self, "tabs"):
            count = self.tabs.count()
            if count:
                self.tabs.setCurrentIndex((self.tabs.currentIndex() + 1) % count)
