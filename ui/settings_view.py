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
    QTextEdit,
    QInputDialog,
)
from PyQt6.QtGui import QDesktopServices
from core.constants import PROVIDERS
from ui.custom_widgets import PremiumCheckBox
from utils.logger import setup_logger
from utils.context_store import get_store as get_context_store

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

    result_ready = pyqtSignal(str, bool, str, object)

    def __init__(self, provider_id, config, parent=None, key_override: str = ""):
        super().__init__(parent)
        self.provider_id = provider_id
        self.config = config
        self.key_override = (key_override or "").strip()

    def run(self):
        import asyncio
        import aiohttp

        # Prefer the in-UI key (what the user just typed) so TEST works even
        # before "APPLY SETTINGS" persists into encrypted storage.
        key = self.key_override or self.config.get_api_key(self.provider_id)
        if self.provider_id == "ollama":
            endpoint = (
                self.config.get("ai.providers.ollama.endpoint")
                or self.config.get_api_key("ollama")
                or "http://localhost:11434"
            )
            if not str(endpoint).startswith("http"):
                endpoint = "http://localhost:11434"

            async def test_ollama():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{endpoint}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status != 200:
                            return {"error": f"Server returned {resp.status}"}
                        return await resp.json()

            try:
                result = asyncio.run(test_ollama())
                if "error" in result:
                    self.result_ready.emit(
                        self.provider_id,
                        False,
                        result.get("error", "Connection failed"),
                        None,
                    )
                    return

                models = result.get("models", []) or []
                model_names = [m.get("name", "") for m in models if m.get("name")]
                if models:
                    model_count = len(models)
                    self.result_ready.emit(
                        self.provider_id,
                        True,
                        f"Connected ({model_count} model{'s' if model_count != 1 else ''})",
                        {"models": model_names, "endpoint": endpoint},
                    )
                else:
                    self.result_ready.emit(
                        self.provider_id,
                        True,
                        "Connected (no models pulled)",
                        {"models": [], "endpoint": endpoint},
                    )
                return
            except asyncio.TimeoutError:
                self.result_ready.emit(self.provider_id, False, "Timeout", None)
                return
            except Exception as e:
                self.result_ready.emit(self.provider_id, False, str(e)[:30], None)
                return

        # Check key format first
        is_valid, validation_message = self.config.validate_key_for_ui(
            self.provider_id, key
        )
        if not is_valid:
            self.result_ready.emit(
                self.provider_id, False, validation_message, None
            )
            return

        # Test endpoints for different providers
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
                # v1beta is the current Gemini API surface for model operations.
                # Using Flash keeps TEST fast and typically within free-tier limits.
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}",
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
            "openai": (
                "https://api.openai.com/v1/chat/completions",
                {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            ),
            "anthropic": (
                "https://api.anthropic.com/v1/messages",
                {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
                },
            ),
            "mistral": (
                "https://api.mistral.ai/v1/chat/completions",
                {
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
            ),
        }

        if self.provider_id not in endpoints:
            self.result_ready.emit(
                self.provider_id, False, "Provider not supported", None
            )
            return

        url, payload = endpoints[self.provider_id]
        headers = {"Content-Type": "application/json"}

        # Configure provider-specific authentication headers
        if self.provider_id == "gemini":
            pass  # Key is already in the URL
        elif self.provider_id == "anthropic":
            headers["x-api-key"] = key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {key}"

        async def _post_json():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    status = resp.status
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"raw": (await resp.text())[:200]}
                    return status, data

        try:
            status, result = asyncio.run(_post_json())
            if status >= 400:
                msg = None
                if isinstance(result, dict):
                    err = result.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message")
                msg = msg or (result.get("raw") if isinstance(result, dict) else None) or "API Error"
                self.result_ready.emit(self.provider_id, False, f"HTTP {status}: {msg}"[:120], result)
                return

            if isinstance(result, dict) and "error" in result:
                self.result_ready.emit(
                    self.provider_id,
                    False,
                    (result["error"].get("message", "API Error") if isinstance(result.get("error"), dict) else "API Error"),
                    result,
                )
                return

            # Heuristic: success for OpenAI-style APIs returns choices/output; Gemini returns candidates.
            if isinstance(result, dict) and (
                "choices" in result
                or "output" in result
                or "candidates" in result
                or "contents" in result
            ):
                self.result_ready.emit(self.provider_id, True, "Connected", result)
            else:
                self.result_ready.emit(self.provider_id, True, "Connected (unexpected payload)", result)
        except asyncio.TimeoutError:
            self.result_ready.emit(self.provider_id, False, "Timeout", None)
        except Exception as e:
            self.result_ready.emit(self.provider_id, False, str(e)[:120], None)


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
        self.provider_detail_labels = {}
        self._test_workers = {}
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
            f"{TEXT_MUTED} font-size: 10px; font-weight: 800; background: transparent;"
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

    @staticmethod
    def _recommended_ollama_model(models, mode: str = "general") -> str:
        if not models:
            return ""

        model_names = [m for m in models if m]
        mode = (mode or "general").lower()

        def pick(markers):
            for marker in markers:
                for model in model_names:
                    if marker in model.lower():
                        return model
            return ""

        if mode == "coding":
            chosen = pick(
                [
                    "coder",
                    "codestral",
                    "deepseek-coder",
                    "codegemma",
                    "qwen2.5-coder",
                    "qwen2.5",
                    "qwen2",
                    "llama3.1",
                    "llama3.2",
                ]
            )
            if chosen:
                return chosen

        chosen = pick(
            [
                "llama3.2",
                "llama3.1",
                "qwen2.5",
                "qwen2",
                "mistral",
                "gemma",
                "phi3",
            ]
        )
        return chosen or model_names[0]

    def _build(self):
        self.setStyleSheet(BG_DARK + SS_INPUT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        hdr = QHBoxLayout()
        t = QLabel("⚙️ SYSTEM CONFIG")
        t.setStyleSheet(
            f"{TEXT_PRIMARY} font-weight: 900; font-size: 12px; background: transparent;"
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

        self.tabs.addTab(self._tab_api(),      "AI ENGINES")
        self.tabs.addTab(self._tab_hotkeys(),   "SHORTCUTS")
        self.tabs.addTab(self._tab_context(),   "CONTEXT")
        self.tabs.addTab(self._tab_capture(),   "HARDWARE")
        self.tabs.addTab(self._tab_ui(),        "DISPLAY")
        self.tabs.addTab(self._tab_stealth(),   "GHOST")
        layout.addWidget(self.tabs)

        self.btn_save = QPushButton("APPLY SETTINGS")
        self.btn_save.setFixedHeight(48)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 24px; font-weight: 900; font-size: 11px;
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
            ("openai", "🟢 OpenAI"),
            ("anthropic", "🟣 Anthropic"),
            ("mistral", "🌬️ Mistral AI"),
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
                "QPushButton { background: rgba(80,85,255,0.18); color: #f8fafc; border-radius: 4px; font-size: 10px; font-weight: bold; border: 1px solid rgba(129,140,248,0.55); padding: 0 8px; text-align: center; } "
                "QPushButton:hover { background: rgba(80,85,255,0.35); color: white; border: 1px solid rgba(165,180,252,0.85); } "
                "QPushButton:pressed { background: rgba(67,56,202,0.55); }"
            )
            tbtn.clicked.connect(lambda _, p=pid: self._test_pid(p))
            top.addWidget(tbtn)
            bl.addLayout(top)

            link_text = "Install Ollama" if pid == "ollama" else "Get API key"
            link = QLabel(f'<a href="{provider_meta.get("url", "")}">{link_text}</a>')
            link.setOpenExternalLinks(False)
            link.setWordWrap(False)
            link.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
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
            link.adjustSize()
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

                detail = QLabel("Not tested")
                detail.setWordWrap(True)
                detail.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
                self.provider_detail_labels[pid] = detail
                bl.addWidget(detail)
            else:
                desc = QLabel("Local Engine (No API Key Required)")
                desc.setStyleSheet(f"color: #64748b; font-size: 10px; font-style: italic; background: transparent;")
                bl.addWidget(desc)

                self.ollama_model_combo = QComboBox()
                self._style_combo(self.ollama_model_combo)
                self.ollama_model_combo.setEnabled(False)
                saved_model = self.config.get("ai.providers.ollama.model", "")
                self.ollama_model_combo.addItem(
                    saved_model or "Test Ollama to load local models"
                )
                bl.addWidget(self.ollama_model_combo)

                detail = QLabel(
                    "Tests the local Ollama server and lists installed models."
                )
                detail.setWordWrap(True)
                detail.setStyleSheet(
                    f"{TEXT_MUTED} font-size: 10px; background: transparent;"
                )
                self.provider_detail_labels[pid] = detail
                bl.addWidget(detail)
                
            l.addWidget(box)

        # Parallel Inference Toggle
        sep = QFrame()
        sep.setStyleSheet(
            "background: rgba(255,255,255,0.05); height: 1px; margin: 15px 0;"
        )
        l.addWidget(sep)

        lbl_parallel = QLabel("⚡ PARALLEL INFERENCE")
        lbl_parallel.setStyleSheet(
            "font-size: 10px; color: #475569; font-weight: 800;"
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
            "stealth": "Reapply Ghost Stealth",
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        w = QWidget()
        w.setMinimumWidth(0)
        l = QVBoxLayout(w)
        l.setContentsMargins(14, 16, 14, 16)
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

        # P2.3: Transcription Language Selector
        lbl_lang = self._make_section_label("TRANSCRIPTION LANGUAGE")
        l.addWidget(lbl_lang)
        self.audio_language = QComboBox()
        self.audio_language.addItems([
            "Auto-detect",
            "English (en)",
            "Hindi (hi)",
            "Spanish (es)",
            "French (fr)",
            "German (de)",
            "Japanese (ja)",
            "Chinese (zh)",
            "Portuguese (pt)",
            "Arabic (ar)",
            "Russian (ru)",
        ])
        lang_map = {
            "": 0, "auto": 0, "en": 1, "hi": 2, "es": 3,
            "fr": 4, "de": 5, "ja": 6, "zh": 7, "pt": 8, "ar": 9, "ru": 10
        }
        saved_lang = self.config.get("capture.audio.language", "")
        self.audio_language.setCurrentIndex(lang_map.get(saved_lang, 0))
        self._style_combo(self.audio_language)
        l.addWidget(self.audio_language)
        lang_desc = QLabel("Language hint for the transcription engine. Auto-detect works well for most cases.")
        lang_desc.setWordWrap(True)
        lang_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(lang_desc)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: rgba(255,255,255,0.05);")
        l.addWidget(sep2)

        # P2.8: ASR Correction Provider Selector
        lbl_corr = self._make_section_label("TRANSCRIPT CORRECTION PROVIDER")
        l.addWidget(lbl_corr)
        self.correction_provider = QComboBox()
        self.correction_provider.addItems(["Auto (Fastest Available)", "groq", "gemini", "cerebras", "together", "ollama"])
        cp_map = {"auto": 0, "groq": 1, "gemini": 2, "cerebras": 3, "together": 4, "ollama": 5}
        saved_cp = self.config.get("capture.audio.correction_provider", "groq")
        self.correction_provider.setCurrentIndex(cp_map.get(saved_cp, 1))
        self._style_combo(self.correction_provider)
        l.addWidget(self.correction_provider)
        corr_desc = QLabel(
            "Provider used to fix ASR typos in transcripts. Kept separate from your main AI so typo correction "
            "doesn't burn quota on the primary model."
        )
        corr_desc.setWordWrap(True)
        corr_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(corr_desc)

        sep3 = QFrame()
        sep3.setFixedHeight(1)
        sep3.setStyleSheet("background: rgba(255,255,255,0.05);")
        l.addWidget(sep3)

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

        self.chk_vision_local_only = PremiumCheckBox("Vision local-only (force Ollama)")
        self.chk_vision_local_only.setChecked(
            bool(self.config.get("ai.vision.local_only", False))
        )
        l.addWidget(self.chk_vision_local_only)

        vlocal_desc = QLabel(
            "Forces screenshot analysis to use only local Ollama vision models. Disables race + paid fallback."
        )
        vlocal_desc.setWordWrap(True)
        vlocal_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(vlocal_desc)

        # Vision Priority + Race Mode
        lbl_vprio = self._make_section_label("VISION PRIORITY (LOW LATENCY)")
        l.addWidget(lbl_vprio)

        self.vision_primary = QComboBox()
        self.vision_secondary = QComboBox()
        vision_options = ["gemini", "ollama", "openai"]
        self.vision_primary.addItems([p.capitalize() for p in vision_options])
        self.vision_secondary.addItems([p.capitalize() for p in vision_options])
        self._style_combo(self.vision_primary)
        self._style_combo(self.vision_secondary)

        saved_order = self.config.get("ai.vision.preferred_providers", ["gemini", "ollama"]) or ["gemini", "ollama"]
        primary = saved_order[0] if len(saved_order) > 0 else "gemini"
        secondary = saved_order[1] if len(saved_order) > 1 else "ollama"
        idx_map = {p: i for i, p in enumerate(vision_options)}
        self.vision_primary.setCurrentIndex(idx_map.get(primary, 0))
        self.vision_secondary.setCurrentIndex(idx_map.get(secondary, 1))

        row_v = QGridLayout()
        row_v.setHorizontalSpacing(10)
        row_v.setVerticalSpacing(8)
        row_v.addWidget(QLabel("Primary"), 0, 0)
        row_v.addWidget(self.vision_primary, 0, 1)
        row_v.addWidget(QLabel("Secondary"), 1, 0)
        row_v.addWidget(self.vision_secondary, 1, 1)
        l.addLayout(row_v)

        self.chk_vision_race = PremiumCheckBox("Race mode (send to both, take fastest)")
        self.chk_vision_race.setChecked(bool(self.config.get("ai.vision.race_enabled", False)))
        l.addWidget(self.chk_vision_race)

        vprio_desc = QLabel(
            "Primary/Secondary sets the order for screenshot analysis. Race mode runs both concurrently and uses the first successful response."
        )
        vprio_desc.setWordWrap(True)
        vprio_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(vprio_desc)

        sep_text = QFrame()
        sep_text.setFixedHeight(1)
        sep_text.setStyleSheet("background: rgba(255,255,255,0.05);")
        l.addWidget(sep_text)

        # Text Priority + Race Mode (manual + audio queries)
        lbl_tprio = self._make_section_label("TEXT PROVIDER PRIORITY (LOW LATENCY)")
        l.addWidget(lbl_tprio)

        self.text_primary = QComboBox()
        self.text_secondary = QComboBox()
        text_options = ["groq", "cerebras", "together", "gemini", "ollama"]
        self.text_primary.addItems([p.capitalize() for p in text_options])
        self.text_secondary.addItems([p.capitalize() for p in text_options])
        self._style_combo(self.text_primary)
        self._style_combo(self.text_secondary)

        saved_text_order = (
            self.config.get("ai.text.preferred_providers", text_options) or text_options
        )
        t1 = saved_text_order[0] if len(saved_text_order) > 0 else "groq"
        t2 = saved_text_order[1] if len(saved_text_order) > 1 else "cerebras"
        tidx = {p: i for i, p in enumerate(text_options)}
        self.text_primary.setCurrentIndex(tidx.get(t1, 0))
        self.text_secondary.setCurrentIndex(tidx.get(t2, 1))

        row_t = QGridLayout()
        row_t.setHorizontalSpacing(10)
        row_t.setVerticalSpacing(8)
        row_t.addWidget(QLabel("Primary"), 0, 0)
        row_t.addWidget(self.text_primary, 0, 1)
        row_t.addWidget(QLabel("Secondary"), 1, 0)
        row_t.addWidget(self.text_secondary, 1, 1)
        l.addLayout(row_t)

        self.chk_text_race = PremiumCheckBox("Race mode for text (use fastest successful)")
        self.chk_text_race.setChecked(bool(self.config.get("ai.text.race_enabled", False)))
        l.addWidget(self.chk_text_race)

        tprio_desc = QLabel(
            "Controls provider order for manual + audio queries. Race mode runs the top providers concurrently (no token streaming)."
        )
        tprio_desc.setWordWrap(True)
        tprio_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(tprio_desc)

        self.chk_text_local_only = PremiumCheckBox("Text local-only (force Ollama)")
        self.chk_text_local_only.setChecked(
            bool(self.config.get("ai.text.local_only", False))
        )
        l.addWidget(self.chk_text_local_only)

        tlocal_desc = QLabel(
            "Forces manual + audio text replies to use only local Ollama. Disables race mode."
        )
        tlocal_desc.setWordWrap(True)
        tlocal_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(tlocal_desc)

        def _apply_local_only_lockouts():
            try:
                v_local = bool(getattr(self, "chk_vision_local_only", None) and self.chk_vision_local_only.isChecked())
                t_local = bool(getattr(self, "chk_text_local_only", None) and self.chk_text_local_only.isChecked())
                if hasattr(self, "chk_paid_vision_fallback"):
                    self.chk_paid_vision_fallback.setEnabled(not v_local)
                if hasattr(self, "vision_primary"):
                    self.vision_primary.setEnabled(not v_local)
                if hasattr(self, "vision_secondary"):
                    self.vision_secondary.setEnabled(not v_local)
                if hasattr(self, "chk_vision_race"):
                    self.chk_vision_race.setEnabled(not v_local)
                if hasattr(self, "text_primary"):
                    self.text_primary.setEnabled(not t_local)
                if hasattr(self, "text_secondary"):
                    self.text_secondary.setEnabled(not t_local)
                if hasattr(self, "chk_text_race"):
                    self.chk_text_race.setEnabled(not t_local)
            except Exception:
                pass

        self.chk_vision_local_only.toggled.connect(lambda _checked: _apply_local_only_lockouts())
        self.chk_text_local_only.toggled.connect(lambda _checked: _apply_local_only_lockouts())
        _apply_local_only_lockouts()
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
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 800; background: transparent;"
        )
        l.addWidget(lbl_gaze)

        self.chk_gaze = PremiumCheckBox("Enable gaze-based window fading")
        self.chk_gaze.setChecked(self.config.get("app.gaze_fade.enabled", True))  # P1.2: default ON
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
            "background: rgba(255,255,255,0.05); height: 1px; margin: 20px 0;"
        )
        l.addWidget(sep)

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

    def _tab_context(self):
        """Session Context tab — custom AI persona / instructions."""
        w = QScrollArea()
        w.setWidgetResizable(True)
        w.setStyleSheet("background: transparent; border: none;")
        c = QWidget()
        l = QVBoxLayout(c)
        l.setContentsMargins(20, 20, 20, 20)
        l.setSpacing(16)
        c.setStyleSheet("background: transparent;")

        # Header
        hdr = QLabel("📝 SESSION CONTEXT")
        hdr.setStyleSheet(
            f"{TEXT_PRIMARY} font-size: 11px; font-weight: 900; background: transparent;"
        )
        l.addWidget(hdr)

        desc = QLabel(
            "Write custom instructions the AI must follow for every response during a session. "
            "Use this to define a role, tech stack, tone, and response style. "
            "Context is saved between app launches so you don't retype it."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(desc)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(255,255,255,0.05);")
        l.addWidget(sep)

        # Presets row
        preset_row = QHBoxLayout()
        preset_lbl = QLabel("PRESETS")
        preset_lbl.setStyleSheet(
            f"{TEXT_MUTED} font-size: 9px; font-weight: 800; background: transparent;"
        )
        preset_row.addWidget(preset_lbl)

        self._ctx_preset_combo = QComboBox()
        self._style_combo(self._ctx_preset_combo)
        self._ctx_preset_combo.setMinimumWidth(180)
        self._refresh_preset_combo()
        preset_row.addWidget(self._ctx_preset_combo, 1)

        load_btn = QPushButton("↓ Load")
        load_btn.setFixedHeight(32)
        load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        load_btn.setStyleSheet(
            "QPushButton { background: rgba(80,85,255,0.18); color: #c0c0ff; border-radius: 6px; "
            "font-size: 10px; font-weight: 800; border: 1px solid rgba(129,140,248,0.4); padding: 0 10px; }"
            "QPushButton:hover { background: rgba(80,85,255,0.3); color: white; }"
        )
        load_btn.clicked.connect(self._load_ctx_preset)
        preset_row.addWidget(load_btn)
        l.addLayout(preset_row)

        # Text area
        self._ctx_edit = QTextEdit()
        self._ctx_edit.setMinimumHeight(160)
        self._ctx_edit.setMaximumHeight(280)
        self._ctx_edit.setPlaceholderText(
            'e.g. "You are a real-time interview assistant for a Full Stack React role. '
            "Give short, direct answers. Use functional patterns (map/filter/reduce). "
            'No basic loops. Prioritize hooks, modularity, and readability."'
        )
        self._ctx_edit.setStyleSheet(
            """
            QTextEdit {
                background: rgba(12, 12, 28, 200);
                color: #d0d0f0;
                border: 1px solid rgba(99, 102, 241, 30);
                border-radius: 8px;
                padding: 12px;
                font-size: 11px;
                font-family: 'Segoe UI', sans-serif;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 1px solid rgba(99, 102, 241, 80);
            }
            """
        )
        # Load current context from app state or store
        ctx_store = get_context_store()
        current_ctx = ""
        if self.app and hasattr(self.app, "state"):
            current_ctx = self.app.state.session_context
        if not current_ctx:
            current_ctx = ctx_store.get_last_context()
        self._ctx_edit.setPlainText(current_ctx)
        self._ctx_edit.textChanged.connect(self._on_ctx_text_changed)
        l.addWidget(self._ctx_edit)

        # Char counter
        self._ctx_char_label = QLabel(f"{len(current_ctx)} / 2000 chars")
        self._ctx_char_label.setStyleSheet(
            f"{TEXT_MUTED} font-size: 9px; background: transparent;"
        )
        l.addWidget(self._ctx_char_label)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        save_preset_btn = QPushButton("☆ Save as Preset")
        save_preset_btn.setFixedHeight(34)
        save_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_preset_btn.setStyleSheet(
            "QPushButton { background: rgba(16,185,129,0.15); color: #6ee7b7; border-radius: 6px; "
            "font-size: 10px; font-weight: 800; border: 1px solid rgba(52,211,153,0.3); padding: 0 12px; }"
            "QPushButton:hover { background: rgba(16,185,129,0.25); color: white; }"
        )
        save_preset_btn.clicked.connect(self._save_ctx_as_preset)
        btn_row.addWidget(save_preset_btn)

        del_preset_btn = QPushButton("🗑 Delete Preset")
        del_preset_btn.setFixedHeight(34)
        del_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_preset_btn.setStyleSheet(
            "QPushButton { background: rgba(220,38,38,0.12); color: #fca5a5; border-radius: 6px; "
            "font-size: 10px; font-weight: 800; border: 1px solid rgba(248,113,113,0.28); padding: 0 12px; }"
            "QPushButton:hover { background: rgba(220,38,38,0.22); color: white; }"
        )
        del_preset_btn.clicked.connect(self._delete_ctx_preset)
        btn_row.addWidget(del_preset_btn)

        clear_btn = QPushButton("× Clear")
        clear_btn.setFixedHeight(34)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,0.04); color: #64748b; border-radius: 6px; "
            "font-size: 10px; font-weight: 700; border: 1px solid rgba(255,255,255,0.08); padding: 0 12px; }"
            "QPushButton:hover { color: #94a3b8; background: rgba(255,255,255,0.07); }"
        )
        clear_btn.clicked.connect(lambda: self._ctx_edit.clear())
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        l.addLayout(btn_row)

        tip = QLabel(
            "⚡ Tip: Use specific, actionable language. The more precise your instructions, "
            "the more consistently the AI will follow them. "
            "Click \"APPLY SETTINGS\" to activate the context for the next session."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #7c6f9e; font-size: 9px; background: transparent; font-style: italic;")
        l.addWidget(tip)

        l.addStretch()
        c.setLayout(l)
        w.setWidget(c)
        return w

    def _refresh_preset_combo(self):
        """Reload preset names into the combo box."""
        if not hasattr(self, "_ctx_preset_combo"):
            return
        store = get_context_store()
        self._ctx_preset_combo.blockSignals(True)
        self._ctx_preset_combo.clear()
        self._ctx_preset_combo.addItem("-- Select a preset --")
        for name in store.get_preset_names():
            self._ctx_preset_combo.addItem(name)
        self._ctx_preset_combo.blockSignals(False)

    def _load_ctx_preset(self):
        idx = self._ctx_preset_combo.currentIndex()
        if idx <= 0:
            return
        name = self._ctx_preset_combo.currentText()
        text = get_context_store().get_preset(name)
        if text:
            self._ctx_edit.setPlainText(text)

    def _save_ctx_as_preset(self):
        text = self._ctx_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Context", "Write some instructions first.")
            return
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            text=self._ctx_preset_combo.currentText()
            if self._ctx_preset_combo.currentIndex() > 0 else ""
        )
        if ok and name.strip():
            get_context_store().save_preset(name.strip(), text)
            self._refresh_preset_combo()
            # Select the new preset in the combo
            idx = self._ctx_preset_combo.findText(name.strip())
            if idx >= 0:
                self._ctx_preset_combo.setCurrentIndex(idx)

    def _delete_ctx_preset(self):
        idx = self._ctx_preset_combo.currentIndex()
        if idx <= 0:
            QMessageBox.information(self, "Delete Preset", "Select a preset to delete first.")
            return
        name = self._ctx_preset_combo.currentText()
        from utils.context_store import DEFAULT_PRESETS
        if name in DEFAULT_PRESETS:
            QMessageBox.information(
                self, "Built-in Preset",
                f'"{name}" is a built-in preset and cannot be deleted.\n'
                "You can overwrite it by saving a preset with the same name."
            )
            return
        result = QMessageBox.question(
            self, "Delete Preset",
            f'Delete preset "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result == QMessageBox.StandardButton.Yes:
            get_context_store().delete_preset(name)
            self._refresh_preset_combo()

    def _on_ctx_text_changed(self):
        if hasattr(self, "_ctx_char_label") and hasattr(self, "_ctx_edit"):
            n = len(self._ctx_edit.toPlainText())
            color = "#fca5a5" if n > 2000 else "#64748b"
            self._ctx_char_label.setText(f"{n} / 2000 chars")
            self._ctx_char_label.setStyleSheet(
                f"color: {color}; font-size: 9px; background: transparent;"
            )

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
        key_override = ""
        if pid in self.api_inputs:
            try:
                key_override = self.api_inputs[pid].text().strip()
            except Exception:
                key_override = ""
        worker = ProviderTestWorker(pid, self.config, self, key_override=key_override)
        self._test_workers[pid] = worker
        worker.result_ready.connect(self._on_provider_test_result)
        worker.finished.connect(lambda p=pid: self._test_workers.pop(p, None))
        worker.start()

    def _on_provider_test_result(self, provider_id, success, message, details):
        self.status_labels[provider_id].setText("✅" if success else "❌")

        if provider_id in self.provider_detail_labels:
            color = "#4ade80" if success else "#fda4af"
            self.provider_detail_labels[provider_id].setStyleSheet(
                f"color: {color}; font-size: 10px; background: transparent;"
            )
            self.provider_detail_labels[provider_id].setText(message)

        if provider_id != "ollama" or not hasattr(self, "ollama_model_combo"):
            return

        models = []
        endpoint = None
        if isinstance(details, dict):
            models = details.get("models", []) or []
            endpoint = details.get("endpoint")

        if provider_id in self.provider_detail_labels and endpoint:
            self.provider_detail_labels[provider_id].setText(f"{message} | {endpoint}")

        self.ollama_model_combo.blockSignals(True)
        self.ollama_model_combo.clear()

        if not success:
            self.ollama_model_combo.addItem("Ollama test failed")
            self.ollama_model_combo.setEnabled(False)
            self.ollama_model_combo.blockSignals(False)
            return

        if not models:
            self.ollama_model_combo.addItem("No local models found")
            self.ollama_model_combo.setEnabled(False)
            self.ollama_model_combo.blockSignals(False)
            return

        self.ollama_model_combo.addItems(models)
        saved_model = self.config.get("ai.providers.ollama.model", "")
        selected_mode = self.config.get("ai.mode", "general")
        target_model = (
            saved_model
            if saved_model in models
            else self._recommended_ollama_model(models, selected_mode)
        )
        target_index = models.index(target_model) if target_model in models else 0
        self.ollama_model_combo.setCurrentIndex(target_index)
        self.ollama_model_combo.setEnabled(True)
        self.ollama_model_combo.blockSignals(False)

    def showEvent(self, event):
        """Reset UI state when settings are opened."""
        super().showEvent(event)
        self.btn_save.setText("APPLY SETTINGS")
        self.btn_save.setEnabled(True)
        self.btn_save.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 24px; font-weight: 900; font-size: 11px;
            }
            QPushButton:hover { background: #6366f1; border: 1px solid rgba(255,255,255,30); }
        """)

    def _save_all(self):
        try:
            self.btn_save.setText("APPLYING...")
            self.btn_save.setEnabled(False)
            self.btn_save.setStyleSheet(
                "background: #312e81; color: #6366f1; border-radius: 24px; font-weight: 900; font-size: 11px;"
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
            if hasattr(self, "chk_vision_local_only"):
                self.config.set(
                    "ai.vision.local_only",
                    self.chk_vision_local_only.isChecked(),
                )
            self.config.set("stealth.enabled", True)

            # Save vision priority and race mode
            if hasattr(self, "vision_primary") and hasattr(self, "vision_secondary"):
                vision_local = bool(self.config.get("ai.vision.local_only", False))
                if vision_local:
                    # Force local-only path without mutating provider enablement.
                    self.config.set("ai.vision.allow_paid_fallback", False)
                    self.config.set("ai.vision.race_enabled", False)
                    self.config.set("ai.vision.preferred_providers", ["ollama"])
                else:
                    options = ["gemini", "ollama", "openai"]
                    p1 = options[self.vision_primary.currentIndex()] if self.vision_primary.currentIndex() < len(options) else "gemini"
                    p2 = options[self.vision_secondary.currentIndex()] if self.vision_secondary.currentIndex() < len(options) else "ollama"
                    order = [p1]
                    if p2 and p2 not in order:
                        order.append(p2)
                    # Append openai only when paid fallback is allowed (prevents useless attempts)
                    allow_paid = bool(self.config.get("ai.vision.allow_paid_fallback", False))
                    if allow_paid and "openai" not in order:
                        order.append("openai")
                    self.config.set("ai.vision.preferred_providers", order)
            if hasattr(self, "chk_vision_race"):
                if not bool(self.config.get("ai.vision.local_only", False)):
                    self.config.set("ai.vision.race_enabled", self.chk_vision_race.isChecked())

            # Save text priority and race mode (manual + audio queries)
            if hasattr(self, "text_primary") and hasattr(self, "text_secondary"):
                text_local = bool(
                    getattr(self, "chk_text_local_only", None)
                    and self.chk_text_local_only.isChecked()
                )
                if text_local:
                    self.config.set("ai.text.race_enabled", False)
                    self.config.set("ai.text.preferred_providers", ["ollama"])
                else:
                    options = ["groq", "cerebras", "together", "gemini", "ollama"]
                    p1 = options[self.text_primary.currentIndex()] if self.text_primary.currentIndex() < len(options) else "groq"
                    p2 = options[self.text_secondary.currentIndex()] if self.text_secondary.currentIndex() < len(options) else "cerebras"
                    order = [p1]
                    if p2 and p2 not in order:
                        order.append(p2)
                    # Append remaining defaults to keep a complete fallback chain.
                    for p in options:
                        if p not in order:
                            order.append(p)
                    self.config.set("ai.text.preferred_providers", order)
            if hasattr(self, "chk_text_race"):
                if not bool(
                    getattr(self, "chk_text_local_only", None)
                    and self.chk_text_local_only.isChecked()
                ):
                    self.config.set("ai.text.race_enabled", self.chk_text_race.isChecked())
            if hasattr(self, "chk_text_local_only"):
                self.config.set("ai.text.local_only", self.chk_text_local_only.isChecked())

            # Save screenshot interval
            if hasattr(self, "screenshot_interval"):
                interval_values = [500, 1000, 2000, 3000, 5000]
                self.config.set(
                    "capture.screen.interval_ms",
                    interval_values[self.screenshot_interval.currentIndex()],
                )

            # P2.3: Save transcription language
            if hasattr(self, "audio_language"):
                lang_codes = ["", "en", "hi", "es", "fr", "de", "ja", "zh", "pt", "ar", "ru"]
                idx = self.audio_language.currentIndex()
                self.config.set("capture.audio.language", lang_codes[idx] if idx < len(lang_codes) else "")

            # P2.8: Save correction provider
            if hasattr(self, "correction_provider"):
                cp_values = ["auto", "groq", "gemini", "cerebras", "together", "ollama"]
                idx = self.correction_provider.currentIndex()
                self.config.set("capture.audio.correction_provider", cp_values[idx] if idx < len(cp_values) else "groq")

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
            if hasattr(self, "ollama_model_combo") and self.ollama_model_combo.isEnabled():
                selected_model = self.ollama_model_combo.currentText().strip()
                if selected_model and "load local models" not in selected_model.lower():
                    self.config.set("ai.providers.ollama.model", selected_model)

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
            if hasattr(self, "chk_focus_on_show"):
                self.config.set(
                    "app.focus_on_show",
                    self.chk_focus_on_show.isChecked(),
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

            # Save and apply session context
            if hasattr(self, "_ctx_edit"):
                ctx_text = self._ctx_edit.toPlainText().strip()[:2000]
                store = get_context_store()
                store.set_last_context(ctx_text)
                if self.app and hasattr(self.app, "state"):
                    self.app.state.session_context = ctx_text
                    if hasattr(self.app, "ai"):
                        self.app.ai.set_session_context(ctx_text)
                    # Mark as manually set — mode switches won't overwrite this
                    if hasattr(self.app, "_context_auto_suggested"):
                        self.app._context_auto_suggested = False

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
