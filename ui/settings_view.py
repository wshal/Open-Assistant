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
from ui.settings._tab_api import ApiTabMixin
from ui.settings._tab_capture import CaptureTabMixin
from ui.settings._tab_context import ContextTabMixin
from ui.settings._tab_hotkeys import HotkeysTabMixin
from ui.settings._tab_stealth import StealthTabMixin
from ui.settings._tab_ui import UiTabMixin

from utils.context_store import get_store as get_context_store

logger = setup_logger(__name__)

# --- Premium Midnight Styling ---
BG_DARK = "background: rgba(15,15,30,245);"
TEXT_PRIMARY = "color: #c0c0ff;"
TEXT_MUTED = "color: #64748b;"
SS_INPUT = """
QLineEdit, QComboBox {
    background: rgba(20,20,40,220);
    color: #e0e0f5;
    border: 1px solid rgba(80,85,255,20);
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
    background: rgba(255,255,255,12);
    color: #94a3b8;
    border-radius: 12px;
    font-weight: 600;
    font-size: 11px;
    padding: 12px 24px;
    border: 1px solid rgba(255,255,255,25);
}}
QPushButton:hover {{
    background: rgba(255,255,255,20);
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


class SettingsView(QWidget, ApiTabMixin, CaptureTabMixin, ContextTabMixin, HotkeysTabMixin, StealthTabMixin, UiTabMixin):
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
        self.test_buttons = {}          # Q10: populated by _tab_api
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
            QTabWidget::pane { border: 1px solid rgba(80,85,255,10); background: transparent; border-radius: 8px; }
            QTabBar::tab { background: rgba(0,0,0,76); color: #778; padding: 10px 20px; font-size: 10px; font-weight: bold; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; }
            QTabBar::tab:selected { background: rgba(80,85,255,20); color: #c0c0ff; border-bottom: 2px solid #6366f1; }
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
        # Q5: Rate-limit — disable TEST button for 3 seconds to prevent spam
        if not hasattr(self, '_test_cooldowns'):
            self._test_cooldowns = {}
        if self._test_cooldowns.get(pid, False):
            logger.debug("[Q5 Debounce] TEST for %s is on cooldown, ignoring click", pid)
            return
        self._test_cooldowns[pid] = True
        # Re-enable after 3 seconds
        QTimer.singleShot(3000, lambda p=pid: self._test_cooldowns.update({p: False}))

        self.status_labels[pid].setText("\u23f3")
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
        logger.info("[Q5] Provider test started for %s (3s cooldown active)", pid)

    def _on_provider_test_result(self, provider_id, success, message, details):
        import datetime as _dt
        self.status_labels[provider_id].setText("\u2705" if success else "\u274c")

        if provider_id in self.provider_detail_labels:
            color = "#4ade80" if success else "#fda4af"
            # Q6: Append last-tested timestamp
            ts = _dt.datetime.now().strftime("%H:%M:%S")
            display_msg = f"{message}  \u00b7  tested {ts}"
            self.provider_detail_labels[provider_id].setStyleSheet(
                f"color: {color}; font-size: 10px; background: transparent;"
            )
            self.provider_detail_labels[provider_id].setText(display_msg)
            logger.info("[Q6 Timestamp] %s test result: %s at %s", provider_id, message, ts)

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
        """Reset UI state and sync from current config when settings are opened."""
        super().showEvent(event)
        self._sync_ui_from_config()
        self.btn_save.setText("APPLY SETTINGS")
        self.btn_save.setEnabled(True)
        self.btn_save.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f46e5, stop:1 #7c3aed); 
                color: white; border-radius: 24px; font-weight: 900; font-size: 11px;
            }
            QPushButton:hover { background: #6366f1; border: 1px solid rgba(255,255,255,30); }
        """)

    def _sync_ui_from_config(self):
        """Pull latest config/state values into UI widgets."""
        logger.debug("[Settings] Syncing UI from current config/state...")

        # 1. Sync AI Mode
        if hasattr(self, "ai_mode"):
            current_mode = self.config.get("ai.mode", "general")
            mode_map = {"general": 0, "interview": 1, "coding": 2, "meeting": 3, "exam": 4, "writing": 5}
            self.ai_mode.blockSignals(True)
            self.ai_mode.setCurrentIndex(mode_map.get(current_mode, 0))
            self.ai_mode.blockSignals(False)

        # 2. Sync Audio Source
        if hasattr(self, "audio_mode"):
            curr = self.config.get("capture.audio.mode", "system")
            self.audio_mode.blockSignals(True)
            self.audio_mode.setCurrentIndex(0 if curr == "system" else 1 if curr == "mic" else 2)
            self.audio_mode.blockSignals(False)

        # 3. Always refresh API key fields from encrypted storage.
        # Bug fix: previous code only repopulated when the field was empty, so
        # imported keys or keys saved outside the UI were never shown on re-open.
        for pid, inp in self.api_inputs.items():
            stored_key = self.config.get_api_key(pid) or ""
            inp.blockSignals(True)
            inp.setText(stored_key)
            inp._original_value = stored_key
            inp.blockSignals(False)
            # Q10: sync button enabled state with refreshed key
            btn = self.test_buttons.get(pid)
            if btn is not None:
                btn.setEnabled(bool(stored_key))

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
            if hasattr(self, "chk_vision_enabled"):
                self.config.set("capture.screen.enabled", self.chk_vision_enabled.isChecked())
            self.config.set("capture.screen.smart_crop", self.chk_smart.isChecked())
            # Phase 2: Transcription engine and chunking
            if hasattr(self, "transcription_provider"):
                tp = "groq" if self.transcription_provider.currentIndex() == 1 else "local"
                self.config.set("capture.audio.transcription_provider", tp)
            if hasattr(self, "chk_chunking"):
                self.config.set("capture.audio.chunking.enabled", self.chk_chunking.isChecked())
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

            # Q21: Save Whisper model size
            if hasattr(self, "whisper_model"):
                _wm_values = ["tiny.en", "base.en", "small.en", "medium.en"]
                idx = self.whisper_model.currentIndex()
                self.config.set(
                    "capture.audio.whisper_model",
                    _wm_values[idx] if idx < len(_wm_values) else "base.en",
                )
                import logging as _log
                _log.getLogger(__name__).info(
                    "[Q21] Whisper model saved: %s (restart required to apply)",
                    _wm_values[idx] if idx < len(_wm_values) else "base.en",
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
                # Also emit our own signals for any other listeners
                self.mode_changed.emit(selected_mode)
                self.audio_source_changed.emit(selected_audio)
                # Explicitly sync StandbyView UI immediately to avoid signal timing issues
                if hasattr(self.app, 'overlay') and hasattr(self.app.overlay, 'standby_view'):
                    self.app.overlay.standby_view.refresh_highlights(mode=selected_mode, audio=selected_audio)

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

