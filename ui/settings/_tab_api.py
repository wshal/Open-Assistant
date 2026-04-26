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
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ApiTabMixin:


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
                # Q7: Row with input + encrypted-storage icon
                key_row = QHBoxLayout()
                key_row.setSpacing(6)

                inp = QLineEdit()
                inp.setEchoMode(QLineEdit.EchoMode.Password)
                inp.setStyleSheet(SS_INPUT)
                inp.setPlaceholderText(f"Enter {name} API Key...")
                inp.setText(self.config.get_api_key(pid) or "")
                # Q4: Store original value so Escape can revert
                inp._original_value = inp.text()
                inp.editingFinished.connect(lambda i=inp: setattr(i, '_original_value', i.text()))
                # Q4: Revert on Escape key
                _orig_key_press = inp.keyPressEvent
                def _make_escape_revert(field, orig_handler):
                    from PyQt6.QtCore import Qt as _Qt
                    def _kp(event, _f=field, _h=orig_handler):
                        if event.key() == _Qt.Key.Key_Escape:
                            _f.setText(getattr(_f, '_original_value', ''))
                            logger.debug("[Q4 Escape] Reverted %s key field to saved value", _f.placeholderText())
                        else:
                            _h(event)
                    return _kp
                inp.keyPressEvent = _make_escape_revert(inp, _orig_key_press)
                # Q8: Tooltip explaining the key
                _tips = {
                    "groq": "Groq Cloud API key. Get one free at console.groq.com. Used for ultra-fast LLM inference.",
                    "gemini": "Google Gemini API key from aistudio.google.com. Used for vision + reasoning tasks.",
                    "cerebras": "Cerebras API key. Fastest token generation on wafer-scale hardware.",
                    "together": "Together AI key for open-source model access (Llama, Mistral, etc.).",
                    "openai": "OpenAI API key from platform.openai.com. Enables GPT-4 and GPT-4o.",
                    "anthropic": "Anthropic Claude key from console.anthropic.com. Best for long-context reasoning.",
                    "mistral": "Mistral AI key. European models with strong coding and multilingual support.",
                }
                inp.setToolTip(_tips.get(pid, f"{name} API key — stored encrypted on this machine."))
                self.api_inputs[pid] = inp
                key_row.addWidget(inp)

                # Q7: Lock icon indicating encrypted storage
                lock_lbl = QLabel("\U0001f512")
                lock_lbl.setStyleSheet("background: transparent; font-size: 12px;")
                lock_lbl.setToolTip("Your API key is stored encrypted (Fernet AES-128) on this machine only. Never sent to any server.")
                key_row.addWidget(lock_lbl)
                bl.addLayout(key_row)

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


