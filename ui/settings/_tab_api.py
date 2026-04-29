"""
AI Engines settings tab (Phase 3 complete).

Features implemented:
  - Q4:  Escape-to-revert on API key fields
  - Q6:  Last-tested timestamp displayed per provider
  - Q7:  Encrypted-storage lock icon + tooltip
  - Q8:  Per-provider tooltip describing the key
  - Q10: TEST button dynamically enabled/disabled when key field changes
  - P3a: Bulk Export — password-protected portable .enc backup
  - P3b: Bulk Import — restores keys from a portable .enc backup
"""

import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QSizePolicy, QMessageBox,
    QComboBox, QInputDialog,
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from core.constants import PROVIDERS
from ui.custom_widgets import PremiumCheckBox
from ui.settings.constants import (
    BG_DARK, TEXT_PRIMARY, TEXT_MUTED, SS_INPUT,
    STYLE_BTN_PRIMARY, STYLE_BTN_SECONDARY,
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

        # Initialise dicts that settings_view.py references by name
        if not hasattr(self, "test_buttons"):
            self.test_buttons = {}          # Q10: track per-provider TEST button

        providers = [
            ("groq",      "🚀 Groq Cloud"),
            ("gemini",    "🧠 Google Gemini"),
            ("cerebras",  "⚡ Cerebras AI"),
            ("together",  "🎭 Together AI"),
            ("openai",    "🟢 OpenAI"),
            ("anthropic", "🟣 Anthropic"),
            ("mistral",   "🌬️ Mistral AI"),
            ("ollama",    "🏠 Local Ollama"),
        ]

        _tips = {
            "groq":      "Groq Cloud API key. Get one free at console.groq.com. Used for ultra-fast LLM inference.",
            "gemini":    "Google Gemini API key from aistudio.google.com. Used for vision + reasoning tasks.",
            "cerebras":  "Cerebras API key. Fastest token generation on wafer-scale hardware.",
            "together":  "Together AI key for open-source model access (Llama, Mistral, etc.).",
            "openai":    "OpenAI API key from platform.openai.com. Enables GPT-4 and GPT-4o.",
            "anthropic": "Anthropic Claude key from console.anthropic.com. Best for long-context reasoning.",
            "mistral":   "Mistral AI key. European models with strong coding and multilingual support.",
        }

        for pid, name in providers:
            provider_meta = PROVIDERS.get(pid, {})
            box = QFrame()
            box.setStyleSheet(
                "background: rgba(255,255,255,5); border-radius: 10px; "
                "padding: 10px; border: 1px solid rgba(255,255,255,7);"
            )
            bl = QVBoxLayout(box)

            # ── Header row: name  status  TEST ──────────────────────────────
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
                "QPushButton { background: rgba(80,85,255,45); color: #f8fafc; "
                "border-radius: 4px; font-size: 10px; font-weight: bold; "
                "border: 1px solid rgba(129,140,248,140); padding: 0 8px; text-align: center; } "
                "QPushButton:hover { background: rgba(80,85,255,89); color: white; "
                "border: 1px solid rgba(165,180,252,216); } "
                "QPushButton:pressed { background: rgba(67,56,202,140); } "
                "QPushButton:disabled { background: rgba(255,255,255,7); "
                "color: rgba(255,255,255,51); border: 1px solid rgba(255,255,255,12); }"
            )
            tbtn.clicked.connect(lambda _, p=pid: self._test_pid(p))
            top.addWidget(tbtn)
            self.test_buttons[pid] = tbtn  # Q10
            bl.addLayout(top)

            # ── "Get API key" / "Install Ollama" link ───────────────────────
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
            link.setStyleSheet("""
                QLabel {
                    color: #bae6fd;
                    font-size: 10px;
                    font-weight: 700;
                    background: rgba(56,189,248,30);
                    border: 1px solid rgba(56,189,248,71);
                    border-radius: 10px;
                    padding: 4px 10px;
                    margin: 0 0 4px 0;
                }
                QLabel:hover {
                    background: rgba(56,189,248,45);
                    border: 1px solid rgba(125,211,252,127);
                }
            """)
            link.adjustSize()
            bl.addWidget(link)

            if pid != "ollama":
                # ── API key row: input + lock icon ───────────────────────────
                key_row = QHBoxLayout()
                key_row.setSpacing(6)

                inp = QLineEdit()
                inp.setEchoMode(QLineEdit.EchoMode.Password)
                inp.setStyleSheet(SS_INPUT)
                inp.setPlaceholderText(f"Enter {name} API Key...")
                inp.setText(self.config.get_api_key(pid) or "")
                inp.setToolTip(_tips.get(pid, f"{name} API key — stored encrypted on this machine."))

                # Q4: store original value so Escape can revert
                inp._original_value = inp.text()
                inp.editingFinished.connect(
                    lambda i=inp: setattr(i, "_original_value", i.text())
                )
                _orig_kp = inp.keyPressEvent

                def _make_escape_revert(field, orig_handler):
                    from PyQt6.QtCore import Qt as _Qt
                    def _kp(event, _f=field, _h=orig_handler):
                        if event.key() == _Qt.Key.Key_Escape:
                            _f.setText(getattr(_f, "_original_value", ""))
                            logger.debug("[Q4] Reverted %s key field", _f.placeholderText())
                        else:
                            _h(event)
                    return _kp

                inp.keyPressEvent = _make_escape_revert(inp, _orig_kp)

                # Q10: enable/disable TEST button based on whether there is a key
                def _update_test_btn(text, btn=tbtn, _pid=pid):
                    has_key = bool(text.strip())
                    btn.setEnabled(has_key)

                inp.textChanged.connect(_update_test_btn)
                # Ollama has no input so initial state is always enabled;
                # for real providers initialise from current field content.
                _update_test_btn(inp.text())

                self.api_inputs[pid] = inp
                key_row.addWidget(inp)

                # Q7: lock icon — encrypted storage indicator
                lock_lbl = QLabel("\U0001f512")
                lock_lbl.setStyleSheet("background: transparent; font-size: 12px;")
                lock_lbl.setToolTip(
                    "Your API key is stored encrypted (AES-128 Fernet) on this machine only. "
                    "Never sent to any server."
                )
                key_row.addWidget(lock_lbl)
                bl.addLayout(key_row)

                # Q6: last-tested status line (populated by _on_provider_test_result)
                detail = QLabel("Not tested")
                detail.setWordWrap(True)
                detail.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
                self.provider_detail_labels[pid] = detail
                bl.addWidget(detail)

            else:
                # Ollama — no API key needed
                desc = QLabel("Local Engine (No API Key Required)")
                desc.setStyleSheet(
                    "color: #64748b; font-size: 10px; font-style: italic; background: transparent;"
                )
                bl.addWidget(desc)

                self.ollama_model_combo = QComboBox()
                self._style_combo(self.ollama_model_combo)
                self.ollama_model_combo.setEnabled(False)
                saved_model = self.config.get("ai.providers.ollama.model", "")
                self.ollama_model_combo.addItem(
                    saved_model or "Test Ollama to load local models"
                )
                bl.addWidget(self.ollama_model_combo)

                detail = QLabel("Tests the local Ollama server and lists installed models.")
                detail.setWordWrap(True)
                detail.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
                self.provider_detail_labels[pid] = detail
                bl.addWidget(detail)

            l.addWidget(box)

        # ── Parallel Inference ───────────────────────────────────────────────
        sep = QFrame()
        sep.setStyleSheet("background: rgba(255,255,255,12); height: 1px; margin: 15px 0;")
        l.addWidget(sep)

        lbl_parallel = QLabel("⚡ PARALLEL INFERENCE")
        lbl_parallel.setStyleSheet("font-size: 10px; color: #475569; font-weight: 800;")
        l.addWidget(lbl_parallel)

        self.chk_parallel = PremiumCheckBox("Enable parallel multi-provider inference")
        self.chk_parallel.setChecked(self.config.get("ai.parallel.enabled", False))
        l.addWidget(self.chk_parallel)

        desc_parallel = QLabel(
            "When enabled, queries are sent to multiple providers simultaneously "
            "and the fastest response is used."
        )
        desc_parallel.setWordWrap(True)
        desc_parallel.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(desc_parallel)

        # ── Bulk API Key Management ──────────────────────────────────────────
        sep_mgmt = QFrame()
        sep_mgmt.setStyleSheet("background: rgba(255,255,255,12); height: 1px; margin: 15px 0;")
        l.addWidget(sep_mgmt)

        lbl_mgmt = QLabel("🔑 API KEY MANAGEMENT")
        lbl_mgmt.setStyleSheet("font-size: 10px; color: #475569; font-weight: 800;")
        l.addWidget(lbl_mgmt)

        desc_mgmt = QLabel(
            "Export all API keys to a portable password-protected backup. "
            "Works on any machine or .exe installation — just use the same password."
        )
        desc_mgmt.setWordWrap(True)
        desc_mgmt.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(desc_mgmt)

        mgmt_row = QHBoxLayout()
        mgmt_row.setSpacing(8)

        btn_export = QPushButton("⬆ Export Keys")
        btn_export.setFixedHeight(30)
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export.setToolTip("Save all API keys to an encrypted .enc backup file")
        btn_export.setStyleSheet(
            "QPushButton { background: rgba(34,197,94,35); color: #86efac; "
            "border-radius: 6px; font-size: 10px; font-weight: bold; "
            "border: 1px solid rgba(34,197,94,89); padding: 0 12px; } "
            "QPushButton:hover { background: rgba(34,197,94,63); color: #bbf7d0; "
            "border: 1px solid rgba(34,197,94,153); }"
        )
        btn_export.clicked.connect(self._export_api_keys)
        mgmt_row.addWidget(btn_export)

        btn_import = QPushButton("⬇ Import Keys")
        btn_import.setFixedHeight(30)
        btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_import.setToolTip("Restore API keys from a previously exported .enc backup")
        btn_import.setStyleSheet(
            "QPushButton { background: rgba(59,130,246,35); color: #93c5fd; "
            "border-radius: 6px; font-size: 10px; font-weight: bold; "
            "border: 1px solid rgba(59,130,246,89); padding: 0 12px; } "
            "QPushButton:hover { background: rgba(59,130,246,63); color: #bfdbfe; "
            "border: 1px solid rgba(59,130,246,153); }"
        )
        btn_import.clicked.connect(self._import_api_keys)
        mgmt_row.addWidget(btn_import)
        mgmt_row.addStretch()
        l.addLayout(mgmt_row)

        c.setLayout(l)
        w.setWidget(c)
        return w

    # ── Bulk Export ──────────────────────────────────────────────────────────

    def _export_api_keys(self):
        """Export all API keys to a portable password-encrypted .enc file."""
        from PyQt6.QtWidgets import QFileDialog

        keys = {pid: inp.text().strip() for pid, inp in self.api_inputs.items() if inp.text().strip()}
        if not keys:
            QMessageBox.information(self, "Export Keys", "No API keys to export.")
            return

        password, ok = QInputDialog.getText(
            self,
            "Set Export Password",
            "Enter a password to protect the backup.\n"
            "You will need this password to import on any machine:",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not password:
            return
        confirm, ok2 = QInputDialog.getText(
            self,
            "Confirm Password",
            "Re-enter the password to confirm:",
            QLineEdit.EchoMode.Password,
        )
        if not ok2 or password != confirm:
            QMessageBox.warning(self, "Password Mismatch", "Passwords did not match. Export cancelled.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Encrypted API Keys", "openassist_keys_backup.enc",
            "OpenAssist Backup (*.enc);;All Files (*)",
        )
        if not path:
            return

        try:
            blob = self.config.secrets.export_keys_portable(password)
            with open(path, "wb") as f:
                f.write(blob)
            QMessageBox.information(
                self, "Export Successful",
                f"Exported {len(keys)} API key(s) to:\n{path}\n\n"
                "Keep your password safe — it cannot be recovered.",
            )
            logger.info("[P3 Export] %d keys exported to %s", len(keys), path)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export keys:\n{e}")
            logger.error("[P3 Export] Failed: %s", e)

    # ── Bulk Import ──────────────────────────────────────────────────────────

    def _import_api_keys(self):
        """Import API keys from a portable password-encrypted .enc backup."""
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Encrypted API Keys", "",
            "OpenAssist Backup (*.enc);;All Files (*)",
        )
        if not path:
            return

        password, ok = QInputDialog.getText(
            self,
            "Enter Import Password",
            "Enter the password used when this backup was exported:",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not password:
            return

        try:
            with open(path, "rb") as f:
                blob = f.read()
            count = self.config.secrets.import_keys_portable(blob, password)
        except ValueError as e:
            QMessageBox.critical(self, "Import Failed", str(e))
            logger.warning("[P3 Import] Bad file/password: %s", e)
            return
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Could not read backup file:\n{e}")
            logger.error("[P3 Import] Unexpected error: %s", e)
            return

        # Refresh in-UI key fields from storage so the user sees the imported values.
        # We intentionally DO NOT update inp._original_value here. This guarantees that
        # the parent SettingsView sees a mismatch between the UI text and the original
        # value, forcing the "APPLY SETTINGS" button to remain enabled until clicked.
        for pid, inp in self.api_inputs.items():
            key = self.config.get_api_key(pid) or ""
            inp.setText(key)

        QMessageBox.information(
            self, "Import Successful",
            f"Imported {count} API key(s).\n"
            "Click \"APPLY SETTINGS\" to persist and activate them.",
        )
        logger.info("[P3 Import] %d keys imported from %s", count, path)
