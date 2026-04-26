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
from utils.context_store import get_store as get_context_store
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ContextTabMixin:


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

