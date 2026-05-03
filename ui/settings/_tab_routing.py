from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QScrollArea, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt
from ui.custom_widgets import PremiumCheckBox
from ui.settings.constants import TEXT_MUTED


class RoutingTabMixin:
    def _tab_routing(self):
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
        mode_desc = QLabel(
            "Controls how OpenAssist prioritizes speed, reasoning, and live context while answering."
        )
        mode_desc.setWordWrap(True)
        mode_desc.setStyleSheet(
            f"{TEXT_MUTED} font-size: 10px; background: transparent;"
        )
        l.addWidget(mode_desc)

        sep_mode = QFrame()
        sep_mode.setFixedHeight(1)
        sep_mode.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep_mode)

        lbl_corr = self._make_section_label("TRANSCRIPT CORRECTION PROVIDER")
        l.addWidget(lbl_corr)
        self.correction_provider = QComboBox()
        self.correction_provider.addItems(
            ["Auto (Fastest Available)", "groq", "gemini", "cerebras", "together", "ollama"]
        )
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

        sep_vision = QFrame()
        sep_vision.setFixedHeight(1)
        sep_vision.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep_vision)

        lbl2 = self._make_section_label("VISION ROUTING")
        l.addWidget(lbl2)

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
        sep_text.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep_text)

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

        self.chk_vision_local_only.toggled.connect(lambda _checked: self._apply_local_only_lockouts())
        self.chk_text_local_only.toggled.connect(lambda _checked: self._apply_local_only_lockouts())
        self._apply_local_only_lockouts()

        l.addStretch()
        scroll.setWidget(w)
        return scroll

    def _apply_local_only_lockouts(self):
        try:
            v_local = bool(
                getattr(self, "chk_vision_local_only", None)
                and self.chk_vision_local_only.isChecked()
            )
            t_local = bool(
                getattr(self, "chk_text_local_only", None)
                and self.chk_text_local_only.isChecked()
            )
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
