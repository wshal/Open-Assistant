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


class CaptureTabMixin:


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
        sep_mode.setStyleSheet("background: rgba(255,255,255,12);")
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
        sep.setStyleSheet("background: rgba(255,255,255,12);")
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
        sep2.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep2)

        # Q21: Whisper model size selector
        lbl_wm = self._make_section_label("WHISPER MODEL SIZE")
        l.addWidget(lbl_wm)
        self.whisper_model = QComboBox()
        _whisper_models = ["tiny.en", "base.en", "small.en", "medium.en"]
        self.whisper_model.addItems([
            "⚡ tiny.en  (fastest, English-only)",
            "✅ base.en  (balanced speed and accuracy)",
            "🔍 small.en (default, best local benchmark result)",
            "🧠 medium.en (best accuracy, heaviest runtime cost)",
        ])
        _wm_map = {m: i for i, m in enumerate(_whisper_models)}
        saved_wm = self.config.get("capture.audio.whisper_model", "small.en")
        self.whisper_model.setCurrentIndex(_wm_map.get(saved_wm, 2))
        self.whisper_model.setToolTip(
            "Whisper model size affects transcription speed vs accuracy.\n"
            "tiny.en: fastest, but benchmarked weakest on long coding questions.\n"
            "base.en: balanced fallback with lower CPU cost.\n"
            "small.en: current default and best local benchmark winner.\n"
            "medium.en: higher accuracy potential, but much slower.\n"
            "⚠️ Changing this requires an app restart to take effect."
        )
        self._style_combo(self.whisper_model)
        l.addWidget(self.whisper_model)
        # Restart notice label (hidden by default, shown on change)
        self._whisper_restart_lbl = QLabel("⚠️ Restart required for model change to take effect")
        self._whisper_restart_lbl.setStyleSheet(
            "color: #f59e0b; font-size: 10px; background: transparent; padding: 2px 0;"
        )
        self._whisper_restart_lbl.setVisible(False)
        l.addWidget(self._whisper_restart_lbl)
        _orig_wm_idx = self.whisper_model.currentIndex()
        def _on_whisper_model_changed(idx):
            self._whisper_restart_lbl.setVisible(idx != _orig_wm_idx)
        self.whisper_model.currentIndexChanged.connect(_on_whisper_model_changed)
        wm_desc = QLabel(
            "Select the Whisper ASR model size. small.en is the current recommended "
            "default for local coding Q&A. Changes apply after restart."
        )
        wm_desc.setWordWrap(True)
        wm_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(wm_desc)

        sep_wm = QFrame()
        sep_wm.setFixedHeight(1)
        sep_wm.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep_wm)

        # Phase 2: Transcription Engine
        lbl_tprov = self._make_section_label("TRANSCRIPTION ENGINE")
        l.addWidget(lbl_tprov)
        self.transcription_provider = QComboBox()
        self.transcription_provider.addItems([
            "⚡ Local (Faster-Whisper)",
            "☁️ Cloud (Groq Whisper — Fastest)",
        ])
        _saved_tp = self.config.get("capture.audio.transcription_provider", "local")
        self.transcription_provider.setCurrentIndex(0 if _saved_tp != "groq" else 1)
        self._style_combo(self.transcription_provider)
        l.addWidget(self.transcription_provider)
        tp_desc = QLabel(
            "Local uses your CPU/GPU with no internet. "
            "Groq Cloud uses the whisper-large-v3 model for sub-100ms transcription (requires Groq API key)."
        )
        tp_desc.setWordWrap(True)
        tp_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(tp_desc)

        self.chk_chunking = PremiumCheckBox("Enable Low-Latency Dynamic Chunking")
        self.chk_chunking.setChecked(bool(self.config.get("capture.audio.chunking.enabled", True)))
        l.addWidget(self.chk_chunking)
        chunking_desc = QLabel(
            "Slices long utterances at natural micro-pauses (2–4 s window) and streams each "
            "chunk to Whisper in parallel. No word-slicing, no overlap deduplication."
        )
        chunking_desc.setWordWrap(True)
        chunking_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(chunking_desc)

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
        sep3.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep3)

        lbl2 = self._make_section_label("VISION ENGINE")
        l.addWidget(lbl2)

        self.chk_vision_enabled = PremiumCheckBox("Enable Vision (OCR)")
        self.chk_vision_enabled.setChecked(self.config.get("capture.screen.enabled", True))
        l.addWidget(self.chk_vision_enabled)
        vision_desc = QLabel("Master switch for screen analysis. When disabled, the AI runs in audio-only mode for minimal latency.")
        vision_desc.setWordWrap(True)
        vision_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(vision_desc)

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
        sep_text.setStyleSheet("background: rgba(255,255,255,12);")
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


