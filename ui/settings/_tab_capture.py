from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QScrollArea, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QMetaObject, Q_ARG
from ui.custom_widgets import PremiumCheckBox
from ui.settings.constants import TEXT_MUTED


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

        lbl = self._make_section_label("PRIMARY AUDIO SOURCE")
        l.addWidget(lbl)
        self.audio_mode = QComboBox()
        self.audio_mode.addItems(
            ["System Speakers", "Microphone Only", "Hybrid (Both)"]
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
        lang_desc = QLabel(
            "Language hint for the transcription engine. Auto-detect works well for most cases."
        )
        lang_desc.setWordWrap(True)
        lang_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(lang_desc)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep2)

        lbl_wm = self._make_section_label("WHISPER MODEL SIZE")
        l.addWidget(lbl_wm)
        self.whisper_model = QComboBox()
        whisper_models = ["tiny.en", "base.en", "small.en", "medium.en"]
        self.whisper_model.addItems([
            "tiny.en  (fastest, English-only)",
            "base.en  (balanced speed and accuracy)",
            "small.en (default, best local benchmark result)",
            "medium.en (best accuracy, heaviest runtime cost)",
        ])
        wm_map = {m: i for i, m in enumerate(whisper_models)}
        saved_wm = self.config.get("capture.audio.whisper_model", "small.en")
        self.whisper_model.setCurrentIndex(wm_map.get(saved_wm, 2))
        self.whisper_model.setToolTip(
            "Whisper model size affects transcription speed vs accuracy.\n"
            "tiny.en: fastest, but weaker on longer coding questions.\n"
            "base.en: balanced fallback with lower CPU cost.\n"
            "small.en: current default and best local benchmark winner.\n"
            "medium.en: higher accuracy potential, but much slower.\n"
            "Changing this requires an app restart to take effect."
        )
        self._style_combo(self.whisper_model)
        l.addWidget(self.whisper_model)

        self._whisper_restart_lbl = QLabel("Restart required for model change to take effect")
        self._whisper_restart_lbl.setStyleSheet(
            "color: #f59e0b; font-size: 10px; background: transparent; padding: 2px 0;"
        )
        self._whisper_restart_lbl.setVisible(False)
        l.addWidget(self._whisper_restart_lbl)
        orig_wm_idx = self.whisper_model.currentIndex()

        def _on_whisper_model_changed(idx):
            self._whisper_restart_lbl.setVisible(idx != orig_wm_idx)

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

        lbl_tprov = self._make_section_label("TRANSCRIPTION ENGINE")
        l.addWidget(lbl_tprov)
        self.transcription_provider = QComboBox()
        self.transcription_provider.addItems([
            "Local (Faster-Whisper)",
            "Cloud (Groq Whisper - Fastest)",
        ])
        saved_tp = self.config.get("capture.audio.transcription_provider", "local")
        self.transcription_provider.setCurrentIndex(0 if saved_tp != "groq" else 1)
        self._style_combo(self.transcription_provider)
        l.addWidget(self.transcription_provider)
        tp_desc = QLabel(
            "Local uses your CPU/GPU with no internet. "
            "Groq Cloud uses whisper-large-v3 for very fast transcription and requires a Groq API key."
        )
        tp_desc.setWordWrap(True)
        tp_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(tp_desc)

        self.chk_chunking = PremiumCheckBox("Enable Low-Latency Dynamic Chunking")
        self.chk_chunking.setChecked(bool(self.config.get("capture.audio.chunking.enabled", True)))
        l.addWidget(self.chk_chunking)
        chunking_desc = QLabel(
            "Slices long utterances at natural micro-pauses and streams each chunk to Whisper in parallel."
        )
        chunking_desc.setWordWrap(True)
        chunking_desc.setStyleSheet(f"{TEXT_MUTED} font-size: 10px; background: transparent;")
        l.addWidget(chunking_desc)

        sep3 = QFrame()
        sep3.setFixedHeight(1)
        sep3.setStyleSheet("background: rgba(255,255,255,12);")
        l.addWidget(sep3)

        lbl2 = self._make_section_label("VISION ENGINE")
        l.addWidget(lbl2)

        self.chk_vision_enabled = PremiumCheckBox("Enable Vision (Screen Capture + OCR)")
        self.chk_vision_enabled.setChecked(self.config.get("capture.screen.enabled", True))
        l.addWidget(self.chk_vision_enabled)
        vision_desc = QLabel(
            "Master switch for screen analysis. When disabled, the AI runs in audio-only mode for minimal latency."
        )
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

    def _refresh_kb_status(self):
        """Update the knowledge base chunk count label."""
        try:
            import threading
            def _count():
                try:
                    rag = getattr(self.app, "rag", None)
                    if rag:
                        rag._ensure_loaded()
                        count = rag.collection.count() if rag.collection else 0
                        from core.constants import DOCS_DIR
                        from pathlib import Path
                        files = list(Path(DOCS_DIR).rglob("*"))
                        n_files = sum(1 for f in files if f.is_file())
                        msg = f"Loaded {count} chunks from {n_files} file(s)"
                    else:
                        msg = "RAG engine not available"
                    if hasattr(self, "_kb_status_lbl"):
                        QMetaObject.invokeMethod(
                            self._kb_status_lbl,
                            "setText",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(str, msg),
                        )
                except Exception:
                    pass
            threading.Thread(target=_count, daemon=True).start()
        except Exception:
            pass

    def _import_knowledge_files(self):
        """File picker -> copy to DOCS_DIR -> re-index in background thread."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import shutil
        import threading
        from core.constants import DOCS_DIR

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Knowledge Files",
            "",
            "Supported Files (*.pdf *.json *.txt *.md *.py *.yaml *.yml);;"
            "PDF Files (*.pdf);;"
            "Q&A JSON (*.json);;"
            "Text Files (*.txt *.md);;"
            "All Files (*.*)",
        )
        if not paths:
            return

        dest_dir = Path(DOCS_DIR)
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for src in paths:
            try:
                dst = dest_dir / Path(src).name
                shutil.copy2(src, dst)
                copied.append(Path(src).name)
            except Exception as e:
                QMessageBox.warning(self, "Copy Error", f"Could not copy {Path(src).name}:\n{e}")

        if not copied:
            return

        def _reindex():
            try:
                from knowledge.ingest import ingest_all
                rag = getattr(self.app, "rag", None)
                if rag:
                    ingest_all(rag, dest_dir)
                    self._refresh_kb_status()
            except Exception:
                pass

        threading.Thread(target=_reindex, daemon=True, name="kb-import").start()

        names = ", ".join(copied[:5])
        if len(copied) > 5:
            names += f" (+{len(copied)-5} more)"
        QMessageBox.information(
            self,
            "Files Imported",
            f"Copied {len(copied)} file(s) to knowledge folder:\n{names}\n\n"
            f"Indexing in background - available on the next query.",
        )

    def _open_knowledge_folder(self):
        """Open the knowledge documents folder in Windows Explorer."""
        from pathlib import Path
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        from core.constants import DOCS_DIR
        p = Path(DOCS_DIR)
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
