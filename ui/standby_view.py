"""
Premium Standby View - OpenAssist AI.

Refined for:
- removed STATUS_ITEMS (STABLE / SYNCED / ACTIVE pills)
- tighter, more intentional vertical rhythm
- consistent section spacing with no double-gaps
- provider bar moved closer to footer
- cleaner section header style
"""

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from utils.logger import setup_logger

logger = setup_logger(__name__)


class StandbyView(QWidget):
    start_clicked = pyqtSignal()
    mode_selected = pyqtSignal(str)
    audio_source_changed = pyqtSignal(str)

    HERO_ICON = "\U0001F9E0"
    MODE_OPTIONS = [
        [("\U0001F9E0", "GENERAL"), ("\U0001F3AF", "INTERVIEW")],
        [("\U0001F4BB", "CODING"), ("\U0001F91D", "MEETING")],
        [("\U0001F393", "EXAM"), ("\u270D\uFE0F", "WRITING")],
    ]
    AUDIO_OPTIONS = [
        ("\U0001F399\uFE0F MIC", "mic"),
        ("\U0001F50A SYSTEM", "system"),
        ("\U0001F310 BOTH", "both"),
    ]

    # ── Shared button styles ─────────────────────────────────────────────────

    STYLE_INACTIVE = """
        QPushButton {
            color: #94a3b8;
            background: rgba(22, 24, 40, 200);
            border: 1px solid rgba(99, 102, 241, 20);
            border-radius: 10px;
            padding: 9px 14px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        QPushButton:hover {
            background: rgba(40, 42, 65, 240);
            border: 1px solid rgba(129, 140, 248, 50);
            color: #c0caff;
        }
    """

    STYLE_ACTIVE = """
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #5b4cf1,
                stop:0.55 #6d4cf6,
                stop:1 #8b5cf6
            );
            color: white;
            border: 1px solid rgba(255, 255, 255, 60);
            border-radius: 10px;
            padding: 9px 14px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1px;
        }
    """

    START_BUTTON_STYLE = """
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #5b4cf1,
                stop:0.55 #7047f5,
                stop:1 #8b5cf6
            );
            color: white;
            border-radius: 26px;
            font-weight: 900;
            font-size: 13px;
            letter-spacing: 3px;
            border: 1px solid rgba(255, 255, 255, 18);
        }
        QPushButton:hover:enabled {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #6a68ff,
                stop:1 #9b6bff
            );
            border: 1px solid rgba(255, 255, 255, 40);
        }
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    """

    START_BUTTON_READY_STYLE = """
        QPushButton {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #10b981,
                stop:1 #059669
            );
            color: white;
            border-radius: 26px;
            font-weight: 900;
            font-size: 13px;
            letter-spacing: 3px;
            border: 1px solid rgba(255, 255, 255, 18);
        }
        QPushButton:hover:enabled {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #34d399,
                stop:1 #10b981
            );
            border: 1px solid rgba(255, 255, 255, 40);
        }
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
    """

    # ── Section header shared style ──────────────────────────────────────────
    _SS_SECTION = (
        "font-size: 9px; color: #475569; font-weight: 900; "
        "letter-spacing: 2.5px; background: transparent;"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_status_widgets = {}
        self.mode_buttons = {}
        self.audio_btns = {}
        self._boot_sync_scheduled = False
        self._boot_sync_logged = False
        self._warmup_done = False  # Latch: once True, no signal can revert READY state

        self._init_ui()
        self._connect_state()
        self._schedule_boot_sync()

    def _connect_state(self):
        parent = self.parent()
        app = getattr(parent, "app", None)
        state = getattr(app, "state", None)
        if state is not None:
            state.mode_changed.connect(self.set_current_mode)
            state.audio_source_changed.connect(self.set_current_audio_source)

    def showEvent(self, event):
        """Final UI sync trigger on window mapping."""
        super().showEvent(event)
        self._schedule_boot_sync()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 20)
        layout.setSpacing(0)
        self.main_layout = layout

        # ── Hero glow block ──────────────────────────────────────────────────
        hero_frame = QFrame()
        hero_frame.setFixedHeight(120)
        hero_frame.setStyleSheet("""
            QFrame {
                background: qradialgradient(
                    cx:0.5, cy:0.45, radius:0.70,
                    fx:0.5, fy:0.38,
                    stop:0 rgba(167, 139, 250, 65),
                    stop:0.40 rgba(99, 102, 241, 22),
                    stop:1 rgba(0, 0, 0, 0)
                );
                border: none;
                border-radius: 24px;
            }
        """)
        hero_inner = QVBoxLayout(hero_frame)
        hero_inner.setContentsMargins(0, 8, 0, 0)
        hero_inner.setSpacing(0)

        self.hero_label = QLabel(self.HERO_ICON)
        self.hero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_label.setStyleSheet("""
            font-size: 72px;
            background: transparent;
            color: #f3e8ff;
        """)
        self.hero_label.setMinimumHeight(80)
        hero_inner.addWidget(
            self.hero_label,
            0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
        )
        hero_inner.addStretch(1)
        layout.addWidget(hero_frame)

        layout.addSpacing(14)

        # ── Status subtitle ──────────────────────────────────────────────────
        self.subtitle = QLabel("NEURAL ENGINE INITIALIZING...")
        self.subtitle.setFixedHeight(16)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet("""
            font-size: 9px;
            color: #7c86cc;
            letter-spacing: 2.8px;
            font-weight: 800;
            background: transparent;
        """)
        layout.addWidget(self.subtitle)

        layout.addSpacing(20)

        # ── Divider ──────────────────────────────────────────────────────────
        layout.addWidget(self._divider())

        layout.addSpacing(16)

        # ── AI MODES ─────────────────────────────────────────────────────────
        lbl_m = QLabel("CAPTURE MODE")
        lbl_m.setStyleSheet(self._SS_SECTION)
        lbl_m.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_m)

        layout.addSpacing(14)

        for row_data in self.MODE_OPTIONS:
            row = QHBoxLayout()
            row.setSpacing(10)
            for icon, name in row_data:
                btn = QPushButton(f"{icon}  {name}")
                btn.setCheckable(True)
                btn.setFixedHeight(38)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(
                    lambda checked=False, n=name.lower(): self._on_mode_btn_clicked(n)
                )
                btn.setStyleSheet(self.STYLE_INACTIVE)
                row.addWidget(btn)
                self.mode_buttons[name.lower()] = btn
            layout.addLayout(row)
            layout.addSpacing(8)

        # ── Context chip ─────────────────────────────────────────────────────
        # Shows auto-suggested or user-active context beneath the mode grid.
        # Hidden until a mode with a matching preset is selected.
        self._ctx_chip = QLabel()
        self._ctx_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ctx_chip.setFixedHeight(20)
        self._ctx_chip.hide()
        layout.addWidget(self._ctx_chip)

        layout.addSpacing(4)

        # ── CAPTURE SOURCE ───────────────────────────────────────────────────
        layout.addWidget(self._divider())

        layout.addSpacing(16)

        lbl_a = QLabel("CAPTURE SOURCE")
        lbl_a.setStyleSheet(self._SS_SECTION)
        lbl_a.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_a)

        layout.addSpacing(10)

        audio_row = QHBoxLayout()
        audio_row.setSpacing(10)
        for label, name in self.AUDIO_OPTIONS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self.STYLE_INACTIVE)
            btn.clicked.connect(
                lambda checked=False, n=name: self._on_audio_btn_clicked(n)
            )
            audio_row.addWidget(btn)
            self.audio_btns[name] = btn
        layout.addLayout(audio_row)

        layout.addSpacing(25)

        # ── Provider status bar ──────────────────────────────────────────────
        self.model_bar = QFrame()
        self.model_bar.setFixedHeight(30)
        self.model_bar.setStyleSheet("""
            QFrame {
                background: rgba(10, 12, 24, 180);
                border-radius: 15px;
                border: 1px solid rgba(99, 102, 241, 18);
            }
        """)
        self.model_bar_layout = QHBoxLayout(self.model_bar)
        self.model_bar_layout.setContentsMargins(14, 0, 14, 0)
        self.model_bar_layout.setSpacing(12)
        self.model_bar_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_bar)
        self.set_provider_statuses({})

        layout.addSpacing(6)

        # P2.4: Update available badge (shown only when a newer version is detected)
        self._update_badge = QLabel()
        self._update_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_badge.setStyleSheet(
            "background: rgba(251,191,36,0.15); color: #fbbf24;"
            " border: 1px solid rgba(251,191,36,0.4); border-radius: 10px;"
            " font-size: 9px; font-weight: 800; letter-spacing: 1px; padding: 3px 10px;"
        )
        self._update_badge.setText("⬆ UPDATE AVAILABLE — github.com/OpenAssist")
        self._update_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_badge.hide()
        layout.addWidget(self._update_badge)
        # Kick off background version check
        QTimer.singleShot(3000, self._check_for_update)

        layout.addStretch(1)

        # ── Footer ───────────────────────────────────────────────────────────
        footer = QVBoxLayout()
        footer.setSpacing(10)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 2px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6a68ff,
                    stop:1 #c15cff
                );
                border-radius: 2px;
            }
        """)
        footer.addWidget(self.progress_bar)

        self.start_btn = QPushButton("START SESSION")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setMinimumWidth(240)
        self.start_btn.setMaximumWidth(400)
        self.start_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(self.START_BUTTON_STYLE)
        self.start_btn.clicked.connect(self.start_clicked.emit)
        footer.addWidget(self.start_btn, 0, Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(footer)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: rgba(255, 255, 255, 0.04);")
        return line

    def _check_for_update(self):
        """P2.4: Non-blocking GitHub releases check — shows badge if update available."""
        import threading

        def _fetch():
            try:
                import urllib.request, json as _json
                REPO = "OpenAssist/OpenAssist"
                url = f"https://api.github.com/repos/{REPO}/releases/latest"
                req = urllib.request.Request(url, headers={"User-Agent": "OpenAssist-Updater/1.0"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = _json.loads(r.read())
                latest_tag = data.get("tag_name", "").lstrip("v")
                try:
                    from core.constants import APP_VERSION as local_ver
                except Exception:
                    local_ver = "0.0.0"

                def _ver_tuple(v):
                    try:
                        return tuple(int(x) for x in str(v).split(".")[:3])
                    except Exception:
                        return (0, 0, 0)

                if _ver_tuple(latest_tag) > _ver_tuple(local_ver):
                    tag = data.get("tag_name", latest_tag)
                    # Update badge text on main thread via QTimer trick
                    QTimer.singleShot(0, lambda: self._show_update_badge(tag))
            except Exception:
                pass  # Non-fatal — silently ignore (offline, rate limit, etc.)

        t = threading.Thread(target=_fetch, daemon=True, name="update-check")
        t.start()

    def _show_update_badge(self, tag: str):
        """Show the update badge with the new version tag."""
        self._update_badge.setText(f"⬆ UPDATE AVAILABLE  v{tag}  — click to visit releases")
        self._update_badge.mousePressEvent = lambda e: self._open_releases()
        self._update_badge.show()

    def _open_releases(self):
        try:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl("https://github.com/OpenAssist/OpenAssist/releases"))
        except Exception:
            pass

    # ── Provider status bar ──────────────────────────────────────────────────

    def set_provider_statuses(self, statuses: dict):
        while self.model_bar_layout.count():
            item = self.model_bar_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not statuses:
            badge = QLabel("WAITING FOR PROVIDERS")
            badge.setStyleSheet(
                "color: #3b4266; font-size: 8px; font-weight: 900; letter-spacing: 1.5px;"
            )
            self.model_bar_layout.addWidget(badge)
            return

        ready_statuses = [
            (pid, info)
            for pid, info in statuses.items()
            if info.get("usable") or info.get("state") in {"active", "cooldown"}
        ]

        if not ready_statuses:
            badge = QLabel("NO PROVIDERS READY")
            badge.setStyleSheet(
                "color: #3b4266; font-size: 8px; font-weight: 900; letter-spacing: 1.5px;"
            )
            self.model_bar_layout.addWidget(badge)
            return

        for pid, info in ready_statuses:
            state = info.get("state", "unknown")
            selected = bool(info.get("selected"))
            color = (
                "#4ade80"
                if state == "active"
                else "#f59e0b"
                if state == "cooldown"
                else "#ef4444"
            )
            label = f"\u25cf {pid.upper()}" if selected else pid.upper()
            badge = QLabel(label)
            if selected:
                badge.setStyleSheet(
                    f"color: {color}; font-size: 8px; font-weight: 900; letter-spacing: 1.2px; "
                    "background: rgba(255,255,255,0.05); "
                    f"border: 1px solid {color}30; border-radius: 9px; padding: 2px 7px;"
                )
            else:
                badge.setStyleSheet(
                    f"color: {color}; font-size: 8px; font-weight: 800; letter-spacing: 1.2px;"
                )
            self.model_bar_layout.addWidget(badge)

    # ── Mode / audio selection ────────────────────────────────────────────────

    def _on_mode_btn_clicked(self, name):
        self.set_current_mode(name)
        self.mode_selected.emit(name)

    def set_current_mode(self, name):
        """Force the active mode button to the highlighted style."""
        if not name:
            return
        target = str(name).strip().lower()
        logger.debug("Injecting Mode Highlight: '%s'", target)
        for mode_name, btn in self.mode_buttons.items():
            active = mode_name == target
            btn.setChecked(active)
            btn.setStyleSheet(self.STYLE_ACTIVE if active else self.STYLE_INACTIVE)

    def _on_audio_btn_clicked(self, name):
        self.set_current_audio_source(name)
        self.audio_source_changed.emit(name)

    def set_current_audio_source(self, name):
        """Force the active audio source button to the highlighted style."""
        if not name:
            return
        target = str(name).strip().lower()
        logger.debug("Injecting Audio Highlight: '%s'", target)
        for source_name, btn in self.audio_btns.items():
            active = source_name == target
            btn.setChecked(active)
            btn.setStyleSheet(self.STYLE_ACTIVE if active else self.STYLE_INACTIVE)

    def show_context_chip(self, preset_name: str | None, applied: bool = True):
        """Update the context chip below the mode grid.

        preset_name=None  → hide the chip (no suggestion for this mode).
        applied=True      → green chip: context was auto-loaded.
        applied=False     → amber chip: suggestion only, user has custom context.
        """
        if not hasattr(self, "_ctx_chip"):
            return
        if not preset_name:
            self._ctx_chip.hide()
            return

        if applied:
            # Green: context is active
            self._ctx_chip.setText(f"✓ Context: {preset_name}")
            self._ctx_chip.setStyleSheet(
                "color: #4ade80; font-size: 8px; font-weight: 800; "
                "letter-spacing: 1.2px; background: rgba(74,222,128,0.08); "
                "border: 1px solid rgba(74,222,128,0.25); border-radius: 8px; "
                "padding: 1px 10px;"
            )
        else:
            # Amber: suggestion available but not loaded (user has custom context)
            self._ctx_chip.setText(f"⚡ Suggested: {preset_name} (Settings › CONTEXT)")
            self._ctx_chip.setStyleSheet(
                "color: #f59e0b; font-size: 8px; font-weight: 700; "
                "letter-spacing: 0.8px; background: rgba(245,158,11,0.07); "
                "border: 1px solid rgba(245,158,11,0.2); border-radius: 8px; "
                "padding: 1px 10px;"
            )
        self._ctx_chip.show()

    # ── Warmup state ─────────────────────────────────────────────────────────

    def set_warmup_status(self, message: str, progress: int = 0, ready: bool = False):
        """Update warmup progress bar and start button state.

        Once ready=True fires the latch closes: subsequent calls with ready=False
        are ignored so deferred background tasks (Whisper, EasyOCR) completing
        AFTER the READY signal cannot revert the 'SESSION READY' button.
        """
        if self._warmup_done and not ready:
            logger.debug(f"Post-READY warmup signal ignored: {message}")
            return

        self.subtitle.setText(message.upper())
        self.progress_bar.setValue(progress)
        self.start_btn.setEnabled(ready)
        if ready:
            self._warmup_done = True
            self.subtitle.setText("ALL SYSTEMS ONLINE")
            self.progress_bar.setValue(100)
            self.start_btn.setText("SESSION READY")
            self.start_btn.setStyleSheet(self.START_BUTTON_READY_STYLE)
        else:
            self.start_btn.setText("START SESSION")
            self.start_btn.setStyleSheet(self.START_BUTTON_STYLE)

    # ── Boot sync ─────────────────────────────────────────────────────────────

    def _apply_initial_highlights(self):
        """Hardened boot sync with forced defaults."""
        mode, audio = self._resolve_initial_selection()
        if not self._boot_sync_logged:
            logger.info("Boot sync: mode='%s', audio='%s'", mode, audio)
            self._boot_sync_logged = True
        self.set_current_mode(mode)
        self.set_current_audio_source(audio)

    def refresh_highlights(self, mode=None, audio=None):
        """Force refresh selection states from explicit values or resolved state."""
        resolved_mode, resolved_audio = self._resolve_initial_selection()
        self.set_current_mode(mode or resolved_mode)
        self.set_current_audio_source(audio or resolved_audio)

    def _resolve_initial_selection(self):
        mode = "general"
        audio = "system"

        parent = self.parent()
        app = getattr(parent, "app", None)

        if app and hasattr(app, "state"):
            state = app.state
            mode = (getattr(state, "mode", None) or mode).strip().lower()
            audio = (getattr(state, "audio_source", None) or audio).strip().lower()

        if app and hasattr(app, "config"):
            mode = (app.config.get("ai.mode", mode) or mode).strip().lower()
            audio = (app.config.get("capture.audio.mode", audio) or audio).strip().lower()

        return mode or "general", audio or "system"

    def _schedule_boot_sync(self):
        if self._boot_sync_scheduled:
            return
        self._boot_sync_scheduled = True
        for delay_ms in (100, 500, 1500):
            QTimer.singleShot(delay_ms, self._apply_initial_highlights)
