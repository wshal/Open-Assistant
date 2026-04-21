"""
Setup wizard â v4.0 final.
Added: Ollama model download with progress bar on page 2.
"""

import asyncio
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QStackedWidget, QWidget, QFrame, QCheckBox,
    QScrollArea, QMessageBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont, QDesktopServices

from core.constants import PROVIDERS
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ââ Ollama Pull Worker ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class OllamaPullWorker(QThread):
    """
    Pull an Ollama model in a background thread.
    Reports progress via signals.
    """
    progress = Signal(str, float)   # (status_message, percent 0-100)
    finished = Signal(bool, str)    # (success, message)

    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.endpoint = endpoint

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success = loop.run_until_complete(self._pull())
        except Exception as e:
            self.finished.emit(False, str(e))
        finally:
            loop.close()

    async def _pull(self) -> bool:
        import aiohttp
        import json

        try:
            # First check if Ollama is running
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        f"{self.endpoint}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status != 200:
                            self.finished.emit(
                                False,
                                "Ollama is not running. Install from https://ollama.com"
                            )
                            return False
                except Exception:
                    self.finished.emit(
                        False,
                        "Cannot connect to Ollama. Install from https://ollama.com"
                    )
                    return False

                # Start pull
                self.progress.emit(f"Downloading {self.model_name}â¦", 0.0)

                async with session.post(
                    f"{self.endpoint}/api/pull",
                    json={"name": self.model_name},
                    timeout=aiohttp.ClientTimeout(total=3600),
                ) as resp:
                    async for line in resp.content:
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            percent = (completed / total * 100) if total > 0 else 0

                            if "pulling" in status.lower():
                                gb = total / 1e9
                                msg = f"Downloading {self.model_name} ({gb:.1f} GB)â¦"
                                self.progress.emit(msg, percent)
                            elif status:
                                self.progress.emit(status, percent)

                        except json.JSONDecodeError:
                            continue

            self.finished.emit(True, f"â {self.model_name} ready!")
            return True

        except Exception as e:
            self.finished.emit(False, f"Pull failed: {e}")
            return False


# ââ Setup Wizard ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class SetupWizard(QDialog):
    """First-run setup wizard."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.api_inputs = {}
        self.completed = False
        self._pull_worker = None

        self.setWindowTitle("OpenAssist AI - Setup")
        # DPI-aware sizing: cap at 660x720 but fit smaller screens
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(660, int(screen.width() * 0.65))
        h = min(720, int(screen.height() * 0.85))
        self.setFixedSize(w, h)
        self.setModal(True)
        self.setStyleSheet("background: #111122;")
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.stack.addWidget(self._page_welcome())
        self.stack.addWidget(self._page_api_keys())
        self.stack.addWidget(self._page_preferences())
        self.stack.addWidget(self._page_done())

    # ââ Page 1: Welcome ââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _page_welcome(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #111122;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 30)
        layout.setSpacing(20)
        layout.addStretch()

        title = QLabel("ð§  OpenAssist AI")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title.setStyleSheet("color: #c0c0ff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Your Free AI Assistant â Setup Wizard")
        subtitle.setFont(QFont("Segoe UI", 13))
        subtitle.setStyleSheet("color: #8888bb;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        features = QLabel(
            "â 12 FREE AI providers (Groq, Gemini, Cerebrasâ¦)\n"
            "â¡ 2,100 tok/s peak inference speed\n"
            "ð¯ Interview Â· Coding Â· Meeting Â· Exam modes\n"
            "ð» Stealth mode (invisible to screen share)\n"
            "ð Knowledge base from your own documents\n"
            "ð¦ Offline mode with local AI (Ollama)"
        )
        features.setFont(QFont("Segoe UI", 12))
        features.setStyleSheet("color: #aaaacc; line-height: 1.8;")
        features.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(features)

        layout.addStretch()

        next_btn = self._nav_button("Get Started â", primary=True)
        next_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        layout.addWidget(next_btn)

        return page

    # ââ Page 2: API Keys + Ollama ââââââââââââââââââââââââââââââââââââââââââ

    def _page_api_keys(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #111122;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(10)

        header = QLabel("ð Add Your Free API Keys")
        header.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        header.setStyleSheet("color: #c0c0ff;")
        layout.addWidget(header)

        hint = QLabel(
            "Click 'Get Key' â sign up free â paste key below. "
            "You only need ONE key to start!"
        )
        hint.setStyleSheet("color: #666688; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # ââ Scrollable provider list âââââââââââââââââââââââââââââââââââââ
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: transparent; width: 5px; }
            QScrollBar::handle:vertical {
                background: rgba(80,80,140,60); border-radius: 2px;
            }
        """)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        cl = QVBoxLayout(container)
        cl.setSpacing(6)

        tiers = [
            ("â¡ Tier 1 â Ultra Fast & Free",     ["groq", "cerebras", "sambanova"]),
            ("ð Tier 2 â High Quality & Free",    ["gemini", "together", "openrouter",
                                                    "mistral", "cohere", "hyperbolic"]),
            ("ð¦ Tier 3 â Local (No key needed)", ["ollama"]),
            ("ð³ Tier 4 â Paid (Optional)",       ["openai", "anthropic"]),
        ]

        for tier_label, pids in tiers:
            lbl = QLabel(tier_label)
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            lbl.setStyleSheet("color: #6666aa; padding: 8px 0 3px 0;")
            cl.addWidget(lbl)
            for pid in pids:
                if pid in PROVIDERS:
                    cl.addWidget(self._provider_row(pid, PROVIDERS[pid]))

        cl.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        # ââ Ollama Local AI section ââââââââââââââââââââââââââââââââââââââ
        ollama_frame = self._ollama_section()
        layout.addWidget(ollama_frame)

        # Navigation
        nav = QHBoxLayout()
        back = self._nav_button("â Back")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        nav.addWidget(back)
        nav.addStretch()

        next_btn = self._nav_button("Next â", primary=True)
        next_btn.clicked.connect(self._save_keys_and_next)
        nav.addWidget(next_btn)
        layout.addLayout(nav)

        return page

    def _ollama_section(self) -> QFrame:
        """
        Ollama local AI section with model download button and progress bar.
        This is the P3 improvement: gives users a UI path to download models.
        """
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: rgba(20, 35, 20, 200);
                border: 1px solid rgba(50, 100, 50, 50);
                border-radius: 8px;
            }
        """)
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(14, 10, 14, 10)
        fl.setSpacing(6)

        # Header
        h_row = QHBoxLayout()
        title = QLabel("ð¦ Local AI (Optional â works offline)")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: #88cc88; border: none;")
        h_row.addWidget(title)
        h_row.addStretch()

        install_btn = QPushButton("Install Ollama â")
        install_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #66aa66;
                border: 1px solid rgba(80,140,80,50); border-radius: 4px;
                padding: 3px 10px; font-size: 10px;
            }
            QPushButton:hover { background: rgba(40,80,40,80); }
        """)
        install_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://ollama.com"))
        )
        h_row.addWidget(install_btn)
        fl.addLayout(h_row)

        desc = QLabel(
            "Run AI locally â no internet, no API key, full privacy.\n"
            "Requires Ollama installed + ~3-5 GB disk space."
        )
        desc.setStyleSheet("color: #447744; font-size: 11px; border: none;")
        fl.addWidget(desc)

        # Model selector + download button row
        model_row = QHBoxLayout()
        model_row.setSpacing(6)

        model_label = QLabel("Model:")
        model_label.setStyleSheet("color: #6688aa; font-size: 11px; border: none;")
        model_label.setFixedWidth(45)
        model_row.addWidget(model_label)

        from PyQt6.QtWidgets import QComboBox
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.addItems([
            "llama3.2:3b   (2 GB â fastest)",
            "llama3.1:8b   (5 GB â balanced)",
            "qwen2.5-coder:7b  (5 GB â best for coding)",
            "deepseek-r1:8b    (5 GB â best reasoning)",
        ])
        self.ollama_model_combo.setStyleSheet("""
            QComboBox {
                background: rgba(20,40,20,200); color: #88cc88;
                border: 1px solid rgba(60,100,60,50); border-radius: 5px;
                padding: 4px 8px; font-size: 11px;
            }
            QComboBox QAbstractItemView {
                background: rgba(15,30,15,250); color: #88cc88;
            }
        """)
        model_row.addWidget(self.ollama_model_combo, 1)

        self.download_btn = QPushButton("â¬ Download")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background: rgba(40,100,40,180); color: #88cc88;
                border: none; border-radius: 5px;
                padding: 6px 14px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background: rgba(55,120,55,200); }
            QPushButton:disabled {
                background: rgba(30,60,30,100); color: #446644;
            }
        """)
        self.download_btn.clicked.connect(self._start_ollama_pull)
        model_row.addWidget(self.download_btn)

        fl.addLayout(model_row)

        # Progress bar (hidden until download starts)
        self.ollama_progress = QProgressBar()
        self.ollama_progress.setRange(0, 100)
        self.ollama_progress.setValue(0)
        self.ollama_progress.setVisible(False)
        self.ollama_progress.setStyleSheet("""
            QProgressBar {
                background: rgba(10,30,10,200); color: #88cc88;
                border: 1px solid rgba(60,100,60,50); border-radius: 4px;
                text-align: center; font-size: 10px; height: 18px;
            }
            QProgressBar::chunk {
                background: rgba(50,140,50,180); border-radius: 3px;
            }
        """)
        fl.addWidget(self.ollama_progress)

        # Status message
        self.ollama_status = QLabel("")
        self.ollama_status.setStyleSheet(
            "color: #668866; font-size: 10px; border: none;"
        )
        self.ollama_status.setVisible(False)
        fl.addWidget(self.ollama_status)

        return frame

    def _start_ollama_pull(self):
        """Start downloading the selected Ollama model."""
        model_raw = self.ollama_model_combo.currentText()
        model_name = model_raw.split()[0]  # Extract "llama3.2:3b" from display text

        endpoint = self.config.get_api_key("ollama") or "http://localhost:11434"
        if not endpoint.startswith("http"):
            endpoint = "http://localhost:11434"

        # Disable button during download
        self.download_btn.setEnabled(False)
        self.download_btn.setText("Downloadingâ¦")
        self.ollama_progress.setVisible(True)
        self.ollama_progress.setValue(0)
        self.ollama_status.setVisible(True)
        self.ollama_status.setText(f"Starting download of {model_name}â¦")

        # Create and start worker
        self._pull_worker = OllamaPullWorker(model_name, endpoint)
        self._pull_worker.progress.connect(self._on_pull_progress)
        self._pull_worker.finished.connect(self._on_pull_finished)
        self._pull_worker.start()

    def _on_pull_progress(self, message: str, percent: float):
        """Update progress bar and status during pull."""
        self.ollama_progress.setValue(int(percent))
        self.ollama_progress.setFormat(f"{percent:.0f}%")
        self.ollama_status.setText(message)

    def _on_pull_finished(self, success: bool, message: str):
        """Handle pull completion."""
        self.download_btn.setEnabled(True)

        if success:
            self.download_btn.setText("â Downloaded")
            self.download_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(30,80,30,150); color: #66aa66;
                    border: 1px solid rgba(50,100,50,60); border-radius: 5px;
                    padding: 6px 14px; font-size: 11px;
                }
            """)
            self.ollama_progress.setValue(100)
            self.ollama_status.setText(message)
        else:
            self.download_btn.setText("â¬ Retry")
            self.ollama_status.setText(f"â {message}")
            self.ollama_status.setStyleSheet(
                "color: #cc6666; font-size: 10px; border: none;"
            )

    def _provider_row(self, pid: str, meta: dict) -> QFrame:
        """Single provider input row."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: rgba(20,20,40,200);
                border: 1px solid rgba(50,50,90,35);
                border-radius: 7px;
            }
        """)
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(12, 7, 12, 7)
        fl.setSpacing(4)

        # Name + info + get key button
        top = QHBoxLayout()
        name = QLabel(f"{meta['icon']}  {meta['name']}")
        name.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        name.setStyleSheet("color: #c0c0dd; border: none;")
        top.addWidget(name)

        info = QLabel(f"{meta['free']}  â¢  {meta['speed']}")
        info.setStyleSheet("color: #445566; font-size: 10px; border: none;")
        top.addWidget(info)
        top.addStretch()

        get_btn = QPushButton("Get Key â")
        get_btn.setFixedHeight(22)
        get_btn.setStyleSheet("""
            QPushButton {
                background: rgba(40,40,80,150); color: #8888bb;
                border: 1px solid rgba(60,60,100,40); border-radius: 4px;
                padding: 2px 8px; font-size: 10px;
            }
            QPushButton:hover { background: rgba(55,55,100,180); }
        """)
        get_btn.clicked.connect(
            lambda _, u=meta["url"]: QDesktopServices.openUrl(QUrl(u))
        )
        top.addWidget(get_btn)
        fl.addLayout(top)

        # Key input
        row = QHBoxLayout()
        inp = QLineEdit()
        inp.setStyleSheet("""
            QLineEdit {
                background: rgba(10,10,25,200); color: #b0b0dd;
                border: 1px solid rgba(60,60,100,50); border-radius: 5px;
                padding: 5px 10px; font-family: monospace; font-size: 11px;
            }
            QLineEdit:focus { border-color: rgba(100,100,200,120); }
        """)
        inp.setFixedHeight(28)
        inp.setPlaceholderText(f"Paste {meta['name']} API keyâ¦")
        inp.setText(self.config.get_api_key(pid) or "")
        inp.setEchoMode(QLineEdit.EchoMode.Password)
        row.addWidget(inp, 1)

        vis = QPushButton("ð")
        vis.setFixedSize(26, 28)
        vis.setStyleSheet("background: transparent; border: none; color: #555577;")
        vis.clicked.connect(lambda _, i=inp: i.setEchoMode(
            QLineEdit.EchoMode.Normal
            if i.echoMode() == QLineEdit.EchoMode.Password
            else QLineEdit.EchoMode.Password
        ))
        row.addWidget(vis)
        fl.addLayout(row)

        self.api_inputs[pid] = inp
        return frame

    # ââ Page 3: Preferences ââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _page_preferences(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #111122;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 24, 40, 20)
        layout.setSpacing(14)

        header = QLabel("âï¸ Preferences")
        header.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        header.setStyleSheet("color: #c0c0ff;")
        layout.addWidget(header)

        check_style = """
            QCheckBox {
                color: #aaaacc; font-size: 12px; spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid rgba(80,80,130,80); border-radius: 3px;
                background: rgba(20,20,40,180);
            }
            QCheckBox::indicator:checked {
                background: rgba(70,70,160,200);
            }
        """

        self.chk_audio = QCheckBox("ð¤ Enable audio capture (live transcription)")
        self.chk_audio.setChecked(True)
        self.chk_audio.setStyleSheet(check_style)
        layout.addWidget(self.chk_audio)

        self.chk_screen = QCheckBox("ðº Enable screen capture (OCR)")
        self.chk_screen.setChecked(True)
        self.chk_screen.setStyleSheet(check_style)
        layout.addWidget(self.chk_screen)

        self.chk_stealth = QCheckBox("ð» Enable stealth mode (invisible to screen recording)")
        self.chk_stealth.setChecked(False)
        self.chk_stealth.setStyleSheet(check_style)
        layout.addWidget(self.chk_stealth)

        self.chk_rag = QCheckBox("ð Enable knowledge base (drop files in knowledge/documents/)")
        self.chk_rag.setChecked(True)
        self.chk_rag.setStyleSheet(check_style)
        layout.addWidget(self.chk_rag)

        self.chk_tray = QCheckBox("Start minimized to system tray")
        self.chk_tray.setChecked(False)
        self.chk_tray.setStyleSheet(check_style)
        layout.addWidget(self.chk_tray)

        layout.addStretch()

        nav = QHBoxLayout()
        back = self._nav_button("â Back")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        nav.addWidget(back)
        nav.addStretch()

        finish = self._nav_button("Finish Setup â", primary=True)
        finish.setStyleSheet(finish.styleSheet().replace("#4444aa", "#338833").replace("#5555bb", "#449944"))
        finish.clicked.connect(self._finish)
        nav.addWidget(finish)
        layout.addLayout(nav)

        return page

    # ââ Page 4: Done âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _page_done(self) -> QWidget:
        page = QWidget()
        page.setStyleSheet("background: #111122;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 30)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        QLabel("â").setParent(None)
        done_icon = QLabel("â")
        done_icon.setFont(QFont("Segoe UI", 40))
        done_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(done_icon)

        done_title = QLabel("You're All Set!")
        done_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        done_title.setStyleSheet("color: #88cc88;")
        done_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(done_title)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #888899; font-size: 12px;")
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addSpacing(16)

        hotkeys = QLabel(
            "â¨ï¸ Hotkeys:\n"
            "Ctrl+\\           â Toggle / pass-through / hide\n"
            "Ctrl+Enter       â Analyze current screen\n"
            "Ctrl+Shift+M     â Switch mode\n"
            "Ctrl+Arrow       â Glide overlay\n"
            "Ctrl+Shift+Z     â Stealth mode"
        )
        hotkeys.setStyleSheet("color: #555577; font-family: monospace; font-size: 11px;")
        hotkeys.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hotkeys)

        layout.addStretch()

        launch = self._nav_button("ð Launch OpenAssist AI", primary=True)
        launch.clicked.connect(self._launch)
        layout.addWidget(launch)

        return page

    # ââ Wizard Logic âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _save_keys_and_next(self):
        count = 0
        for pid, inp in self.api_inputs.items():
            key = inp.text().strip().strip('"').strip("'")
            if key:
                self.config.set_api_key(pid, key)
                count += 1

        if count == 0:
            reply = QMessageBox.question(
                self,
                "No API Keys",
                "No API keys entered.\n\n"
                "You can still use Ollama (local AI) if installed.\n"
                "Continue without cloud AI?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.stack.setCurrentIndex(2)

    def _finish(self):
        self.config.set("capture.audio.enabled", self.chk_audio.isChecked())
        self.config.set("capture.screen.enabled", self.chk_screen.isChecked())
        self.config.set("stealth.enabled", self.chk_stealth.isChecked())
        self.config.set("rag.enabled", self.chk_rag.isChecked())
        self.config.set("app.start_minimized", self.chk_tray.isChecked())
        self.config.secrets.set("setup_complete", True)

        keys = self.config.secrets.get_all_keys()
        names = [PROVIDERS[p]["name"] for p in keys if p in PROVIDERS]
        if names:
            self.summary_label.setText(
                f"{len(names)} provider(s) configured:\n" + ", ".join(names)
            )
        else:
            self.summary_label.setText(
                "No cloud providers. Using Ollama (local) only."
            )

        self.completed = True
        self.stack.setCurrentIndex(3)

    def _launch(self):
        self.accept()

    # ââ Helpers ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    @staticmethod
    def _nav_button(text: str, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        if primary:
            btn.setStyleSheet("""
                QPushButton {
                    background: #4444aa; color: white; border: none;
                    border-radius: 8px; padding: 10px 30px;
                    font-weight: bold; font-size: 12px;
                }
                QPushButton:hover { background: #5555bb; }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    color: #8888bb; background: transparent;
                    border: 1px solid #444466; border-radius: 8px;
                    padding: 8px 20px; font-size: 12px;
                }
                QPushButton:hover { background: rgba(50,50,80,80); }
            """)
        return btn
