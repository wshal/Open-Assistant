"""UI-level stealth management â coordinates all stealth components."""

import threading
import time
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from stealth.window_cloaker import WindowCloaker
from stealth.input_simulator import InputSimulator
from utils.platform_utils import ProcessUtils
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StealthUIManager(QObject):
    """Manages stealth mode UI state and auto-detection."""

    stealth_activated = pyqtSignal()
    stealth_deactivated = pyqtSignal()
    screen_share_detected = pyqtSignal(bool)

    def __init__(self, config, window):
        super().__init__()
        self.config = config
        self.window = window
        self.enabled = config.get("stealth.enabled", False)
        self.auto_hide = config.get("stealth.auto_hide_on_share", True)

        self.cloaker = WindowCloaker(config)
        self.cloaker.attach(window)

        self.typer = InputSimulator(config)

        # Auto-detect timer
        self._monitor_timer = QTimer()
        self._monitor_timer.timeout.connect(self._check_screen_share)
        self._was_sharing = False

    def start_monitoring(self):
        """Start monitoring for screen sharing apps."""
        if self.auto_hide:
            self._monitor_timer.start(5000)  # Check every 5 seconds
            logger.info("ð» Stealth monitor started")

    def stop_monitoring(self):
        self._monitor_timer.stop()

    def activate(self):
        """Activate stealth mode."""
        self.enabled = True
        self.cloaker.cloak()
        self.window.setWindowOpacity(
            self.config.get("stealth.low_opacity", 0.12)
        )
        self.stealth_activated.emit()
        logger.info("ð» Stealth mode activated")

    def deactivate(self):
        """Deactivate stealth mode."""
        self.enabled = False
        self.cloaker.uncloak()
        self.window.setWindowOpacity(
            self.config.get("app.opacity", 0.94)
        )
        self.stealth_deactivated.emit()
        logger.info("ð Stealth mode deactivated")

    def toggle(self):
        if self.enabled:
            self.deactivate()
        else:
            self.activate()

    def type_response(self, text: str, callback=None):
        """Type a response using natural simulation."""
        if self.typer.enabled:
            self.typer.type_text(text, callback)
        else:
            import pyperclip
            pyperclip.copy(text)

    def _check_screen_share(self):
        """Periodically check for screen sharing."""
        is_sharing = ProcessUtils.is_screen_sharing_active()
        if is_sharing != self._was_sharing:
            self._was_sharing = is_sharing
            self.screen_share_detected.emit(is_sharing)

            if is_sharing and not self.enabled:
                logger.info("â ï¸ Screen sharing detected â auto-activating stealth")
                self.activate()
            elif not is_sharing and self.enabled:
                logger.info("â Screen sharing ended â auto-deactivating stealth")
                self.deactivate()

    @property
    def is_active(self) -> bool:
        return self.enabled