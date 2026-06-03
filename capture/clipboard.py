"""Clipboard monitoring."""

import time
import threading
import pyperclip
from PyQt6.QtCore import QObject, pyqtSignal
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ClipboardMonitor(QObject):
    clipboard_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._running = False
        self._last_content = ""
        self._thread = None
    
    def start(self):
        self._running = True
        try:
            self._last_content = pyperclip.paste() or ""
        except Exception as e:
            logger.warning(f"Clipboard read at startup failed: {e}")
            self._last_content = ""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
    
    def _monitor_loop(self):
        while self._running:
            try:
                current = pyperclip.paste() or ""
                if len(current) > 1000000:
                    current = current[:1000000]
                if current != self._last_content and current:
                    self._last_content = current
                    self.clipboard_changed.emit(current)
            except Exception as e:
                logger.warning(f"Clipboard monitor read failed: {e}")
                time.sleep(2)
            time.sleep(1)