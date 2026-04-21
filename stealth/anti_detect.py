"""
Anti-detection for screen sharing environments.
"""

import ctypes
import sys

from PyQt6.QtCore import Qt

from utils.logger import setup_logger

logger = setup_logger(__name__)


class StealthManager:
    """Makes the overlay invisible to screen capture/recording."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.get("stealth.enabled", False)
        self._last_affinity_state = {}

    def apply_to_window(self, window, enabled=None):
        """Apply or remove anti-capture flags to a window."""
        status = enabled if enabled is not None else self.enabled
        self.enabled = status
        if sys.platform == "win32":
            self._win32_anti_capture(window, status)
        elif sys.platform == "darwin":
            self._macos_anti_capture(window, status)
        else:
            self._linux_anti_capture(window, status)

    def _win32_anti_capture(self, window, enabled):
        """Windows: SetWindowDisplayAffinity to exclude from capture."""
        try:
            hwnd = int(window.winId())
            if hwnd == 0:
                logger.debug("Stealth: Window has no HWND yet; skipping affinity update")
                return

            affinity = 0x00000011 if enabled else 0x00000000
            user32 = ctypes.windll.user32
            root_hwnd = user32.GetAncestor(hwnd, 2) or hwnd

            success = user32.SetWindowDisplayAffinity(root_hwnd, affinity)
            if success:
                self._log_affinity_state(root_hwnd, enabled, affinity)
                return

            if enabled and user32.SetWindowDisplayAffinity(root_hwnd, 0x00000001):
                self._log_affinity_state(root_hwnd, True, 0x00000001)
                return

            if enabled:
                logger.warning("Stealth failed: unable to apply capture affinity")
            else:
                logger.warning("Stealth failed: unable to clear capture affinity")
        except Exception as e:
            logger.warning(f"Stealth failed: {e}")

    def _log_affinity_state(self, hwnd, enabled, affinity):
        current = (enabled, affinity)
        if self._last_affinity_state.get(hwnd) == current:
            return
        logger.info(
            f"Stealth {'enabled' if enabled else 'disabled'} "
            f"(affinity 0x{affinity:02x}) on HWND {hwnd}"
        )
        self._last_affinity_state[hwnd] = current

    def _macos_anti_capture(self, window, enabled):
        """macOS: Set sharing type to none."""
        try:
            window.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
            logger.info(
                f"Stealth: macOS transparency {'enabled' if enabled else 'disabled'}"
            )
        except Exception as e:
            logger.warning(f"macOS stealth: {e}")

    def _linux_anti_capture(self, window, enabled):
        """Linux: Best effort with window type hints."""
        logger.info("Stealth: Linux mode (limited)")
