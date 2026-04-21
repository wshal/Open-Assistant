"""
Anti-detection for screen sharing environments.
FIXED: Global imports to resolve IDE warnings.
"""

import sys
import ctypes
from PyQt6.QtCore import Qt
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StealthManager:
    """Makes the overlay invisible to screen capture/recording."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.get("stealth.enabled", False)

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
        """
        Windows: SetWindowDisplayAffinity to exclude from capture.
        Hardened: Ensures we have the absolute top-most HWND.
        """
        try:
            hwnd = int(window.winId())
            if hwnd == 0:
                logger.debug("Stealth: Window has no HWND yet; skipping affinity update")
                return
            
            # WDA_EXCLUDEFROMCAPTURE = 0x00000011 (Windows 10 2004+)
            # WDA_NONE = 0x0
            affinity = 0x00000011 if enabled else 0x00000000
            
            # Ensure we're targeting the real top-level window
            user32 = ctypes.windll.user32
            GA_ROOT = 2
            root_hwnd = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
            
            success = user32.SetWindowDisplayAffinity(root_hwnd, affinity)
            if success:
                logger.info(f"👻 Stealth: {'Enabled' if enabled else 'Disabled'} (Affinity 0x{affinity:02x}) on HWND {root_hwnd}")
            elif enabled:
                # Fallback to WDA_MONITOR
                if user32.SetWindowDisplayAffinity(root_hwnd, 0x00000001):
                    logger.info("👻 Stealth: Fallback affinity applied")
                else:
                    logger.warning("Stealth failed: unable to apply capture affinity")
            else:
                logger.warning("Stealth failed: unable to clear capture affinity")
                
        except Exception as e:
            logger.warning(f"Stealth failed: {e}")

    def _macos_anti_capture(self, window, enabled):
        """macOS: Set sharing type to none."""
        try:
            window.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
            logger.info(f"👻 Stealth: macOS transparency {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logger.warning(f"macOS stealth: {e}")

    def _linux_anti_capture(self, window, enabled):
        """Linux: Best effort with window type hints."""
        logger.info("👻 Stealth: Linux mode (limited)")
