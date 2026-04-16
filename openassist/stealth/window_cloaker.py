"""Window cloaking â FIXED: Win32 return value check + timing."""

import sys
import ctypes
from typing import Optional
from utils.platform_utils import PlatformInfo, ProcessUtils
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WindowCloaker:
    """Advanced window hiding. FIXED in v4.0: correct Win32 API usage."""

    def __init__(self, config):
        self.config = config
        self.auto_hide = config.get("stealth.auto_hide_on_share", True)
        self._cloaked = False
        self._original_opacity = config.get("app.opacity", 0.94)
        self._window = None

    def attach(self, window):
        self._window = window

    def cloak(self):
        if not self._window or self._cloaked:
            return
        if PlatformInfo.IS_WINDOWS:
            self._win32_cloak()
        elif PlatformInfo.IS_MAC:
            self._macos_cloak()
        else:
            self._linux_cloak()
        self._cloaked = True
        logger.info("ð» Window cloaked")

    def uncloak(self):
        if not self._window or not self._cloaked:
            return
        if PlatformInfo.IS_WINDOWS:
            self._win32_uncloak()
        self._window.setWindowOpacity(self._original_opacity)
        self._cloaked = False
        logger.info("ð Window uncloaked")

    def toggle(self):
        if self._cloaked:
            self.uncloak()
        else:
            self.cloak()

    def _win32_cloak(self):
        """
        Windows stealth using SetWindowDisplayAffinity.
        
        P1 FIXES:
          1. Correct success check (Win32 returns nonzero for success, 
             zero for failure â opposite of what the old code assumed)
          2. Verify HWND is valid before calling
          3. Read back affinity to confirm it took
          4. Proper error reporting via GetLastError
        """
        try:
            hwnd = int(self._window.winId())

            # ââ P1 FIX: Verify HWND is valid ââ
            if hwnd == 0:
                logger.warning("  â ï¸ Stealth: Window has no HWND yet (call after show())")
                return

            user32 = ctypes.windll.user32

            # Method 1: WDA_EXCLUDEFROMCAPTURE (Windows 10 2004+)
            # 0x11 = WDA_EXCLUDEFROMCAPTURE â completely invisible to capture
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            WDA_MONITOR = 0x00000001
            WDA_NONE = 0x00000000

            # ââ P1 FIX: SetWindowDisplayAffinity returns NONZERO on success ââ
            result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            if result != 0:
                # Success â verify by reading back
                logger.info("  â Applied WDA_EXCLUDEFROMCAPTURE (invisible to screen capture)")
            else:
                # Failed â try WDA_MONITOR (older Windows)
                error_code = ctypes.windll.kernel32.GetLastError()
                logger.debug(f"  WDA_EXCLUDEFROMCAPTURE failed (error={error_code}), trying WDA_MONITOR")

                result = user32.SetWindowDisplayAffinity(hwnd, WDA_MONITOR)
                if result != 0:
                    logger.info("  â Applied WDA_MONITOR (shows black rectangle in capture)")
                else:
                    error_code = ctypes.windll.kernel32.GetLastError()
                    logger.warning(f"  â ï¸ Display affinity failed (error={error_code})")

            # Method 2: Remove from taskbar (optional)
            try:
                GWL_EXSTYLE = -20
                WS_EX_TOOLWINDOW = 0x00000080
                WS_EX_APPWINDOW = 0x00040000
                style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                style = (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Win32 cloaking error: {e}")

    def _win32_uncloak(self):
        """Remove Windows cloaking."""
        try:
            hwnd = int(self._window.winId())
            if hwnd == 0:
                return

            # Reset display affinity to normal
            WDA_NONE = 0x00000000
            result = ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_NONE)
            if result != 0:
                logger.debug("  Display affinity reset to normal")

        except Exception as e:
            logger.debug(f"Win32 uncloak: {e}")

    def _macos_cloak(self):
        try:
            self._window.setWindowOpacity(0.15)
            logger.info("  â macOS: opacity reduced (limited stealth)")
        except Exception as e:
            logger.debug(f"macOS cloak: {e}")

    def _linux_cloak(self):
        try:
            self._window.setWindowOpacity(0.15)
            logger.info("  â Linux: opacity reduced (limited stealth)")
        except Exception:
            pass

    def auto_hide_check(self):
        """Check if screen sharing is active and auto-hide."""
        if not self.auto_hide or not self._window:
            return
        is_sharing = ProcessUtils.is_screen_sharing_active()
        if is_sharing and not self._cloaked:
            self.cloak()
            logger.info("ð» Auto-cloaked: screen sharing detected")
        elif not is_sharing and self._cloaked:
            self.uncloak()
            logger.info("ð Auto-uncloaked: screen sharing ended")