"""Always-on anti-capture helpers for overlay windows."""

import ctypes
import sys

from PyQt6.QtCore import Qt

from utils.logger import setup_logger

logger = setup_logger(__name__)

GA_ROOT = 2
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class StealthManager:
    """Apply the strongest supported anti-capture behavior per platform."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.get("stealth.enabled", True)
        self._last_affinity_state = {}
        self._status = {
            "state": "limited",
            "platform": self.protection_level(),
            "message": "Stealth manager initialized",
            "last_error": 0,
            "last_affinity": None,
        }

    def apply_to_window(self, window, enabled=None):
        """Apply or refresh anti-capture flags on a window."""
        status = enabled if enabled is not None else self.enabled
        self.enabled = status
        if sys.platform == "win32":
            self._win32_anti_capture(window, status)
        elif sys.platform == "darwin":
            self._macos_anti_capture(window, status)
        else:
            self._linux_anti_capture(window, status)

    @staticmethod
    def protection_level() -> str:
        """Report the expected protection strength for the current platform."""
        if sys.platform == "win32":
            return "strong"
        if sys.platform == "darwin":
            return "limited"
        return "limited"

    def should_hide_for_screen_share(self) -> bool:
        """Hide the HUD only when the platform lacks strong anti-capture support."""
        return self.protection_level() != "strong"

    def get_status(self) -> dict:
        """Return the latest stealth protection status for diagnostics/UI."""
        return dict(self._status)

    def _set_status(self, state: str, message: str, last_error: int = 0, last_affinity=None):
        self._status = {
            "state": state,
            "platform": self.protection_level(),
            "message": message,
            "last_error": int(last_error or 0),
            "last_affinity": last_affinity,
        }

    @staticmethod
    def _resolve_root_hwnd(window):
        hwnd = int(window.winId())
        if hwnd == 0:
            return 0
        user32 = ctypes.windll.user32
        return user32.GetAncestor(hwnd, GA_ROOT) or hwnd

    @staticmethod
    def _last_error() -> int:
        try:
            return int(ctypes.windll.kernel32.GetLastError())
        except Exception:
            return 0

    def _set_affinity(self, hwnd: int, affinity: int) -> bool:
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity))

    def _win32_anti_capture(self, window, enabled):
        """Windows: prefer WDA_EXCLUDEFROMCAPTURE, fallback to WDA_MONITOR."""
        try:
            root_hwnd = self._resolve_root_hwnd(window)
            if root_hwnd == 0:
                logger.debug("Stealth: Window has no HWND yet; skipping affinity update")
                self._set_status(
                    "error",
                    "Window handle unavailable during stealth refresh",
                )
                return

            if not enabled:
                if self._set_affinity(root_hwnd, WDA_NONE):
                    self._log_affinity_state(root_hwnd, False, WDA_NONE)
                    self._set_status(
                        "unprotected",
                        "Stealth guard removed",
                        last_affinity=WDA_NONE,
                    )
                else:
                    err = self._last_error()
                    logger.warning(
                        "Stealth failed: unable to clear capture affinity (error=%s)",
                        err,
                    )
                    self._set_status(
                        "error",
                        "Failed to clear capture affinity",
                        last_error=err,
                    )
                return

            if self._set_affinity(root_hwnd, WDA_EXCLUDEFROMCAPTURE):
                self._log_affinity_state(
                    root_hwnd, True, WDA_EXCLUDEFROMCAPTURE
                )
                self._set_status(
                    "protected",
                    "Capture exclusion active",
                    last_affinity=WDA_EXCLUDEFROMCAPTURE,
                )
                return

            exclude_error = self._last_error()
            logger.debug(
                "Stealth: WDA_EXCLUDEFROMCAPTURE failed on HWND %s (error=%s), trying fallback",
                root_hwnd,
                exclude_error,
            )

            if self._set_affinity(root_hwnd, WDA_MONITOR):
                self._log_affinity_state(root_hwnd, True, WDA_MONITOR)
                self._set_status(
                    "fallback",
                    "Monitor-only capture fallback active",
                    last_error=exclude_error,
                    last_affinity=WDA_MONITOR,
                )
                return

            fallback_error = self._last_error()
            logger.warning(
                "Stealth failed: unable to apply capture affinity (exclude_error=%s fallback_error=%s)",
                exclude_error,
                fallback_error,
            )
            self._set_status(
                "error",
                "No capture affinity mode could be applied",
                last_error=fallback_error or exclude_error,
            )
        except Exception as e:
            logger.warning(f"Stealth failed: {e}")
            self._set_status("error", f"Stealth exception: {e}")

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
        """macOS: limited best-effort behavior only."""
        try:
            window.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
            logger.info(
                f"Stealth: macOS transparency {'enabled' if enabled else 'disabled'}"
            )
            self._set_status(
                "limited",
                "Best-effort macOS stealth applied",
            )
        except Exception as e:
            logger.warning(f"macOS stealth: {e}")
            self._set_status("error", f"macOS stealth failed: {e}")

    def _linux_anti_capture(self, window, enabled):
        """Linux: best-effort only; compositor behavior varies."""
        logger.info("Stealth: Linux mode (limited)")
        self._set_status(
            "limited",
            "Best-effort Linux stealth applied",
        )
