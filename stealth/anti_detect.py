"""Always-on anti-capture helpers for overlay windows."""

import ctypes
from ctypes import c_void_p
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor

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
        self._ghost_cursors = {}       # window → GhostCursorWidget child
        self._ghost_cursor_active = False
        self._blank_cursor_applied = False
        self._static_cursor = None     # StaticCursorOverlay top-level window
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

    # ── Ghost Cursor ─────────────────────────────────────────────────────

    def _get_ghost_cursor(self, window):
        """Get or create a GhostCursorWidget for *window*.

        The widget is a child of *window*, so it shares the same HWND and
        has no DWM compositor surface (no transparent box).
        """
        wid = id(window)
        cursor = self._ghost_cursors.get(wid)
        if cursor is not None:
            return cursor
        try:
            from stealth.ghost_cursor import GhostCursorWidget
            cursor = GhostCursorWidget(window)
            self._ghost_cursors[wid] = cursor
            # Clean up when window is destroyed — restore real cursor if active
            window.destroyed.connect(
                lambda _=None, w=wid: self._on_ghost_window_destroyed(w)
            )
            logger.debug("Ghost cursor child widget created for %s", type(window).__name__)
            return cursor
        except Exception as e:
            logger.warning("Failed to create ghost cursor: %s", e)
            return None

    def _on_ghost_window_destroyed(self, wid):
        """Called when a window with a ghost cursor child is destroyed.

        Removes the stale dict entry AND restores the real cursor if
        the ghost cursor was active — prevents the cursor from being
        stuck blank forever.
        """
        self._ghost_cursors.pop(wid, None)
        if self._ghost_cursor_active:
            self.deactivate_ghost_cursor()

    def activate_ghost_cursor(self, window):
        """Hide the real cursor and show the ghost cursor inside *window*.

        Called when the mouse enters an overlay window while stealth is active.
        """
        if not self.enabled:
            return
        if not self.config.get("stealth.ghost_cursor", True):
            return
        if self._ghost_cursor_active:
            return

        cursor = self._get_ghost_cursor(window)
        if cursor is None:
            return

        self._ghost_cursor_active = True

        # Show static cursor overlay at the entry point so viewers see it parked
        try:
            entry_pos = QCursor.pos()
            if self._static_cursor is None:
                from stealth.ghost_cursor import StaticCursorOverlay
                self._static_cursor = StaticCursorOverlay()
            # Update cursor image and DPI-specific hotspot
            self._static_cursor.update_cursor()
            hspot = self._static_cursor.hotspot()
            # Position overlay so cursor hotspot is exactly at entry point
            self._static_cursor.move(entry_pos - hspot)
            self._static_cursor.show()
            self._static_cursor.raise_()
        except Exception as e:
            logger.warning("Failed to show static cursor overlay: %s", e)

        # Hide the real cursor application-wide while over the protected window
        from PyQt6.QtWidgets import QApplication
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
        self._blank_cursor_applied = True
        cursor.activate()

    def deactivate_ghost_cursor(self):
        """Restore the real cursor and hide all ghost cursors.

        Called when the mouse leaves an overlay window or the window hides.
        """
        if not self._ghost_cursor_active:
            return
        self._ghost_cursor_active = False

        # Hide the static cursor overlay
        if self._static_cursor is not None:
            try:
                self._static_cursor.hide()
            except Exception as e:
                logger.debug("Failed to hide static cursor overlay: %s", e)

        # Iterate a snapshot — destroyed signal could mutate the dict
        for cursor in list(self._ghost_cursors.values()):
            try:
                if cursor.is_active:
                    cursor.deactivate()
            except RuntimeError:
                # C++ object already deleted (window destroyed concurrently)
                pass
        if self._blank_cursor_applied:
            from PyQt6.QtWidgets import QApplication
            QApplication.restoreOverrideCursor()
            self._blank_cursor_applied = False

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

    def apply_app_stealth(self):
        """Apply app-level stealth policies (e.g., hiding from Dock/Taskbar)."""
        if sys.platform == "darwin":
            # Native AppKit behavior (Dynamic loading to prevent IDE errors on Windows)
            try:
                import importlib

                objc = importlib.import_module("objc")
                appkit = importlib.import_module("AppKit")
                NSApp = appkit.NSApplication.sharedApplication()
                # NSApplicationActivationPolicyAccessory = 1
                if NSApp.activationPolicy() != 1:
                    NSApp.setActivationPolicy_(1)
                    logger.info(
                        "Stealth: macOS app hidden from Dock (Accessory policy applied)."
                    )
            except Exception as e:
                logger.debug(f"Stealth: Failed to apply macOS app-level stealth: {e}")

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
        """macOS: AppKit native stealth — hide from screen recording via NSWindowSharingNone."""
        try:
            # NOTE: WindowTransparentForInput was WRONG here — it disables mouse/keyboard
            # input but does nothing to prevent screen capture. Removed.

            # Native AppKit behavior (Dynamic loading to prevent IDE errors on Windows)
            try:
                import importlib
                objc = importlib.import_module('objc')
                appkit = importlib.import_module('AppKit')

                hwnd = int(window.winId())
                if hwnd != 0:
                    # winId on macOS PyQt returns a pointer to the NSView
                    ns_view = objc.objc_object(c_void_p=c_void_p(hwnd))
                    ns_window = ns_view.window()
                    if ns_window is not None:
                        # NSWindowSharingNone = 0  → hidden from all screen capture APIs
                        # NSWindowSharingReadOnly = 1 → visible to capture (default)
                        ns_window.setSharingType_(0 if enabled else 1)

                        can_join = appkit.NSWindowCollectionBehaviorCanJoinAllSpaces
                        move_to_active = appkit.NSWindowCollectionBehaviorMoveToActiveSpace
                        transient = appkit.NSWindowCollectionBehaviorTransient

                        behavior = can_join | move_to_active
                        if enabled:
                            behavior |= transient

                        ns_window.setCollectionBehavior_(behavior)
                        logger.info("Stealth: macOS AppKit NSWindow behaviors applied (NSWindowSharingNone).")
            except ImportError:
                logger.debug(
                    "Stealth: pyobjc not installed. Skipping native macOS window behaviors."
                )
            except Exception as e:
                logger.debug(
                    f"Stealth: Failed to apply native macOS window behaviors: {e}"
                )

            logger.info(
                f"Stealth: macOS stealth {'enabled' if enabled else 'disabled'}"
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
