"""Cross-platform utilities for Windows, macOS, Linux."""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple, Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PlatformInfo:
    """Detect and expose platform capabilities."""

    OS = platform.system()  # Windows, Darwin, Linux
    IS_WINDOWS = OS == "Windows"
    IS_MAC = OS == "Darwin"
    IS_LINUX = OS == "Linux"
    ARCH = platform.machine()  # x86_64, arm64, etc.
    IS_64BIT = sys.maxsize > 2**32
    IS_FROZEN = getattr(sys, 'frozen', False)  # Running as EXE

    @staticmethod
    def get_app_data_dir() -> Path:
        """Get platform-specific application data directory."""
        if PlatformInfo.IS_WINDOWS:
            base = Path(os.environ.get("APPDATA", os.path.expanduser("~")))
        elif PlatformInfo.IS_MAC:
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        app_dir = base / "OpenAssistAI"
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir

    @staticmethod
    def get_config_dir() -> Path:
        """Get platform-specific config directory."""
        if PlatformInfo.IS_WINDOWS:
            base = Path(os.environ.get("APPDATA", os.path.expanduser("~")))
        elif PlatformInfo.IS_MAC:
            base = Path.home() / "Library" / "Preferences"
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        cfg_dir = base / "OpenAssistAI"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir

    @staticmethod
    def get_cache_dir() -> Path:
        """Get platform-specific cache directory."""
        if PlatformInfo.IS_WINDOWS:
            base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
        elif PlatformInfo.IS_MAC:
            base = Path.home() / "Library" / "Caches"
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = base / "OpenAssistAI"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def get_resource_path(relative_path: str) -> Path:
        """Get absolute path to resource, works for dev and PyInstaller."""
        if PlatformInfo.IS_FROZEN:
            base = Path(sys._MEIPASS)
        else:
            base = Path(__file__).parent.parent
        return base / relative_path

    @staticmethod
    def ensure_directories():
        """Create all required directories."""
        dirs = [
            "data/vectordb",
            "data/cache",
            "knowledge/documents",
            "knowledge/templates",
            "logs",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


class ScreenInfo:
    """Screen and display utilities."""

    @staticmethod
    def get_screen_count() -> int:
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                return len(app.screens())
        except Exception:
            pass
        return 1

    @staticmethod
    def get_primary_screen_size() -> Tuple[int, int]:
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                screen = app.primaryScreen()
                if screen:
                    geo = screen.geometry()
                    return geo.width(), geo.height()
        except Exception:
            pass
        return 1920, 1080

    @staticmethod
    def get_scale_factor() -> float:
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                screen = app.primaryScreen()
                if screen:
                    return screen.devicePixelRatio()
        except Exception:
            pass
        return 1.0


class WindowUtils:
    """Native window helpers used for packaging/runtime polish."""

    @staticmethod
    def ensure_topmost(window) -> bool:
        """Re-assert topmost status for overlay windows.

        Qt's ``WindowStaysOnTopHint`` usually works, but Windows can still drop
        the z-order in edge cases after focus changes, monitor switches, or
        shell-level window churn. This helper refreshes the HWND to ``TOPMOST``
        without moving or resizing the window.
        """
        if not PlatformInfo.IS_WINDOWS or window is None:
            return False

        try:
            import ctypes

            hwnd = int(window.winId())
            if hwnd == 0:
                return False

            user32 = ctypes.windll.user32
            HWND_TOPMOST = -1
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            SWP_NOOWNERZORDER = 0x0200
            SWP_NOSENDCHANGING = 0x0400

            return bool(
                user32.SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE
                    | SWP_NOSIZE
                    | SWP_NOACTIVATE
                    | SWP_NOOWNERZORDER
                    | SWP_NOSENDCHANGING,
                )
            )
        except Exception as e:
            logger.debug(f"Ensure topmost skipped: {e}")
            return False

    @staticmethod
    def hide_from_taskbar(window) -> bool:
        """Hide a Qt window from the Windows taskbar.

        This mirrors Electron's ``setSkipTaskbar(true)`` behavior by turning the
        native HWND into a tool window and forcing Windows to refresh the frame.
        Safe no-op on non-Windows platforms or before the native handle exists.
        """
        if not PlatformInfo.IS_WINDOWS or window is None:
            return False

        if getattr(window, "_openassist_taskbar_hidden", False):
            return True

        try:
            import ctypes

            hwnd = int(window.winId())
            if hwnd == 0:
                return False

            user32 = ctypes.windll.user32
            GA_ROOT = 2
            GWL_EXSTYLE = -20
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_APPWINDOW = 0x00040000
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020

            root_hwnd = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
            style = user32.GetWindowLongW(root_hwnd, GWL_EXSTYLE)
            updated_style = (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
            if updated_style != style:
                user32.SetWindowLongW(root_hwnd, GWL_EXSTYLE, updated_style)
                user32.SetWindowPos(
                    root_hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE
                    | SWP_NOSIZE
                    | SWP_NOZORDER
                    | SWP_NOACTIVATE
                    | SWP_FRAMECHANGED,
                )

            window._openassist_taskbar_hidden = True
            return True
        except Exception as e:
            logger.debug(f"Hide from taskbar skipped: {e}")
            return False


class ProcessUtils:
    """Process and system utilities."""

    @staticmethod
    def get_active_window_title() -> str:
        """Get the title of the currently active/focused window."""
        try:
            if PlatformInfo.IS_WINDOWS:
                return ProcessUtils._win_active_title()
            elif PlatformInfo.IS_MAC:
                return ProcessUtils._mac_active_title()
            else:
                return ProcessUtils._linux_active_title()
        except Exception:
            return ""

    @staticmethod
    def _win_active_title() -> str:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    @staticmethod
    def _mac_active_title() -> str:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip()

    @staticmethod
    def _linux_active_title() -> str:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip()

    @staticmethod
    def get_active_window_rect() -> Optional[Tuple[int, int, int, int]]:
        """Get (x, y, width, height) of the active window."""
        try:
            if PlatformInfo.IS_WINDOWS:
                import ctypes
                from ctypes import wintypes

                class RECT(ctypes.Structure):
                    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

                hwnd = ctypes.windll.user32.GetForegroundWindow()
                rect = RECT()
                ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
            elif PlatformInfo.IS_LINUX:
                result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
                    capture_output=True, text=True, timeout=2
                )
                parts = {}
                for line in result.stdout.strip().split('\n'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        parts[k] = int(v)
                return (parts.get("X", 0), parts.get("Y", 0),
                        parts.get("WIDTH", 800), parts.get("HEIGHT", 600))
        except Exception:
            pass
        return None

    @staticmethod
    def is_screen_sharing_active() -> bool:
        """Detect if screen sharing is likely active."""
        try:
            if PlatformInfo.IS_WINDOWS:
                # Check for common screen sharing processes
                import psutil
                sharing_apps = {
                    "zoom.exe", "teams.exe", "slack.exe", "discord.exe",
                    "obs64.exe", "obs32.exe", "webex.exe", "gotomeeting.exe",
                    "screencastomatic.exe", "loom.exe"
                }
                for proc in psutil.process_iter(['name']):
                    if proc.info['name'] and proc.info['name'].lower() in sharing_apps:
                        return True
            elif PlatformInfo.IS_MAC:
                result = subprocess.run(
                    ["pgrep", "-f", "screencapturekit|zoom|teams|slack|obs"],
                    capture_output=True, timeout=2
                )
                return result.returncode == 0
        except Exception:
            pass
        return False

    @staticmethod
    def get_running_apps() -> list:
        """Get list of running application names."""
        try:
            import psutil
            return list(set(
                p.info['name'] for p in psutil.process_iter(['name'])
                if p.info['name']
            ))
        except Exception:
            return []

    @staticmethod
    def open_url(url: str):
        """Open URL in default browser."""
        import webbrowser
        webbrowser.open(url)

    @staticmethod
    def open_folder(path: str):
        """Open folder in file manager."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if PlatformInfo.IS_WINDOWS:
            os.startfile(str(p))
        elif PlatformInfo.IS_MAC:
            subprocess.run(["open", str(p)])
        else:
            subprocess.run(["xdg-open", str(p)])


class AudioDevices:
    """Audio device discovery."""

    @staticmethod
    def get_input_devices() -> list:
        """List available audio input devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            inputs = []
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    inputs.append({
                        "index": i,
                        "name": d['name'],
                        "channels": d['max_input_channels'],
                        "sample_rate": d['default_samplerate'],
                    })
            return inputs
        except Exception:
            return []

    @staticmethod
    def get_default_input() -> Optional[int]:
        """Get default input device index."""
        try:
            import sounddevice as sd
            return sd.default.device[0]
        except Exception:
            return None


class GPUInfo:
    """GPU detection for model optimization."""

    @staticmethod
    def has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def has_mps() -> bool:
        """Apple Metal Performance Shaders."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except Exception:
            return False

    @staticmethod
    def get_gpu_name() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "CPU"

    @staticmethod
    def get_optimal_device() -> str:
        if GPUInfo.has_cuda():
            return "cuda"
        elif GPUInfo.has_mps():
            return "mps"
        return "cpu"

    @staticmethod
    def get_optimal_compute_type() -> str:
        if GPUInfo.has_cuda():
            return "float16"
        return "int8"
