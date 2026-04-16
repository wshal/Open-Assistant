"""Natural typing simulation â paste responses as if typed."""

import time
import random
import threading
from typing import Optional
from utils.platform_utils import PlatformInfo
from utils.logger import setup_logger

logger = setup_logger(__name__)


class InputSimulator:
    """Simulate natural human typing for pasting AI responses."""

    def __init__(self, config):
        self.enabled = config.get("stealth.natural_typing", True)
        self.wpm = config.get("stealth.typing_speed_wpm", 80)
        self._typing = False
        self._cancel = False

        # Calculate delay per character
        chars_per_minute = self.wpm * 5  # Average word = 5 chars
        self.base_delay = 60.0 / chars_per_minute  # seconds per char

    def type_text(self, text: str, callback=None):
        """Type text with natural timing. Non-blocking."""
        if not self.enabled:
            self._instant_paste(text)
            return

        self._cancel = False
        thread = threading.Thread(
            target=self._type_loop, args=(text, callback), daemon=True
        )
        thread.start()

    def cancel(self):
        """Cancel ongoing typing simulation."""
        self._cancel = True

    def _type_loop(self, text: str, callback=None):
        """Simulate typing character by character."""
        self._typing = True

        try:
            if PlatformInfo.IS_WINDOWS:
                self._type_windows(text)
            elif PlatformInfo.IS_MAC:
                self._type_mac(text)
            else:
                self._type_linux(text)
        except Exception as e:
            logger.error(f"Typing simulation failed: {e}")
            self._instant_paste(text)
        finally:
            self._typing = False
            if callback:
                callback()

    def _type_windows(self, text: str):
        """Windows: use SendInput for natural typing."""
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32

            INPUT_KEYBOARD = 1
            KEYEVENTF_UNICODE = 0x0004
            KEYEVENTF_KEYUP = 0x0002

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk", wintypes.WORD),
                    ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
                ]

            class INPUT(ctypes.Structure):
                class _INPUT(ctypes.Union):
                    _fields_ = [("ki", KEYBDINPUT)]
                _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

            for char in text:
                if self._cancel:
                    break

                # Key down
                inp = INPUT()
                inp.type = INPUT_KEYBOARD
                inp._input.ki.wScan = ord(char)
                inp._input.ki.dwFlags = KEYEVENTF_UNICODE
                user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

                # Key up
                inp._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

                # Natural delay
                time.sleep(self._natural_delay(char))

        except Exception as e:
            logger.debug(f"Win32 typing fallback: {e}")
            self._type_pynput(text)

    def _type_mac(self, text: str):
        """macOS: use pynput."""
        self._type_pynput(text)

    def _type_linux(self, text: str):
        """Linux: use xdotool or pynput."""
        try:
            import subprocess
            for char in text:
                if self._cancel:
                    break
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "0", char],
                    timeout=1
                )
                time.sleep(self._natural_delay(char))
        except Exception:
            self._type_pynput(text)

    def _type_pynput(self, text: str):
        """Fallback: use pynput for cross-platform typing."""
        try:
            from pynput.keyboard import Controller
            kb = Controller()

            for char in text:
                if self._cancel:
                    break
                kb.type(char)
                time.sleep(self._natural_delay(char))
        except Exception as e:
            logger.error(f"pynput typing failed: {e}")
            self._instant_paste(text)

    def _natural_delay(self, char: str) -> float:
        """Generate natural-feeling typing delay."""
        delay = self.base_delay

        # Vary by character type
        if char in ' \t':
            delay *= random.uniform(0.5, 0.8)  # Faster for spaces
        elif char in '\n':
            delay *= random.uniform(1.5, 3.0)  # Pause at line breaks
        elif char in '.,;:!?':
            delay *= random.uniform(1.0, 2.0)  # Slight pause at punctuation
        elif char in '()[]{}':
            delay *= random.uniform(1.2, 1.8)
        else:
            delay *= random.uniform(0.7, 1.4)  # Normal variance

        # Occasional longer pause (thinking)
        if random.random() < 0.02:
            delay += random.uniform(0.3, 0.8)

        return max(delay, 0.01)

    @staticmethod
    def _instant_paste(text: str):
        """Fallback: instant clipboard paste."""
        try:
            import pyperclip
            pyperclip.copy(text)
            # Simulate Ctrl+V
            if PlatformInfo.IS_WINDOWS:
                import ctypes
                ctypes.windll.user32.keybd_event(0x11, 0, 0, 0)  # Ctrl down
                ctypes.windll.user32.keybd_event(0x56, 0, 0, 0)  # V down
                ctypes.windll.user32.keybd_event(0x56, 0, 2, 0)  # V up
                ctypes.windll.user32.keybd_event(0x11, 0, 2, 0)  # Ctrl up
            else:
                from pynput.keyboard import Controller, Key
                kb = Controller()
                with kb.pressed(Key.ctrl if not PlatformInfo.IS_MAC else Key.cmd):
                    kb.tap('v')
        except Exception:
            pass

    @property
    def is_typing(self) -> bool:
        return self._typing