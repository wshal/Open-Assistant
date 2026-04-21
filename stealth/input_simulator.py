"""
Stealth Input Simulator — v5.1 (Direct-to-IDE Injection).
Uses Windows WM_CHAR messages for background-safe, stealthy typing.
"""

import time
import ctypes
import threading
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Windows API Constants
WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_RETURN = 0x0D

class InputSimulator:
    def __init__(self, config):
        self.config = config
        self.typing_speed = config.get("stealth.typing_delay_ms", 20) / 1000.0
        self._is_typing = False
        self._stop_requested = False

    def type_text(self, text: str, target_hwnd: int):
        """Asynchronously types text into a specific window HWND."""
        if not target_hwnd or not text:
            logger.warning("Sim: No target window or text provided.")
            return

        # Strip markdown syntax (```code```) if it's a code block
        clean_text = self._clean_markdown(text)
        
        thread = threading.Thread(
            target=self._do_type, 
            args=(clean_text, target_hwnd), 
            daemon=True
        )
        thread.start()

    def _do_type(self, text: str, hwnd: int):
        self._is_typing = True
        self._stop_requested = False
        logger.info(f"Sim: Starting stealth type to HWND {hwnd}")

        # CRLF Guard: Final normalization before injection
        # This prevents double-newlines if the AI already sent \r\n
        text = text.replace("\r\n", "\n")

        try:
            for char in text:
                if self._stop_requested:
                    break
                
                # Universal Line-Endings: Normalize \n for Windows
                if char == "\n":
                    # Send \r followed by \n — standard Windows ENTER
                    ctypes.windll.user32.PostMessageW(hwnd, WM_CHAR, 0x0D, 0) # \r
                    ctypes.windll.user32.PostMessageW(hwnd, WM_CHAR, 0x0A, 0) # \n
                else:
                    # Send WM_CHAR for unicode support
                    ctypes.windll.user32.PostMessageW(hwnd, WM_CHAR, ord(char), 0)
                
                # Variable delay for human-like rhythm
                time.sleep(self.typing_speed * (0.8 + 0.4 * (time.time() % 1)))
                
        except Exception as e:
            logger.error(f"Sim: Typing failure: {e}")
        finally:
            self._is_typing = False
            logger.info("Sim: Typing complete.")

    def stop(self):
        self._stop_requested = True

    def _clean_markdown(self, text: str) -> str:
        """Removes markdown code fences for cleaner IDE injection."""
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    @staticmethod
    def get_foreground_window() -> int:
        """Helper to capture the current active window handle."""
        return ctypes.windll.user32.GetForegroundWindow()