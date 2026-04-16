"""Context manager for screen, audio, clipboard."""

import time
from collections import deque
from typing import Dict


class ContextManager:
    def __init__(self, config):
        self.max_chars = config.get("performance.token_budget", 6000) * 4
        self.screen_text = ""
        self.audio_text = ""
        self.clipboard = ""
        self.conversation = []

    def update_screen(self, text: str):
        if text and text != self.screen_text:
            self.screen_text = text

    def update_audio(self, text: str):
        if text:
            self.audio_text += " " + text
            words = self.audio_text.split()
            if len(words) > 3000:
                self.audio_text = " ".join(words[-2000:])

    def update_clipboard(self, text: str):
        self.clipboard = text

    def get_context(self) -> Dict[str, str]:
        limit = self.max_chars // 3
        return {
            "screen": self.screen_text[:limit],
            "audio": self.audio_text[-limit:],
            "clipboard": self.clipboard[:500],
        }

    def clear(self):
        self.screen_text = ""
        self.audio_text = ""
        self.clipboard = ""
        self.conversation.clear()