"""Mode manager - loads all built-in modes."""

from modes.base import Mode
from modes.general import GeneralMode
from modes.interview import InterviewMode
from modes.coding import CodingMode
from modes.meeting import MeetingMode
from modes.writing import WritingMode
from modes.exam import ExamMode
from utils.logger import setup_logger

logger = setup_logger(__name__)

_ALL_MODES = [GeneralMode, InterviewMode, CodingMode, MeetingMode, WritingMode, ExamMode]


class ModeManager:
    def __init__(self, config):
        self.config = config
        self._modes = {}
        for cls in _ALL_MODES:
            try:
                m = cls()          # Modes take no constructor args
                self._modes[m.name] = m
            except Exception as e:
                logger.warning(f"Could not load mode {cls.__name__}: {e}")

        default = config.get("ai.mode", "general")
        self._current_name = default if default in self._modes else next(iter(self._modes), "general")
        logger.info(f"Modes loaded: {list(self._modes)} | active: {self._current_name}")

    @property
    def current(self) -> Mode:
        if not self._modes:
            # Emergency fallback: return a bare Mode so nothing crashes
            return Mode(name="general", display="General Assistant", icon="🤖")
        return self._modes.get(self._current_name, next(iter(self._modes.values())))


    def switch(self, name: str):
        if name in self._modes:
            self._current_name = name
            return self._modes[name]
        else:
            logger.warning(f"Unknown mode: {name}")
            return self.current

    def cycle(self):
        keys = list(self._modes)
        idx = keys.index(self._current_name) if self._current_name in keys else 0
        self._current_name = keys[(idx + 1) % len(keys)]

    def auto_detect(self, screen_text="", audio_text="", window_category=""):
        text = (screen_text + " " + audio_text + " " + window_category).lower()
        hints = {
            "interview": ["interview", "tell me about", "strengths", "career"],
            "coding":    ["def ", "class ", "import ", "error:", "traceback"],
            "exam":      ["multiple choice", "answer:", "exam", "quiz"],
            "meeting":   ["agenda", "meeting", "action items", "standup"],
            "writing":   ["paragraph", "essay", "article", "draft"],
        }
        for mode_name, keywords in hints.items():
            if any(kw in text for kw in keywords):
                return mode_name
        return None

    @property
    def all_modes(self):
        return list(self._modes.values())
