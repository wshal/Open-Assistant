"""Mode manager — loads all built-in modes and exposes them as ModeProfiles."""

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
        self._modes: dict[str, Mode] = {}
        for cls in _ALL_MODES:
            try:
                m = cls()
                self._modes[m.name] = m
            except Exception as e:
                logger.warning(f"Could not load mode {cls.__name__}: {e}")

        default = config.get("ai.mode", "general")
        self._current_name = default if default in self._modes else next(iter(self._modes), "general")
        logger.info(f"Modes loaded: {list(self._modes)} | active: {self._current_name}")

    # ── Core accessors ───────────────────────────────────────────────────────

    @property
    def current(self) -> Mode:
        """The active Mode object — use this everywhere instead of config.get('ai.mode')."""
        if not self._modes:
            return Mode(name="general", display="General Assistant", icon="🤖")
        return self._modes.get(self._current_name, next(iter(self._modes.values())))

    @property
    def profile(self) -> Mode:
        """Alias for current — reads more clearly at call sites."""
        return self.current

    def get_profile(self, name: str) -> Mode:
        """Look up any mode's profile by name; falls back to current."""
        return self._modes.get(name, self.current)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def switch(self, name: str) -> Mode:
        if name in self._modes:
            self._current_name = name
            profile = self._modes[name]
            logger.info(
                f"Mode switched → {name.upper()} | "
                f"providers={profile.preferred_providers[:3]} | "
                f"sensitivity={profile.detector_sensitivity} | "
                f"ollama_hint={profile.ollama_model_hint}"
            )
            return profile
        else:
            logger.warning(f"Unknown mode: {name}")
            return self.current

    def cycle(self):
        keys = list(self._modes)
        idx = keys.index(self._current_name) if self._current_name in keys else 0
        self._current_name = keys[(idx + 1) % len(keys)]
        return self.current

    # ── Auto-detection ────────────────────────────────────────────────────────

    def auto_detect(self, screen_text="", audio_text="", window_category="") -> str | None:
        """Heuristic mode detection from live context. Returns mode name or None."""
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

    # ── Bulk accessors ────────────────────────────────────────────────────────

    @property
    def all_modes(self) -> list[Mode]:
        return list(self._modes.values())

    @property
    def current_name(self) -> str:
        return self._current_name
