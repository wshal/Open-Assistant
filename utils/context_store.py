"""
Session Context Store — OpenAssist AI.

Persists named preset instructions and the last-used custom context
across app restarts. Stored in data/context_presets.json (plain JSON,
not encrypted — these are user-authored instructions, not secrets).
"""

import json
import threading
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)

_STORE_PATH = Path("data/context_presets.json")

# Built-in general presets shipped with the app.
# Users can overwrite or add their own on top of these.
DEFAULT_PRESETS = {
    "Job Interview": (
        "You are a real-time interview assistant. Your job is to help me answer "
        "interview questions clearly, confidently, and concisely. Give direct answers "
        "first, then brief supporting detail. Use bullet points for multi-part answers. "
        "Keep responses short enough to speak aloud naturally in 30–60 seconds."
    ),
    "Presentation": (
        "You are a presentation coach and slide assistant. Help me explain concepts "
        "clearly to an audience. Use plain language, strong analogies, and memorable "
        "structure. Keep responses concise and speaker-friendly — avoid jargon unless "
        "I ask for technical depth."
    ),
    "Negotiation": (
        "You are a negotiation strategist. When I describe a situation, help me "
        "understand the leverage points, identify what the other party wants, and "
        "suggest concise, confident responses. Be tactical, not verbose. "
        "Frame advice as specific phrases I can use."
    ),
    "Practice": (
        "You are a practice partner. I'm drilling concepts, problems, or skills. "
        "Give me short, precise feedback. If I'm wrong, explain why briefly and give "
        "the correct answer. If I'm right, confirm it and optionally add one useful tip. "
        "Keep energy high and responses fast."
    ),
    "Exam": (
        "You are an exam assistant with screen access. When you see a question, "
        "give the direct answer first, then a one-line explanation. For MCQ, state "
        "the correct option immediately. Be accurate and concise — no filler."
    ),
    "Code Review": (
        "You are a senior software engineer reviewing code. Identify bugs, security "
        "issues, and performance problems first. Provide fixes in fenced code blocks. "
        "Use functional patterns where appropriate. Explain the 'why' in one line. "
        "No unnecessary commentary."
    ),
    "Meeting Copilot": (
        "You are a real-time meeting assistant. Track key points, action items, "
        "decisions, and suggested responses as the conversation unfolds. "
        "Use bullet points only. Be ultra-concise and real-time appropriate."
    ),
}

# Maps each built-in AI mode to its best-matching default context preset.
# Used by auto-suggest: switching mode auto-loads the paired context.
# None means no auto-suggestion for that mode.
MODE_CONTEXT_MAP: dict[str, str | None] = {
    "general":   None,
    "interview": "Job Interview",
    "coding":    "Code Review",
    "meeting":   "Meeting Copilot",
    "exam":      "Exam",
    "writing":   "Presentation",
}


def get_suggested_preset_for_mode(mode: str) -> tuple[str | None, str]:
    """Return (preset_name, preset_text) for the given mode, or (None, '') if none."""
    name = MODE_CONTEXT_MAP.get((mode or "").lower())
    if not name:
        return None, ""
    text = DEFAULT_PRESETS.get(name, "")
    return name, text


class ContextStore:
    """Thread-safe store for session context presets and the last-used context."""

    def __init__(self, path: Path = _STORE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict = {"presets": {}, "last_context": ""}
        self._load()

    # ── Public API ───────────────────────────────────────────────────────────

    def get_presets(self) -> dict:
        """Return all presets (built-ins merged with user-saved ones)."""
        with self._lock:
            merged = dict(DEFAULT_PRESETS)
            merged.update(self._data.get("presets", {}))
            return merged

    def get_preset_names(self) -> list:
        return list(self.get_presets().keys())

    def get_preset(self, name: str) -> str:
        return self.get_presets().get(name, "")

    def save_preset(self, name: str, text: str):
        """Save a user-defined preset (overwrites existing if same name)."""
        name = name.strip()
        if not name or not text.strip():
            return
        with self._lock:
            self._data.setdefault("presets", {})[name] = text.strip()
            self._persist()
        logger.info(f"Context preset saved: '{name}'")

    def delete_preset(self, name: str):
        """Delete a user-defined preset (built-ins cannot be deleted)."""
        with self._lock:
            self._data.get("presets", {}).pop(name, None)
            self._persist()

    def get_last_context(self) -> str:
        with self._lock:
            return self._data.get("last_context", "")

    def set_last_context(self, text: str):
        """Persist the last-used custom context so it reloads on next launch."""
        with self._lock:
            self._data["last_context"] = text.strip()
            self._persist()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                raw = self._path.read_text(encoding="utf-8")
                self._data = json.loads(raw)
        except Exception as e:
            logger.warning(f"ContextStore load failed (using defaults): {e}")
            self._data = {"presets": {}, "last_context": ""}

    def _persist(self):
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"ContextStore save failed: {e}")


# Module-level singleton — shared across all callers
_store: ContextStore | None = None


def get_store() -> ContextStore:
    global _store
    if _store is None:
        _store = ContextStore()
    return _store
