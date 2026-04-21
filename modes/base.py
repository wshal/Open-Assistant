"""Base mode class — full ModeProfile declarative contract."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Mode:
    """
    ModeProfile — single declarative object consumed by AIEngine, ContextRanker,
    PromptBuilder, router, and detector.

    Every field here has a safe default so existing subclasses that don't set
    a field won't break.
    """
    # ── Identity ────────────────────────────────────────────────────────────
    name: str
    display: str
    icon: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)

    # ── Capture behaviour ───────────────────────────────────────────────────
    auto_screen: bool = False
    auto_audio: bool = False

    # ── Context priority: higher number = included first in prompt ───────────
    context_weights: Dict[str, int] = field(default_factory=lambda: {
        "screen": 2, "audio": 2, "rag": 1
    })

    # ── Per-source char budgets for prompt construction ──────────────────────
    context_limits: Dict[str, int] = field(default_factory=lambda: {
        "screen": 4000, "audio": 2500, "rag": 2000
    })

    # ── Provider routing ─────────────────────────────────────────────────────
    preferred_tier: str = "balanced"
    preferred_providers: List[str] = field(default_factory=list)

    # ── Local model hint (shown/logged on mode switch) ───────────────────────
    ollama_model_hint: str = "llama3"

    # ── Detector tuning ─────────────────────────────────────────────────────
    # 0.0 = almost never auto-trigger, 1.0 = very aggressive
    detector_sensitivity: float = 0.5

    # ── Audio VAD tuning ─────────────────────────────────────────────────────
    # Milliseconds of silence before speech segment is sent to Whisper.
    # Lower = faster response but more false-splits on natural speech pauses.
    # Recommended range: 400ms (fast modes) to 1500ms (high-accuracy modes).
    vad_silence_ms: int = 900  # 900ms default — balanced between speed and accuracy

    # ── Context dominance flags ──────────────────────────────────────────────
    audio_dominant: bool = False   # audio is the primary live signal
    vision_dominant: bool = False  # screen/OCR is the primary live signal

    # ── Quick-answer behaviour ───────────────────────────────────────────────
    # The query sent to the LLM when the user hits the quick-answer hotkey.
    quick_answer_query: str = (
        "Using the latest live context, give a quick answer. "
        "First summarise the current situation briefly, then give the "
        "most useful immediate response in 2-4 bullets."
    )
    # Optional short format instruction appended to the quick-answer system prompt.
    quick_answer_format: str = (
        "FORMAT:\n- Quick Summary\n- Best Immediate Answer\n- Next Move"
    )

    # ── Response style ───────────────────────────────────────────────────────
    response_format: str = ""          # injected into system prompt
    max_response_tokens: int = 4096
    custom_instructions: str = ""

    # ════════════════════════════════════════════════════════════════════════
    # Overridable hooks
    # ════════════════════════════════════════════════════════════════════════

    def auto_prompt(self, screen_text: str) -> Optional[str]:
        """Generate automatic prompt from screen content. Override in subclasses."""
        return None

    def audio_prompt(self, transcript: str) -> Optional[str]:
        """Generate automatic prompt from audio. Override in subclasses."""
        return None

    def refine_query(self, query: str, context: dict) -> str:
        """Refine user query with mode-specific context. Override in subclasses."""
        return query

    def post_process(self, response: str) -> str:
        """Post-process AI response. Override in subclasses."""
        return response

    def get_system_addendum(self) -> str:
        """Additional system prompt instructions for this mode."""
        parts = []
        if self.response_format:
            parts.append(f"FORMAT:\n{self.response_format}")
        if self.custom_instructions:
            parts.append(f"CUSTOM INSTRUCTIONS:\n{self.custom_instructions}")
        return "\n\n".join(parts)

    # ── Convenience helpers ──────────────────────────────────────────────────

    def weight(self, source: str) -> int:
        """Return context priority weight for a given source."""
        return self.context_weights.get(source, 1)

    def limit(self, source: str) -> int:
        """Return char budget for a given source."""
        return self.context_limits.get(source, 2000)