"""Base mode class with shared functionality."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Mode:
    """Base mode definition."""
    name: str
    display: str
    icon: str
    description: str = ""
    auto_screen: bool = False
    auto_audio: bool = False
    custom_instructions: str = ""
    keywords: List[str] = field(default_factory=list)
    preferred_tier: str = "balanced"
    preferred_providers: List[str] = field(default_factory=list)
    response_format: str = ""  # Injected into system prompt
    max_response_tokens: int = 4096

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