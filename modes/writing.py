"""Writing assistant mode â grammar, style, tone."""

from typing import Optional
from modes.base import Mode


class WritingMode(Mode):
    def __init__(self):
        super().__init__(
            name="writing",
            display="Writing Assistant",
            icon="âï¸",
            description="Grammar, style, and content improvement",
            auto_screen=False,
            auto_audio=False,
            preferred_tier="balanced",
            preferred_providers=["gemini", "cohere", "together"],
            keywords=["document", "essay", "email", "report", "blog", "article", "write"],
            response_format=(
                "âï¸ **Improvements**:\n"
                "Show specific before â after changes\n\n"
                "ð **Suggestions**:\n"
                "- Clarity improvements\n"
                "- Tone adjustments\n"
                "- Structure enhancements\n\n"
                "ð¯ **Tone Check**:\n"
                "Is the tone appropriate? Formal/Casual/Professional?\n\n"
                "ð **Readability**:\n"
                "Approximate reading level and suggestions\n\n"
                "Show exact rewrites. Be specific, not vague."
            ),
        )

    def refine_query(self, query: str, context: dict) -> str:
        screen = context.get("screen", "")
        if screen:
            word_count = len(screen.split())
            return (
                f"{query}\n\n"
                f"Text on screen ({word_count} words):\n"
                f"---\n{screen[:3000]}\n---"
            )
        return query

    def auto_prompt(self, screen_text: str) -> Optional[str]:
        # Don't auto-analyze writing â wait for user request
        return None