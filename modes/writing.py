"""Writing assistant mode — grammar, style, tone.

Context priority: screen (selected text) > rag (style refs) > audio.
Provider preference: quality models for richer rewrites.
"""

from typing import Optional
from modes.base import Mode


class WritingMode(Mode):
    def __init__(self):
        super().__init__(
            name="writing",
            display="Writing Assistant",
            icon="✍️",
            description="Grammar, style, and content improvement",
            auto_screen=False,
            auto_audio=False,
            # Selected/visible text is the primary material; audio is rarely used
            context_weights={"screen": 3, "rag": 2, "audio": 1},
            context_limits={"screen": 5000, "rag": 2500, "audio": 500},
            # Quality models produce richer rewrites
            preferred_tier="balanced",
            preferred_providers=["gemini", "together", "groq", "cerebras", "ollama"],
            ollama_model_hint="mistral",
            detector_sensitivity=0.3,
            audio_dominant=False,
            vision_dominant=True,
            # Quick answer: screen text → immediate improvement suggestion
            quick_answer_query=(
                "You are a professional editor. "
                "Look at the text visible on screen. "
                "Give an immediate, specific improvement suggestion: "
                "rewrite the weakest sentence, improve the opening, or fix the most obvious issue. "
                "Show a before → after example. Be specific, not generic."
            ),
            quick_answer_format=(
                "FORMAT:\n"
                "- ✏️ What to improve\n"
                "- Before: [original]\n"
                "- After: [rewrite]\n"
                "- Why it's better"
            ),
            keywords=["document", "essay", "email", "report", "blog", "article", "write"],
            response_format=(
                "✏️ **Improvements**:\n"
                "Show specific before → after changes\n\n"
                "📝 **Suggestions**:\n"
                "- Clarity improvements\n"
                "- Tone adjustments\n"
                "- Structure enhancements\n\n"
                "🎯 **Tone Check**:\n"
                "Is the tone appropriate? Formal/Casual/Professional?\n\n"
                "📊 **Readability**:\n"
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
        # Don't auto-analyze writing — wait for user request
        return None