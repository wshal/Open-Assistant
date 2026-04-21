"""Coding assistant mode — technical problems, best practices, review."""

from typing import Optional
from modes.base import Mode


class CodingMode(Mode):
    def __init__(self):
        super().__init__(
            name="coding",
            display="Coding Mode",
            icon="💻",
            description="Technical problem solving and code review",
            auto_screen=True,
            auto_audio=False,
            preferred_tier="large",
            preferred_providers=["gemini", "together", "mistral"],
            keywords=[
                "python", "javascript", "java", "cpp", "coding",
                "function", "class", "async", "algorithm",
                "complexity", "big o", "leak", "segfault",
            ],
            response_format=(
                "💻 **Implementation**\n"
                "```[lang]\n// Code goes here\n```\n\n"
                "⚙️ **Complexity**: O(?) time | O(?) space\n\n"
                "💡 **Key Insights**:\n"
                "  - Point 1\n"
                "  - Point 2\n\n"
                "🧪 **Test Cases**: 1..2..3"
            ),
        )

    def auto_prompt(self, screen_text: str) -> Optional[str]:
        # Detect code-related errors or logic
        if any(err in screen_text.lower() for err in ["traceback", "error:", "exception"]):
            return "A code error or traceback was detected. Please analyze and provide a fix."
        return None

    def refine_query(self, query: str, context: dict) -> str:
        # Add language detection hint
        lang = context.get("detected_language", "unknown")
        if lang != "unknown":
            return f"Context: User is working in {lang}.\n\n{query}"
        return query