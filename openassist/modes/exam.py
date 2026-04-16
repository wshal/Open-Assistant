"""Exam/Assessment assistant mode — MCQ solver, detailed explanations."""

from typing import Optional
from modes.base import Mode


class ExamMode(Mode):
    def __init__(self):
        super().__init__(
            name="exam",
            display="Exam Mode",
            icon="📍",
            description="Accelerated MCQ solving and conceptual explanations",
            auto_screen=True,
            auto_audio=False,
            preferred_tier="quality",
            preferred_providers=["gemini", "together", "sambanova"],
            keywords=[
                "multiple choice", "answer", "select the",
                "which of following", "true or false", "exam",
                "quiz", "assessment", "question 1",
            ],
            response_format=(
                "✅ **Recommended Answer**: [A/B/C/D]\n\n"
                "🧠 **Explanation**: 1-2 sentences explaining why.\n\n"
                "🔍 **Confidence**: [Low/Med/High]"
            ),
        )

    def auto_prompt(self, screen_text: str) -> Optional[str]:
        # Custom logic to skip non-question text
        if "?" in screen_text or "Answer:" in screen_text:
            return "Exam question detected. Identify the correct answer and explain briefly."
        return None