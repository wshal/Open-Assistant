"""Exam/Assessment assistant mode — MCQ solver, detailed explanations.

Context priority: screen (visible question) > rag (knowledge) > audio.
Provider preference: accuracy-focused models.
Detector: high sensitivity — answer fast.
"""

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
            # Visible question is the entire input signal
            context_weights={"screen": 3, "rag": 2, "audio": 1},
            context_limits={"screen": 4000, "rag": 2500, "audio": 500},
            # Accuracy first — sambanova/gemini for quality
            preferred_tier="quality",
            preferred_providers=["gemini", "together", "sambanova", "groq", "ollama"],
            ollama_model_hint="llama3",
            # Most aggressive — grab questions the moment they appear
            detector_sensitivity=0.7,
            audio_dominant=False,
            vision_dominant=True,
            # Quick answer: direct answer to visible question
            quick_answer_query=(
                "You are an exam assistant. "
                "Look at the question visible on screen. "
                "For MCQ: state the correct answer letter first, then a 1-sentence explanation. "
                "For open questions: give the most accurate direct answer in 2-3 sentences. "
                "Confidence: HIGH / MEDIUM / LOW based on question clarity."
            ),
            quick_answer_format=(
                "FORMAT:\n"
                "- ✅ Answer: [letter or direct answer]\n"
                "- 🧠 Why: [1-2 sentences]\n"
                "- 🔍 Confidence: [HIGH/MEDIUM/LOW]"
            ),
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
        if "?" in screen_text or "Answer:" in screen_text:
            return "Exam question detected. Identify the correct answer and explain briefly."
        return None