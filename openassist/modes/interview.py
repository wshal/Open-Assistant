"""Interview assistant mode — STAR method, real-time coaching."""

import re
from typing import Optional
from modes.base import Mode


class InterviewMode(Mode):
    def __init__(self):
        super().__init__(
            name="interview",
            display="Interview Mode",
            icon="🎯",
            description="Real-time interview coaching with STAR answers",
            auto_screen=True,
            auto_audio=True,
            preferred_tier="balanced",
            preferred_providers=["gemini", "groq", "together"],
            keywords=[
                "interview", "behavioral", "technical", "tell me about",
                "describe a time", "what is your", "why do you", "walk me through",
                "strengths", "weaknesses", "experience", "challenge",
            ],
            response_format=(
                "🎯 **Key Points** (3-5 bullets to mention)\n\n"
                "📍 **STAR Answer** (if behavioral):\n"
                "  - **S**ituation: ...\n"
                "  - **T**ask: ...\n"
                "  - **A**ction: ...\n"
                "  - **R**esult: ...\n\n"
                "💡 **Technical Detail** (if technical question)\n\n"
                "🗣️ **Say This**: 1-2 strong opening sentences\n\n"
                "❓ **Ask Them**: 1 follow-up question to show engagement\n\n"
                "⚠️ Keep it scannable — user reads while talking!"
            ),
        )

        # Track asked questions to avoid repeats
        self._asked_questions = set()

    def auto_prompt(self, screen_text: str) -> Optional[str]:
        """Detect and respond to interview questions on screen."""
        question = self._extract_question(screen_text)
        if question and question not in self._asked_questions:
            self._asked_questions.add(question)
            return (
                f"Interview question detected:\n\n\"{question}\"\n\n"
                "Provide structured answer points using STAR method if behavioral. "
                "Be concise — the candidate needs to read quickly."
            )
        return None

    def audio_prompt(self, transcript: str) -> Optional[str]:
        """Respond to interviewer speech."""
        # Only trigger on substantial new speech
        words = transcript.split()
        if len(words) < 8:
            return None

        recent = " ".join(words[-60:])
        if self._is_question(recent):
            return (
                f"The interviewer just asked:\n\n\"{recent}\"\n\n"
                "Suggest a strong, structured response. "
                "Include key talking points and a concise sample answer."
            )
        return None

    def refine_query(self, query: str, context: dict) -> str:
        audio = context.get("audio", "")
        if audio:
            recent_audio = " ".join(audio.split()[-100:])
            return (
                f"{query}\n\n"
                f"Recent conversation transcript:\n{recent_audio}"
            )
        return query

    def _extract_question(self, text: str) -> Optional[str]:
        """Extract the most likely interview question from screen text."""
        lines = text.strip().split('\n')
        questions = []

        for line in lines:
            line = line.strip()
            if len(line) < 15:
                continue

            lower = line.lower()

            # Direct question patterns
            if '?' in line:
                questions.append(line)
                continue

            # Behavioral patterns
            behavioral = [
                "tell me about a time", "describe a situation",
                "give an example", "walk me through",
                "how would you", "how do you", "how did you",
                "what would you do", "what is your",
                "why are you", "why do you", "why should we",
                "where do you see", "what are your",
            ]
            if any(lower.startswith(p) or f" {p}" in lower for p in behavioral):
                questions.append(line)
                continue

            # Numbered questions
            if re.match(r'^(Q?\d+[\.\):]|Question\s+\d+)', line, re.IGNORECASE):
                questions.append(line)

        if questions:
            # Return the most recent/longest question
            return max(questions[-3:], key=len)
        return None

    @staticmethod
    def _is_question(text: str) -> bool:
        """Check if text contains a question."""
        lower = text.lower().strip()
        if '?' in text:
            return True
        question_starts = [
            "what", "how", "why", "when", "where", "who",
            "can you", "could you", "tell me", "describe",
            "explain", "do you", "have you", "would you",
        ]
        return any(lower.startswith(q) for q in question_starts)