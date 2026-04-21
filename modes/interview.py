"""Interview assistant mode — STAR method, real-time coaching.

Context priority: recent audio > screen question > concise answer framing.
"""

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
            # Audio first — the interviewer speaks; screen shows question text
            context_weights={"audio": 3, "screen": 2, "rag": 1},
            context_limits={"audio": 3500, "screen": 2000, "rag": 1500},
            preferred_tier="balanced",
            preferred_providers=["groq", "cerebras", "gemini", "together", "ollama"],
            ollama_model_hint="llama3",
            # Slightly more aggressive — pick up interview questions early
            detector_sensitivity=0.65,
            # Tight VAD: interviewer pauses are short; fire transcription fast
            vad_silence_ms=500,
            audio_dominant=True,
            vision_dominant=False,
            # Quick answer: audio transcript first, then screen question, then advice
            quick_answer_query=(
                "You are an interview coach. Use the recent audio transcript as the "
                "primary source to understand what the interviewer asked. "
                "If a question is visible on screen, treat it as supporting context. "
                "Give a concise, interview-ready response the candidate can speak aloud immediately. "
                "Lead with the key point, then 2-3 supporting bullets."
            ),
            quick_answer_format=(
                "FORMAT:\n"
                "- 🎤 What to say first (1 sentence)\n"
                "- 📌 Key points (2-3 bullets)\n"
                "- ⚡ Closing signal (optional)"
            ),
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
                f"Interview question detected on screen:\n\n\"{question}\"\n\n"
                "Provide structured answer points using STAR method if behavioral. "
                "Be concise — the candidate needs to read quickly."
            )
        return None

    def audio_prompt(self, transcript: str) -> Optional[str]:
        """Respond to interviewer speech — audio is the primary signal."""
        words = transcript.split()
        if len(words) < 8:
            return None

        recent = " ".join(words[-60:])
        if self._is_question(recent):
            return (
                f"The interviewer just asked (via audio):\n\n\"{recent}\"\n\n"
                "Suggest a strong, structured response. "
                "Include key talking points and a concise sample answer."
            )
        return None

    def refine_query(self, query: str, context: dict) -> str:
        """Audio first, then screen question as support."""
        audio = context.get("audio", "")
        screen = context.get("screen", "")
        parts = [query]
        if audio:
            recent_audio = " ".join(audio.split()[-100:])
            parts.append(f"Recent interview audio:\n{recent_audio}")
        if screen:
            parts.append(f"Screen question (if visible):\n{screen[:500]}")
        return "\n\n".join(parts)

    def _extract_question(self, text: str) -> Optional[str]:
        """Extract the most likely interview question from screen text."""
        lines = text.strip().split('\n')
        questions = []

        for line in lines:
            line = line.strip()
            if len(line) < 15:
                continue

            lower = line.lower()

            if '?' in line:
                questions.append(line)
                continue

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

            if re.match(r'^(Q?\d+[\.):)]|Question\s+\d+)', line, re.IGNORECASE):
                questions.append(line)

        if questions:
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