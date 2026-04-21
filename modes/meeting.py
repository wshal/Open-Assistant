"""Meeting assistant mode — notes, action items, summaries.

Context priority: audio > screen (rarely useful) > rag.
Provider preference: fastest general models (groq/cerebras first).
"""

from typing import Optional
from modes.base import Mode


class MeetingMode(Mode):
    def __init__(self):
        super().__init__(
            name="meeting",
            display="Meeting Assistant",
            icon="📅",
            description="Real-time meeting notes and action items",
            auto_screen=False,
            auto_audio=True,
            # Audio is everything in a meeting; screen context is rarely relevant
            context_weights={"audio": 3, "screen": 1, "rag": 1},
            context_limits={"audio": 4000, "screen": 1000, "rag": 1000},
            # Fast general models — meetings need low latency, not deep reasoning
            preferred_tier="fast",
            preferred_providers=["groq", "cerebras", "together", "gemini", "ollama"],
            ollama_model_hint="llama3",
            detector_sensitivity=0.55,
            # Fast VAD: meeting conversation has short pauses between speakers
            vad_silence_ms=600,
            audio_dominant=True,
            vision_dominant=False,
            # Quick answer: decisions and action items — the two things that matter most
            quick_answer_query=(
                "You are a real-time meeting copilot. "
                "From the recent audio transcript, extract immediately: "
                "1) The most recent topic or decision discussed "
                "2) Any action items mentioned (who does what by when) "
                "3) If someone asked a question that needs a response, suggest one. "
                "Be ultra-concise. Bullet points only."
            ),
            quick_answer_format=(
                "FORMAT:\n"
                "- 📋 Current topic / decision\n"
                "- ✅ Action items (WHO: WHAT by WHEN)\n"
                "- 💬 Suggested response (if asked something)"
            ),
            keywords=["meeting", "agenda", "action item", "sync", "standup", "retrospective"],
            response_format=(
                "📋 **Key Discussion Points**\n"
                "- ...\n\n"
                "✅ **Action Items**\n"
                "- [ ] WHO: does WHAT by WHEN\n\n"
                "🎯 **Decisions Made**\n"
                "- ...\n\n"
                "💬 **Suggested Response** (if user is asked something)\n\n"
                "⏰ **Follow-ups Needed**\n"
                "- ...\n\n"
                "Ultra-concise. Bullet points only. Update incrementally."
            ),
        )

        self._summary_parts = []
        self._action_items = []
        self._decisions = []

    def audio_prompt(self, transcript: str) -> Optional[str]:
        words = transcript.split()
        if len(words) < 15:
            return None

        recent = " ".join(words[-80:])
        return (
            f"Meeting transcript update:\n\n\"{recent}\"\n\n"
            "Extract: 1) Key points discussed 2) Any action items (who/what/when) "
            "3) Decisions made 4) If someone asked the user a question, suggest a response. "
            "Be extremely concise."
        )

    def refine_query(self, query: str, context: dict) -> str:
        audio = context.get("audio", "")
        if audio:
            return (
                f"{query}\n\n"
                f"Meeting transcript so far:\n{audio[-2000:]}"
            )
        return query

    def post_process(self, response: str) -> str:
        """Track action items and decisions from responses."""
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- [ ]') or line.startswith('- [x]'):
                if line not in self._action_items:
                    self._action_items.append(line)
        return response

    def get_accumulated_notes(self) -> str:
        """Get all accumulated meeting notes."""
        parts = []
        if self._action_items:
            parts.append("✅ **All Action Items:**\n" + "\n".join(self._action_items))
        if self._decisions:
            parts.append("🎯 **All Decisions:**\n" + "\n".join(self._decisions))
        return "\n\n".join(parts) or "No notes accumulated yet."

    def clear_notes(self):
        self._summary_parts.clear()
        self._action_items.clear()
        self._decisions.clear()