"""Meeting assistant mode â notes, action items, summaries."""

from typing import Optional
from modes.base import Mode


class MeetingMode(Mode):
    def __init__(self):
        super().__init__(
            name="meeting",
            display="Meeting Assistant",
            icon="ð",
            description="Real-time meeting notes and action items",
            auto_screen=False,
            auto_audio=True,
            preferred_tier="fast",
            preferred_providers=["groq", "cerebras", "gemini"],
            keywords=["meeting", "agenda", "action item", "sync", "standup", "retrospective"],
            response_format=(
                "ð **Key Discussion Points**\n"
                "- ...\n\n"
                "â **Action Items**\n"
                "- [ ] WHO: does WHAT by WHEN\n\n"
                "ð¯ **Decisions Made**\n"
                "- ...\n\n"
                "ð¬ **Suggested Response** (if user is asked something)\n\n"
                "â° **Follow-ups Needed**\n"
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
            parts.append("â **All Action Items:**\n" + "\n".join(self._action_items))
        if self._decisions:
            parts.append("ð¯ **All Decisions:**\n" + "\n".join(self._decisions))
        return "\n\n".join(parts) or "No notes accumulated yet."

    def clear_notes(self):
        self._summary_parts.clear()
        self._action_items.clear()
        self._decisions.clear()