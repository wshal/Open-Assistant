"""General assistant mode."""

from modes.base import Mode


class GeneralMode(Mode):
    def __init__(self):
        super().__init__(
            name="general",
            display="General Assistant",
            icon="ð¤",
            description="All-purpose AI assistant",
            auto_screen=False,
            auto_audio=False,
            preferred_tier="balanced",
            keywords=["help", "question", "explain"],
            response_format=(
                "- Be concise and direct\n"
                "- Use bullet points for lists\n"
                "- Code in fenced blocks with language tags\n"
                "- No unnecessary preamble"
            ),
        )

    def refine_query(self, query: str, context: dict) -> str:
        screen = context.get("screen", "")
        if screen and len(query) < 20:
            return f"{query}\n\nContext from screen:\n{screen[:1000]}"
        return query