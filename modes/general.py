"""General assistant mode — balanced, fast, all-purpose."""

from modes.base import Mode


class GeneralMode(Mode):
    def __init__(self):
        super().__init__(
            name="general",
            display="General Assistant",
            icon="🤖",
            description="All-purpose AI assistant",
            auto_screen=False,
            auto_audio=False,
            # Context: balanced — both screen and audio are useful
            context_weights={"screen": 2, "audio": 2, "rag": 1},
            context_limits={"screen": 4000, "audio": 2500, "rag": 2000},
            # Prefer fast providers; quality is secondary for general queries
            preferred_tier="balanced",
            preferred_providers=["groq", "cerebras", "together", "gemini", "ollama"],
            ollama_model_hint="llama3",
            detector_sensitivity=0.5,
            audio_dominant=False,
            vision_dominant=False,
            # Quick answer: balanced — use whichever context is available
            quick_answer_query=(
                "Using the latest live context, give a quick, direct answer. "
                "Prefer the most recent audio or screen evidence. "
                "Keep it extremely concise and actionable."
            ),
            quick_answer_format=(
                "FORMAT:\n- Quick Summary\n- Best Immediate Answer\n- Next Move"
            ),
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