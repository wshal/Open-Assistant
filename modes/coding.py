"""Coding assistant mode — technical problems, best practices, review.

Context priority: screen > rag > audio.
Provider preference: quality-first (gemini/together for larger context windows).
Ollama hint: codellama or qwen2.5-coder for local coding assistance.
"""

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
            # Screen is everything in coding; audio is rarely helpful
            context_weights={"screen": 3, "rag": 2, "audio": 1},
            context_limits={"screen": 5000, "rag": 3000, "audio": 1000},
            # Quality models for code — larger context window matters
            preferred_tier="large",
            preferred_providers=["gemini", "together", "groq", "cerebras", "ollama"],
            # Local coding specialist model
            ollama_model_hint="codellama",
            detector_sensitivity=0.4,
            audio_dominant=False,
            vision_dominant=True,
            # Quick answer: what's on screen → fix or explain
            quick_answer_query=(
                "You are a senior software engineer. "
                "Look at the code visible on screen. "
                "If there is an error or traceback, identify the root cause and the exact fix. "
                "If there is code without an error, explain what it does and suggest any immediate improvements. "
                "Be direct — code fix first, explanation after."
            ),
            quick_answer_format=(
                "FORMAT:\n"
                "- 🐛 Issue / What I see\n"
                "- 🔧 Fix (code block if needed)\n"
                "- 💡 Why"
            ),
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
        if any(err in screen_text.lower() for err in ["traceback", "error:", "exception"]):
            return "A code error or traceback was detected. Please analyze and provide a fix."
        return None

    def refine_query(self, query: str, context: dict) -> str:
        lang = context.get("detected_language", "unknown")
        screen = context.get("screen", "")
        parts = [query]
        if lang != "unknown":
            parts.insert(0, f"Context: User is working in {lang}.")
        if screen and len(screen) > 50:
            parts.append(f"Code on screen:\n```\n{screen[:3000]}\n```")
        return "\n\n".join(parts)