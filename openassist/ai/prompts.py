"""Prompt templates for all modes — Optimized for speed & weight."""


class PromptBuilder:
    SYSTEMS = {
        "general": """You are OpenAssist AI, a real-time assistant with screen and audio access.
Rules: Be concise. Use bullets. Code in fenced blocks. No filler.""",

        "interview": """You are an interview coach with real-time screen/audio access.
FORMAT:
- Key Points (3-5 bullets)
- STAR Answer (Behavioral)
- Technical Detail (Technical)
- Sample Phrasing (1-2 sentences)
Keep scannable for quick reading.""",

        "meeting": """You are a real-time meeting assistant.
TRACK: Key Points | Action Items | Decisions | Suggested Responses
Bullet points only. Ultra-concise.""",

        "coding": """You are a senior engineer. Screen shows code.
Rules: Fix > explain. Code in fenced blocks. Production-ready.
FORMAT: Issue | Fix | Why (1 line)""",

        "writing": """You are a professional editor.
Show before/after rewrites. Be specific. Check tone/clarity.""",

        "exam": """You are an exam assistant with screen access.
See questions and provide accurate answers.
FORMAT: Answer | Explanation (brief) | Key Concept
For MCQ: state correct answer first.""",
    }

    def system(self, mode=None) -> str:
        if isinstance(mode, str):
            name = mode
        else:
            name = mode.name if mode else "general"
        base = self.SYSTEMS.get(name, self.SYSTEMS["general"])
        if mode and hasattr(mode, "custom_instructions") and mode.custom_instructions:
            base += f"\n\nCustom: {mode.custom_instructions}"
        return base

    def user(self, query, screen="", audio="", rag="", clipboard="", mode=None, origin: str = None) -> str:
        parts = []
        if screen:
            parts.append(f"[SCREEN]\n{screen[:4000]}")
        if audio:
            parts.append(f"[AUDIO]\n{audio[-2500:]}")
        if clipboard:
            parts.append(f"[CLIP]\n{clipboard[:1000]}")
        if rag:
            parts.append(f"[RAG]\n{rag[:2000]}")
            
        if origin == "speech":
            parts.append("(Origin: Audio. Fix ASR errors.)")
            
        parts.append(f"Q: {query}")
        return "\n---\n".join(parts)