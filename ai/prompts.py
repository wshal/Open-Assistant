"""Prompt templates for all modes — Optimized for speed & weight."""

from typing import Any, Optional, List, Dict


class ContextRanker:
    """Ranks context importance for prompt inclusion."""

    PRIORITY = {
        "screen": 3,  # Highest - most current
        "audio": 2,  # Medium - recent speech
        "rag": 1,  # Low - knowledge base
    }

    LIMITS = {
        "screen": 4000,
        "audio": 2500,
        "rag": 2000,
    }

    @classmethod
    def rank(cls, contexts: List[str]) -> List[tuple]:
        """Return ranked (source, content, priority) tuples."""
        ranked = []
        source_map = {
            "screen": contexts[0] if len(contexts) > 0 else "",
            "audio": contexts[1] if len(contexts) > 1 else "",
            "rag": contexts[2] if len(contexts) > 2 else "",
        }

        for src, content in source_map.items():
            if content:
                ranked.append((src, content, cls.PRIORITY.get(src, 0)))

        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    @classmethod
    def limit(cls, source: str, content: str) -> str:
        """Apply token limits per source."""
        limit = cls.LIMITS.get(source, 1000)
        return content[:limit]


class PromptBuilder:
    SYSTEMS = {
        "general": """You are OpenAssist AI, a real-time assistant with screen and audio access.
Rules:
- Prioritize what is visible on the current screen.
- Use recent audio and session context as supporting evidence.
- Distinguish clearly between observed facts and inference.
- Be concise, useful, and action-oriented.
- Use bullets when scanning is easier. Code in fenced blocks. No filler.""",
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

    # P2: Fine-grained prompt packs per use case
    PROMPT_PACKS = {
        "meeting": {
            "system": """You are a real-time meeting copilot.
CONTEXT TRACKING:
- What was the question asked?
- What are suggested responses?
- Action items from discussion
- Key decisions made

RULES:
- Ultra-concise bullet points
- Real-time appropriate (fast responses)
- Mark confidence: HIGH/MEDIUM/LOW when uncertain
- Flag if context is unclear""",
            "user_template": "[Meeting Context]\n{screen}\n\n[Recent Audio]\n{audio}\n\n[Question/Topic]\n{query}",
        },
        "coding": {
            "system": """You are a senior software engineer reviewing code.
CODE REVIEW FORMAT:
- Issue: [Bug/Vulnerability/Code Smell]
- Fix: [Suggested patch in fenced code]
- Risk: [LOW/MEDIUM/HIGH]
- Why: [1 sentence explanation]

PRIORITIES:
1. Security vulnerabilities
2. Logic errors
3. Performance issues
4. Code style/naming

OUTPUT: Production-ready code when providing fixes.""",
            "user_template": "[Code Context]\n{screen}\n\n[Question]\n{query}\n\n[Available Knowledge]\n{rag}",
        },
        "interview": {
            "system": """You are an expert interview coach.
RESPONSE FORMAT:
- Direct Answer: [1-2 sentences]
- Evidence: [Supporting detail or example]
- Delivery: [Suggested phrasing/tone]

CONFIDENCE MARKERS:
- HIGH: Clear question with sufficient context
- MEDIUM: Partial context, clarify if needed
- LOW: Unclear question, ask for clarification

Keep responses scannable for interview pressure.""",
            "user_template": "[Screen/Question]\n{screen}\n\n[Audio Context]\n{audio}\n\n[Query]\n{query}",
        },
        "writing": {
            "system": """You are a professional editor and writing coach.
REVIEW FORMAT:
- Before: [Original text]
- After: [Improved version]
- Why: [1 sentence rationale]

CHECKLIST:
- Tone consistency
- Clarity improvements
- Grammar/punctuation
- Engagement/flow

Be specific with suggestions.""",
            "user_template": "[Original Text]\n{screen}\n\n[Writing Goal]\n{query}\n\n[Additional Context]\n{rag}",
        },
        "research": {
            "system": """You are a research assistant with screen access.
OUTPUT FORMAT:
- Summary: [2-3 sentences]
- Key Findings: [Bulleted list]
- Source Attribution: [Where info came from]
- Confidence: [HIGH/MEDIUM/LOW]

RULES:
- Distinguish observed (screen) vs inferred
- Flag if OCR might be inaccurate
- Prefer specific data over general claims""",
            "user_template": "[Screen/Content]\n{screen}\n\n[Research Query]\n{query}\n\n[Knowledge Base]\n{rag}",
        },
    }

    # Uncertainty markers for weak context
    UNCERTAINTY_MARKERS = {
        "screen": "[Screen may be partial or OCR may contain errors]",
        "audio": "[Audio may be incomplete or contain ASR errors]",
        "rag": "[Knowledge base results may be outdated]",
    }

    def system(self, mode=None) -> str:
        if isinstance(mode, str):
            name = mode
        else:
            name = mode.name if mode else "general"

        # P2: Use fine-grained prompt pack if available
        if name in self.PROMPT_PACKS:
            return self.PROMPT_PACKS[name]["system"]

        base = self.SYSTEMS.get(name, self.SYSTEMS["general"])
        if mode and hasattr(mode, "custom_instructions") and mode.custom_instructions:
            base += f"\n\nCustom: {mode.custom_instructions}"
        return base

    def build_from_pack(
        self,
        mode: str,
        screen: str = "",
        audio: str = "",
        rag: str = "",
        query: str = "",
        nexus: Dict[str, Any] = None,
    ) -> tuple:
        """P2: Build system and user prompts from prompt pack."""
        pack = self.PROMPT_PACKS.get(mode)
        if not pack:
            return self.system(mode), self.user(query, screen, audio, rag, nexus=nexus)

        # Build user prompt from template
        user_prompt = pack["user_template"].format(
            screen=screen[:4000] if screen else "",
            audio=audio[-2500:] if audio else "",
            rag=rag[:2000] if rag else "",
            query=query,
        )
        
        if nexus:
            user_prompt = f"[ACTIVE WINDOW: {nexus.get('active_window', 'Unknown')}]\n" + user_prompt

        return pack["system"], user_prompt

    def user(
        self,
        query,
        screen="",
        audio="",
        rag="",
        clipboard="",
        mode=None,
        origin: str = None,
        nexus: Dict[str, Any] = None,
    ) -> str:
        """Build user prompt with ranked context and uncertainty markers."""
        parts = []

        # Context ranking and limiting
        contexts = [screen, audio, rag]
        ranked = ContextRanker.rank(contexts)
        
        if nexus:
            parts.append(f"[ENVIRONMENT]\nActive Window: {nexus.get('active_window', 'Unknown')}\nHistory Depth: {nexus.get('history_depth_secs', 60)}s")

        for src, content, _ in ranked:
            if src == "screen":
                limited = ContextRanker.limit("screen", content)
                parts.append(f"[SCREEN]\n{limited}")
            elif src == "audio":
                limited = ContextRanker.limit("audio", content)
                parts.append(f"[AUDIO]\n{limited}")
            elif src == "rag":
                limited = ContextRanker.limit("rag", content)
                parts.append(f"[RAG]\n{limited}")

        if clipboard:
            parts.append(f"[CLIP]\n{clipboard[:1000]}")

        if origin == "speech":
            parts.append("(Origin: Audio. Fix ASR errors.)")
        elif origin == "screen_analysis":
            parts.append(
                "[TASK]\nAnalyze the attached screenshot first. Treat the image as the primary source of truth. "
                "Use [SCREEN] as OCR support, and use [AUDIO] and environment context only as support. "
                "If the screenshot or OCR is partial, say what is visible, what it likely means, and the best next action.\n"
                "FORMAT:\n- What I See\n- What It Means\n- What To Do Next"
            )
        elif origin == "manual":
            parts.append(
                "[TASK]\nAnswer the user's question using the current live session context. "
                "Prefer the most recent on-screen evidence when relevant."
            )

        parts.append(f"Q: {query}")
        return "\n---\n".join(parts)

    @staticmethod
    def format_response(mode: str, content: str) -> str:
        """Apply mode-specific formatting contract to response."""
        contracts = {
            "coding": "Issue | Fix | Why (1 line)",
            "interview": "Key Points | STAR Answer | Technical Detail | Sample Phrasing",
            "meeting": "Key Points | Action Items | Decisions | Suggested Responses",
            "exam": "Answer | Explanation | Key Concept",
        }

        contract = contracts.get(mode, "")
        if contract:
            return f"[Format: {contract}]\n\n{content}"
        return content
