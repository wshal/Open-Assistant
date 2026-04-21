"""Prompt templates for all modes — fully mode-profile-aware."""

from typing import Any, Optional, List, Dict, Union


class ContextRanker:
    """
    Ranks context sources for prompt inclusion.

    Accepts either a mode name string (backward-compat) or a Mode object.
    When a Mode object is provided, it reads context_weights and context_limits
    directly from the profile — no scattered if/else.
    """

    # Fallback static priority (used when no mode object is available)
    _DEFAULT_WEIGHTS = {
        "screen": 2,
        "audio":  2,
        "rag":    1,
    }
    _DEFAULT_LIMITS = {
        "screen": 4000,
        "audio":  2500,
        "rag":    2000,
    }

    # Legacy string-based priority table (kept for backward compatibility)
    MODE_PRIORITY = {
        "interview": {"audio": 3, "screen": 2, "rag": 1},
        "meeting":   {"audio": 3, "screen": 1, "rag": 1},
        "coding":    {"screen": 3, "rag": 2, "audio": 1},
        "writing":   {"screen": 3, "rag": 2, "audio": 1},
        "exam":      {"screen": 3, "rag": 2, "audio": 1},
    }

    @classmethod
    def _resolve_weights(cls, mode) -> Dict[str, int]:
        """Resolve priority weights from a Mode object or string."""
        if mode is None:
            return cls._DEFAULT_WEIGHTS
        if hasattr(mode, "context_weights"):
            return mode.context_weights
        # Fallback: mode is a string
        return cls.MODE_PRIORITY.get(str(mode).lower(), cls._DEFAULT_WEIGHTS)

    @classmethod
    def _resolve_limits(cls, mode) -> Dict[str, int]:
        """Resolve char limits from a Mode object or string."""
        if mode is None:
            return cls._DEFAULT_LIMITS
        if hasattr(mode, "context_limits"):
            return mode.context_limits
        return cls._DEFAULT_LIMITS

    @classmethod
    def rank(cls, contexts: List[str], mode=None) -> List[tuple]:
        """
        Return ranked (source, content, priority) tuples.
        `mode` can be a Mode object or a string mode name.
        """
        weights = cls._resolve_weights(mode)
        source_map = {
            "screen": contexts[0] if len(contexts) > 0 else "",
            "audio":  contexts[1] if len(contexts) > 1 else "",
            "rag":    contexts[2] if len(contexts) > 2 else "",
        }
        ranked = [
            (src, content, weights.get(src, 1))
            for src, content in source_map.items()
            if content
        ]
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    @classmethod
    def limit(cls, source: str, content: str, mode=None) -> str:
        """
        Apply char limits per source.
        `mode` can be a Mode object or None.
        """
        limits = cls._resolve_limits(mode)
        cap = limits.get(source, cls._DEFAULT_LIMITS.get(source, 1000))
        return content[:cap]


class PromptBuilder:
    SYSTEMS = {
        "general": """You are OpenAssist AI, a real-time assistant with screen and audio access.
Rules:
- Prefer a fast, direct answer. Use screen/audio context only when it is relevant to the question.
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
PRIORITY:
- Use the most recent audio first.
- Use the visible on-screen question as support if present.
- Keep answer framing concise and interview-ready.
Keep scannable for quick reading.""",
        "meeting": """You are a real-time meeting assistant.
TRACK: Key Points | Action Items | Decisions | Suggested Responses
Bullet points only. Ultra-concise.
Prefer fast, direct responses over long analysis.""",
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
            "user_template": "[Recent Audio]\n{audio}\n\n[Meeting Context]\n{screen}\n\n[Question/Topic]\n{query}",
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

PRIORITY:
- Recent audio is the primary source.
- Visible interview question on screen is secondary support.
- Keep framing concise and easy to speak aloud.

Keep responses scannable for interview pressure.""",
            "user_template": "[Recent Audio]\n{audio}\n\n[Screen Question]\n{screen}\n\n[Query]\n{query}",
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

    UNCERTAINTY_MARKERS = {
        "screen": "[Screen may be partial or OCR may contain errors]",
        "audio":  "[Audio may be incomplete or contain ASR errors]",
        "rag":    "[Knowledge base results may be outdated]",
    }

    @staticmethod
    def _is_general_knowledge_query(query: str) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False

        contextual_markers = (
            "this ", "current ", "shown ", "visible ", "on screen",
            "in this code", "in the code", "this code", "this function",
            "this file", "line ", "error", "traceback", "bug", "fix",
            "function do", "what does this",
        )
        if any(marker in q for marker in contextual_markers):
            return False

        starters = (
            "what is ", "what are ", "who is ", "who are ",
            "explain ", "define ", "tell me about ",
            "give me an example", "show me an example",
            "how does ", "how do ",
        )
        if any(q.startswith(prefix) for prefix in starters):
            return True

        generic_markers = (
            "example of", "context api", "useeffect", "react",
            "javascript", "typescript", "python", "api", "hook",
        )
        return len(q.split()) <= 10 and any(marker in q for marker in generic_markers)

    def system(self, mode=None) -> str:
        if isinstance(mode, str):
            name = mode
        else:
            name = mode.name if mode and hasattr(mode, "name") else "general"

        if name in self.PROMPT_PACKS:
            return self.PROMPT_PACKS[name]["system"]

        base = self.SYSTEMS.get(name, self.SYSTEMS["general"])
        if mode and hasattr(mode, "custom_instructions") and mode.custom_instructions:
            base += f"\n\nCustom: {mode.custom_instructions}"
        return base

    def build_from_pack(
        self,
        mode,
        screen: str = "",
        audio: str = "",
        rag: str = "",
        query: str = "",
        nexus: Dict[str, Any] = None,
    ) -> tuple:
        """Build system and user prompts from prompt pack, respecting Mode limits."""
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        pack = self.PROMPT_PACKS.get(mode_name)
        if not pack:
            return self.system(mode), self.user(query, screen, audio, rag, nexus=nexus)

        # Use mode-specific limits if available
        screen_limit = mode.limit("screen") if hasattr(mode, "limit") else 4000
        audio_limit  = mode.limit("audio")  if hasattr(mode, "limit") else 2500
        rag_limit    = mode.limit("rag")    if hasattr(mode, "limit") else 2000

        user_prompt = pack["user_template"].format(
            screen=screen[:screen_limit] if screen else "",
            audio=audio[-audio_limit:]   if audio  else "",
            rag=rag[:rag_limit]          if rag    else "",
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
        history: str = "",
    ) -> str:
        """Build user prompt with mode-profile-aware context ranking and limits."""
        parts = []
        suppress_live_context = origin == "manual" and self._is_general_knowledge_query(query)

        # Conversation history — injected first so the model anchors on prior context
        # before reading the current query. Critical for follow-up queries.
        if history:
            parts.append(f"[CONVERSATION HISTORY]\n{history}")

        # Context ranking — accepts Mode object or string
        contexts = [screen, audio, rag]
        ranked = ContextRanker.rank(contexts, mode=mode)

        if nexus and not suppress_live_context:
            parts.append(
                f"[ENVIRONMENT]\nActive Window: {nexus.get('active_window', 'Unknown')}\n"
                f"History Depth: {nexus.get('history_depth_secs', 60)}s"
            )

        for src, content, _ in ranked:
            if suppress_live_context and src in {"screen", "audio"}:
                continue
            # Use mode-aware limits
            limited = ContextRanker.limit(src, content, mode=mode)
            if src == "screen":
                parts.append(f"[SCREEN]\n{limited}")
            elif src == "audio":
                parts.append(f"[AUDIO]\n{limited}")
            elif src == "rag":
                parts.append(f"[RAG]\n{limited}")

        if clipboard:
            parts.append(f"[CLIP]\n{clipboard[:1000]}")

        if origin == "speech":
            parts.append("(Origin: Audio. Fix ASR errors.)")
        elif origin == "screen_analysis":
            parts.append(
                "[TASK]\nAnalyse the attached screenshot first. Treat the image as the primary source of truth. "
                "Use [SCREEN] as OCR support, and use [AUDIO] and environment context only as support. "
                "If the screenshot or OCR is partial, say what is visible, what it likely means, and the best next action.\n"
                "FORMAT:\n- What I See\n- What It Means\n- What To Do Next"
            )
        elif origin == "manual":
            if suppress_live_context:
                parts.append(
                    "[TASK]\nAnswer the user's question directly from general knowledge. "
                    "Do not mention the current screen, codebase, active window, or live session unless the question clearly asks about them."
                )
            else:
                parts.append(
                    "[TASK]\nAnswer the user's question using the current live session context. "
                    "Prefer the most recent on-screen evidence when relevant. "
                    "If the question is generic and the live context is unrelated, answer directly without talking about the unrelated context."
                )
        elif origin == "quick":
            # Mode-specific quick-answer format injected here
            fmt = ""
            if mode and hasattr(mode, "quick_answer_format"):
                fmt = f"\n{mode.quick_answer_format}"
            parts.append(
                "[TASK]\nGive the fastest useful context answer using the most recent live context. "
                "Prioritise context in the order it appears above (highest weight first). "
                f"Keep it extremely concise and actionable.{fmt}"
            )

        parts.append(f"Q: {query}")
        return "\n---\n".join(parts)

    @staticmethod
    def format_response(mode, content: str) -> str:
        """Apply mode-specific formatting contract to response."""
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        contracts = {
            "coding":    "Issue | Fix | Why (1 line)",
            "interview": "Key Points | STAR Answer | Technical Detail | Sample Phrasing",
            "meeting":   "Key Points | Action Items | Decisions | Suggested Responses",
            "exam":      "Answer | Explanation | Key Concept",
        }
        contract = contracts.get(mode_name, "")
        if contract:
            return f"[Format: {contract}]\n\n{content}"
        return content
