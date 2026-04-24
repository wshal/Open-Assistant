"""Auto-detect questions on screen with interaction triggers.

Full-Auto Meeting Interaction:
- Questions (what is, how do, why does)
- Coding (implement, code, function, algorithm)
- Casual (hello, agree, think, thanks)
- Follow-ups (elaborate, that makes sense)
"""

import re
import time
import numpy as np
from collections import deque
from typing import Optional, List
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DetectionResult:
    """P2: Confidence-gated detection result with provenance."""

    triggered: bool
    confidence: float
    trigger_type: str  # "question", "coding", "casual", "followup"
    source: str  # "screen", "audio", "rag"
    detected_text: str
    language: Optional[str] = None  # optional code/stack hint (e.g., "react", "typescript", "api")
    is_code: bool = False
    auto_response_threshold: float = 0.7

    def should_auto_respond(self) -> bool:
        """Determine if auto-response should be triggered based on confidence."""
        return self.triggered and self.confidence >= self.auto_response_threshold

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "confidence": self.confidence,
            "trigger_type": self.trigger_type,
            "source": self.source,
            "detected_text": self.detected_text,
            "language": self.language,
            "is_code": self.is_code,
            "auto_respond": self.should_auto_respond(),
        }


class QuestionDetector:
    def __init__(self, config):
        self.enabled = config.get("detection.auto_detect_questions", True)
        self.min_words = config.get("detection.min_words", 3)

        # Question patterns
        self.question_patterns = config.get(
            "detection.question_patterns",
            [
                "what is",
                "explain",
                "how do",
                "why does",
                "describe",
                "implement",
                "write a",
                "solve",
                "define",
                "compare",
                "can you",
                "could you",
                "tell me",
                "what if",
                "how should",
                "do you",
                "are you",
                "should i",
                "would you",
            ],
        )

        # Coding patterns
        self.coding_patterns = [
            "implement",
            "write code",
            "create function",
            "how to build",
            "logic for",
            "coding",
            "algorithm",
            "code a",
            "write a",
            "function",
            "class",
            "debug",
            "fix",
            "error",
            "exception",
            "stack trace",
            "traceback",
            "hook",
            "component",
            "react",
            "next.js",
            "nextjs",
            "react router",
            "redux",
            "zustand",
            "context api",
            "useeffect",
            "usestate",
            "usememo",
            "usecallback",
            "useref",
            "usecontext",
            "jsx",
            "tsx",
            "vite",
            "webpack",
            "babel",
            "eslint",
            "prettier",
            "npm",
            "yarn",
            "pnpm",
            "typescript",
            "tsconfig",
            "tsc",
            "generic",
            "generics",
            "union type",
            "intersection type",
            "type narrowing",
            "rest api",
            "endpoint",
            "route",
            "router",
            "http",
            "https",
            "status code",
            "cors",
            "jwt",
            "oauth",
            "props",
            "state",
            "css",
            "html",
            "tailwind",
            "css grid",
            "flexbox",
            "responsive",
            "media query",
            "browser",
            "frontend",
            "backend",
            "database",
            "query",
            "api",
            "json",
            "dom",
        ]

        # Casual/Engagement patterns
        self.casual_patterns = [
            "hello",
            "hi everyone",
            "good morning",
            "good afternoon",
            "let's start",
            "i agree",
            "that makes sense",
            "thank you",
            "thanks",
            "sure",
            "okay",
            "yeah",
            "good point",
            "agree",
            "how is it",
            "how are you",
            "good to see",
            "let's begin",
        ]

        # Follow-up patterns
        self.followup_patterns = [
            "elaborate",
            "more detail",
            "explain further",
            "that makes sense",
            "got it",
            "i see",
            "understood",
        ]

        # Debounce: prevent rapid triggers
        self._last_trigger_time = 0
        self._debounce_seconds = 3.0
        self._last_text = ""
        self.fragment_buffer = deque(maxlen=4)
        self.question_prefixes = [
            "what ",
            "how ",
            "why ",
            "where ",
            "when ",
            "who ",
            "is ",
            "are ",
            "does ",
            "do ",
            "did ",
            "should ",
            "can ",
            "could ",
            "would ",
            "may ",
            "shall ",
            "which ",
            "whom ",
            "whose ",
            "tell me",
            "explain",
            "describe",
            "can we ",
            "could we ",
            "is there ",
            "are there ",
            "hey can you",
            "i was wondering",
            "do you know",
            "i'd like to know",
            "could you please",
            "would you mind",
            "i have a question",
        ]
        self.fragment_continuations = [
            "and ",
            "or ",
            "but ",
            "also ",
            "else ",
            "then ",
            "so ",
            "because ",
            "for ",
            "to ",
            "with ",
            "about ",
            "that ",
            "this ",
        ]

        # Sensitivity: 1.0 = Max triggers, 0.1 = Strict
        self.sensitivity = config.get("detection.sensitivity", 0.5)

        # Zero-Shot Semantic Intent Anchors
        self._semantic_enabled = config.get("detection.semantic_enabled", True)
        self._semantic_threshold = float(config.get("detection.semantic_threshold", 0.35))
        self._anchors_initialized = False
        self._anchor_q = None
        self._anchor_s = None

        self.auto_response_threshold = config.get(
            "detection.auto_response_threshold", 0.7
        )
        self.current_source = "screen"  # Track current detection source
        self.current_mode = config.get("ai.mode", "general")
        self._mode_weights = {
            "general": {"q": 1.0, "code": 1.0, "casual": 1.0},
            "coding": {"q": 1.0, "code": 1.5, "casual": 0.5},
            "interview": {"q": 1.3, "code": 0.8, "casual": 0.8},
            "meeting": {"q": 1.1, "code": 0.5, "casual": 1.1},
            "exam": {"q": 1.5, "code": 0.5, "casual": 0.3},
        }

        # Interim (live) detection guardrails (Option 2)
        self._interim_enabled = bool(config.get("detection.interim.enabled", True))
        self._interim_min_words = int(config.get("detection.interim.min_words", 4) or 4)
        self._interim_stability_ms = int(config.get("detection.interim.stability_ms", 900) or 900)
        self._interim_min_confidence = float(config.get("detection.interim.min_confidence", 0.75) or 0.75)
        self._interim_cooldown_s = float(config.get("detection.interim.cooldown_s", 6.0) or 6.0)
        self._interim_require_question_signal = bool(
            config.get("detection.interim.require_question_signal", True)
        )
        self._interim_last_key = ""
        self._interim_first_seen_at = 0.0
        self._interim_last_trigger_at = 0.0

    @staticmethod
    def _split_clauses(text: str) -> List[str]:
        parts = re.split(r"[.?!,;:\n]+", text or "")
        return [p.strip() for p in parts if p and p.strip()]

    @staticmethod
    def _code_keywords() -> dict:
        # Centralized markers so detect_with_confidence() can attach hints without
        # re-running stateful detection logic.
        return {
            "react": [
                "react",
                "next.js",
                "nextjs",
                "jsx",
                "tsx",
                "component",
                "props",
                "hook",
                "useState",
                "useEffect",
                "useMemo",
                "useCallback",
                "useRef",
                "useContext",
                "context api",
                "react router",
                "redux",
                "zustand",
            ],
            "web": [
                "html",
                "css",
                "dom",
                "tailwind",
                "css grid",
                "flexbox",
                "responsive",
                "media query",
                "frontend",
                "browser",
                "vite",
                "webpack",
            ],
            "api": [
                "rest api",
                "endpoint",
                "route",
                "router",
                "fetch",
                "axios",
                "graphql",
                "json",
                "http",
                "https",
                "status 200",
                "status code",
                "cors",
                "jwt",
                "oauth",
                "openapi",
                "swagger",
                "websocket",
                "sse",
            ],
            "python": ["def ", "import ", "class ", ".py", "python", "flask", "django", "fastapi"],
            "javascript": [
                "function ",
                "const ",
                "let ",
                "var ",
                "=>",
                ".js",
                "javascript",
                "console.log",
            ],
            "typescript": [
                "interface ",
                "type ",
                ": string",
                ": number",
                ".ts",
                "typescript",
                "tsconfig",
                "tsc",
                "generic",
                "generics",
                "any",
            ],
            "java": ["public class", "void main", "System.out", ".java", "java", "spring"],
            "cpp": ["#include", "std::", "int main", "cout", "c++"],
            "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT ", "CREATE TABLE", "database", "query"],
            "go": ["func ", "package ", "fmt.", "golang", "goroutine"],
            "rust": ["fn ", "let mut", "impl ", "pub fn", "rust", "cargo"],
        }

    def detect_language_hint(self, text: str) -> Optional[str]:
        """Best-effort stack/language hint for routing/logging (pure function)."""
        if not text:
            return None
        lower = text.lower()
        for lang, markers in self._code_keywords().items():
            if any(str(m).lower() in lower for m in markers):
                return lang
        return None

    @staticmethod
    def _norm_key(text: str) -> str:
        t = (text or "").lower().strip()
        t = re.sub(r"\s+", " ", t)
        if t.endswith("?"):
            t = t[:-1].strip()
        return t

    def detect_interim_with_guardrails(self, text: str) -> Optional[str]:
        """Return a stable question clause from partial ASR, or None.

        Guardrails:
        - Must look question-like (prefix/pattern/?), unless disabled
        - Must meet a higher confidence threshold
        - Must be stable for a time window
        - Hard cooldown after firing to prevent duplicate/misfires
        """
        if not self._interim_enabled or not self.enabled:
            return None
        raw = (text or "").strip()
        if not raw:
            return None

        candidate = (self._extract_question_clause(raw) or "").strip()
        if not candidate:
            return None

        lower = candidate.lower().strip()
        words = lower.replace("?", "").split()
        if len(words) < max(self._interim_min_words, 2):
            return None

        has_signal = (
            candidate.strip().endswith("?")
            or any(lower.startswith(prefix) for prefix in self.question_prefixes)
            or any(pattern in lower for pattern in self.question_patterns)
        )
        if self._interim_require_question_signal and not has_signal:
            return None

        confidence = self._calculate_confidence(candidate)
        if confidence < self._interim_min_confidence:
            return None

        now = time.time()
        if self._interim_last_trigger_at and (now - self._interim_last_trigger_at) < self._interim_cooldown_s:
            return None

        key = self._norm_key(candidate)
        if not key:
            return None

        if key != self._interim_last_key:
            self._interim_last_key = key
            self._interim_first_seen_at = now
            return None

        stable_ms = (now - self._interim_first_seen_at) * 1000.0
        if stable_ms < float(self._interim_stability_ms):
            return None

        self._interim_last_trigger_at = now
        return candidate

    def _extract_question_clause(self, text: str) -> str:
        """Extract the most question-like clause from a longer transcript.

        Keeps the query short and cache-friendly while the full transcript can
        still be passed as audio_context/screen_context.
        """
        raw = (text or "").strip()
        if not raw:
            return raw

        # Always start by splitting into clauses once (centralized splitting)
        clauses = self._split_clauses(raw)
        if not clauses:
            return raw

        def _looks_question_like(clause_text: str) -> bool:
            lower = clause_text.lower().strip()
            return any(lower.startswith(p) for p in self.question_prefixes) or any(
                pat in lower for pat in self.question_patterns
            )

        # Priority 1: If we have an explicit question mark anywhere, anchor to the last one.
        if "?" in raw:
            # Find which clause contains the last question mark
            for clause in reversed(clauses):
                if "?" in clause:
                    # Return this clause (already has ?)
                    return clause.strip()
            # Fallback: return last clause with ?
            return clauses[-1].strip() + "?"

        # Priority 2: No question mark - find best question-like clause
        for clause in reversed(clauses):
            if _looks_question_like(clause):
                return clause.strip()

        # Priority 3: Fallback - return first clause (usually contains the core)
        return clauses[0].strip()

    def set_mode(self, mode_id: str):
        """Update detection bias based on current Task mode."""
        self.current_mode = mode_id
        logger.debug(f"Detector Mode set to: {mode_id}")

    def detect_with_confidence(
        self, text: str, source: str = "screen"
    ) -> DetectionResult:
        """P2: Detect with confidence score and provenance tracking."""
        self.current_source = source

        # Call existing detect logic
        result = self.detect(text)

        if result:
            # Calculate confidence based on pattern match
            confidence = self._calculate_confidence(result)
            trigger_type = self._classify_trigger(result)

            # Attach best-effort language hints without re-running stateful detection.
            language = self.detect_language_hint(result) or self.detect_language_hint(text)
            lower_result = result.lower()
            is_code = bool(language) or any(p in lower_result for p in self.coding_patterns)
            if is_code and trigger_type == "question":
                trigger_type = "coding"

            detection = DetectionResult(
                triggered=True,
                confidence=confidence,
                trigger_type=trigger_type,
                source=source,
                detected_text=result,
                language=language,
                is_code=is_code,
                auto_response_threshold=self.auto_response_threshold,
            )

            logger.debug(
                f"QuestionDetector: {trigger_type} detected "
                f"(conf={confidence:.2f}, auto={detection.should_auto_respond()})"
            )
            return detection

        return DetectionResult(
            triggered=False,
            confidence=0.0,
            trigger_type="none",
            source=source,
            detected_text="",
            auto_response_threshold=self.auto_response_threshold,
        )

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for the detection."""
        lower = text.lower()
        words = lower.split()

        score = 0.0

        # P0: Trust Whisper's punctuation. If it explicitly punctuates with a ?, it's a question.
        if text.strip().endswith("?"):
            score += 0.8
        elif "?" in text:
            score += 0.4

        # Strong signals - Check if ANY sentence in the text starts with a prefix
        # This handles multi-sentence queries like "I am working on this. How do I fix it?"
        clauses = self._split_clauses(lower)
        has_prefix = False
        for clause in clauses:
            if any(clause.startswith(prefix) for prefix in self.question_prefixes):
                has_prefix = True
                break
                
        if has_prefix:
            score += 0.4
            
        if any(pattern in lower for pattern in self.question_patterns):
            score += 0.3

        # Context signals
        if (
            any(pattern in lower for pattern in self.coding_patterns)
            and len(words) >= 3
        ):
            score += 0.2
        if any(pattern in lower for pattern in self.followup_patterns):
            score += 0.2

        score = min(score, 1.0)

        # Tier 2: Zero-Shot Semantic Verification for ambiguous cases
        if self._semantic_enabled and 0.2 <= score < self.auto_response_threshold:
            try:
                from ai.embedding_manager import EmbeddingManager
                manager = EmbeddingManager()
                
                if not self._anchors_initialized:
                    # Lazy load anchors
                    self._anchor_q = manager.embed("Can you explain how this works or help me with this problem?")
                    self._anchor_s = manager.embed("I am just talking to my coworker about this issue.")
                    self._anchors_initialized = True
                    
                if self._anchor_q is not None and self._anchor_s is not None:
                    vec = manager.embed(text)
                    if vec is not None:
                        sim_q = float(np.dot(vec, self._anchor_q))
                        sim_s = float(np.dot(vec, self._anchor_s))
                        
                        # If it's mathematically closer to a question AND exceeds the minimum similarity threshold
                        if sim_q > sim_s and sim_q > self._semantic_threshold:
                            logger.debug(f"Semantic Intent Match: sim_q={sim_q:.2f} > sim_s={sim_s:.2f}")
                            # Boost confidence above the auto-response threshold
                            score = max(score, self.auto_response_threshold + 0.05)
            except Exception as e:
                logger.debug(f"Semantic verification failed: {e}")

        return min(score, 1.0)

    def learn_from_query(self, query: str):
        """Learn a successful query prefix to dynamically improve detection."""
        extracted = self._extract_question_clause(query)
        text = (extracted or "").strip().lower()
        if not text:
            return

        looks_question_like = (
            "?" in (query or "")
            or any(text.startswith(prefix) for prefix in self.question_prefixes)
            or any(pattern in text for pattern in self.question_patterns)
        )
        if not looks_question_like:
            return

        words = text.replace("?", "").strip().split()
        if len(words) >= 2:
            prefix = " ".join(words[:2]) + " "
            if prefix not in self.question_prefixes:
                logger.info(f"QuestionDetector: Learned dynamic prefix: '{prefix}'")
                self.question_prefixes.append(prefix)
            
            if len(words) >= 3:
                prefix_3 = " ".join(words[:3]) + " "
                if prefix_3 not in self.question_prefixes:
                    logger.info(f"QuestionDetector: Learned dynamic prefix: '{prefix_3}'")
                    self.question_prefixes.append(prefix_3)

    def _classify_trigger(self, text: str) -> str:
        """Classify the type of trigger."""
        lower = text.lower()
        if any(p in lower for p in self.coding_patterns):
            return "coding"
        if any(p in lower for p in self.followup_patterns):
            return "followup"
        if any(p in lower for p in self.casual_patterns):
            return "casual"
        return "question"

    def detect(self, text: str) -> Optional[str]:
        """Find interaction triggers in text.

        This detector is stateful: it combines adjacent speech fragments to
        handle split questions and continuation fragments.
        """
        if not self.enabled or not text:
            return None

        text = text.strip()
        if len(text) < 2:
            return None

        lower = text.lower()
        words = lower.split()
        self.fragment_buffer.append(text)

        # Try a combined detection over the current buffer first.
        candidate = self._detect_from_buffer()
        if candidate:
            self.fragment_buffer.clear()
            candidate = self._extract_question_clause(candidate)
            if self._is_debounced(candidate):
                return None
            logger.debug(f"QuestionDetector: Triggered from buffer: {candidate!r}")
            self._update_trigger(candidate)
            return candidate

        # Hold short or clearly continued fragments so split questions can be joined.
        if self._should_buffer_fragment(text, lower, words):
            logger.debug(f"QuestionDetector: Buffering fragment: {text!r}")
            return None

        # Final pass on the current segment only.
        if self._is_question_candidate(text, lower, words):
            self.fragment_buffer.clear()
            extracted = self._extract_question_clause(text)
            if self._is_debounced(extracted):
                return None
            logger.debug(f"QuestionDetector: Triggered direct segment: {extracted!r}")
            self._update_trigger(extracted)
            return extracted

        # Keep the fragment if it looks like a continuation phrase.
        if self._looks_like_continuation(text, lower):
            logger.debug(f"QuestionDetector: Waiting for continuation: {text!r}")
            return None

        if len(self.fragment_buffer) == self.fragment_buffer.maxlen:
            self.fragment_buffer.popleft()

        return None

    def _detect_from_buffer(self) -> Optional[str]:
        if not self.fragment_buffer:
            return None

        fragments = list(self.fragment_buffer)
        for size in range(len(fragments), 0, -1):
            candidate = " ".join(fragments[-size:]).strip()
            lower = candidate.lower()
            words = lower.split()
            if self._is_question_candidate(candidate, lower, words):
                return candidate
        return None

    def _is_question_candidate(self, text: str, lower: str, words: List[str]) -> bool:
        if len(words) < 2:
            return False

        weights = self._mode_weights.get(self.current_mode, self._mode_weights["general"])
        score = 0.0

        # 1. Strong Triggers (Auto-Pass or High Score)
        if text.strip().endswith("?"):
            score += 1.0 * weights["q"]
        elif "?" in text:
            score += 0.8 * weights["q"]
            
        starts_with_prefix = any(lower.startswith(prefix) for prefix in self.question_prefixes)
        clause_starts_with_prefix = False
        if not starts_with_prefix:
            for clause in self._split_clauses(lower):
                if any(clause.startswith(prefix) for prefix in self.question_prefixes):
                    clause_starts_with_prefix = True
                    break

        if starts_with_prefix:
            score += 0.6 * weights["q"]
        elif clause_starts_with_prefix:
            # Treat mid-sentence clause-start questions as equally strong.
            score += 0.6 * weights["q"]
        if any(pattern in lower for pattern in self.question_patterns):
            score += 0.5 * weights["q"]

        # 2. Contextual Triggers
        if (
            any(pattern in lower for pattern in self.coding_patterns)
            and len(words) >= 3
        ):
            score += 0.5 * weights["code"]
        if any(pattern in lower for pattern in self.followup_patterns):
            score += 0.4 * weights["q"]
        if (
            any(pattern in lower for pattern in self.casual_patterns)
            and len(words) >= self.min_words
        ):
            score += 0.3 * weights["casual"]

        # 3. Sensitivity Moderation
        # High sensitivity (1.0) -> requires low score (0.2)
        # Low sensitivity (0.1) -> requires high score (0.9)
        required_score = 1.0 - (self.sensitivity * 0.85)

        return score >= required_score

    def _should_buffer_fragment(self, text: str, lower: str, words: List[str]) -> bool:
        if self._has_question_mark(text):
            return False

        if len(words) < self.min_words and any(
            lower.startswith(prefix) for prefix in self.question_prefixes
        ):
            return True

        return self._looks_like_continuation(text, lower)

    def _looks_like_continuation(self, text: str, lower: str) -> bool:
        if self._has_sentence_terminal(text):
            return False
        return any(lower.startswith(marker) for marker in self.fragment_continuations)

    def _has_sentence_terminal(self, text: str) -> bool:
        return bool(re.search(r"[.?!]$", text.strip()))

    def _has_question_mark(self, text: str) -> bool:
        return "?" in text

    def _is_debounced(self, text: str) -> bool:
        now = time.time()
        if (
            text == self._last_text
            and (now - self._last_trigger_time) < self._debounce_seconds
        ):
            logger.debug("QuestionDetector: Debounced rapid trigger")
            return True
        return False

    def _update_trigger(self, text: str):
        """Update debounce state."""
        self._last_text = text
        self._last_trigger_time = time.time()

    def detect_code_question(self, text: str) -> Optional[dict]:
        """Detect coding questions with language hints."""
        if not text:
            return None

        detected_lang = self.detect_language_hint(text)

        # Find the question
        question = self.detect(text)
        if question:
            return {
                "question": question,
                "language": detected_lang,
                "is_code": detected_lang is not None,
            }
        return None
