"""Auto-detect questions on screen with interaction triggers.

Full-Auto Meeting Interaction:
- Questions (what is, how do, why does)
- Coding (implement, code, function, algorithm)
- Casual (hello, agree, think, thanks)
- Follow-ups (elaborate, that makes sense)
"""

import re
import time
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
            "hook",
            "component",
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
            confidence = self._calculate_confidence(text)
            trigger_type = self._classify_trigger(result)

            detection = DetectionResult(
                triggered=True,
                confidence=confidence,
                trigger_type=trigger_type,
                source=source,
                detected_text=result,
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

        # Strong signals
        if "?" in text:
            score += 0.3
        if any(lower.startswith(prefix) for prefix in self.question_prefixes):
            score += 0.2
        if any(pattern in lower for pattern in self.question_patterns):
            score += 0.2

        # Context signals
        if (
            any(pattern in lower for pattern in self.coding_patterns)
            and len(words) >= 3
        ):
            score += 0.15
        if any(pattern in lower for pattern in self.followup_patterns):
            score += 0.1

        # Normalize by sensitivity (lower sensitivity = higher confidence required)
        adjusted = score * (0.5 + self.sensitivity * 0.5)
        return min(adjusted, 1.0)

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
            if self._is_debounced(text):
                return None
            logger.debug(f"QuestionDetector: Triggered direct segment: {text!r}")
            self._update_trigger(text)
            return text

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
        if "?" in text:
            score += 0.8 * weights["q"]
        if any(lower.startswith(prefix) for prefix in self.question_prefixes):
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

        code_keywords = {
            "python": ["def ", "import ", "class ", ".py", "python"],
            "javascript": [
                "function ",
                "const ",
                "let ",
                "var ",
                "=>",
                ".js",
                "javascript",
            ],
            "typescript": [
                "interface ",
                "type ",
                ": string",
                ": number",
                ".ts",
                "typescript",
            ],
            "java": ["public class", "void main", "System.out", ".java", "java"],
            "cpp": ["#include", "std::", "int main", "cout", "c++"],
            "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT ", "CREATE TABLE"],
            "go": ["func ", "package ", "fmt.", "golang"],
            "rust": ["fn ", "let mut", "impl ", "pub fn", "rust"],
        }

        detected_lang = None
        lower = text.lower()
        for lang, markers in code_keywords.items():
            if any(m.lower() in lower for m in markers):
                detected_lang = lang
                break

        # Find the question
        question = self.detect(text)
        if question:
            return {
                "question": question,
                "language": detected_lang,
                "is_code": detected_lang is not None,
            }
        return None
