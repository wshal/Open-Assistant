"""
Incremental Context Assembly — P2.1
Builds context incrementally across turns instead of re-sending the full
screen/audio text every time.  Tracks entities mentioned across queries
and only pushes diffs when the screen changes substantially.
"""

import re
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Simple entity extraction (reuses same pattern set as the cache layer)
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"),          # Proper nouns (two+ words)
    re.compile(r"\b([A-Z][a-zA-Z]{2,})\b"),                         # PascalCase / tech names (React, Django, TypeScript)
    re.compile(r"\b([A-Z]{2,})\b"),                                 # Acronyms (HTTP, API, SQL)
    re.compile(r"`([^`]+)`"),                                       # Code identifiers
    re.compile(r'"([^"]{3,40})"'),                                  # Quoted phrases
    re.compile(r"\b(\w+(?:Error|Exception|Warning))\b"),            # Exception names
    re.compile(r"\b(def\s+\w+|class\s+\w+|import\s+\w+)\b"),       # Python declarations
]

_NOISE_ENTITIES: Set[str] = {
    "The", "This", "That", "There", "These", "Those", "When", "Where",
    "What", "Why", "How", "Which", "Then", "Also", "Just",
}


def _extract_entities(text: str) -> Set[str]:
    """Extract named entities and keywords from text."""
    entities: Set[str] = set()
    for pat in _ENTITY_PATTERNS:
        for match in pat.finditer(text):
            candidate = match.group(1).strip()
            if len(candidate) >= 3 and candidate not in _NOISE_ENTITIES:
                entities.add(candidate)
    return entities


def _similarity(a: str, b: str) -> float:
    """Fast text similarity ratio using SequenceMatcher."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:500], b[:500]).ratio()


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    P2.1: Incremental Context Assembly.

    Maintains a rolling state of:
      - The last screen text (to detect significant diffs)
      - Entities mentioned across the current session
      - The assembled diff-based screen context to inject into prompts

    Usage:
        builder = ContextBuilder(config)
        # When a new query arrives:
        ctx = builder.build(query, current_screen_text)
        # ctx["screen"] will contain only the relevant, non-redundant portion
    """

    def __init__(self, config):
        self._config = config
        self._similarity_threshold: float = float(
            config.get("ai.context.similarity_threshold", 0.85)
        )
        self._max_screen_chars: int = int(
            config.get("ai.context.max_screen_chars", 2000)
        )
        self._entity_ttl_s: float = float(
            config.get("ai.context.entity_ttl_s", 600)  # 10 min
        )

        # State
        self._last_screen: str = ""
        self._last_build_time: float = 0.0
        self._session_entities: Dict[str, float] = {}  # entity -> last_seen_ts
        self._turns: int = 0

        logger.info("ContextBuilder initialised (P2.1)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, query: str, current_screen: str) -> Dict[str, str]:
        """
        Build incremental context for the current query.

        Returns a dict with:
          - "screen"   : optimal screen context (full or diff)
          - "entities" : comma-separated entities seen across session
          - "mode"     : "full" | "diff" | "cached"  (for logging/telemetry)
        """
        now = time.time()
        screen = (current_screen or "").strip()
        query_entities = _extract_entities(query)

        # Update session entities
        for e in query_entities:
            self._session_entities[e] = now
        self._evict_stale_entities(now)

        # Decide: full, diff, or cached
        if not self._last_screen:
            mode = "full"
            screen_ctx = screen[: self._max_screen_chars]
            logger.debug(
                f"[ContextBuilder] Turn {self._turns+1}: first screen context "
                f"({len(screen_ctx)} chars)"
            )
        else:
            sim = _similarity(self._last_screen, screen)
            # Also guard against length-ratio growth: if the new screen is >15%
            # longer than the last, there is new content even if similarity is high.
            len_old = len(self._last_screen) or 1
            len_new = len(screen) or 1
            length_ratio = len_new / len_old
            content_grew = length_ratio > 1.15  # 15% threshold

            if sim >= self._similarity_threshold and not content_grew:
                # Screen is nearly identical — reuse cached, send nothing new
                mode = "cached"
                screen_ctx = self._last_screen[: self._max_screen_chars]
                logger.debug(
                    f"[ContextBuilder] Turn {self._turns+1}: screen unchanged "
                    f"(similarity={sim:.2f}, length_ratio={length_ratio:.2f}), "
                    f"reusing cached context"
                )
            else:
                # Build a diff — only the lines that changed
                mode = "diff"
                screen_ctx = self._build_diff(self._last_screen, screen)
                logger.debug(
                    f"[ContextBuilder] Turn {self._turns+1}: screen changed "
                    f"(similarity={sim:.2f}, length_ratio={length_ratio:.2f}), "
                    f"injecting diff ({len(screen_ctx)} chars)"
                )

        # Update state
        if screen:
            self._last_screen = screen
        self._last_build_time = now
        self._turns += 1

        active_entities = sorted(self._session_entities.keys())
        entity_str = ", ".join(active_entities[:30]) if active_entities else ""

        if entity_str:
            logger.debug(
                f"[ContextBuilder] Session entities tracked: {entity_str[:120]}"
            )

        return {
            "screen": screen_ctx,
            "entities": entity_str,
            "mode": mode,
        }

    def reset(self):
        """Call on new session to clear all state."""
        self._last_screen = ""
        self._last_build_time = 0.0
        self._session_entities.clear()
        self._turns = 0
        logger.info("[ContextBuilder] State reset for new session")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_stale_entities(self, now: float) -> None:
        stale = [
            e for e, ts in self._session_entities.items()
            if now - ts > self._entity_ttl_s
        ]
        for e in stale:
            del self._session_entities[e]
        if stale:
            logger.debug(
                f"[ContextBuilder] Evicted {len(stale)} stale entities"
            )

    def _build_diff(self, old: str, new: str) -> str:
        """
        Extract lines from `new` that are NOT in `old` (the diff).
        Prepends a '--- screen diff ---' marker so the AI knows it's a delta.
        """
        old_lines: Set[str] = {l.strip() for l in old.splitlines() if l.strip()}
        new_lines = [l for l in new.splitlines() if l.strip()]
        added = [l for l in new_lines if l.strip() not in old_lines]

        if not added:
            # No textual diff found — send the full new context
            return new[: self._max_screen_chars]

        diff_text = "\n".join(added)
        header = "[Screen update — only showing changed lines]\n"
        result = (header + diff_text)[: self._max_screen_chars]
        return result


# ---------------------------------------------------------------------------
# P3.4: Attention-Based Context Pruner
# ---------------------------------------------------------------------------

class ContextPruner:
    """
    P3.4: Attention-Based Context Pruning.

    Splits the screen text into logical blocks (paragraphs / code sections)
    and scores each block against the user query using keyword overlap and
    token density. Drops blocks whose relevance falls below a threshold,
    keeping only the most relevant sections.

    This extends the *effective* context window — instead of blindly
    truncating at N chars, we send N chars of the most useful content.
    """

    def __init__(self, config):
        self._config = config
        self._enabled: bool = bool(config.get("ai.pruner.enabled", True))
        self._min_block_lines: int = int(config.get("ai.pruner.min_block_lines", 2))
        self._top_k_blocks: int = int(config.get("ai.pruner.top_k_blocks", 8))
        self._min_score: float = float(config.get("ai.pruner.min_score", 0.05))

        if self._enabled:
            logger.info(
                f"[P3.4 Pruner] Initialised — top_k={self._top_k_blocks}, "
                f"min_score={self._min_score}"
            )
        else:
            logger.info("[P3.4 Pruner] Disabled via config")

    def prune(self, screen_text: str, query: str) -> str:
        """
        Return a pruned version of screen_text keeping only the blocks
        most relevant to query.  Falls back to original text on failure.
        """
        if not self._enabled or not screen_text or not query:
            return screen_text

        try:
            blocks = self._split_blocks(screen_text)
            if len(blocks) <= 2:
                # Too small to worth pruning
                return screen_text

            query_tokens = self._tokenize(query)
            if not query_tokens:
                return screen_text

            scored = []
            for block in blocks:
                score = self._score(block, query_tokens)
                scored.append((score, block))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Keep top-k blocks that meet the minimum relevance threshold
            kept = [blk for score, blk in scored[:self._top_k_blocks] if score >= self._min_score]

            if not kept:
                logger.debug("[P3.4 Pruner] No blocks met min_score — returning full screen")
                return screen_text

            # Preserve original order for readability
            kept_set = set(id(b) for b in kept)
            ordered = [blk for blk in blocks if id(blk) in kept_set]

            pruned = "\n\n".join(ordered)
            logger.debug(
                f"[P3.4 Pruner] Pruned {len(blocks)} blocks → {len(kept)} kept "
                f"({len(pruned)}/{len(screen_text)} chars, "
                f"top_score={scored[0][0]:.2f})"
            )
            return pruned

        except Exception as e:
            logger.warning(f"[P3.4 Pruner] Pruning failed (returning original): {e}")
            return screen_text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split_blocks(self, text: str) -> List[str]:
        """Split screen text into logical blocks on blank lines or section markers."""
        # Split on blank lines first
        raw = re.split(r"\n{2,}", text)
        blocks = []
        for chunk in raw:
            lines = [l for l in chunk.splitlines() if l.strip()]
            if len(lines) >= self._min_block_lines:
                blocks.append(chunk.strip())
            elif lines:
                # Single-line blocks — group with previous block if possible
                if blocks:
                    blocks[-1] = blocks[-1] + "\n" + chunk.strip()
                else:
                    blocks.append(chunk.strip())
        return blocks if blocks else [text]

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Lowercase word tokens, strip punctuation, remove very short words."""
        tokens = re.findall(r"\b[a-zA-Z_]\w{2,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "this", "that", "with", "from", "are",
            "was", "has", "have", "can", "will", "what", "how", "why",
        }
        return {t for t in tokens if t not in stopwords}

    def _score(self, block: str, query_tokens: Set[str]) -> float:
        """Score a block by keyword overlap with query tokens."""
        block_tokens = self._tokenize(block)
        if not block_tokens:
            return 0.0
        overlap = len(query_tokens & block_tokens)
        # Normalise by sqrt of block length to prefer denser matches
        density = overlap / (len(block_tokens) ** 0.5)
        return round(density, 4)