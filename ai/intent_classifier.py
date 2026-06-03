"""Embedding-based intent classifier for Auto Mode speech classification.

Uses the shared EmbeddingManager (BAAI/bge-small-en-v1.5) to classify incoming
speech transcripts into intent categories via cosine similarity against
pre-embedded reference exemplars.

Design principles:
  - AUGMENTS regex, does not replace it.  Regex stays as the fast first-pass;
    embeddings break ties and handle novel phrasing.
  - Lazy-initialized: reference embeddings are computed on first call, not at
    import time, so startup cost is zero if Auto Mode is never used.
  - Thread-safe: uses a lock around the one-time init of reference vectors.
  - Self-improving: learns from live session outcomes with strict quality gates.
  - Self-healing: age-based decay + compaction prevent corruption over time.
  - Builtin-dominant: hardcoded exemplars are always weighted higher than
    learned ones, so bad learned data can never fully corrupt the system.

Intent categories:
  - QUESTION:  An actionable question or command the user wants answered.
  - SETUP:    A preamble / context-setting statement (should be buffered).
  - GREETING: A greeting or acknowledgement (short response or ignore).
  - FOLLOWUP: A continuation of a prior question (should be merged).
"""

from __future__ import annotations

import json
import logging
import threading
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ai.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

# Paths
_DATA_DIR = Path("./data")
_LEARNED_INTENTS_FILE = _DATA_DIR / "learned_intents.jsonl"

# Learning configuration
VALID_INTENTS = {"question", "setup", "greeting", "followup"}

# Quality gates for learning
_MIN_WORDS = 5                # Minimum words to accept as exemplar
_MIN_ALPHA_RATIO = 0.60       # At least 60% of chars must be alphabetic
_MAX_FILLER_RATIO = 0.30      # At most 30% filler words
_MAX_EXEMPLARS_PER_CAT = 100  # Cap per category to prevent unbounded growth
_DIVERSITY_THRESHOLD = 0.92   # Reject if cosine sim to ANY existing > this
_DEDUPE_EXACT_THRESHOLD = 0.98  # Near-exact duplicate rejection
_CONFIDENCE_MARGIN = 0.03     # Both regex and embeddings must agree with margin

# Rate limiting
_MAX_LEARNS_PER_SESSION = 10  # Max new exemplars per app session
_LEARN_COOLDOWN_S = 30.0      # Minimum seconds between learning same category

# Decay
_MAX_AGE_DAYS = 30            # Remove learned exemplars older than this
_COMPACTION_KEEP = 80         # During compaction, keep top N per category

# Weighting
_BUILTIN_WEIGHT = 2.0         # Builtin exemplars count double in centroid calc

# Filler words that indicate noisy/garbled audio
_FILLER_WORDS_SINGLE = frozenset({
    "uh", "um", "umm", "uhh", "ah", "ahh", "hmm", "hm", "er", "erm",
    "like", "yeah", "yep", "yah", "ya", "ok", "okay", "right", "so",
    "well", "just", "actually", "basically", "literally", "honestly",
})
_FILLER_PHRASES = (
    "you know", "i mean", "sort of", "kind of",
)

# Reference exemplars per intent category

_QUESTION_EXEMPLARS = [
    "What is the difference between CSS Grid and Flexbox?",
    "How does React handle state management with hooks?",
    "Why would you choose a NoSQL database over a relational one?",
    "When should you use memoization in React?",
    "Where does the borrow checker enforce ownership in Rust?",
    "Which caching strategy is best for a high-traffic news website?",
    "Who is responsible for rotating JWT tokens in a microservices setup?",
    "Can you explain the difference between TCP and UDP?",
    "Could you walk me through the authentication flow?",
    "Would you recommend using GraphQL over REST for mobile APIs?",
    "Should we use horizontal or vertical scaling for this database?",
    "Do you prefer composition over inheritance in object-oriented design?",
    "Is it safe to store JWT tokens in browser local storage?",
    "Explain how you would implement a caching layer.",
    "Describe the tradeoffs between monolith and microservices.",
    "Compare the time complexity of quicksort and mergesort.",
    "Tell me about the SOLID principles in software design.",
    "Define what a closure is in JavaScript.",
    "Write a function that reverses a linked list.",
    "What is a closure in JavaScript?",
    "How do I merge two dictionaries in Python?",
    "What is React Router and how does it work?",
    "Explain lifetimes in Rust briefly.",
    "What is the expected time complexity in big O notation and what is the worst case scenario if the tree is unbalanced?",
    "Could you explain how JWT tokens work and what are the potential security risks?",
]

_SETUP_EXEMPLARS = [
    "Let's pivot to some CSS basics.",
    "Alright, let's talk about scaling.",
    "So, moving on to the next topic.",
    "Now let's discuss system design.",
    "Okay, next up we have React.",
    "Let's shift gears and talk about databases.",
    "Imagine we are designing a new public-facing API for our mobile app.",
    "Suppose you have a distributed system with eventual consistency.",
    "Consider a scenario where the database is under heavy read traffic.",
    "Picture a large-scale e-commerce platform during Black Friday.",
    "I was looking at your resume and I see you've used React quite a bit.",
    "I notice you have experience with Kubernetes.",
    "Looking at your background, you've worked with microservices.",
    "Based on your experience with cloud infrastructure.",
    "In a typical sprint, disagreements can happen.",
    "When building a secure REST API, authentication is critical.",
    "Code reviews are a big part of our engineering culture here.",
    "A lot of developers get confused between CSS Grid and Flexbox.",
    "Most teams struggle with state management in large React apps.",
    "We are building a high-traffic news website.",
    "Our system currently handles about ten thousand requests per second.",
    "The frontend team has been using TypeScript for the past two years.",
    "I'd like to ask a quick algorithmic question.",
]

_GREETING_EXEMPLARS = [
    "Hello there, can you help me with some coding?",
    "Hey, how are you doing?",
    "Hi, I need some help.",
    "Good morning!",
    "Thanks for joining us today.",
    "Welcome to the interview.",
    "Nice to meet you.",
    "Can you hear me okay?",
    "Let me know when you're ready.",
    "Sure, go ahead.",
    "Okay, got it.",
    "That makes sense.",
    "Right, I understand.",
    "Great, thanks.",
]

_FOLLOWUP_EXEMPLARS = [
    "How did you approach the situation and what was the outcome?",
    "Can you give a specific example?",
    "What happened after that?",
    "And what was the result?",
    "Could you elaborate on that?",
    "Why did you choose that approach?",
    "How did the team react?",
    "What would you do differently?",
    "Besides just checking for syntax errors.",
    "And what about the performance implications?",
    "Can you walk me through the implementation details?",
    "What were the tradeoffs?",
]

_BUILTIN_EXEMPLARS = {
    "question": _QUESTION_EXEMPLARS,
    "setup": _SETUP_EXEMPLARS,
    "greeting": _GREETING_EXEMPLARS,
    "followup": _FOLLOWUP_EXEMPLARS,
}


# Text quality assessment

def _text_quality_ok(text: str) -> tuple[bool, str]:
    """Check if text is clean enough to learn from.

    Returns (ok, reason); reason is empty when ok=True.
    Rejects:
      - Too few real words
      - Too many filler words (STT noise)
      - Too few alphabetic characters (garbled audio)
      - Repeated word patterns (STT stuttering)
      - System/UI messages that leaked through
    """
    if not text or not text.strip():
        return False, "empty"

    words = text.strip().split()
    word_count = len(words)

    if word_count < _MIN_WORDS:
        return False, f"too short ({word_count} words)"

    # Check alphabetic ratio; garbled audio has lots of punctuation/symbols
    alpha_chars = sum(1 for c in text if c.isalpha())
    total_chars = max(len(text.replace(" ", "")), 1)
    alpha_ratio = alpha_chars / total_chars
    if alpha_ratio < _MIN_ALPHA_RATIO:
        return False, f"low alpha ratio ({alpha_ratio:.2f})"

    # Check filler word ratio (single words + multi-word phrases)
    lower_words = [w.lower().strip(".,!?;:-_'\"") for w in words]
    lower_joined = " ".join(lower_words)
    filler_count = sum(1 for w in lower_words if w in _FILLER_WORDS_SINGLE)
    # Count multi-word filler phrases (each counts as the number of words it contains)
    for phrase in _FILLER_PHRASES:
        filler_count += lower_joined.count(phrase) * len(phrase.split())
    filler_ratio = filler_count / max(word_count, 1)
    if filler_ratio > _MAX_FILLER_RATIO:
        return False, f"too many fillers ({filler_ratio:.0%})"

    # Check for stuttering/repetition (STT error pattern)
    if word_count >= 6:
        windows = [tuple(lower_words[i:i+3]) for i in range(len(lower_words) - 2)]
        if len(windows) > len(set(windows)):
            return False, "repetitive pattern (STT stutter)"

    # Reject system/UI messages that leaked through
    system_patterns = (
        "click-through enabled",
        "click-through disabled",
        "press ctrl+m to restore interaction",
        "connection lost",
        "reconnecting",
    )
    lower = text.lower()
    if any(p in lower for p in system_patterns):
        return False, "system/UI message"

    # Check unique word ratio; very repetitive text is suspicious
    unique_ratio = len(set(lower_words)) / max(word_count, 1)
    if word_count >= 8 and unique_ratio < 0.50:
        return False, f"low unique word ratio ({unique_ratio:.2f})"

    return True, ""


@dataclass(frozen=True)
class IntentScores:
    """Cosine similarity scores for each intent category (0.0 to 1.0)."""
    question: float
    setup: float
    greeting: float
    followup: float

    @property
    def best_intent(self) -> str:
        """Return the intent with the highest score."""
        scores = {
            "question": self.question,
            "setup": self.setup,
            "greeting": self.greeting,
            "followup": self.followup,
        }
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    @property
    def best_score(self) -> float:
        return max(self.question, self.setup, self.greeting, self.followup)

    @property
    def is_confident(self) -> bool:
        """True when the best intent has a meaningful margin over the runner-up."""
        scores = sorted(
            [self.question, self.setup, self.greeting, self.followup],
            reverse=True,
        )
        return (scores[0] - scores[1]) >= _CONFIDENCE_MARGIN

    def __repr__(self) -> str:
        return (
            f"IntentScores(question={self.question:.3f}, setup={self.setup:.3f}, "
            f"greeting={self.greeting:.3f}, followup={self.followup:.3f} "
            f"-> {self.best_intent})"
        )


@dataclass
class _LearnedExemplar:
    """A single learned exemplar persisted to disk."""
    text: str
    intent: str
    confidence: float
    learned_at: float  # Unix timestamp
    source: str = "auto"  # "auto" = system-labeled, "manual" = user-labeled


class IntentClassifier:
    """Embedding-based intent classifier using cosine similarity centroids.

    Self-improving with safeguards:
      - Text quality filter rejects garbled/noisy STT output
      - Rate limiting prevents flood from bad sessions
      - Age-based decay removes stale exemplars on startup
      - Compaction keeps only the most diverse entries
      - Builtin exemplars always weighted 2x over learned ones
      - Config toggle: learning is off by default

    Usage::

        classifier = IntentClassifier()
        scores = classifier.classify("What is a closure in JavaScript?")

        # After outcome observed (only when learning is enabled):
        classifier.learn("What is a closure in JavaScript?", "question",
                         confidence=0.85)
    """

    _instances: dict[type, object] = {}
    _lock = threading.Lock()

    def __new__(cls) -> "IntentClassifier":
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[cls] = instance
            return cls._instances[cls] # type: ignore[return-value]

    # Initialization

    def _ensure_initialized(self) -> bool:
        """Lazy-init: compute reference centroids on first classify() call."""
        if self._initialized:
            return self._centroids is not None

        with self._lock:
            if self._initialized:
                return self._centroids is not None

            try:
                from ai.embedding_manager import EmbeddingManager
                mgr = EmbeddingManager()
                mgr.warmup()

                # Per-category vector stores
                self._builtin_vectors: dict[str, list[np.ndarray]] = {
                    k: [] for k in VALID_INTENTS
                }
                self._learned_vectors: dict[str, list[np.ndarray]] = {
                    k: [] for k in VALID_INTENTS
                }
                self._learned_texts: dict[str, list[str]] = {
                    k: [] for k in VALID_INTENTS
                }
                self._all_texts: dict[str, list[str]] = {
                    k: [] for k in VALID_INTENTS
                }
                self._centroids: dict[str, np.ndarray] = {}
                self._learned_count: dict[str, int] = {k: 0 for k in VALID_INTENTS}

                # H-2: Mutation lock guards every read/write of the learned
                # vector/text/centroid state. Separate from cls._lock which only
                # serialises the one-time singleton bootstrap. RLock so
                # _recompute_centroids() (called under the lock) can be invoked
                # from already-locked sites.
                self._mutation_lock = threading.RLock()

                # Session-level rate limiting
                self._session_learn_count = 0
                self._last_learn_time: dict[str, float] = {k: 0.0 for k in VALID_INTENTS}

                # Embed builtin exemplars
                for name, exemplars in _BUILTIN_EXEMPLARS.items():
                    for text in exemplars:
                        vec = mgr.embed(text)
                        if vec is not None:
                            self._builtin_vectors[name].append(vec)
                            self._all_texts[name].append(text)

                # Load + decay + compact learned exemplars
                learned_count = self._load_and_compact_learned(mgr)

                # Compute initial centroids (with builtin weighting)
                self._recompute_centroids()

                self._mgr = mgr
                self._initialized = True

                total_builtin = sum(len(v) for v in _BUILTIN_EXEMPLARS.values())
                total = sum(
                    len(self._builtin_vectors[k]) + len(self._learned_vectors[k])
                    for k in VALID_INTENTS
                )
                logger.info(
                    "IntentClassifier ready: %d categories, %d builtin + %d learned = %d total",
                    len(self._centroids), total_builtin, learned_count, total,
                )
                return True

            except Exception as e:
                logger.warning("IntentClassifier init failed: %s", e)
                self._centroids = None
                self._mgr = None
                self._initialized = True
                return False

    def _recompute_centroids(self) -> None:
        """Recompute category centroids with builtin-dominant weighting.

        Builtin exemplars are weighted 2x compared to learned ones.
        This ensures bad learned data can never fully corrupt the system:
        the hardcoded exemplars always anchor the centroid.
        """
        temp_centroids = {}
        for name in VALID_INTENTS:
            builtin_vecs = self._builtin_vectors.get(name, [])
            learned_vecs = self._learned_vectors.get(name, [])

            if not builtin_vecs and not learned_vecs:
                continue

            # Weight builtins 2x by repeating them in the average
            all_vecs = []
            for v in builtin_vecs:
                all_vecs.append(v * _BUILTIN_WEIGHT)
            for v in learned_vecs:
                all_vecs.append(v)

            if all_vecs:
                centroid = np.mean(all_vecs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                temp_centroids[name] = centroid
        with self._mutation_lock:
            self._centroids = temp_centroids

    # Persistence + Decay + Compaction

    def _load_and_compact_learned(self, mgr: "EmbeddingManager") -> int:
        """Load learned exemplars, remove expired ones, compact if needed.

        Decay: entries older than _MAX_AGE_DAYS are discarded.
        Compaction: if a category has more than _COMPACTION_KEEP entries,
        keep only the most diverse ones (highest min-distance to others).
        """
        if not _LEARNED_INTENTS_FILE.exists():
            return 0

        now = _time.time()
        max_age_s = _MAX_AGE_DAYS * 86400
        entries_by_cat: dict[str, list[dict]] = {k: [] for k in VALID_INTENTS}
        discarded_expired = 0

        try:
            with open(_LEARNED_INTENTS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        intent = entry.get("intent", "")
                        learned_at = float(entry.get("learned_at", 0))
                        if intent not in VALID_INTENTS:
                            continue

                        # Decay: skip entries older than max age
                        age = now - learned_at
                        if age > max_age_s:
                            discarded_expired += 1
                            continue

                        entries_by_cat[intent].append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.warning("Failed to read learned intents: %s", e)
            return 0

        if discarded_expired:
            logger.info(
                "IntentClassifier: expired %d learned exemplars (>%d days old)",
                discarded_expired, _MAX_AGE_DAYS,
            )

        # Embed and optionally compact each category
        loaded = 0
        needs_rewrite = discarded_expired > 0
        kept_entries: list[dict] = []

        for cat, entries in entries_by_cat.items():
            if not entries:
                continue

            # Compact if over limit: keep only the most diverse
            if len(entries) > _COMPACTION_KEEP:
                entries = self._compact_entries(entries, mgr, cat)
                needs_rewrite = True

            for entry in entries:
                text = entry.get("text", "").strip()
                if not text:
                    continue
                # Re-validate quality on load
                ok, reason = _text_quality_ok(text)
                if not ok:
                    needs_rewrite = True
                    continue

                vec = mgr.embed(text)
                if vec is not None:
                    self._learned_vectors[cat].append(vec)
                    self._learned_texts[cat].append(text)
                    self._all_texts[cat].append(text)
                    self._learned_count[cat] += 1
                    loaded += 1
                    kept_entries.append(entry)

        # Rewrite file if we discarded anything (compaction / decay)
        if needs_rewrite:
            self._rewrite_learned_file(kept_entries)

        if loaded:
            logger.info("Loaded %d learned exemplars from %s", loaded, _LEARNED_INTENTS_FILE)
        return loaded

    def _compact_entries(
        self, entries: list[dict], mgr: "EmbeddingManager", cat: str
    ) -> list[dict]:
        """Keep the most diverse entries when a category exceeds the cap.

        Strategy: greedily select entries that maximize minimum distance
        to already-selected entries. This keeps the exemplar set spread
        across the semantic space instead of clustering in one area.
        """
        # Embed all entries
        embedded = []
        for entry in entries:
            text = entry.get("text", "").strip()
            if not text:
                continue
            vec = mgr.embed(text)
            if vec is not None:
                embedded.append((entry, vec))

        if len(embedded) <= _COMPACTION_KEEP:
            return [e for e, _ in embedded]

        # Sort by confidence descending, then by recency
        embedded.sort(
            key=lambda x: (
                -float(x[0].get("confidence", 0)),
                -float(x[0].get("learned_at", 0)),
            )
        )

        n = len(embedded)
        selected_idx_set: set[int] = {0}  # Start with highest confidence
        
        # min_distances[i] will store the minimum distance from candidate i to the selected set.
        # Initially, the selected set only contains index 0.
        min_distances = [1.0 - float(np.dot(embedded[i][1], embedded[0][1])) for i in range(n)]

        while len(selected_idx_set) < _COMPACTION_KEEP and len(selected_idx_set) < n:
            best_idx = -1
            best_min_dist = -1.0
            for i in range(n):
                if i in selected_idx_set:
                    continue
                if min_distances[i] > best_min_dist:
                    best_min_dist = min_distances[i]
                    best_idx = i
            
            if best_idx >= 0:
                selected_idx_set.add(best_idx)
                # Update min_distances incrementally with the newly selected element
                best_vec = embedded[best_idx][1]
                for i in range(n):
                    if i not in selected_idx_set:
                        dist_to_new = 1.0 - float(np.dot(embedded[i][1], best_vec))
                        if dist_to_new < min_distances[i]:
                            min_distances[i] = dist_to_new
            else:
                break

        compacted = [embedded[i][0] for i in sorted(selected_idx_set)]
        logger.info(
            "IntentClassifier: compacted %r from %d to %d exemplars",
            cat, len(embedded), len(compacted),
        )
        return compacted

    def _rewrite_learned_file(self, entries: list[dict]) -> None:
        """Atomically rewrite the learned intents file (compaction/decay)."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = _LEARNED_INTENTS_FILE.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # Atomic replace
            tmp_path.replace(_LEARNED_INTENTS_FILE)
            logger.info(
                "IntentClassifier: rewrote learned intents (%d entries)", len(entries)
            )
        except Exception as e:
            logger.warning("Failed to rewrite learned intents: %s", e)

    def _persist_exemplar(self, exemplar: _LearnedExemplar) -> None:
        """Append a single learned exemplar to disk (JSONL format).

        H7 FIX: Serialize file writes under _mutation_lock to prevent two
        concurrent learn() calls from interleaving JSONL lines.
        """
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "text": exemplar.text,
                "intent": exemplar.intent,
                "confidence": round(exemplar.confidence, 4),
                "learned_at": exemplar.learned_at,
                "source": exemplar.source,
            }
            with self._mutation_lock:
                with open(_LEARNED_INTENTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to persist learned exemplar: %s", e)

    # Learning

    def learn(
        self,
        text: str,
        intent: str,
        *,
        confidence: float = 0.0,
        source: str = "auto",
    ) -> bool:
        """Learn a new exemplar from a live session outcome.

        Quality gates applied (in order):
          1. Text quality check (garbled/noisy/filler rejection)
          2. Valid intent category
          3. Session rate limit (max 10 per session)
          4. Category cooldown (30s between same category)
          5. Per-category cap not exceeded
          6. Not a near-exact duplicate
          7. Semantically diverse enough

        Args:
            text: The transcript text to learn from.
            intent: The observed intent label ("question", "setup", etc.)
            confidence: How confident the system was (0-1).
            source: "auto" for system-labeled, "manual" for user corrections.

        Returns:
            True if the exemplar was accepted and persisted, False if rejected.
        """
        if not self._ensure_initialized():
            return False

        text = (text or "").strip()
        intent = (intent or "").lower().strip()

        # Gate 1: Text quality
        ok, reason = _text_quality_ok(text)
        if not ok:
            logger.debug("learn() rejected: %s: %r", reason, text[:60])
            return False

        # Gate 2: Valid intent
        if intent not in VALID_INTENTS:
            logger.debug("learn() rejected: invalid intent %r", intent)
            return False

        # Gate 3: Session rate limit
        if self._session_learn_count >= _MAX_LEARNS_PER_SESSION:
            logger.debug("learn() rejected: session limit reached (%d)", _MAX_LEARNS_PER_SESSION)
            return False

        # Gate 4: Category cooldown
        now = _time.time()
        last_learn = self._last_learn_time.get(intent, 0.0)
        if (now - last_learn) < _LEARN_COOLDOWN_S:
            logger.debug("learn() rejected: category %r cooldown (%.0fs remaining)",
                         intent, _LEARN_COOLDOWN_S - (now - last_learn))
            return False

        # Gate 5: Per-category cap
        if self._learned_count.get(intent, 0) >= _MAX_EXEMPLARS_PER_CAT:
            logger.debug("learn() rejected: category %r at capacity (%d)", intent, _MAX_EXEMPLARS_PER_CAT)
            return False

        # Embed the candidate
        vec = self._mgr.embed(text)
        if vec is None:
            return False

        # H-2/H-3: All remaining gates inspect mutable state, then mutate it as
        # a group. Hold the mutation lock for the whole accept block so a
        # concurrent classify() / learn() / reset_learned() cannot observe a
        # half-updated triplet (vectors, texts, all_texts) or a stale centroid.
        with self._mutation_lock:
            # Gate 6: Exact/near-duplicate check (across ALL categories)
            for cat_name, cat_texts in self._all_texts.items():
                if text.lower() in [t.lower() for t in cat_texts]:
                    logger.debug("learn() rejected: exact duplicate in %r", cat_name)
                    return False

            # Gate 7: Semantic diversity; reject if too similar to existing
            existing_vecs = (
                list(self._builtin_vectors.get(intent, []))
                + list(self._learned_vectors.get(intent, []))
            )
            if existing_vecs:
                similarities = [float(np.dot(vec, ev)) for ev in existing_vecs]
                max_sim = max(similarities) if similarities else 0.0

                if max_sim >= _DEDUPE_EXACT_THRESHOLD:
                    logger.debug("learn() rejected: near-duplicate (sim=%.3f)", max_sim)
                    return False
                if max_sim >= _DIVERSITY_THRESHOLD:
                    logger.debug("learn() rejected: insufficient diversity (sim=%.3f)", max_sim)
                    return False

            # All gates passed; accept
            self._learned_vectors[intent].append(vec)
            self._learned_texts[intent].append(text)
            self._all_texts[intent].append(text)
            self._learned_count[intent] = self._learned_count.get(intent, 0) + 1
            self._session_learn_count += 1
            self._last_learn_time[intent] = now

            # H-3: Drift invariant — vectors and texts must move together.
            if len(self._learned_vectors[intent]) != len(self._learned_texts[intent]):
                logger.error(
                    "IntentClassifier drift: %r vectors=%d texts=%d (rolling back)",
                    intent,
                    len(self._learned_vectors[intent]),
                    len(self._learned_texts[intent]),
                )
                # Roll back the last append on the longer side.
                while len(self._learned_vectors[intent]) > len(self._learned_texts[intent]):
                    self._learned_vectors[intent].pop()
                while len(self._learned_texts[intent]) > len(self._learned_vectors[intent]):
                    self._learned_texts[intent].pop()
                return False

            self._recompute_centroids()

            total = sum(
                len(self._builtin_vectors[k]) + len(self._learned_vectors[k])
                for k in VALID_INTENTS
            )

        exemplar = _LearnedExemplar(
            text=text, intent=intent, confidence=confidence,
            learned_at=now, source=source,
        )
        self._persist_exemplar(exemplar)

        logger.info(
            "Learned new %r exemplar (confidence=%.2f, session=%d/%d, total=%d): %r",
            intent, confidence, self._session_learn_count, _MAX_LEARNS_PER_SESSION,
            total, text[:80],
        )
        return True

    # Management

    def reset_learned(self) -> None:
        """Wipe all learned exemplars and start fresh.

        Removes the persisted file and clears in-memory learned data.
        Builtin exemplars are preserved; the system is never worse than
        the hardcoded baseline.
        """
        if not self._ensure_initialized():
            return

        # H-2: Reset mutates the same fields that learn() touches; serialise.
        with self._mutation_lock:
            # Clear in-memory learned data
            for cat in VALID_INTENTS:
                self._learned_vectors[cat] = []
                self._learned_texts[cat] = []
                self._learned_count[cat] = 0
                # Rebuild _all_texts from builtins only
                self._all_texts[cat] = list(_BUILTIN_EXEMPLARS.get(cat, []))

            self._session_learn_count = 0
            self._last_learn_time = {k: 0.0 for k in VALID_INTENTS}

            # Recompute centroids (builtin-only)
            self._recompute_centroids()

        # Remove the file
        try:
            if _LEARNED_INTENTS_FILE.exists():
                _LEARNED_INTENTS_FILE.unlink()
            logger.info("IntentClassifier: all learned exemplars reset")
        except Exception as e:
            logger.warning("Failed to delete learned intents file: %s", e)

    def get_stats(self) -> dict:
        """Return statistics about the current exemplar set."""
        if not self._ensure_initialized():
            return {"available": False}

        stats = {"available": True, "categories": {}}
        with self._mutation_lock:
            session_learns = self._session_learn_count
            learned_counts = dict(self._learned_count)
            builtin_counts = {name: len(self._builtin_vectors.get(name, [])) for name in VALID_INTENTS}

        for name in VALID_INTENTS:
            builtin = builtin_counts[name]
            learned = learned_counts.get(name, 0)
            stats["categories"][name] = {
                "builtin": builtin,
                "learned": learned,
                "total": builtin + learned,
                "capacity_remaining": _MAX_EXEMPLARS_PER_CAT - learned,
            }
        stats["total_exemplars"] = sum(
            d["total"] for d in stats["categories"].values()
        )
        stats["session_learns"] = session_learns
        stats["session_limit"] = _MAX_LEARNS_PER_SESSION
        stats["max_age_days"] = _MAX_AGE_DAYS
        return stats

    # Classification

    def classify(self, text: str) -> IntentScores | None:
        """Classify a text into intent categories.

        Returns IntentScores with cosine similarities, or None if embeddings
        are unavailable.
        """
        if not text or not text.strip():
            return None

        if not self._ensure_initialized():
            return None

        vec = self._mgr.embed(text.strip())
        if vec is None:
            return None

        # H-2: Snapshot centroids under the mutation lock to avoid reading a
        # half-updated dict mid-learn(). The dot products themselves are
        # computed outside the lock to keep the critical section minimal.
        with self._mutation_lock:
            centroids_snapshot = dict(self._centroids)

        scores = {}
        for name, centroid in centroids_snapshot.items():
            sim = float(np.dot(vec, centroid))
            scores[name] = max(0.0, sim)

        return IntentScores(
            question=scores.get("question", 0.0),
            setup=scores.get("setup", 0.0),
            greeting=scores.get("greeting", 0.0),
            followup=scores.get("followup", 0.0),
        )

    def is_likely_setup(self, text: str, *, regex_says_setup: bool = False) -> bool:
        """Combined regex + embedding check for setup statements."""
        scores = self.classify(text)
        if scores is None:
            return regex_says_setup

        if regex_says_setup:
            if scores.best_intent in {"question", "greeting", "followup"} and scores.is_confident:
                best_score = getattr(scores, scores.best_intent)
                margin = best_score - scores.setup
                if margin >= 0.05:
                    return False
            return True

        if scores.best_intent == "setup" and scores.is_confident:
            margin = scores.setup - scores.question
            if margin >= 0.05:
                return True
        return False

    def is_likely_question(self, text: str, *, regex_says_question: bool = False) -> bool:
        """Combined regex + embedding check for actionable questions."""
        scores = self.classify(text)
        if scores is None:
            return regex_says_question

        if regex_says_question:
            if scores.best_intent == "setup" and scores.is_confident:
                margin = scores.setup - scores.question
                if margin >= 0.05:
                    return False
            return True

        if scores.best_intent == "question" and scores.is_confident:
            margin = scores.question - scores.setup
            if margin >= 0.05:
                return True
        return False
