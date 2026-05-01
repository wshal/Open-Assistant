"""
Short-query response cache (P1).

Goal: return instantly for repeated small questions when the live context
fingerprint hasn't changed materially.

Four-Tier Lookup:
  Tier 1 - Exact match      (fast path,    hash lookup,         ~0ms)
  Tier 2 - Semantic Sig     (smart path,   intent+entities,     ~0.1ms)
  Tier 3 - Embedding Sim    (smarter path, MiniLM ONNX cosine,  ~15ms)
  Tier 4 - Fuzzy Jaccard    (safety path,  token overlap,       ~1ms, optional)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Q13: Telemetry (import lazily to avoid circular deps at module load)
try:
    from utils.telemetry import telemetry as _telemetry
except Exception:
    _telemetry = None

# Suppress the noisy HuggingFace Hub unauthenticated-request warning.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="Cannot enable progress bars.*")

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Embedding persistence paths (inside data/cache/ — already gitignored)
# ---------------------------------------------------------------------------
_EMBED_VECTORS_PATH = Path("data/cache/embed_vectors.npy")
_EMBED_META_PATH    = Path("data/cache/embed_meta.json")
_EMBED_MODEL_NAME   = "BAAI/bge-small-en-v1.5"   # 24MB ONNX, best quality/size ratio


# ---------------------------------------------------------------------------
# P2: Canonical Entity Map
# Aliases → canonical key. Word-boundary regex is used at lookup time.
# ---------------------------------------------------------------------------
_CANONICAL_ENTITIES: Dict[str, List[str]] = {
    # React ecosystem
    "react":        ["react", "reactjs", "react.js", "react library", "react framework"],
    "nextjs":       ["nextjs", "next.js", "next framework"],
    "hooks":        ["hook", "hooks", "react hook"],
    "usestate":     ["usestate", "use state"],
    "useeffect":    ["useeffect", "use effect"],
    "usememo":      ["usememo", "use memo"],
    "usecallback":  ["usecallback", "use callback"],
    "usereducer":   ["usereducer", "use reducer"],
    "usecontext":   ["usecontext", "use context"],
    "useref":       ["useref", "use ref"],
    "component":    ["component", "components", "react component", "functional component", "class component"],
    "props":        ["props", "properties", "react props"],
    "state":        ["state", "react state"],
    "context api":  ["context api", "react context"],
    "redux":        ["redux", "redux store", "reducer", "action"],
    "zustand":      ["zustand"],
    "react router": ["react router", "router", "routing"],

    # API & Web
    "rest api":   ["rest api", "rest endpoint", "restful api", "api endpoint", "api"],
    "graphql":    ["graphql", "graph ql", "apollo"],
    "websocket":  ["websocket", "web socket", "socket.io", "sse"],
    "cors":       ["cors", "cors error", "cors issue", "cross origin"],
    "jwt":        ["jwt", "json web token", "auth token"],
    "oauth":      ["oauth", "oauth2"],
    "tailwind":   ["tailwind", "tailwindcss"],
    "css":        ["css", "style", "styling", "flexbox", "grid"],
    "html":       ["html", "dom", "document"],
    "browser":    ["browser", "chrome", "firefox", "safari", "client side"],
    "vite":       ["vite", "vite.js"],
    "webpack":    ["webpack", "bundler", "babel"],

    # Languages & Tools
    "typescript": ["typescript", "tsconfig"],
    "javascript": ["javascript", "nodejs"],
    "python":     ["python", "fastapi", "flask", "django"],
    "generics":   ["generics", "generic types", "type generic"],
    "database":   ["database", "sql", "postgresql", "mongodb", "prisma", "drizzle"],

    # Alternate frameworks
    "vue":     ["vue", "vue.js", "vuejs", "pinia", "nuxt"],
    "svelte":  ["svelte", "sveltekit"],
    "angular": ["angular", "angularjs"],
}

# Use word-boundary patterns for intent matching to avoid e.g. "fix" matching "fixed"
_INTENT_PATTERNS: Dict[str, List[re.Pattern]] = {
    "EXPLAIN": [
        re.compile(r"\bwhat is\b"), re.compile(r"\bwhat are\b"),
        re.compile(r"\bexplain\b"), re.compile(r"\btell me about\b"),
        re.compile(r"\bdescribe\b"), re.compile(r"\bdefine\b"),
    ],
    "HOW_TO": [
        re.compile(r"\bhow do i\b"), re.compile(r"\bhow do\b"),
        re.compile(r"\bhow to\b"), re.compile(r"\bsteps to\b"),
        re.compile(r"\bimplement\b"), re.compile(r"\bbuild\b"),
        re.compile(r"\bcreate\b"), re.compile(r"\bsetup\b"),
    ],
    "TROUBLESHOOT": [
        re.compile(r"\bwhy is\b"), re.compile(r"\bfix\b"),
        re.compile(r"\bdebug\b"), re.compile(r"\berror\b"),
        re.compile(r"\bnot working\b"), re.compile(r"\bproblem\b"),
        re.compile(r"\bissue\b"), re.compile(r"\bfail\b"),
    ],
    "COMPARE": [
        re.compile(r"\bvs\b"), re.compile(r"\bversus\b"),
        re.compile(r"\bdifference between\b"), re.compile(r"\bcompare\b"),
    ],
}


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _norm_query(query: str) -> str:
    q = (query or "").strip().lower()
    q = _WS_RE.sub(" ", q)
    return q


def _is_simple_ascii(text: str) -> bool:
    if not text:
        return False
    try:
        text.encode("ascii")
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class CacheKey:
    mode: str
    query: str
    context_fp: str
    history_fp: str


@dataclass
class CacheEntry:
    response: str
    provider: str
    created_at: float
    expires_at: float


# ---------------------------------------------------------------------------
# Tier 3: Embedding-based semantic similarity
# ---------------------------------------------------------------------------

@dataclass
class _EmbedRecord:
    """One entry in the embedding index."""
    mode: str
    context_fp: str
    cache_query: str    # normalized — used to reconstruct CacheKey
    history_fp: str


class EmbeddingTier:
    """
    Tier 3 semantic cache backed by ONNX MiniLM embeddings.

    Model:  BAAI/bge-small-en-v1.5  (~24MB ONNX, auto-downloaded on first use)
    Speed:  ~10-15ms per query on CPU
    Benefit: Catches paraphrased questions that signature+fuzzy both miss,
             e.g. 'centering a div' ↔ 'horizontal alignment issue'
    """

    def __init__(
        self,
        threshold: float = 0.88,
        vectors_path: Path = _EMBED_VECTORS_PATH,
        meta_path: Path = _EMBED_META_PATH,
        ttl_s: float = 120.0,
    ):
        self.threshold = threshold
        self.ttl_s = ttl_s
        self._vectors_path = Path(vectors_path)
        self._meta_path = Path(meta_path)

        self._model = None          # None = not loaded yet, False = failed, model otherwise
        self._model_lock = threading.Lock()
        self._data_lock = threading.RLock()
        # Q18: separate write-lock to prevent concurrent np.save races
        self._persist_lock = threading.Lock()
        self._persist_timer: threading.Timer = None
        self._persist_thread: Optional[threading.Thread] = None

        # In-memory index: parallel lists for fast numpy batch cosine similarity
        self._vectors: List[np.ndarray] = []   # each shape (384,) float32
        self._records: List[_EmbedRecord] = []
        self._timestamps: List[float] = []     # created_at for TTL pruning

        self._dirty = False   # True when vectors need to be flushed to disk
        # Q18: Start periodic 30s auto-persist background timer
        self._schedule_persist_timer()
        with self._data_lock:
            self._load_persisted()

    # ------------------------------------------------------------------
    # Model loading (lazy, thread-safe)
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Pre-load the model at app startup so first query has zero extra latency."""
        from ai.embedding_manager import EmbeddingManager
        EmbeddingManager().warmup()

    def _embed(self, text: str) -> Optional[np.ndarray]:
        from ai.embedding_manager import EmbeddingManager
        return EmbeddingManager().embed(text)

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    def add(self, query: str, record: _EmbedRecord) -> None:
        vec = self._embed(query)
        if vec is None:
            return
        with self._data_lock:
            self._vectors.append(vec)
            self._records.append(record)
            self._timestamps.append(time.time())
            self._dirty = True
        # Q18: Do NOT call _persist() here — it would deadlock because _do_persist
        # also needs _data_lock from a different thread. The 30s auto-timer handles persistence.

    def find(
        self,
        query: str,
        mode: str,
        context_fp: str,
    ) -> Optional[_EmbedRecord]:
        """Return the best matching record above threshold, or None."""
        if not self._vectors:
            return None
        vec = self._embed(query)
        if vec is None:
            return None

        now = time.time()
        with self._data_lock:
            if not self._vectors:
                return None
            matrix = np.stack(self._vectors)  # (N, D) — already L2-normalized
            records = list(self._records)
            timestamps = list(self._timestamps)

        # Cosine similarity = dot product (since both sides are normalized)
        scores = matrix @ vec  # (N,)

        # Filter by mode + context scope and TTL
        best_score = self.threshold - 1e-9  # below threshold
        best_idx = -1
        for i, (rec, ts) in enumerate(zip(records, timestamps)):
            if rec.mode != mode or rec.context_fp != context_fp:
                continue
            if (now - ts) > self.ttl_s:
                continue
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i

        if best_idx >= 0:
            logger.debug(
                f"EmbeddingTier: hit (cosine={best_score:.3f}) for query: {query[:50]!r}"
            )
            return records[best_idx]
        return None

    def prune_expired(self) -> None:
        """Remove entries older than TTL — called periodically from _prune()."""
        with self._data_lock:
            if not self._timestamps:
                return
            now = time.time()
            keep = [i for i, ts in enumerate(self._timestamps) if (now - ts) <= self.ttl_s]
            if len(keep) == len(self._timestamps):
                return
            self._vectors = [self._vectors[i] for i in keep]
            self._records = [self._records[i] for i in keep]
            self._timestamps = [self._timestamps[i] for i in keep]
            self._dirty = True
        # Q18: _persist() removed from here — called by 30s auto-timer instead

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._dirty or not self._vectors:
            return
        with self._persist_lock:
            if self._persist_thread and self._persist_thread.is_alive():
                return
        # Q18: Run in background thread to avoid blocking response path
        def _do_persist():
            try:
                with self._data_lock:
                    if not self._vectors:
                        return
                    vecs = np.stack(self._vectors)
                    meta = [
                        {
                            "mode": r.mode,
                            "context_fp": r.context_fp,
                            "cache_query": r.cache_query,
                            "history_fp": r.history_fp,
                            "ts": ts,
                        }
                        for r, ts in zip(self._records, self._timestamps)
                    ]
                self._vectors_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(self._vectors_path), vecs)
                self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                with self._data_lock:
                    self._dirty = False
                logger.debug("[Q18 Persist] Embedding index saved in background (%d vectors)", len(vecs))
            except Exception as e:
                logger.debug("EmbeddingTier: background persist failed: %s", e)

        thread = threading.Thread(target=_do_persist, daemon=True, name="embed-persist")
        self._persist_thread = thread
        thread.start()

    def _schedule_persist_timer(self, interval: float = 30.0) -> None:
        """Q18: Periodic background persist every 30s — restarts itself."""
        def _tick():
            try:
                self._persist()
            finally:
                self._schedule_persist_timer(interval)  # reschedule
        self._persist_timer = threading.Timer(interval, _tick)
        self._persist_timer.daemon = True
        self._persist_timer.start()

    def _load_persisted(self) -> None:
        try:
            if not self._vectors_path.exists() or not self._meta_path.exists():
                return
            raw_meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            raw_vecs = np.load(str(self._vectors_path))
            if len(raw_vecs) != len(raw_meta):
                logger.debug("EmbeddingTier: length mismatch on load — resetting.")
                return
            now = time.time()
            for i, m in enumerate(raw_meta):
                ts = float(m.get("ts", 0))
                if (now - ts) > self.ttl_s:
                    continue  # skip expired on load
                self._vectors.append(raw_vecs[i].astype(np.float32))
                self._records.append(_EmbedRecord(
                    mode=m["mode"],
                    context_fp=m["context_fp"],
                    cache_query=m["cache_query"],
                    history_fp=m["history_fp"],
                ))
                self._timestamps.append(ts)
            logger.info(f"EmbeddingTier: Loaded {len(self._vectors)} persisted embeddings.")
        except Exception as e:
            logger.debug(f"EmbeddingTier: load failed (fresh start): {e}")
            self._vectors = []
            self._records = []
            self._timestamps = []


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------

class ShortQueryCache:
    """
    Four-tier in-memory + disk-backed cache.

    Tier 1 - Exact match:      O(1) hash,     ~0ms
    Tier 2 - Semantic sig:     intent+entity, ~0.1ms
    Tier 3 - Embedding sim:    MiniLM ONNX,   ~15ms CPU  (default ON)
    Tier 4 - Fuzzy Jaccard:    token overlap, ~1ms       (optional)
    """

    def __init__(
        self,
        ttl_s: float = 120.0,
        max_items: int = 128,
        enable_fuzzy: bool = False,
        fuzzy_threshold: float = 0.85,
        enable_semantic: bool = True,
        enable_embedding: bool = True,
        embedding_threshold: float = 0.88,
    ):
        self.ttl_s = float(ttl_s or 0)
        self.max_items = int(max_items or 0)
        self.enable_fuzzy = bool(enable_fuzzy)
        self.fuzzy_threshold = float(fuzzy_threshold or 0.85)
        self.enable_semantic = bool(enable_semantic)
        self.enable_embedding = bool(enable_embedding)

        self._items: dict[CacheKey, CacheEntry] = {}
        # Composite key: (signature, mode, context_fp) → CacheKey
        self._semantic_items: dict[Tuple[str, str, str], CacheKey] = {}
        self._lru: list[CacheKey] = []
        # Q17: LRU list for semantic_items (prevents unbounded growth)
        self._semantic_lru: list[Tuple[str, str, str]] = []
        self._max_semantic_items: int = 512

        # Tier 3: embedding index
        self._embed: Optional[EmbeddingTier] = None
        if self.enable_embedding:
            self._embed = EmbeddingTier(
                threshold=float(embedding_threshold or 0.88),
                ttl_s=self.ttl_s,
            )
            
            import queue
            self._embed_queue: queue.Queue = queue.Queue(maxsize=100)
            self._embed_worker = threading.Thread(
                target=self._embed_worker_loop,
                daemon=True,
                name="embed-worker",
            )
            self._embed_worker.start()

    def _embed_worker_loop(self) -> None:
        while True:
            try:
                query, rec = self._embed_queue.get()
                if self._embed:
                    self._embed.add(query, rec)
            except Exception as e:
                logger.debug(f"Embed indexing error: {e}")
            finally:
                self._embed_queue.task_done()

    def warmup(self) -> None:
        """Pre-load the embedding model during app warmup — zero cold-start latency."""
        if self._embed:
            threading.Thread(
                target=self._embed.warmup,
                daemon=True,
                name="embed-warmup",
            ).start()

    @staticmethod
    def context_fingerprint(
        *,
        window_id: str = "",
        active_window: str = "",
        screen: str = "",
        audio: str = "",
        screen_hash: str = "",
    ) -> str:
        wid = (str(window_id or "")).strip()[:24]
        aw = (active_window or "").strip()[:80]
        # P1.3: Use the perceptual screen_hash if available, otherwise fallback to text
        if screen_hash:
            sc = f"hash:{(screen_hash or '')[:8]}"
        else:
            sc = (screen or "").strip()[:400]
        # Audio deliberately excluded — highly volatile in live meetings.
        # Cache is stable as long as the visual/window context remains the same.
        return _sha1(f"wid={wid}||aw={aw}||sc={sc}")

    @staticmethod
    def history_fingerprint(last_query: str = "", last_response: str = "") -> str:
        lq = (last_query or "").strip()[:120]
        lr = (last_response or "").strip()[:200]
        return _sha1(f"lq={lq}||lr={lr}")

    def _prune(self) -> None:
        if not self._items:
            return
        now = time.time()
        expired = [k for k, v in self._items.items() if v.expires_at <= now]
        for k in expired:
            self._items.pop(k, None)
        if expired:
            self._lru = [k for k in self._lru if k in self._items]

        if self.max_items <= 0:
            self._items.clear()
            self._lru.clear()
            self._semantic_items.clear()
            self._semantic_lru.clear()  # Q17: sync LRU list
            return

        while len(self._items) > self.max_items and self._lru:
            k = self._lru.pop(0)
            self._items.pop(k, None)

        # Garbage-collect orphaned semantic entries and sync LRU list (Q17)
        active_keys = set(self._items.keys())
        self._semantic_items = {sk: ck for sk, ck in self._semantic_items.items() if ck in active_keys}
        # Rebuild LRU to match surviving keys (removes dangling references)
        surviving_sem = set(self._semantic_items.keys())
        self._semantic_lru = [sk for sk in self._semantic_lru if sk in surviving_sem]

        # Prune embedding tier (runs cheaply — just checks timestamps)
        if self._embed:
            self._embed.prune_expired()

    def get_with_tier(
        self,
        *,
        mode: str,
        query: str,
        context_fp: str,
        history_fp: str,
        boost_context: Optional[str] = None,  # Q16: prior-turn entity keywords for score boosting
    ) -> "tuple[Optional[CacheEntry], int]":
        """Like get() but also returns which tier was hit (1-4, or 0 for miss).

        Q16: boost_context is a space-separated string of entity keywords from the
        prior turn (e.g. 'React useState hooks').  When provided, Tier 4 fuzzy
        scoring applies a 1.2x multiplier to candidates whose query overlaps with
        those tokens, increasing cache hit rate for topic-related follow-ups.
        """
        if self.ttl_s <= 0 or self.max_items <= 0:
            return None, 0
        self._prune()

        # ── Tier 1: Exact match ──────────────────────────────────────────────
        key = CacheKey(str(mode or "general"), _norm_query(query), context_fp, history_fp)
        entry = self._items.get(key)
        if entry and entry.expires_at > time.time():
            self._touch_lru(key)
            if _telemetry:
                _telemetry.record_cache_hit(tier=1)
            return entry, 1

        # ── Tier 2: Semantic Signature ────────────────────────────────────────
        if self.enable_semantic:
            signature = self._get_semantic_signature(query)
            sem_lookup_key = (signature, key.mode, key.context_fp)
            semantic_key = self._semantic_items.get(sem_lookup_key)
            if semantic_key:
                entry = self._items.get(semantic_key)
                if entry and entry.expires_at > time.time():
                    self._touch_lru(semantic_key)
                    if _telemetry:
                        _telemetry.record_cache_hit(tier=2)
                    return entry, 2

        # ── Tier 3: Embedding Similarity ──────────────────────────────────────
        if self._embed:
            rec = self._embed.find(query, mode=key.mode, context_fp=key.context_fp)
            if rec:
                embed_key = CacheKey(rec.mode, rec.cache_query, rec.context_fp, rec.history_fp)
                entry = self._items.get(embed_key)
                if entry and entry.expires_at > time.time():
                    self._touch_lru(embed_key)
                    if _telemetry:
                        _telemetry.record_cache_hit(tier=3)
                    return entry, 3

        # ── Tier 4: Conservative Fuzzy ─────────────────────────────────────────
        if not self.enable_fuzzy:
            if _telemetry:
                _telemetry.record_cache_miss()
            return None, 0

        nq = _norm_query(query)
        if len(nq) < 4 or len(nq) > 240 or not _is_simple_ascii(nq):
            if _telemetry:
                _telemetry.record_cache_miss()
            return None, 0

        q_tokens = set(nq.split())
        if not q_tokens:
            if _telemetry:
                _telemetry.record_cache_miss()
            return None, 0
        # Q16: Build boost token set from prior-turn entity context
        boost_tokens: set = set()
        if boost_context:
            boost_tokens = set(_norm_query(boost_context).split())
            boost_tokens.discard("")  # remove empty strings
        best_key = None
        best_score = 0.0
        for k in list(self._items.keys()):
            if k.mode != key.mode or k.context_fp != context_fp or k.history_fp != history_fp:
                continue
            kt = set(k.query.split())
            if not kt:
                continue
            inter = len(q_tokens & kt)
            union = len(q_tokens | kt)
            score = inter / union if union else 0.0
            # Q16: Boost score if candidate shares entity tokens with prior turn
            if boost_tokens and kt & boost_tokens:
                score *= 1.2
                logger.debug(
                    "[Q16 Boost] score %.3f boosted to %.3f for '%s'",
                    score / 1.2, score, k.query[:60],
                )
            if score > best_score:
                best_score = score
                best_key = k
        if best_key and best_score >= self.fuzzy_threshold:
            e = self._items.get(best_key)
            if e and e.expires_at > time.time():
                if _telemetry:
                    _telemetry.record_cache_hit(tier=4)
                return e, 4

        if _telemetry:
            _telemetry.record_cache_miss()
        return None, 0

    def get(
        self,
        *,
        mode: str,
        query: str,
        context_fp: str,
        history_fp: str,
        boost_context: Optional[str] = None,
    ) -> Optional[CacheEntry]:
        """Backward-compatible wrapper — delegates to get_with_tier()."""
        entry, _ = self.get_with_tier(
            mode=mode, query=query, context_fp=context_fp, history_fp=history_fp,
            boost_context=boost_context,
        )
        return entry

    def set(
        self,
        *,
        mode: str,
        query: str,
        context_fp: str,
        history_fp: str,
        response: str,
        provider: str,
    ) -> None:
        if self.ttl_s <= 0 or self.max_items <= 0:
            return
        nq = _norm_query(query)
        if not nq:
            return
        key = CacheKey(str(mode or "general"), nq, context_fp, history_fp)
        now = time.time()
        entry = CacheEntry(
            response=(response or ""),
            provider=(provider or ""),
            created_at=now,
            expires_at=now + self.ttl_s,
        )

        # Tier 1: Exact
        self._items[key] = entry

        # Q17: Evict oldest semantic entry when cap is exceeded
        if self.enable_semantic:
            try:
                signature = self._get_semantic_signature(query)
                sem_key = (signature, key.mode, key.context_fp)
                if sem_key not in self._semantic_items:
                    # New entry — check cap
                    while len(self._semantic_items) >= self._max_semantic_items and self._semantic_lru:
                        oldest = self._semantic_lru.pop(0)
                        self._semantic_items.pop(oldest, None)
                        logger.debug("[Q17 LRU] Evicted semantic_item key: %s", oldest)
                self._semantic_items[sem_key] = key
                # Track LRU order for this sem_key
                try:
                    self._semantic_lru.remove(sem_key)
                except ValueError:
                    pass
                self._semantic_lru.append(sem_key)
            except Exception:
                pass

        # Tier 3: Embedding (runs in background thread — never blocks the response path)
        if self._embed:
            rec = _EmbedRecord(
                mode=key.mode,
                context_fp=key.context_fp,
                cache_query=nq,
                history_fp=key.history_fp,
            )
            import queue
            try:
                self._embed_queue.put((query, rec), timeout=0.1)
            except queue.Full:
                pass

        self._touch_lru(key)
        self._prune()

    def _touch_lru(self, key: CacheKey) -> None:
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def _get_semantic_signature(self, query: str) -> str:
        """P2: Versioned semantic signature = Intent + sorted canonical entities."""
        lower = (query or "").lower().strip()

        # 1. Detect intent using compiled word-boundary patterns
        intent = "GENERAL"
        for intent_name, patterns in _INTENT_PATTERNS.items():
            if any(p.search(lower) for p in patterns):
                intent = intent_name
                break

        # 2. Extract canonical entities (word-boundary match to prevent partials)
        found_entities: Set[str] = set()
        for canonical, aliases in _CANONICAL_ENTITIES.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", lower):
                    found_entities.add(canonical)
                    break

        # 3. Build versioned signature — no trailing colon when no entities
        if found_entities:
            entity_str = ":".join(sorted(found_entities))
            return f"v1:{intent}:{entity_str}"
        return f"v1:{intent}"
