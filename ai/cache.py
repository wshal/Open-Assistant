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
    "redux":        ["redux", "store", "reducer", "action"],
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

        # In-memory index: parallel lists for fast numpy batch cosine similarity
        self._vectors: List[np.ndarray] = []   # each shape (384,) float32
        self._records: List[_EmbedRecord] = []
        self._timestamps: List[float] = []     # created_at for TTL pruning

        self._dirty = False   # True when vectors need to be flushed to disk
        self._load_persisted()

    # ------------------------------------------------------------------
    # Model loading (lazy, thread-safe)
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Pre-load the model at app startup so first query has zero extra latency."""
        self._load_model()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            # Fallback chain:
            #   1. fastembed      — ONNX-native, ~24MB, fastest (fails on Python 3.14 due to py-rust-stemmers)
            #   2. sentence-transformers ONNX backend — requires `optimum` package
            #   3. sentence-transformers PyTorch      — always works, ~30ms CPU, still background-threaded

            # Path 1: fastembed (best)
            try:
                from fastembed import TextEmbedding  # type: ignore
                self._model = TextEmbedding(model_name=_EMBED_MODEL_NAME)
                logger.info(f"✅ Embedding model loaded via fastembed ONNX ({_EMBED_MODEL_NAME})")
                return
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"fastembed unavailable: {e}")

            # Path 2: sentence-transformers with ONNX backend
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    backend="onnx",
                )
                logger.info("✅ Embedding model loaded via sentence-transformers (ONNX backend)")
                return
            except Exception:
                pass

            # Path 3: sentence-transformers PyTorch (universal fallback)
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("✅ Embedding model loaded via sentence-transformers (PyTorch)")
                return
            except Exception as e:
                self._model = False  # sentinel: stop retrying
                logger.warning(f"Embedding model unavailable — Tier 3 disabled: {e}")

    def _embed(self, text: str) -> Optional[np.ndarray]:
        self._load_model()
        if not self._model:
            return None
        try:
            if hasattr(self._model, "embed"):
                # fastembed API
                vecs = list(self._model.embed([text]))
            else:
                # sentence-transformers API
                vecs = self._model.encode([text], normalize_embeddings=True)
            v = np.array(vecs[0], dtype=np.float32)
            # L2-normalize for cosine similarity via dot product
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-8 else v
        except Exception as e:
            logger.debug(f"Embedding error: {e}")
            return None

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    def add(self, query: str, record: _EmbedRecord) -> None:
        vec = self._embed(query)
        if vec is None:
            return
        self._vectors.append(vec)
        self._records.append(record)
        self._timestamps.append(time.time())
        self._dirty = True
        self._persist()

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
        matrix = np.stack(self._vectors)  # (N, D) — already L2-normalized

        # Cosine similarity = dot product (since both sides are normalized)
        scores = matrix @ vec  # (N,)

        # Filter by mode + context scope and TTL
        best_score = self.threshold - 1e-9  # below threshold
        best_idx = -1
        for i, (rec, ts) in enumerate(zip(self._records, self._timestamps)):
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
            return self._records[best_idx]
        return None

    def prune_expired(self) -> None:
        """Remove entries older than TTL — called periodically from _prune()."""
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
        self._persist()

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._dirty or not self._vectors:
            return
        try:
            self._vectors_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(self._vectors_path), np.stack(self._vectors))
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
            self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._dirty = False
        except Exception as e:
            logger.debug(f"EmbeddingTier: persist failed: {e}")

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

        # Tier 3: embedding index
        self._embed: Optional[EmbeddingTier] = None
        if self.enable_embedding:
            self._embed = EmbeddingTier(
                threshold=float(embedding_threshold or 0.88),
                ttl_s=self.ttl_s,
            )

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
        active_window: str = "",
        screen: str = "",
        audio: str = "",
    ) -> str:
        aw = (active_window or "").strip()[:80]
        sc = (screen or "").strip()[:400]
        # Audio deliberately excluded — highly volatile in live meetings.
        # Cache is stable as long as the visual/window context remains the same.
        return _sha1(f"aw={aw}||sc={sc}")

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
            return

        while len(self._items) > self.max_items and self._lru:
            k = self._lru.pop(0)
            self._items.pop(k, None)

        # Garbage-collect orphaned semantic entries
        active_keys = set(self._items.keys())
        self._semantic_items = {sk: ck for sk, ck in self._semantic_items.items() if ck in active_keys}

        # Prune embedding tier (runs cheaply — just checks timestamps)
        if self._embed:
            self._embed.prune_expired()

    def get(
        self,
        *,
        mode: str,
        query: str,
        context_fp: str,
        history_fp: str,
    ) -> Optional[CacheEntry]:
        if self.ttl_s <= 0 or self.max_items <= 0:
            return None
        self._prune()

        # ── Tier 1: Exact match ─────────────────────────────────────────
        key = CacheKey(str(mode or "general"), _norm_query(query), context_fp, history_fp)
        entry = self._items.get(key)
        if entry and entry.expires_at > time.time():
            self._touch_lru(key)
            return entry

        # ── Tier 2: Semantic Signature ──────────────────────────────────
        if self.enable_semantic:
            signature = self._get_semantic_signature(query)
            sem_lookup_key = (signature, key.mode, key.context_fp)
            semantic_key = self._semantic_items.get(sem_lookup_key)
            if semantic_key:
                entry = self._items.get(semantic_key)
                if entry and entry.expires_at > time.time():
                    self._touch_lru(semantic_key)
                    return entry

        # ── Tier 3: Embedding Similarity ────────────────────────────────
        if self._embed:
            rec = self._embed.find(query, mode=key.mode, context_fp=key.context_fp)
            if rec:
                embed_key = CacheKey(rec.mode, rec.cache_query, rec.context_fp, rec.history_fp)
                entry = self._items.get(embed_key)
                if entry and entry.expires_at > time.time():
                    self._touch_lru(embed_key)
                    return entry

        # ── Tier 4: Conservative Fuzzy ──────────────────────────────────
        if not self.enable_fuzzy:
            return None

        nq = _norm_query(query)
        if len(nq) < 4 or len(nq) > 240 or not _is_simple_ascii(nq):
            return None

        q_tokens = set(nq.split())
        if not q_tokens:
            return None
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
            if score > best_score:
                best_score = score
                best_key = k
        if best_key and best_score >= self.fuzzy_threshold:
            e = self._items.get(best_key)
            if e and e.expires_at > time.time():
                return e
        return None

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

        # Tier 2: Semantic signature
        if self.enable_semantic:
            try:
                signature = self._get_semantic_signature(query)
                sem_key = (signature, key.mode, key.context_fp)
                self._semantic_items[sem_key] = key
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
            threading.Thread(
                target=self._embed.add,
                args=(query, rec),
                daemon=True,
                name="embed-index",
            ).start()

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
