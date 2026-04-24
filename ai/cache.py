"""
Short-query response cache (P1).

Goal: return instantly for repeated small questions when the live context
fingerprint hasn't changed materially.

Three-Tier Lookup:
  Tier 1 - Exact match (fast path, hash lookup)
  Tier 2 - Semantic signature match (smart path, intent + canonical entities)
  Tier 3 - Conservative fuzzy Jaccard match (safety path, optional)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple


_WS_RE = re.compile(r"\s+")


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

# ---------------------------------------------------------------------------
# P2: Intent Patterns (compiled regex with word boundaries — no false positives
# like "fix" matching "fixed", or "build" matching "rebuilding")
# ---------------------------------------------------------------------------
_INTENT_PATTERNS: Dict[str, List[re.Pattern]] = {
    "EXPLAIN": [
        re.compile(r"\bwhat is\b"),
        re.compile(r"\bwhat are\b"),
        re.compile(r"\bexplain\b"),
        re.compile(r"\btell me about\b"),
        re.compile(r"\bdescribe\b"),
        re.compile(r"\bdefine\b"),
    ],
    "HOW_TO": [
        re.compile(r"\bhow do i\b"),
        re.compile(r"\bhow do\b"),
        re.compile(r"\bhow to\b"),
        re.compile(r"\bsteps to\b"),
        re.compile(r"\bimplement\b"),
        re.compile(r"\bbuild\b"),
        re.compile(r"\bcreate\b"),
        re.compile(r"\bsetup\b"),
    ],
    "TROUBLESHOOT": [
        re.compile(r"\bwhy is\b"),
        re.compile(r"\bfix\b"),
        re.compile(r"\bdebug\b"),
        re.compile(r"\berror\b"),
        re.compile(r"\bnot working\b"),
        re.compile(r"\bproblem\b"),
        re.compile(r"\bissue\b"),
        re.compile(r"\bfail\b"),
    ],
    "COMPARE": [
        re.compile(r"\bvs\b"),
        re.compile(r"\bversus\b"),
        re.compile(r"\bdifference between\b"),
        re.compile(r"\bcompare\b"),
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


class ShortQueryCache:
    """
    Three-tier in-memory cache.

    Tier 1 - Exact match: always checked, O(1) hash lookup.
    Tier 2 - Semantic signature: Intent + Canonical Entities, scoped by mode+context.
    Tier 3 - Conservative fuzzy Jaccard (optional, ASCII-only, ≤240 chars).
    """

    def __init__(
        self,
        ttl_s: float = 120.0,
        max_items: int = 128,
        enable_fuzzy: bool = False,
        fuzzy_threshold: float = 0.85,
        enable_semantic: bool = True,
    ):
        self.ttl_s = float(ttl_s or 0)
        self.max_items = int(max_items or 0)
        self.enable_fuzzy = bool(enable_fuzzy)
        self.fuzzy_threshold = float(fuzzy_threshold or 0.85)
        self.enable_semantic = bool(enable_semantic)
        self._items: dict[CacheKey, CacheEntry] = {}
        # Composite key: (signature, mode, context_fp) → CacheKey
        # Scoped per mode+context to prevent cross-contamination.
        self._semantic_items: dict[Tuple[str, str, str], CacheKey] = {}
        self._lru: list[CacheKey] = []

    @staticmethod
    def context_fingerprint(
        *,
        active_window: str = "",
        screen: str = "",
        audio: str = "",
    ) -> str:
        aw = (active_window or "").strip()[:80]
        sc = (screen or "").strip()[:400]
        # P2: Audio deliberately excluded — highly volatile in live meetings.
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

        # Tier 1: Exact match
        key = CacheKey(str(mode or "general"), _norm_query(query), context_fp, history_fp)
        entry = self._items.get(key)
        if entry and entry.expires_at > time.time():
            self._touch_lru(key)
            return entry

        # Tier 2: Semantic Signature (Smart Path)
        if self.enable_semantic:
            signature = self._get_semantic_signature(query)
            sem_lookup_key = (signature, key.mode, key.context_fp)
            semantic_key = self._semantic_items.get(sem_lookup_key)
            if semantic_key:
                entry = self._items.get(semantic_key)
                if entry and entry.expires_at > time.time():
                    self._touch_lru(semantic_key)
                    return entry

        # Tier 3: Conservative Fuzzy (Safety Path)
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
        self._items[key] = entry

        # Tier 2: also store by semantic signature (scoped by mode + context)
        if self.enable_semantic:
            try:
                signature = self._get_semantic_signature(query)
                sem_key = (signature, key.mode, key.context_fp)
                self._semantic_items[sem_key] = key
            except Exception:
                pass  # Never fail caching due to signature errors

        self._touch_lru(key)
        self._prune()

    def _touch_lru(self, key: CacheKey) -> None:
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def _get_semantic_signature(self, query: str) -> str:
        """P2: Versioned semantic signature = Intent + sorted canonical entities.

        Guarantees:
        - 'React hooks' and 'hooks in React' → same signature (sorted)
        - 'fix' does NOT match 'fixed' (word-boundary regex patterns)
        - Empty entity list → no trailing colon (clean fallback)
        """
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
