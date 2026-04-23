"""
Short-query response cache (P1).

Goal: return instantly for repeated small questions when the live context
fingerprint hasn't changed materially.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Optional


_WS_RE = re.compile(r"\s+")


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
    Tiny in-memory cache.

    - Exact match: always.
    - Conservative fuzzy match: optional, ASCII-only, short queries only.
    """

    def __init__(
        self,
        ttl_s: float = 25.0,
        max_items: int = 128,
        enable_fuzzy: bool = False,
        fuzzy_threshold: float = 0.92,
    ):
        self.ttl_s = float(ttl_s or 0)
        self.max_items = int(max_items or 0)
        self.enable_fuzzy = bool(enable_fuzzy)
        self.fuzzy_threshold = float(fuzzy_threshold or 0.92)
        self._items: dict[CacheKey, CacheEntry] = {}
        self._lru: list[CacheKey] = []

    @staticmethod
    def context_fingerprint(
        *,
        active_window: str = "",
        screen: str = "",
        audio: str = "",
    ) -> str:
        # Keep it stable, cheap, and privacy-preserving (hashes only).
        aw = (active_window or "").strip()[:80]
        # Use small snippets: enough to detect material changes without bloating.
        sc = (screen or "").strip()[:400]
        au = (audio or "").strip()[-250:]
        return _sha1(f"aw={aw}||sc={sc}||au={au}")

    @staticmethod
    def history_fingerprint(last_query: str = "", last_response: str = "") -> str:
        # Hash only short snippets to avoid heavy work and avoid caching across
        # materially different conversational states.
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
            return

        while len(self._items) > self.max_items and self._lru:
            k = self._lru.pop(0)
            self._items.pop(k, None)

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

        key = CacheKey(str(mode or "general"), _norm_query(query), context_fp, history_fp)
        entry = self._items.get(key)
        if entry and entry.expires_at > time.time():
            # Touch LRU.
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            return entry

        if not self.enable_fuzzy:
            return None

        nq = _norm_query(query)
        if len(nq) < 4 or len(nq) > 80 or not _is_simple_ascii(nq):
            return None

        # Conservative fuzzy: same mode + same context/history only.
        # Score via Jaccard on tokens, then require high overlap.
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
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)
        self._prune()

