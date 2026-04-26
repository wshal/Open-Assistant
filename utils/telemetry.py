"""
utils/telemetry.py  --  Q13: Lightweight in-process telemetry counters.

Design goals:
  - Zero dependencies beyond stdlib (thread-safe via threading.Lock)
  - No disk I/O in the hot path
  - Accessible from anywhere via module-level singleton

Usage:
    from utils.telemetry import telemetry
    telemetry.record_cache_hit(tier=1)
    telemetry.record_llm_first_token(ms=340.5)
    summary = telemetry.summary()          # dict of stats
"""

import threading
import time
from collections import deque
from typing import Dict, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


class _Histogram:
    """Rolling-window histogram (last N samples)."""

    def __init__(self, maxlen: int = 200):
        self._lock = threading.Lock()
        self._samples: deque = deque(maxlen=maxlen)

    def record(self, value: float) -> None:
        with self._lock:
            self._samples.append(value)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            data = list(self._samples)
        if not data:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(data)
        n = len(s)
        return {
            "count": n,
            "mean": round(sum(s) / n, 1),
            "p50": round(s[int(n * 0.50)], 1),
            "p90": round(s[int(n * 0.90)], 1),
            "min": round(s[0], 1),
            "max": round(s[-1], 1),
        }


class _Counter:
    """Thread-safe integer counter."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0

    def increment(self, by: int = 1) -> None:
        with self._lock:
            self._value += by

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class Telemetry:
    """
    Central telemetry store for OpenAssist.

    Tracks:
      - cache_hits[tier]  (int counters, tier = 1..4)
      - cache_misses      (int counter)
      - llm_first_token   (histogram, ms)
      - total_latency     (histogram, ms)
      - screen_ocr        (histogram, ms)
      - asr               (histogram, ms)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Cache counters
        self.cache_hits: Dict[int, _Counter] = {1: _Counter(), 2: _Counter(), 3: _Counter(), 4: _Counter()}
        self.cache_misses = _Counter()
        self.clipboard_context_uses = _Counter()

        # Latency histograms (all values in ms)
        self.llm_first_token = _Histogram()
        self.total_latency = _Histogram()
        self.screen_ocr = _Histogram()
        self.asr = _Histogram()

        logger.info("[Q13 Telemetry] Telemetry module initialised")

    # ── Recording helpers ─────────────────────────────────────────────────────

    def record_cache_hit(self, tier: int) -> None:
        """Call when a cache lookup succeeds at a specific tier (1–4)."""
        if tier in self.cache_hits:
            self.cache_hits[tier].increment()
            logger.debug("[Q13 Telemetry] cache_hit tier=%d (total=%d)", tier, self.cache_hits[tier].value)

    def record_cache_miss(self) -> None:
        self.cache_misses.increment()
        logger.debug("[Q13 Telemetry] cache_miss (total=%d)", self.cache_misses.value)

    def record_llm_first_token(self, ms: float) -> None:
        self.llm_first_token.record(ms)
        logger.debug("[Q13 Telemetry] llm_first_token=%.0fms", ms)

    def record_total_latency(self, ms: float) -> None:
        self.total_latency.record(ms)
        logger.debug("[Q13 Telemetry] total_latency=%.0fms", ms)

    def record_screen_ocr(self, ms: float) -> None:
        self.screen_ocr.record(ms)
        logger.debug("[Q13 Telemetry] screen_ocr=%.0fms", ms)

    def record_asr(self, ms: float) -> None:
        self.asr.record(ms)
        logger.debug("[Q13 Telemetry] asr=%.0fms", ms)

    def record_clipboard_use(self) -> None:
        self.clipboard_context_uses.increment()
        logger.info("[Q13 Telemetry] clipboard_context_use #%d", self.clipboard_context_uses.value)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a dict snapshot of all metrics."""
        total_hits = sum(c.value for c in self.cache_hits.values())
        total_requests = total_hits + self.cache_misses.value
        hit_rate = round(total_hits / total_requests * 100, 1) if total_requests else 0.0

        return {
            "uptime_s": round(time.time() - self._start_time, 0),
            "cache": {
                "hits_t1": self.cache_hits[1].value,
                "hits_t2": self.cache_hits[2].value,
                "hits_t3": self.cache_hits[3].value,
                "hits_t4": self.cache_hits[4].value,
                "misses": self.cache_misses.value,
                "hit_rate_pct": hit_rate,
            },
            "latency_ms": {
                "llm_first_token": self.llm_first_token.stats(),
                "total": self.total_latency.stats(),
                "screen_ocr": self.screen_ocr.stats(),
                "asr": self.asr.stats(),
            },
            "clipboard_context_uses": self.clipboard_context_uses.value,
        }

    def log_summary(self) -> None:
        """Log a human-readable summary at INFO level."""
        s = self.summary()
        c = s["cache"]
        logger.info(
            "[Q13 Telemetry] Session summary | uptime=%.0fs | "
            "cache_hits T1=%d T2=%d T3=%d T4=%d misses=%d hit_rate=%.1f%% | "
            "llm_first_token p50=%.0fms p90=%.0fms | "
            "total p50=%.0fms p90=%.0fms | clipboard_uses=%d",
            s["uptime_s"],
            c["hits_t1"], c["hits_t2"], c["hits_t3"], c["hits_t4"], c["misses"], c["hit_rate_pct"],
            s["latency_ms"]["llm_first_token"]["p50"],
            s["latency_ms"]["llm_first_token"]["p90"],
            s["latency_ms"]["total"]["p50"],
            s["latency_ms"]["total"]["p90"],
            s["clipboard_context_uses"],
        )


# ── Module-level singleton ────────────────────────────────────────────────────
telemetry = Telemetry()
