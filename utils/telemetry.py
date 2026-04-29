"""
Lightweight in-process telemetry counters and histograms.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict

from utils.logger import setup_logger

logger = setup_logger(__name__)


class _Histogram:
    def __init__(self, maxlen: int = 200):
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=maxlen)

    def record(self, value: float) -> None:
        with self._lock:
            self._samples.append(float(value))

    def stats(self) -> Dict[str, float]:
        with self._lock:
            data = list(self._samples)
        if not data:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
        data.sort()
        n = len(data)
        return {
            "count": n,
            "mean": round(sum(data) / n, 1),
            "p50": round(data[min(n - 1, int(n * 0.50))], 1),
            "p90": round(data[min(n - 1, int(n * 0.90))], 1),
            "min": round(data[0], 1),
            "max": round(data[-1], 1),
        }


class _Counter:
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


class _NamedCounters:
    def __init__(self):
        self._lock = threading.Lock()
        self._values: Dict[str, int] = {}

    def increment(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0) + by

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._values)


class Telemetry:
    def __init__(self):
        self._start_time = time.time()
        # Persistent lock for cer_by_engine dict mutations (Bug 1 fix)
        self._cer_lock = threading.Lock()

        self.cache_hits: Dict[int, _Counter] = {1: _Counter(), 2: _Counter(), 3: _Counter(), 4: _Counter()}
        self.cache_misses = _Counter()
        self.clipboard_context_uses = _Counter()

        self.llm_first_token = _Histogram()
        self.total_latency = _Histogram()
        self.screen_ocr = _Histogram()
        self.ocr_cer = _Histogram()
        self.ocr_cer_by_engine: Dict[str, _Histogram] = {}
        self.asr = _Histogram()
        self.roi_width = _Histogram()
        self.roi_height = _Histogram()
        self.roi_area = _Histogram()

        self.ocr_backend_success = _NamedCounters()
        self.ocr_backend_fallback_success = _NamedCounters()
        self.ocr_backend_failures = _NamedCounters()
        self.roi_sources = _NamedCounters()
        self.cache_evictions = _NamedCounters()

        logger.info("[Telemetry] module initialised")

    def record_cache_hit(self, tier: int) -> None:
        if tier in self.cache_hits:
            self.cache_hits[tier].increment()

    def record_cache_miss(self) -> None:
        self.cache_misses.increment()

    def record_llm_first_token(self, ms: float) -> None:
        self.llm_first_token.record(ms)

    def record_total_latency(self, ms: float) -> None:
        self.total_latency.record(ms)

    def record_screen_ocr(self, ms: float, engine: str = "") -> None:
        self.screen_ocr.record(ms)
        if engine:
            logger.debug("[Telemetry] screen_ocr %.1fms engine=%s", ms, engine)

    def record_ocr_cer(self, cer: float, engine: str = "") -> None:
        """Record OCR character error rate for a run."""
        self.ocr_cer.record(cer)
        if engine:
            engine_key = str(engine).lower()
            with self._cer_lock:  # Bug 1 fix: use persistent instance lock
                if engine_key not in self.ocr_cer_by_engine:
                    self.ocr_cer_by_engine[engine_key] = _Histogram()
                self.ocr_cer_by_engine[engine_key].record(cer)

    def record_asr(self, ms: float) -> None:
        self.asr.record(ms)

    def record_clipboard_use(self) -> None:
        self.clipboard_context_uses.increment()

    def record_ocr_backend(self, engine: str, outcome: str = "success") -> None:
        engine = str(engine or "unknown")
        if outcome == "success":
            self.ocr_backend_success.increment(engine)
        elif outcome == "fallback_success":
            self.ocr_backend_fallback_success.increment(engine)
        else:
            self.ocr_backend_failures.increment(engine)

    def record_roi(self, width: int, height: int, source: str = "unknown") -> None:
        width = max(0, int(width))
        height = max(0, int(height))
        self.roi_width.record(width)
        self.roi_height.record(height)
        self.roi_area.record(width * height)
        self.roi_sources.increment(str(source or "unknown"))

    def record_cache_eviction(self, namespace: str) -> None:
        self.cache_evictions.increment(str(namespace or "unknown"))

    def summary(self) -> Dict[str, Any]:
        total_hits = sum(counter.value for counter in self.cache_hits.values())
        total_requests = total_hits + self.cache_misses.value
        hit_rate = round(total_hits / total_requests * 100, 1) if total_requests else 0.0

        ocr_cer_by_engine_stats = {}
        for engine, hist in self.ocr_cer_by_engine.items():
            ocr_cer_by_engine_stats[engine] = hist.stats()

        return {
            "uptime_s": round(time.time() - self._start_time, 0),
            "cache": {
                "hits_t1": self.cache_hits[1].value,
                "hits_t2": self.cache_hits[2].value,
                "hits_t3": self.cache_hits[3].value,
                "hits_t4": self.cache_hits[4].value,
                "misses": self.cache_misses.value,
                "hit_rate_pct": hit_rate,
                "evictions": self.cache_evictions.snapshot(),
            },
            "latency_ms": {
                "llm_first_token": self.llm_first_token.stats(),
                "total": self.total_latency.stats(),
                "screen_ocr": self.screen_ocr.stats(),
                "asr": self.asr.stats(),
            },
            "ocr": {
                "cer": self.ocr_cer.stats(),
                "cer_by_engine": ocr_cer_by_engine_stats,
                "backend_success": self.ocr_backend_success.snapshot(),
                "backend_fallback_success": self.ocr_backend_fallback_success.snapshot(),
                "backend_failures": self.ocr_backend_failures.snapshot(),
            },
            "roi": {
                "width": self.roi_width.stats(),
                "height": self.roi_height.stats(),
                "area": self.roi_area.stats(),
                "sources": self.roi_sources.snapshot(),
            },
            "clipboard_context_uses": self.clipboard_context_uses.value,
        }

    def log_summary(self) -> None:
        summary = self.summary()
        logger.info(
            "[Telemetry] uptime=%ss hit_rate=%.1f%% screen_ocr_p50=%.0fms roi_samples=%d",
            summary["uptime_s"],
            summary["cache"]["hit_rate_pct"],
            summary["latency_ms"]["screen_ocr"]["p50"],
            summary["roi"]["width"]["count"],
        )


telemetry = Telemetry()
