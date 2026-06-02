"""Base provider with stats tracking and rate limiting."""

import time
import threading
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass, field
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Stats:
    name: str
    requests: int = 0
    errors: int = 0
    total_time: float = 0.0
    total_tokens: int = 0
    last_latency: float = 0.0
    tps: float = 0.0
    # Issue #13: parallel inference races mutate counters/tps; serialise updates.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, tokens: int, latency: float):
        with self._lock:
            self.requests += 1
            self.total_tokens += int(tokens or 0)
            self.total_time += float(latency or 0.0)
            self.last_latency = float(latency or 0.0)
            if latency > 0:
                self.tps = self.tps * 0.7 + (tokens / latency) * 0.3

    def record_error(self):
        """Thread-safe error counter increment."""
        with self._lock:
            self.errors += 1

    @property
    def success_rate(self):
        t = self.requests + self.errors
        return self.requests / max(t, 1)

    @property
    def avg_latency(self):
        return self.total_time / max(self.requests, 1)


class BaseProvider(ABC):
    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.pcfg = config.get(f"ai.providers.{name}", {})
        self.enabled = bool(self.pcfg) and self.pcfg.get("enabled", True)
        self.max_tokens = self.pcfg.get("max_tokens", 4096)
        self.speed = self.pcfg.get("speed", 5)
        self.quality = self.pcfg.get("quality", 5)
        self.rpm = self.pcfg.get("rpm", 60)
        self.models = self.pcfg.get("models", {})
        self.default_tier = self.pcfg.get("default", "balanced")
        self.stats = Stats(name)
        self._req_times = []
        self._disabled_until = 0.0
        self._disabled_reason = ""
        self._disabled_at = 0.0
        self.failure_cooldown = self.pcfg.get("failure_cooldown", 30)
        # Issue #13: serialise _req_times mutations across parallel/race callers.
        self._rate_lock = threading.Lock()
        # Track last successful request time for idle connection detection.
        self._last_request_time: float = 0.0

    def get_model(self, tier: str = None) -> str:
        """
        RESOLVED: Fallback strategy for simple vs tiered configs.
        Checks for 'models' dict first, then falls back to singular 'model' key.
        """
        # 1. Check tiered models if requested
        t = tier or self.default_tier
        model = self.models.get(t) if self.models else None
        
        # 2. Sequential fallback through common tiers
        if not model and self.models:
            for fallback_t in ["balanced", "fast", "reasoning"]:
                if fallback_t in self.models:
                    model = self.models[fallback_t]
                    break
        
        # 3. GLOBAL FALLBACK: Singular 'model' key (Used in simple config.yaml)
        if not model:
            model = self.pcfg.get("model")
            
        # 4. FINAL FAILSAFE: First item in models dict
        if not model and self.models:
            model = list(self.models.values())[0]
            
        return model or ""

    def has_model(self, tier: str = None) -> bool:
        return bool(self.get_model(tier))

    def supports_health_check(self) -> bool:
        return type(self).health_check is not BaseProvider.health_check

    def is_disabled(self) -> bool:
        return time.time() < self._disabled_until

    def cooldown_remaining_s(self) -> int:
        return max(0, int(self._disabled_until - time.time()))

    def disabled_reason(self) -> str:
        return str(self._disabled_reason or "").strip()

    def disable(self, seconds: int = None, reason: str = ""):
        duration = seconds if seconds is not None else self.failure_cooldown
        now = time.time()
        with self._rate_lock:  # Guard against torn reads in _pre_request's is_disabled()
            self._disabled_until = now + duration
            self._disabled_at = now
            self._disabled_reason = str(reason or "").strip()
        logger.warning(
            f"{self.name}: temporarily disabled for {duration}s after failure"
            + (f" ({self._disabled_reason})" if self._disabled_reason else "")
        )

    def check_rate(self) -> bool:
        # Issue #13: lock both the prune and the read so a parallel _pre_request
        # can't append between this thread's prune and its capacity check.
        now = time.time()
        with self._rate_lock:
            self._req_times = [t for t in self._req_times if now - t < 60]
            return len(self._req_times) < self.rpm and not self.is_disabled()

    def idle_seconds(self) -> float:
        """Seconds since the last successful request (0.0 if never called)."""
        if self._last_request_time == 0.0:
            return 0.0
        return time.time() - self._last_request_time

    _MAX_REQ_TIMES = 1000  # Safety cap to prevent unbounded list growth

    def _pre_request(self):
        # Issue #13: atomic check + append. Previously two threads could both
        # pass check_rate() before either recorded, exceeding the RPM ceiling.
        now = time.time()
        with self._rate_lock:
            self._req_times = [t for t in self._req_times if now - t < 60]
            # Safety cap: prevent pathological growth if pruning is starved
            if len(self._req_times) > self._MAX_REQ_TIMES:
                self._req_times = self._req_times[-self._MAX_REQ_TIMES:]
            if len(self._req_times) >= self.rpm or self.is_disabled():
                raise Exception(
                    f"{self.name}: rate limit ({self.rpm} RPM) or temporary cooldown"
                )
            self._req_times.append(now)
            self._last_request_time = now

    @abstractmethod
    async def generate(self, system: str, user: str, tier: str = None) -> str:
        pass

    @abstractmethod
    async def generate_stream(
        self, system: str, user: str, tier: str = None
    ) -> AsyncGenerator[str, None]:
        pass

    def supports_vision(self) -> bool:
        return False

    async def analyze_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        tier: str = None,
    ) -> str:
        raise NotImplementedError(f"{self.name} does not support image analysis")

    def supports_vision_stream(self) -> bool:
        return False

    async def analyze_image_stream(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        tier: str = None,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError(f"{self.name} does not support streaming image analysis")

    async def health_check(self) -> bool:
        """Lightweight availability probe. Does NOT consume RPM quota.

        Subclasses that support a dedicated ping/status endpoint should override
        this. The default simply reflects whether the provider is enabled and
        not in a cooldown period.
        """
        return self.enabled and not self.is_disabled()

    def __repr__(self):
        return f"<{self.name} spd={self.speed} qual={self.quality} ok={self.stats.success_rate:.0%}>"
