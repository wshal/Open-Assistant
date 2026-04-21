"""Base provider with stats tracking and rate limiting."""

import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from dataclasses import dataclass
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

    def record(self, tokens: int, latency: float):
        self.requests += 1
        self.total_tokens += tokens
        self.total_time += latency
        self.last_latency = latency
        if latency > 0:
            self.tps = self.tps * 0.7 + (tokens / latency) * 0.3

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
        self.failure_cooldown = self.pcfg.get("failure_cooldown", 30)

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

    def disable(self, seconds: int = None):
        duration = seconds if seconds is not None else self.failure_cooldown
        self._disabled_until = time.time() + duration
        logger.warning(
            f"{self.name}: temporarily disabled for {duration}s after failure"
        )

    def check_rate(self) -> bool:
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60]
        return len(self._req_times) < self.rpm and not self.is_disabled()

    def _pre_request(self):
        if not self.check_rate():
            raise Exception(
                f"{self.name}: rate limit ({self.rpm} RPM) or temporary cooldown"
            )
        self._req_times.append(time.time())

    @abstractmethod
    async def generate(self, system: str, user: str, tier: str = None) -> str:
        pass

    @abstractmethod
    async def generate_stream(
        self, system: str, user: str, tier: str = None
    ) -> AsyncGenerator[str, None]:
        pass

    async def health_check(self) -> bool:
        try:
            r = await self.generate("Say ok.", "ok", "fast")
            return bool(r)
        except Exception:
            return False

    def __repr__(self):
        return f"<{self.name} spd={self.speed} qual={self.quality} ok={self.stats.success_rate:.0%}>"
