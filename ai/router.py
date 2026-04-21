"""Smart router â picks optimal provider per request."""

from typing import Optional, List, Tuple
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SmartRouter:
    def __init__(self, config, providers: dict):
        self.config = config
        self.providers = providers
        self.strategy = config.get("ai.strategy", "smart")
        self.offline_first = config.get("ai.offline_first", True)  # P2: Prefer local
        self.task_map = config.get("ai.router.task_routing", {})
        self.fallback = config.get("ai.router.fallback_order", [])
        self._rr = 0

    def _is_offline_available(self) -> bool:
        """Check if local/offline provider is available."""
        return "ollama" in self.providers and self.providers["ollama"].enabled

    def select(
        self,
        task: str = "general",
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        tier: str = None,
        exclude: List[str] = None,
        preferred: List[str] = None,
    ) -> Tuple[Optional[BaseProvider], str]:
        exclude = exclude or []
        preferred = preferred or []
        avail = {
            n: p
            for n, p in self.providers.items()
            if p.enabled and p.check_rate() and n not in exclude
        }

        if not avail:
            return None, ""

        # Ensure tier is respected when a provider supports it
        if tier:
            tier_support = {n: p for n, p in avail.items() if p.has_model(tier)}
            if tier_support:
                avail = tier_support
            else:
                logger.debug(
                    f"Router: no available provider supports requested tier '{tier}', using all available providers"
                )

        # Fixed provider
        if self.strategy == "fixed":
            fixed = self.config.get("ai.fixed_provider", "")
            if fixed:
                if fixed in avail:
                    return avail[fixed], self._selected_tier(avail[fixed], tier)
                logger.warning(
                    f"Router fixed provider '{fixed}' unavailable; falling back to smart routing"
                )

        if self.strategy == "fastest":
            p = max(avail.values(), key=lambda x: x.speed)
            return p, self._selected_tier(p, tier or "fast")

        if self.strategy == "roundrobin":
            names = list(avail.keys())
            self._rr = (self._rr + 1) % len(names)
            provider = avail[names[self._rr]]
            return provider, self._selected_tier(provider, tier or "balanced")

        if self.strategy == "fallback":
            for n in self.fallback:
                if n in avail:
                    provider = avail[n]
                    return provider, self._selected_tier(provider, tier or "balanced")
            provider = next(iter(avail.values()))
            return provider, self._selected_tier(provider, tier or "balanced")

        # P2: Offline-first strategy - prefer local (Ollama), then fastest cloud
        if self.strategy == "offline" or (
            self.offline_first and self.strategy == "smart"
        ):
            if "ollama" in avail:
                logger.debug("Router: Using offline-first (Ollama)")
                return avail["ollama"], self._selected_tier(
                    avail["ollama"], tier or "balanced"
                )
            if self.strategy == "offline":
                provider = max(avail.values(), key=lambda x: x.speed)
                return provider, self._selected_tier(provider, tier or "balanced")

        # Smart strategy
        return self._smart(avail, task, prefer_speed, prefer_quality, tier, preferred)

    def _smart(self, avail, task, prefer_speed, prefer_quality, tier, preferred):
        task_preferred = self.task_map.get(task, self.task_map.get("general", []))
        scored = []

        for name, p in avail.items():
            score = 0.0
            if name in preferred:
                score += (len(preferred) - preferred.index(name)) * 20
            if name in task_preferred:
                score += (len(task_preferred) - task_preferred.index(name)) * 10
            if prefer_speed:
                score += p.speed * 5 + p.quality * 1
            elif prefer_quality:
                score += p.speed * 1 + p.quality * 5
            else:
                score += p.speed * 3 + p.quality * 3

            if p.stats.requests > 0:
                score += p.stats.success_rate * 10
                if p.stats.tps > 100:
                    score += 5

            remaining = p.rpm - len(p._req_times)
            if remaining < p.rpm * 0.1:
                score -= 15

            scored.append((name, p, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        name, provider, score = scored[0]

        t = tier or self._tier_for_task(task)
        if not provider.models.get(t):
            t = provider.default_tier

        return provider, t

    @staticmethod
    def _selected_tier(provider: BaseProvider, tier: str = None) -> str:
        if provider.has_model(tier):
            return tier or provider.default_tier
        return provider.default_tier

    @staticmethod
    def _tier_for_task(task):
        return {
            "quick": "fast",
            "meeting": "fast",
            "coding": "code",
            "reasoning": "reasoning",
            "interview": "balanced",
            "exam": "balanced",
            "writing": "balanced",
            "general": "balanced",
        }.get(task, "balanced")

    def get_stats(self):
        return {
            n: {
                "requests": p.stats.requests,
                "errors": p.stats.errors,
                "latency": f"{p.stats.avg_latency:.2f}s",
                "tps": f"{p.stats.tps:.0f}",
                "success": f"{p.stats.success_rate:.0%}",
                "speed": p.speed,
                "quality": p.quality,
            }
            for n, p in self.providers.items()
        }

    def get_provider_health(self):
        health = {}
        for name, p in self.providers.items():
            if not p.enabled:
                state = "disabled"
            elif p.is_disabled():
                state = "cooldown"
            elif not p.check_rate():
                state = "rate_limited"
            else:
                state = "active"

            health[name] = {
                "state": state,
                "requests": p.stats.requests,
                "errors": p.stats.errors,
                "success": f"{p.stats.success_rate:.0%}",
            }
        return health
