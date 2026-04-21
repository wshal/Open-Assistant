"""Rate limiter for API providers."""

import time
import asyncio
from collections import defaultdict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    def __init__(self):
        self._windows = defaultdict(list)  # provider -> [timestamps]
        self._limits = {}  # provider -> (rpm, rpd)

    def configure(self, provider: str, rpm: int = 60, rpd: int = 10000):
        self._limits[provider] = (rpm, rpd)

    def can_request(self, provider: str) -> bool:
        now = time.time()
        rpm, rpd = self._limits.get(provider, (60, 10000))

        # Clean old entries
        self._windows[provider] = [
            t for t in self._windows[provider] if now - t < 86400
        ]

        minute_count = sum(1 for t in self._windows[provider] if now - t < 60)
        day_count = len(self._windows[provider])

        return minute_count < rpm and day_count < rpd

    def record(self, provider: str):
        self._windows[provider].append(time.time())

    async def wait_if_needed(self, provider: str):
        while not self.can_request(provider):
            await asyncio.sleep(1)
        self.record(provider)