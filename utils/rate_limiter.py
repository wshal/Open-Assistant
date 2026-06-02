"""Rate limiter for API providers."""

import time
import asyncio
import threading
from collections import defaultdict
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    # Issue #24: Shared rate-limit state was mutated without a lock, and the
    # public ``wait_if_needed`` did can_request() + record() as separate ops,
    # so parallel callers could all pass the check before any recorded. All
    # mutations now go through ``try_record`` under a single RLock.
    def __init__(self):
        self._windows = defaultdict(list)  # provider -> [timestamps]
        self._limits = {}  # provider -> (rpm, rpd)
        self._lock = threading.RLock()

    def configure(self, provider: str, rpm: int = 60, rpd: int = 10000):
        rpm = int(rpm)
        rpd = int(rpd)
        if rpm < 1:
            raise ValueError("rpm must be at least 1")
        if rpd < 1:
            raise ValueError("rpd must be at least 1")
        with self._lock:
            self._limits[provider] = (rpm, rpd)

    def can_request(self, provider: str) -> bool:
        now = time.time()
        with self._lock:
            rpm, rpd = self._limits.get(provider, (60, 10000))
            window = [t for t in self._windows[provider] if now - t < 86400]
            self._windows[provider] = window
            minute_count = sum(1 for t in window if now - t < 60)
            return minute_count < rpm and len(window) < rpd

    def record(self, provider: str):
        with self._lock:
            now = time.time()
            self._windows[provider] = [
                t for t in self._windows[provider] if now - t < 86400
            ]
            self._windows[provider].append(now)

    def try_record(self, provider: str) -> bool:
        """Atomically check capacity and record a request if allowed."""
        now = time.time()
        with self._lock:
            rpm, rpd = self._limits.get(provider, (60, 10000))
            window = [t for t in self._windows[provider] if now - t < 86400]
            minute_count = sum(1 for t in window if now - t < 60)
            if minute_count >= rpm or len(window) >= rpd:
                self._windows[provider] = window
                return False
            window.append(now)
            self._windows[provider] = window
            return True

    async def wait_if_needed(self, provider: str, max_wait_s: float = 30.0):
        """Wait until a request slot is available, or raise after max_wait_s.

        Args:
            provider:   Provider name to check capacity for.
            max_wait_s: Maximum seconds to wait before raising. Defaults to 30.
        """
        max_wait_s = max(0.0, float(max_wait_s))
        deadline = time.time() + max_wait_s
        while not self.try_record(provider):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise Exception(
                    f"Rate limiter: provider '{provider}' has no capacity "
                    f"after waiting {max_wait_s:.0f}s — quota may be exhausted"
                )
            await asyncio.sleep(min(1.0, remaining))
