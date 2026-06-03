"""Rate limiter for API providers."""

import time
import asyncio
import threading
from collections import defaultdict, deque
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    # Issue #24: Shared rate-limit state was mutated without a lock, and the
    # public ``wait_if_needed`` did can_request() + record() as separate ops,
    # so parallel callers could all pass the check before any recorded. All
    # mutations now go through ``try_record`` under a single RLock.
    def __init__(self):
        # Keep separate sliding windows so each check is O(k) over the
        # current window, not O(n) over a long history list.
        self._windows_24h = defaultdict(deque)  # provider -> deque[timestamps]
        self._windows_1m = defaultdict(deque)   # provider -> deque[timestamps]
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
            window_24h = self._windows_24h[provider]
            window_1m = self._windows_1m[provider]
            cutoff_24h = now - 86400
            while window_24h and window_24h[0] < cutoff_24h:
                window_24h.popleft()
            cutoff_1m = now - 60
            while window_1m and window_1m[0] < cutoff_1m:
                window_1m.popleft()
            return len(window_1m) < rpm and len(window_24h) < rpd

    def record(self, provider: str):
        with self._lock:
            now = time.time()
            window_24h = self._windows_24h[provider]
            window_1m = self._windows_1m[provider]
            cutoff_24h = now - 86400
            while window_24h and window_24h[0] < cutoff_24h:
                window_24h.popleft()
            window_24h.append(now)
            window_1m.append(now)

    def try_record(self, provider: str) -> bool:
        """Atomically check capacity and record a request if allowed."""
        now = time.time()
        with self._lock:
            rpm, rpd = self._limits.get(provider, (60, 10000))
            window_24h = self._windows_24h[provider]
            window_1m = self._windows_1m[provider]
            cutoff_24h = now - 86400
            while window_24h and window_24h[0] < cutoff_24h:
                window_24h.popleft()
            cutoff_1m = now - 60
            while window_1m and window_1m[0] < cutoff_1m:
                window_1m.popleft()
            if len(window_1m) >= rpm or len(window_24h) >= rpd:
                return False
            window_24h.append(now)
            window_1m.append(now)
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
