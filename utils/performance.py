"""Performance monitoring."""

import time
import functools
import psutil
from threading import Timer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceMonitor:
    def __init__(self, config):
        self.max_memory = config.get("performance.max_memory_mb", 1024)
        self.process = psutil.Process()

    def get_stats(self) -> dict:
        return {
            "memory_mb": round(self.process.memory_info().rss / 1024 / 1024, 1),
            "cpu_percent": round(self.process.cpu_percent(), 1),
        }


def debounce(wait: float):
    def decorator(fn):
        timer = None
        @functools.wraps(fn)
        def debounced(*args, **kwargs):
            nonlocal timer
            if timer:
                timer.cancel()
            timer = Timer(wait, fn, args, kwargs)
            timer.start()
        return debounced
    return decorator