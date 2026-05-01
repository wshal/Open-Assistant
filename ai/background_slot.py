import asyncio
import threading
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BackgroundGenerationSlot:
    """Holds at most one in-flight background generation task.
    When a new task is assigned, the old one is cancelled.
    """
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._cancel_flag = threading.Event()

    def assign(self, coro, loop: asyncio.AbstractEventLoop) -> None:
        """Cancel existing BG task and start new one."""
        self.cancel()
        self._cancel_flag = threading.Event()
        try:
            self._task = asyncio.run_coroutine_threadsafe(coro, loop)
            logger.debug("[BG Slot] Assigned new background generation task")
        except Exception as e:
            logger.error(f"[BG Slot] Failed to assign task: {e}")

    def cancel(self) -> None:
        """Cancel the current background task if running."""
        if self._cancel_flag:
            self._cancel_flag.set()  # signal the BG coroutine to stop
        if self._task and not self._task.done():
            try:
                self._task.cancel()
                logger.debug("[BG Slot] Cancelled existing background task")
            except Exception:
                pass
        self._task = None

    def is_cancelled(self) -> bool:
        return self._cancel_flag.is_set()
