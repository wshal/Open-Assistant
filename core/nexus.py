"""
ContextNexus — v5.1 (Universal Intelligence Hub).
Time-aligned coordinator for Audio, Screen, and Process context.
"""

import time
import collections
import threading
from typing import Dict, Any, List, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ContextNexus:
    def __init__(self, config):
        self.config = config
        self.max_age = config.get("nexus.max_age_seconds", 60)
        self._lock = threading.Lock()
        
        # Buffers for different context streams
        # Format: (timestamp, source_type, data)
        self.history = collections.deque(maxlen=200)
        
        # State tracking for quick snapshots
        self.active_window = "Unknown"
        self._last_ocr = ""
        self._last_audio = ""

    def push(self, source: str, data: Any):
        """Pushes a new context event into the Nexus."""
        now = time.time()
        
        with self._lock:
            # Handle specific state updates
            if source == "window":
                self.active_window = data
            elif source == "screen":
                self._last_ocr = data
            elif source == "audio":
                self._last_audio = data
                
            self.history.append((now, source, data))
            
            # Clean up stale entries if buffer exceeds memory
            self._cleanup_stale()

    def _cleanup_stale(self):
        now = time.time()
        # Internal helper, called within lock usually, but let's be safe
        while self.history and (now - self.history[0][0]) > self.max_age:
            self.history.popleft()

    def get_snapshot(self) -> Dict[str, Any]:
        """Provides a structured multi-source snapshot for AI consumption."""
        now = time.time()
        
        with self._lock:
            self._cleanup_stale()
            
            # Filter sequences (Thread Safe under lock)
            audio_feed = [d for t, s, d in self.history if s == "audio"]
            ocr_feed = [d for t, s, d in self.history if s == "screen"]
            
            return {
                "timestamp": now,
                "active_window": self.active_window,
                "recent_audio": " ".join(audio_feed[-5:]), 
                "full_audio_history": " ".join(audio_feed),
                "latest_ocr": self._last_ocr,
                "ocr_context": " \n ".join(ocr_feed[-3:]),
                "history_depth_secs": self.max_age
            }

    def clear(self):
        with self._lock:
            self.history.clear()
            self.active_window = "Unknown"
            self._last_ocr = ""
            self._last_audio = ""
