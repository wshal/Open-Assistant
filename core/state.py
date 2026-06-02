"""
Centralized Application State Model — v5.1.
Single source of truth for UI synchronization and engine behavior.
"""

import threading

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread, QCoreApplication
from typing import Dict, Any


class AppState(QObject):
    # Signals for UI synchronization
    state_changed = pyqtSignal(str, object)  # key, value
    mode_changed = pyqtSignal(str)
    audio_source_changed = pyqtSignal(str)
    muted_changed = pyqtSignal(bool)
    capturing_changed = pyqtSignal(bool)
    stealth_changed = pyqtSignal(bool)
    hud_mode_changed = pyqtSignal(bool)  # True = Mini, False = Overlay
    session_context_changed = pyqtSignal(str)
    _dispatch_requested = pyqtSignal(object)

    def __init__(self, config=None):
        super().__init__()
        # Issue #21: AppState is mutated from audio, AI, hotkey, warmup and UI
        # threads. _lock serialises field reads/writes; _dispatch() ensures Qt
        # signal emissions and config writes run on the owner thread.
        self._lock = threading.RLock()
        self._config = config
        self._dispatch_requested.connect(lambda fn: fn())

        # Internal State
        self._mode = "general"
        self._audio_source = "system"
        self._is_muted = False
        self._is_capturing = False
        self._is_stealth = True
        self._is_mini = False
        self._target_window_id = 0
        self._provider_health = {}
        self._session_context = ""  # Custom per-session instructions

        # Load initials from config if provided
        if config:
            self._mode = config.get("ai.mode", "general")
            self._audio_source = config.get("capture.audio.mode", "system")
            self._is_muted = bool(config.get("capture.audio.muted", False))
            self._is_stealth = config.get("stealth.enabled", True)

    # ── Thread-affinity helpers ──────────────────────────────────────────────
    def _on_owner_thread(self) -> bool:
        app = QCoreApplication.instance()
        return bool(app) and QThread.currentThread() == self.thread()

    def _dispatch(self, fn):
        """Run ``fn`` on the QObject's owning thread, immediately if already there.

        Falls back to a direct synchronous call when no QCoreApplication exists
        (e.g., unit-test environments) so that setters are never silent no-ops.
        """
        app = QCoreApplication.instance()
        if not app:
            # No Qt event loop — apply directly (test environments).
            fn()
        elif self._on_owner_thread():
            fn()
        else:
            self._dispatch_requested.emit(fn)

    # Properties with signal emission
    @property
    def target_window_id(self):
        with self._lock:
            return self._target_window_id

    @target_window_id.setter
    def target_window_id(self, val):
        def apply():
            with self._lock:
                self._target_window_id = val
            self.state_changed.emit("target_window_id", val)
        self._dispatch(apply)

    @property
    def mode(self):
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, val):
        def apply():
            with self._lock:
                if self._mode == val:
                    return
                self._mode = val
                if self._config:
                    self._config.set("ai.mode", val)
            self.mode_changed.emit(val)
            self.state_changed.emit("mode", val)
        self._dispatch(apply)

    @property
    def audio_source(self):
        with self._lock:
            return self._audio_source

    @audio_source.setter
    def audio_source(self, val):
        def apply():
            with self._lock:
                if self._audio_source == val:
                    return
                self._audio_source = val
                if self._config:
                    self._config.set("capture.audio.mode", val)
            self.audio_source_changed.emit(val)
            self.state_changed.emit("audio_source", val)
        self._dispatch(apply)

    @property
    def is_muted(self):
        with self._lock:
            return self._is_muted

    @is_muted.setter
    def is_muted(self, val):
        def apply():
            with self._lock:
                if self._is_muted == val:
                    return
                self._is_muted = val
            self.muted_changed.emit(val)
            self.state_changed.emit("is_muted", val)
        self._dispatch(apply)

    @property
    def is_capturing(self):
        with self._lock:
            return self._is_capturing

    @is_capturing.setter
    def is_capturing(self, val):
        def apply():
            with self._lock:
                if self._is_capturing == val:
                    return
                self._is_capturing = val
            self.capturing_changed.emit(val)
            self.state_changed.emit("is_capturing", val)
        self._dispatch(apply)

    @property
    def is_stealth(self):
        with self._lock:
            return self._is_stealth

    @is_stealth.setter
    def is_stealth(self, val):
        def apply():
            with self._lock:
                if self._is_stealth == val:
                    return
                self._is_stealth = val
                if self._config:
                    self._config.set("stealth.enabled", val)
            self.stealth_changed.emit(val)
            self.state_changed.emit("is_stealth", val)
        self._dispatch(apply)

    @property
    def is_mini(self):
        with self._lock:
            return self._is_mini

    @is_mini.setter
    def is_mini(self, val):
        def apply():
            with self._lock:
                if self._is_mini == val:
                    return
                self._is_mini = val
            self.hud_mode_changed.emit(val)
            self.state_changed.emit("is_mini", val)
        self._dispatch(apply)

    def update_provider_health(self, health: Dict[str, Any]):
        def apply():
            with self._lock:
                self._provider_health = health
            self.state_changed.emit("provider_health", health)
        self._dispatch(apply)

    @property
    def provider_health(self):
        with self._lock:
            return self._provider_health

    @property
    def session_context(self) -> str:
        with self._lock:
            return self._session_context

    @session_context.setter
    def session_context(self, val: str):
        val = (val or "").strip()

        def apply():
            with self._lock:
                if self._session_context == val:
                    return
                self._session_context = val
            self.session_context_changed.emit(val)
            self.state_changed.emit("session_context", val)
        self._dispatch(apply)
