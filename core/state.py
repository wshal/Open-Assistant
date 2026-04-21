"""
Centralized Application State Model — v5.1.
Single source of truth for UI synchronization and engine behavior.
"""

from PyQt6.QtCore import QObject, pyqtSignal
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
    
    def __init__(self, config=None):
        super().__init__()
        self._config = config
        
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
            self._is_stealth = config.get("stealth.enabled", True)
            

    # Properties with signal emission
    @property
    def target_window_id(self): return self._target_window_id
    @target_window_id.setter
    def target_window_id(self, val):
        self._target_window_id = val
        self.state_changed.emit("target_window_id", val)

    @property
    def mode(self): return self._mode
    @mode.setter
    def mode(self, val):
        if self._mode != val:
            self._mode = val
            if self._config: self._config.set("ai.mode", val)
            self.mode_changed.emit(val)
            self.state_changed.emit("mode", val)

    @property
    def audio_source(self): return self._audio_source
    @audio_source.setter
    def audio_source(self, val):
        if self._audio_source != val:
            self._audio_source = val
            if self._config: self._config.set("capture.audio.mode", val)
            self.audio_source_changed.emit(val)
            self.state_changed.emit("audio_source", val)

    @property
    def is_muted(self): return self._is_muted
    @is_muted.setter
    def is_muted(self, val):
        if self._is_muted != val:
            self._is_muted = val
            self.muted_changed.emit(val)
            self.state_changed.emit("is_muted", val)

    @property
    def is_capturing(self): return self._is_capturing
    @is_capturing.setter
    def is_capturing(self, val):
        if self._is_capturing != val:
            self._is_capturing = val
            self.capturing_changed.emit(val)
            self.state_changed.emit("is_capturing", val)

    @property
    def is_stealth(self): return self._is_stealth
    @is_stealth.setter
    def is_stealth(self, val):
        if self._is_stealth != val:
            self._is_stealth = val
            if self._config: self._config.set("stealth.enabled", val)
            self.stealth_changed.emit(val)
            self.state_changed.emit("is_stealth", val)

    @property
    def is_mini(self): return self._is_mini
    @is_mini.setter
    def is_mini(self, val):
        if self._is_mini != val:
            self._is_mini = val
            self.hud_mode_changed.emit(val)
            self.state_changed.emit("is_mini", val)

    def update_provider_health(self, health: Dict[str, Any]):
        self._provider_health = health
        self.state_changed.emit("provider_health", health)

    @property
    def provider_health(self): return self._provider_health

    @property
    def session_context(self) -> str:
        return self._session_context

    @session_context.setter
    def session_context(self, val: str):
        val = (val or "").strip()
        if self._session_context != val:
            self._session_context = val
            self.session_context_changed.emit(val)
            self.state_changed.emit("session_context", val)
