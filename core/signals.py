"""
Signal architecture documentation.

DESIGN DECISION (v4.0):
    We use DIRECT signal connections between components rather than
    a centralized SignalBus. Here's why:

    Direct connections (current approach):
        self.ai.response_chunk.connect(self.overlay.append_response)

    Centralized bus (rejected for now):
        signals.response_chunk.connect(self.overlay.append_response)
        self.ai.response_chunk.connect(signals.response_chunk.emit)

    Direct is better because:
      1. Fewer indirection layers = easier debugging
      2. Signal connections are explicit in core/app.py._connect_signals()
      3. No risk of bus becoming a god object
      4. Type safety â each component defines its own signal signatures

    The SignalBus pattern would be useful IF we add:
      - Plugin system (plugins subscribe to bus events)
      - Multiple overlay windows (each subscribes independently)
      - Event logging/replay (bus can record all events)

    For now, core/app.py._connect_signals() is the single place
    where all wiring is documented. That's clean enough.

FUTURE:
    If plugins are added, create a PluginSignalBus here that
    plugins register with. Keep core signals direct.
"""

# Intentionally empty â all signals are on their respective QObject classes:
#   AIEngine:       response_chunk, response_complete, error_occurred, provider_info
#   ScreenCapture:  text_captured
#   AudioCapture:   transcription_ready, level
#   WindowDetector: window_changed, category_changed
#   OverlayWindow:  user_query