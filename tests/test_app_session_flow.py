import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.app import OpenAssistApp


class HistoryStub:
    def __init__(self):
        self.started = 0
        self.state = (
            0,
            1,
            {
                "query": "example question",
                "response": "example response",
            },
        )

    def start_new_session(self):
        self.started += 1

    def get_state(self):
        return self.state

    def get_last(self, n):
        return [
            SimpleNamespace(
                query="example question",
                response="example response",
                provider="groq",
                latency=1234,
            )
        ][:n]


class AudioStub:
    def __init__(self):
        self.clear_calls = 0
        self.muted = False
        self.toggle_calls = 0

    def clear(self):
        self.clear_calls += 1

    def toggle(self):
        self.toggle_calls += 1
        self.muted = not self.muted
        return self.muted


class OverlayStub:
    def __init__(self):
        self.stack = SimpleNamespace(setCurrentIndex=self._set_index)
        self.indices = []
        self.mode_updates = []
        self.transcript_updates = []
        self.completed = []
        self.history_updates = []
        self.hide_calls = 0
        self.show_calls = 0
        self.raise_calls = 0
        self.activate_calls = 0
        self.onboarding_calls = 0
        self.status_updates = []
        self.opacity_updates = []

    def _set_index(self, index):
        self.indices.append(index)

    def update_mode(self, mode):
        self.mode_updates.append(mode)

    def update_transcript(self, text):
        self.transcript_updates.append(text)

    def on_complete(self, text, query=None):
        self.completed.append((text, query))

    def update_history_state(self, *state):
        self.history_updates.append(state)

    def hide(self):
        self.hide_calls += 1

    def show(self):
        self.show_calls += 1

    def raise_(self):
        self.raise_calls += 1

    def activateWindow(self):
        self.activate_calls += 1

    def update_audio_state(self, muted):
        pass

    def update_status(self, **kwargs):
        self.status_updates.append(kwargs)

    def setWindowOpacity(self, value):
        self.opacity_updates.append(value)

    def start_session_ui(self):
        pass

    def end_session_ui(self):
        pass

    def show_onboarding(self):
        self.onboarding_calls += 1


class MiniOverlayStub:
    def __init__(self):
        self.mode_updates = []
        self.completed = []
        self.ready_calls = 0
        self.history_updates = []
        self.hide_calls = 0
        self.show_calls = 0
        self.opacity_updates = []

    def update_mode(self, mode):
        self.mode_updates.append(mode)

    def on_complete(self, text, query=None):
        self.completed.append((text, query))

    def set_ready(self):
        self.ready_calls += 1

    def update_history_state(self, *state):
        self.history_updates.append(state)

    def hide(self):
        self.hide_calls += 1

    def show(self):
        self.show_calls += 1

    def update_audio_state(self, muted):
        pass

    def setWindowOpacity(self, value):
        self.opacity_updates.append(value)


class ConfigStub:
    def __init__(self):
        self._settings = {"ai.mode": "general"}

    def get(self, path, default=None):
        return self._settings.get(path, default)

    def set(self, path, value):
        self._settings[path] = value


class StateStub:
    def __init__(self):
        self.mode = "general"
        self.is_muted = False
        self.is_mini = False
        self.target_window_id = None
        self.is_capturing = False


class OpenAssistAppSessionFlowTests(unittest.TestCase):
    def _build_app(self, mini_mode=False):
        app = OpenAssistApp.__new__(OpenAssistApp)
        app.history = HistoryStub()
        app.audio = AudioStub()
        app.overlay = OverlayStub()
        app.mini_overlay = MiniOverlayStub()
        app.config = ConfigStub()
        app.state = StateStub()
        app.mini_mode = mini_mode
        app.session_active = False
        app._last_query = "stale query"
        app.ai = SimpleNamespace(_providers={"groq": SimpleNamespace(enabled=True)})
        app.stealth = SimpleNamespace(apply_to_window=lambda window, enabled: None)
        return app

    def test_start_new_session_resets_state_and_updates_ui(self):
        app = self._build_app()

        OpenAssistApp.start_new_session(app)

        self.assertEqual(app.history.started, 1)
        self.assertEqual(app.audio.clear_calls, 1)
        self.assertEqual(app._last_query, "")
        self.assertTrue(app.session_active)
        self.assertEqual(app.overlay.indices, [1])
        self.assertEqual(app.overlay.transcript_updates[-1], "Listening for context...")
        self.assertEqual(app.overlay.mode_updates[-1], "general")
        self.assertEqual(app.mini_overlay.mode_updates[-1], "general")
        self.assertEqual(app.mini_overlay.ready_calls, 1)
        self.assertTrue(app.state.is_capturing)

    def test_history_sync_uses_dict_entries_for_mini_overlay(self):
        app = self._build_app(mini_mode=True)

        OpenAssistApp._sync_history_ui(app)

        self.assertEqual(len(app.overlay.history_updates), 1)
        self.assertEqual(len(app.mini_overlay.history_updates), 1)
        self.assertEqual(
            app.mini_overlay.completed[-1], ("example response", "example question")
        )

    def test_end_session_resets_state_and_returns_to_standby(self):
        app = self._build_app()
        app.session_active = True

        OpenAssistApp.end_session(app)

        self.assertFalse(app.session_active)
        self.assertEqual(app.audio.clear_calls, 1)
        self.assertEqual(app._last_query, "")
        self.assertEqual(app.overlay.indices, [0])
        self.assertEqual(app.overlay.transcript_updates[-1], "Ready...")
        self.assertFalse(app.state.is_capturing)

    def test_toggle_mini_mode_switches_overlays(self):
        app = self._build_app(mini_mode=False)

        self.assertFalse(app.mini_mode)

        OpenAssistApp.toggle_mini_mode(app)
        self.assertTrue(app.mini_mode)

        OpenAssistApp.toggle_mini_mode(app)
        self.assertFalse(app.mini_mode)

    def test_switch_mode_changes_mode_in_overlay(self):
        app = self._build_app()

        OpenAssistApp.switch_mode(app, "coding")

        self.assertEqual(app.overlay.mode_updates[-1], "coding")
        self.assertEqual(app.mini_overlay.mode_updates[-1], "coding")

    def test_toggle_audio_toggles_audio_state(self):
        app = self._build_app()

        OpenAssistApp.toggle_audio(app)
        self.assertTrue(app.audio.muted)

        OpenAssistApp.toggle_audio(app)
        self.assertFalse(app.audio.muted)

    def test_show_initial_window_respects_start_minimized(self):
        app = self._build_app()
        app.config.set("onboarding.completed", True)
        app.config.set("app.start_minimized", True)

        OpenAssistApp._show_initial_window(app)

        self.assertEqual(app.overlay.hide_calls, 1)
        self.assertEqual(app.mini_overlay.hide_calls, 1)
        self.assertEqual(app.overlay.show_calls, 0)

    def test_show_initial_window_forces_onboarding(self):
        app = self._build_app()
        app.config.set("onboarding.completed", False)
        app.config.set("app.start_minimized", True)

        OpenAssistApp._show_initial_window(app)

        self.assertEqual(app.overlay.show_calls, 1)
        self.assertEqual(app.overlay.onboarding_calls, 1)
        self.assertEqual(app.overlay.raise_calls, 1)
        self.assertEqual(app.overlay.activate_calls, 1)

    def test_open_settings_returns_to_full_overlay(self):
        app = self._build_app(mini_mode=True)

        OpenAssistApp.open_settings(app)

        self.assertEqual(app.mini_overlay.hide_calls, 1)
        self.assertEqual(app.overlay.indices, [2])
        self.assertEqual(app.overlay.show_calls, 1)
        self.assertEqual(app.overlay.raise_calls, 1)
        self.assertEqual(app.overlay.activate_calls, 1)

    def test_response_complete_uses_app_capture_state_for_status(self):
        app = self._build_app()
        app.state.is_capturing = True
        app.config.set("capture.audio.enabled", True)
        app.config.set("capture.screen.enabled", True)

        OpenAssistApp._on_response_complete(app, "done")

        update = app.overlay.status_updates[-1]
        self.assertTrue(update["capture_audio"])
        self.assertTrue(update["capture_screen"])
        self.assertEqual(update["provider"], "groq")
        self.assertEqual(update["latency_ms"], 1234)

    def test_toggle_stealth_mode_updates_state_and_window_opacity(self):
        app = self._build_app()
        app.state.is_stealth = False
        app.config.set("stealth.low_opacity", 0.75)
        app.config.set("app.opacity", 0.94)

        OpenAssistApp.toggle_stealth_mode(app)

        self.assertTrue(app.state.is_stealth)
        self.assertEqual(app.overlay.opacity_updates[-1], 0.75)
        self.assertEqual(app.mini_overlay.opacity_updates[-1], 0.75)

        OpenAssistApp.toggle_stealth_mode(app)

        self.assertFalse(app.state.is_stealth)
        self.assertEqual(app.overlay.opacity_updates[-1], 0.94)
        self.assertEqual(app.mini_overlay.opacity_updates[-1], 0.94)


if __name__ == "__main__":
    unittest.main()
