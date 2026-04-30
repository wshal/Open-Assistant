import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        self.restart_calls = 0

    def clear(self):
        self.clear_calls += 1

    def toggle(self):
        self.toggle_calls += 1
        self.muted = not self.muted
        return self.muted

    def restart(self):
        self.restart_calls += 1


class OverlayStub:
    def __init__(self):
        self.current_widget = "standby"
        self.stack = SimpleNamespace(
            setCurrentIndex=self._set_index,
            currentIndex=lambda: 0,
            currentWidget=lambda: self.current_widget,
        )
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
        self.analysis_badges = []
        self.refresh_calls = []
        self.chat_view = "chat"
        self.settings_view = "settings"
        self.standby_view = "standby"
        self.visible = True
        self.opacity = 1.0
        # Internal state fields accessed directly by start_new_session / end_session
        self._current_query = ""
        self._raw_buffer = ""
        self._is_streaming = False
        self.response_area = SimpleNamespace(clear=lambda: None)

    def _set_index(self, index):
        self.indices.append(index)

    def update_mode(self, mode):
        self.mode_updates.append(mode)

    def update_transcript(self, text, state="auto"):
        self.transcript_updates.append(text)

    def on_complete(self, text, query=None):
        self.completed.append((text, query))

    def update_history_state(self, *state):
        self.history_updates.append(state)

    def hide(self):
        self.hide_calls += 1
        self.visible = False

    def show(self):
        self.show_calls += 1
        self.visible = True

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
        self.opacity = value

    def isVisible(self):
        return self.visible

    def windowOpacity(self):
        return self.opacity

    def start_session_ui(self):
        pass

    def end_session_ui(self):
        pass

    def show_onboarding(self):
        self.onboarding_calls += 1

    def set_analysis_provider_badge(self, provider=None, pending=False):
        self.analysis_badges.append({"provider": provider, "pending": pending})

    def show_chat_view(self):
        self.indices.append(1)

    def show_standby_view(self):
        self.indices.append(0)

    def show_settings_view(self):
        self.indices.append(2)

    def refresh_standby_state(self, mode=None, audio=None):
        self.refresh_calls.append({"mode": mode, "audio": audio})


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
        self.reset_calls = 0
        self.save_calls = 0

    def get(self, path, default=None):
        return self._settings.get(path, default)

    def set(self, path, value):
        self._settings[path] = value

    def save(self):
        self.save_calls += 1

    def reset_all(self):
        self.reset_calls += 1


class StateStub:
    def __init__(self, config=None):
        self._config = config
        self.mode = "general"
        self.is_muted = False
        self.is_mini = False
        self.target_window_id = None
        self.is_capturing = False
        self._audio_source = "system"
        self.session_context = ""  # Added for session context feature

    @property
    def audio_source(self):
        return self._audio_source

    @audio_source.setter
    def audio_source(self, value):
        self._audio_source = value
        if self._config:
            self._config.set("capture.audio.mode", value)


class ModeManagerStub:
    """Minimal stub for ModeManager — supports switch() and profile access."""
    def __init__(self):
        self._current_name = "general"

    def switch(self, name):
        self._current_name = name
        from types import SimpleNamespace
        return SimpleNamespace(
            name=name,
            detector_sensitivity=0.5,
            ollama_model_hint="llama3",
            vad_silence_ms=900,
        )

    @property
    def current(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            name=self._current_name,
            detector_sensitivity=0.5,
            ollama_model_hint="llama3",
            vad_silence_ms=900,
        )

    @property
    def current_name(self):
        return self._current_name


class OpenAssistAppSessionFlowTests(unittest.TestCase):
    def _build_app(self, mini_mode=False):
        app = SimpleNamespace(
            history=HistoryStub(),
            audio=AudioStub(),
            overlay=OverlayStub(),
            mini_overlay=MiniOverlayStub(),
            config=ConfigStub(),
            state=None,
            mini_mode=mini_mode,
            session_active=False,
            _last_query="stale query",
            _generation_epoch=0,
            _screen_analysis_pending=False,
            _click_through=False,
            loop=SimpleNamespace(is_running=lambda: True),
            _do_analyze_screen=lambda: __import__("asyncio").sleep(0),
            modes=ModeManagerStub(),
            ai=SimpleNamespace(
                _providers={"groq": SimpleNamespace(enabled=True)},
                cancel=lambda: None,
                _rag_cache={},
                clear_rag_prefetch=lambda: None,
                set_session_context=lambda ctx: None,
                detector=SimpleNamespace(
                    set_mode=lambda m: None,
                    question_prefixes=["what ", "how ", "why ", "can ", "could "],
                    question_patterns=["what is", "how do", "can you", "could you"],
                ),
            ),
            rag=SimpleNamespace(_cache={}, stop=lambda: None),
            nexus=SimpleNamespace(clear=lambda: None),
            hotkeys=SimpleNamespace(stop=lambda: None, reset_state=lambda: None),
            screen=SimpleNamespace(),
            qt_app=SimpleNamespace(quit=lambda: None),
            stealth=SimpleNamespace(apply_to_window=lambda window, enabled: None),
        )
        app.state = StateStub(app.config)
        app._context_auto_suggested = False  # auto-suggest flag
        app._context_store = SimpleNamespace(
            get_last_context=lambda: "",
            set_last_context=lambda t: None,
        )
        app._apply_window_effects = lambda window: OpenAssistApp._apply_window_effects(
            app, window
        )
        app._apply_ui_only = lambda: OpenAssistApp._apply_ui_only(app)
        app._active_view = lambda: OpenAssistApp._active_view(app)
        app._hud_focus_enabled = lambda: OpenAssistApp._hud_focus_enabled(app)
        app._present_window = lambda window, focus=False: OpenAssistApp._present_window(
            app, window, focus
        )
        app._sync_history_ui = lambda: OpenAssistApp._sync_history_ui(app)
        app._stop_background_tasks = lambda: None
        app._sync_state_from_config = lambda: OpenAssistApp._sync_state_from_config(app)
        app._should_ignore_final_transcript = lambda text: OpenAssistApp._should_ignore_final_transcript(app, text)
        app._looks_question_like_transcript = lambda text: OpenAssistApp._looks_question_like_transcript(app, text)
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

    def test_quick_answer_prefers_cached_context_without_fresh_capture(self):
        app = self._build_app()
        app._ai_lock_ready = SimpleNamespace(wait=lambda timeout=2: True)
        app.loop = object()
        app.session_active = True  # P0.2: guard requires active session
        app.nexus = SimpleNamespace(
            get_snapshot=lambda: {
                "recent_audio": "latest meeting question",
                "full_audio_history": "latest meeting question",
                "latest_ocr": "visible code",
            }
        )
        app.audio = SimpleNamespace(get_transcript=lambda: "")
        captures = []
        app.screen = SimpleNamespace(capture_context=lambda: captures.append(True))
        quick_calls = []

        async def fake_quick(snapshot, screen_context="", audio_context=""):
            quick_calls.append(
                {
                    "screen": screen_context,
                    "audio": audio_context,
                    "snapshot": snapshot,
                }
            )

        app.ai = SimpleNamespace(generate_quick_response=fake_quick)

        with patch(
            "core.app.asyncio.run_coroutine_threadsafe",
            side_effect=lambda coro, loop: __import__("asyncio").run(coro),
        ):
            OpenAssistApp.quick_answer(app)

        self.assertEqual(captures, [])
        self.assertEqual(quick_calls[0]["audio"], "latest meeting question")
        self.assertEqual(quick_calls[0]["screen"], "visible code")
        self.assertIn(
            "Quick answer using cached audio + cached screen...",
            app.overlay.transcript_updates,
        )

    def test_quick_answer_announces_fallback_capture_when_cached_context_missing(self):
        app = self._build_app()
        app._ai_lock_ready = SimpleNamespace(wait=lambda timeout=2: True)
        app.loop = object()
        app.session_active = True  # P0.2: guard requires active session
        app.nexus = SimpleNamespace(
            get_snapshot=lambda: {
                "recent_audio": "",
                "full_audio_history": "",
                "latest_ocr": "",
            },
            push=lambda source, value: None,
        )
        app.audio = SimpleNamespace(get_transcript=lambda: "")

        async def fake_capture_context():
            return "fresh screen"

        app.screen = SimpleNamespace(capture_context=fake_capture_context)

        async def fake_quick(snapshot, screen_context="", audio_context=""):
            return ""

        app.ai = SimpleNamespace(generate_quick_response=fake_quick)

        with patch(
            "core.app.asyncio.run_coroutine_threadsafe",
            side_effect=lambda coro, loop: __import__("asyncio").run(coro),
        ):
            OpenAssistApp.quick_answer(app)

        self.assertIn(
            "Quick answer missing cached context. Refreshing screen once...",
            app.overlay.transcript_updates,
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
        self.assertEqual(len(app.overlay.history_updates), 1)
        self.assertEqual(len(app.mini_overlay.history_updates), 1)
        self.assertEqual(
            app.mini_overlay.completed[-1], ("example response", "example question")
        )

        OpenAssistApp.toggle_mini_mode(app)
        self.assertFalse(app.mini_mode)

    def test_switch_mode_changes_mode_in_overlay(self):
        app = self._build_app()

        OpenAssistApp.switch_mode(app, "coding")

        self.assertEqual(app.overlay.mode_updates[-1], "coding")
        self.assertEqual(app.mini_overlay.mode_updates[-1], "coding")

    def test_audio_source_ui_change_persists_and_restarts_audio(self):
        app = self._build_app()

        OpenAssistApp._on_audio_source_ui_change(app, "mic")

        self.assertEqual(app.state.audio_source, "mic")
        self.assertEqual(app.config.get("capture.audio.mode"), "mic")
        self.assertEqual(app.config.save_calls, 1)
        self.assertEqual(app.audio.restart_calls, 1)

    def test_toggle_audio_toggles_audio_state(self):
        app = self._build_app()

        OpenAssistApp.toggle_audio(app)
        self.assertTrue(app.audio.muted)

        OpenAssistApp.toggle_audio(app)
        self.assertFalse(app.audio.muted)

    def test_short_final_transcript_fragment_is_ignored(self):
        app = self._build_app()
        app.session_active = True
        pushed = []
        generated = []
        app.nexus = SimpleNamespace(push=lambda source, value: pushed.append((source, value)))
        app.generate_response = lambda *args, **kwargs: generated.append((args, kwargs))
        app.audio = SimpleNamespace(get_last_transcription_metrics=lambda: {})
        app.ai.detector = SimpleNamespace(
            question_prefixes=["what ", "how ", "why "],
            question_patterns=["what is", "how do"],
            detect_with_confidence=lambda text, source="audio": SimpleNamespace(
                triggered=False,
                confidence=0.0,
                detected_text="",
                should_auto_respond=lambda: False,
            ),
        )

        OpenAssistApp._on_transcription(app, "API.")

        self.assertEqual(pushed, [])
        self.assertEqual(generated, [])
        self.assertEqual(app.overlay.transcript_updates, [])

    def test_continuation_fragment_transcript_is_ignored(self):
        app = self._build_app()
        app.session_active = True
        resets = []
        app.nexus = SimpleNamespace(push=lambda source, value: None)
        app.generate_response = lambda *args, **kwargs: None
        app.audio = SimpleNamespace(get_last_transcription_metrics=lambda: {})
        app.ai.detector = SimpleNamespace(
            question_prefixes=["what ", "how ", "why "],
            question_patterns=["what is", "how do"],
            reset_fragment_buffer=lambda reason="": resets.append(reason),
            detect_with_confidence=lambda text, source="audio": SimpleNamespace(
                triggered=False,
                confidence=0.0,
                detected_text="",
                should_auto_respond=lambda: False,
            ),
        )

        OpenAssistApp._on_transcription(app, "for Cloud Helps.")

        self.assertEqual(app.overlay.transcript_updates, [])
        self.assertEqual(resets, ["ignored-final-fragment"])

    def test_short_question_transcript_still_reaches_detector(self):
        app = self._build_app()
        app.session_active = True
        pushed = []
        generated = []
        detector_calls = []
        app.nexus = SimpleNamespace(push=lambda source, value: pushed.append((source, value)), get_snapshot=lambda: {})
        app.generate_response = lambda *args, **kwargs: generated.append((args, kwargs))
        app.audio = SimpleNamespace(get_last_transcription_metrics=lambda: {})
        app.ai.detector = SimpleNamespace(
            question_prefixes=["what ", "how ", "why "],
            question_patterns=["what is", "how do"],
            detect_with_confidence=lambda text, source="audio": (
                detector_calls.append((text, source)) or SimpleNamespace(
                    triggered=True,
                    confidence=1.0,
                    detected_text=text,
                    should_auto_respond=lambda: True,
                )
            ),
        )

        OpenAssistApp._on_transcription(app, "Why?")

        self.assertEqual(pushed, [("audio", "Why?")])
        self.assertEqual(detector_calls, [("Why?", "audio")])
        self.assertEqual(len(generated), 1)

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

    def test_show_active_overlay_reasserts_topmost_before_raising(self):
        app = self._build_app(mini_mode=False)

        with patch("core.app.WindowUtils.ensure_topmost") as ensure_topmost:
            result = OpenAssistApp._show_active_overlay(app)

        self.assertIs(result, app.overlay)
        ensure_topmost.assert_called_once_with(app.overlay)
        self.assertEqual(app.overlay.show_calls, 1)
        self.assertEqual(app.overlay.raise_calls, 1)
        self.assertEqual(app.overlay.activate_calls, 0)

    def test_present_window_only_focuses_when_requested(self):
        app = self._build_app()

        with patch("core.app.WindowUtils.ensure_topmost") as ensure_topmost:
            OpenAssistApp._present_window(app, app.overlay, focus=False)
            OpenAssistApp._present_window(app, app.overlay, focus=True)

        self.assertEqual(ensure_topmost.call_count, 2)
        self.assertEqual(app.overlay.show_calls, 2)
        self.assertEqual(app.overlay.raise_calls, 2)
        self.assertEqual(app.overlay.activate_calls, 1)

    def test_show_active_overlay_respects_focus_on_show_config(self):
        app = self._build_app(mini_mode=False)

        app.config.set("app.focus_on_show", False)
        OpenAssistApp._show_active_overlay(app)
        self.assertEqual(app.overlay.activate_calls, 0)

        app.config.set("app.focus_on_show", True)
        OpenAssistApp._show_active_overlay(app)
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

    def test_screen_analysis_badge_updates_during_flow(self):
        app = self._build_app()
        app.session_active = True
        app._generation_epoch = 0

        with patch(
            "core.app.asyncio.run_coroutine_threadsafe",
            side_effect=lambda coro, loop: coro.close(),
        ):
            OpenAssistApp.analyze_current_screen(app)
        self.assertEqual(
            app.overlay.analysis_badges[-1], {"provider": None, "pending": True}
        )

        app._screen_analysis_pending = True
        OpenAssistApp._on_response_complete(app, "done")
        self.assertEqual(
            app.overlay.analysis_badges[-1],
            {"provider": "groq", "pending": False},
        )

    def test_history_navigation_switches_settings_tabs_when_settings_open(self):
        app = self._build_app()
        tab_moves = []
        app.overlay.settings_view = SimpleNamespace(
            select_prev_tab=lambda: tab_moves.append("prev"),
            select_next_tab=lambda: tab_moves.append("next"),
        )
        app.overlay.current_widget = app.overlay.settings_view

        OpenAssistApp.history_prev(app)
        OpenAssistApp.history_next(app)

        self.assertEqual(tab_moves, ["prev", "next"])

    def test_scroll_routes_to_settings_view_when_settings_open(self):
        app = self._build_app()
        scrolls = []
        app.overlay.settings_view = SimpleNamespace(
            scroll_up=lambda: scrolls.append("up"),
            scroll_down=lambda: scrolls.append("down"),
        )
        app.overlay.current_widget = app.overlay.settings_view

        OpenAssistApp.scroll_up(app)
        OpenAssistApp.scroll_down(app)

        self.assertEqual(scrolls, ["up", "down"])

    def test_toggle_stealth_mode_enforces_stealth_state_and_window_opacity(self):
        app = self._build_app()
        app.state.is_stealth = False
        app.config.set("stealth.low_opacity", 0.75)
        app.config.set("app.opacity", 0.94)

        OpenAssistApp.toggle_stealth_mode(app)

        self.assertTrue(app.state.is_stealth)
        self.assertEqual(app.overlay.opacity_updates[-1], 0.75)
        self.assertEqual(app.mini_overlay.opacity_updates[-1], 0.75)

        OpenAssistApp.toggle_stealth_mode(app)

        self.assertTrue(app.state.is_stealth)
        self.assertEqual(app.overlay.opacity_updates[-1], 0.75)
        self.assertEqual(app.mini_overlay.opacity_updates[-1], 0.75)

    def test_factory_reset_falls_back_to_onboarding_when_restart_unavailable(self):
        app = self._build_app()
        stopped = []
        cleared = []
        shown = []

        app._stop_runtime_for_reset = lambda: stopped.append(True)
        app._clear_factory_reset_artifacts = lambda: cleared.append(True)
        app._restart_app = lambda: False
        app._show_onboarding_after_reset = lambda: shown.append(True)

        OpenAssistApp.factory_reset(app)

        self.assertEqual(stopped, [True])
        self.assertEqual(cleared, [True])
        self.assertEqual(shown, [True])

    def test_toggle_overlay_hides_and_shows_without_click_through(self):
        app = self._build_app()
        toggles = []

        def fake_toggle_click_through():
            app._click_through = not app._click_through
            toggles.append(app._click_through)

        app.toggle_click_through = fake_toggle_click_through

        OpenAssistApp.toggle_overlay(app)
        self.assertEqual(toggles, [])
        self.assertFalse(app.overlay.visible)

        OpenAssistApp.toggle_overlay(app)
        self.assertEqual(toggles, [])
        self.assertTrue(app.overlay.visible)
        self.assertEqual(app.overlay.activate_calls, 0)

        OpenAssistApp.toggle_overlay(app)
        self.assertEqual(toggles, [])
        self.assertFalse(app.overlay.visible)

    def test_show_onboarding_after_reset_restarts_runtime_services(self):
        app = self._build_app()
        app.is_running = False
        timer_starts = []
        audio_starts = []
        hotkey_starts = []
        warmups = []

        app._nexus_timer = SimpleNamespace(start=lambda ms: timer_starts.append(ms))
        app.audio = SimpleNamespace(start=lambda: audio_starts.append(True))
        app.hotkeys = SimpleNamespace(start=lambda: hotkey_starts.append(True))
        app._background_warmup = lambda: warmups.append(True)

        OpenAssistApp._show_onboarding_after_reset(app)

        self.assertTrue(app.is_running)
        self.assertEqual(timer_starts, [3000])
        self.assertEqual(audio_starts, [True])
        self.assertEqual(hotkey_starts, [True])
        self.assertEqual(warmups, [True])
        self.assertEqual(app.overlay.onboarding_calls, 1)


if __name__ == "__main__":
    unittest.main()
