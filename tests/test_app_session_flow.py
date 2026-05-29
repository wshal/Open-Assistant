import unittest
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.app import OpenAssistApp


class HistoryStub:
    def __init__(self):
        self.entries = []
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
        self.entries = []

    def get_state(self):
        return self.state

    def get_last(self, n):
        if self.entries:
            return [
                SimpleNamespace(
                    query=entry["query"],
                    response=entry["response"],
                    provider=entry["provider"],
                    latency=entry.get("latency", 0),
                    metadata=entry.get("metadata", {}),
                )
                for entry in self.entries[-n:]
            ]
        return [
            SimpleNamespace(
                query="example question",
                response="example response",
                provider="groq",
                latency=1234,
                metadata={},
            )
        ][:n]

    def add(self, query, response, provider, mode="general", latency=0.0, metadata=None):
        self.entries.append(
            {
                "query": query,
                "response": response,
                "provider": provider,
                "mode": mode,
                "latency": latency,
                "metadata": metadata or {},
            }
        )

    def save(self):
        pass  # no-op for tests


class AudioStub:
    def __init__(self):
        self.clear_calls = 0
        self.muted = False
        self.toggle_calls = 0
        self.restart_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.ensure_session_ready_calls = 0
        self.ensure_session_ready_result = True
        self.trace_context_calls = []

    def clear(self):
        self.clear_calls += 1

    def toggle(self):
        self.toggle_calls += 1
        self.muted = not self.muted
        return self.muted

    def restart(self):
        self.restart_calls += 1

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1

    def ensure_session_ready(self):
        self.ensure_session_ready_calls += 1
        return self.ensure_session_ready_result

    def set_trace_context(self, session_id="", session_started_at=0.0):
        self.trace_context_calls.append((session_id, session_started_at))


class OverlayStub:
    def __init__(self):
        self.current_widget = "standby"
        self.stack = SimpleNamespace(
            setCurrentIndex=self._set_index,
            setCurrentWidget=self._set_widget,
            currentIndex=lambda: 0,
            currentWidget=lambda: self.current_widget,
        )
        self.indices = []
        self.mode_updates = []
        self.transcript_updates = []
        self.completed = []
        self.appended = []
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
        self._active_response_query = ""
        self._active_response_text = ""
        self._active_response_streaming = False
        self.response_area = SimpleNamespace(clear=lambda: None)
        self.restore_calls = 0
        self.clear_calls = 0

    def _set_index(self, index):
        self.indices.append(index)

    def _set_widget(self, widget):
        self.current_widget = widget
        self.indices.append(widget)

    def update_mode(self, mode):
        self.mode_updates.append(mode)

    def update_transcript(self, text, state="auto"):
        self.transcript_updates.append(text)

    def on_complete(self, text, query=None, cache_tier: int = 0, provider: str = ""):
        self.completed.append((text, query))
        if query is not None:
            self._active_response_query = query
        self._active_response_text = text or ""
        self._active_response_streaming = False

    def append_response(self, text):
        self.appended.append(text)

    def update_history_state(self, *state):
        self.history_updates.append(state)

    def begin_active_response(self, query):
        self._active_response_query = query
        self._active_response_text = ""
        self._active_response_streaming = False

    def capture_active_response_chunk(self, text):
        self._active_response_text += text
        self._active_response_streaming = True

    def restore_active_response_view(self):
        self.restore_calls += 1
        return bool(self._active_response_query or self._active_response_text)

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

    def set_click_through(self, enabled):
        self.click_through = enabled

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
        self.show_chat_view()

    def end_session_ui(self):
        self.show_standby_view()

    def show_onboarding(self):
        self.onboarding_calls += 1

    def set_analysis_provider_badge(self, provider=None, pending=False):
        self.analysis_badges.append({"provider": provider, "pending": pending})

    def show_chat_view(self):
        self.current_widget = self.chat_view
        self.indices.append(1)

    def show_standby_view(self):
        self.current_widget = self.standby_view
        self.indices.append(0)

    def show_settings_view(self):
        self.current_widget = self.settings_view
        self.indices.append(2)

    def clear_session_response(self):
        self.clear_calls += 1
        self._current_query = ""
        self._raw_buffer = ""
        self._is_streaming = False
        self._active_response_query = ""
        self._active_response_text = ""
        self._active_response_streaming = False

    def refresh_standby_state(self, mode=None, audio=None):
        self.refresh_calls.append({"mode": mode, "audio": audio})

    def update_auto_mode_state(self, enabled, connected, reconnecting=False, fallback=False):
        self.auto_mode_updates = getattr(self, "auto_mode_updates", [])
        self.auto_mode_updates.append({"enabled": enabled, "connected": connected})


class MiniOverlayStub:
    def __init__(self):
        self.mode_updates = []
        self.completed = []
        self.appended = []
        self.ready_calls = 0
        self.history_updates = []
        self.hide_calls = 0
        self.show_calls = 0
        self.opacity_updates = []
        self.warmup_updates = []
        self._active_response_query = ""
        self._active_response_text = ""
        self._active_response_streaming = False
        self.restore_calls = 0
        self.clear_calls = 0

    def update_mode(self, mode):
        self.mode_updates.append(mode)

    def on_complete(self, text, query=None, cache_tier: int = 0, provider: str = ""):
        self.completed.append((text, query))
        if query is not None:
            self._active_response_query = query
        self._active_response_text = text or ""
        self._active_response_streaming = False

    def append_response(self, text):
        self.appended.append(text)

    def set_ready(self):
        self.ready_calls += 1

    def clear_session_response(self):
        self.clear_calls += 1
        self._active_response_query = ""
        self._active_response_text = ""
        self._active_response_streaming = False

    def update_history_state(self, *state):
        self.history_updates.append(state)

    def begin_active_response(self, query):
        self._active_response_query = query
        self._active_response_text = ""
        self._active_response_streaming = False

    def capture_active_response_chunk(self, text):
        self._active_response_text += text
        self._active_response_streaming = True

    def restore_active_response_view(self):
        self.restore_calls += 1
        return bool(self._active_response_query or self._active_response_text)

    def update_warmup_status(self, m, p, r):
        self.warmup_updates.append((m, p, r))

    def hide(self):
        self.hide_calls += 1

    def show(self):
        self.show_calls += 1

    def update_audio_state(self, muted):
        pass

    def set_click_through(self, enabled):
        self.click_through = enabled

    def setWindowOpacity(self, value):
        self.opacity_updates.append(value)

    def isVisible(self):
        return False


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
    def setUp(self):
        pass

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
            _screen_share_active=False,
            _screen_share_hidden_window=None,
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
            stealth=SimpleNamespace(
                apply_to_window=lambda window, enabled: None,
                should_hide_for_screen_share=lambda: False,
            ),
        )
        app.state = StateStub(app.config)
        app.run_on_ui_thread = SimpleNamespace(emit=lambda callback: callback())
        app._history_navigation_active = False
        app._context_auto_suggested = False  # auto-suggest flag
        app._context_store = SimpleNamespace(
            get_last_context=lambda: "",
            set_last_context=lambda t: None,
        )
        app._apply_window_effects = lambda window: OpenAssistApp._apply_window_effects(
            app, window
        )
        app._apply_ui_only = lambda: OpenAssistApp._apply_ui_only(app)
        app._refresh_window_invariants = lambda window=None: OpenAssistApp._refresh_window_invariants(
            app, window
        )
        app._active_view = lambda: OpenAssistApp._active_view(app)
        app._show_active_overlay = lambda: OpenAssistApp._show_active_overlay(app)
        app._hud_focus_enabled = lambda: OpenAssistApp._hud_focus_enabled(app)
        app._present_window = lambda window, focus=False: OpenAssistApp._present_window(
            app, window, focus
        )
        app._sync_history_ui = lambda: OpenAssistApp._sync_history_ui(app)
        app._stop_background_tasks = lambda: None
        app._sync_state_from_config = lambda: OpenAssistApp._sync_state_from_config(app)
        app._should_ignore_final_transcript = lambda text: OpenAssistApp._should_ignore_final_transcript(app, text)
        app._looks_like_acknowledgement = lambda text: OpenAssistApp._looks_like_acknowledgement(text)
        app._looks_question_like_transcript = lambda text: OpenAssistApp._looks_question_like_transcript(app, text)
        app._looks_like_clipped_query_fragment = lambda text: OpenAssistApp._looks_like_clipped_query_fragment(text)
        app._carry_forward_incomplete_audio_query = lambda text: OpenAssistApp._carry_forward_incomplete_audio_query(app, text)
        app.generate_response = lambda query, origin, ctx: None
        app._reset_turn_local_state = lambda reason="": OpenAssistApp._reset_turn_local_state(app, reason)
        app._log_turn_waterfall_summary = lambda provider=None, request_metadata=None, stage_timings=None: OpenAssistApp._log_turn_waterfall_summary(
            app,
            provider=provider,
            request_metadata=request_metadata,
            stage_timings=stage_timings,
        )
        app._auto_mode_requested = lambda: OpenAssistApp._auto_mode_requested(app)
        return app

    def test_start_new_session_resets_state_and_updates_ui(self):
        app = self._build_app()

        OpenAssistApp.start_new_session(app)

        self.assertEqual(app.history.started, 1)
        self.assertEqual(app.audio.clear_calls, 1)
        self.assertEqual(app.audio.ensure_session_ready_calls, 1)
        self.assertEqual(len(app.audio.trace_context_calls), 1)
        self.assertEqual(app._last_query, "")
        self.assertTrue(app.session_active)
        self.assertEqual(app.overlay.indices, [1])
        self.assertEqual(app.overlay.current_widget, app.overlay.chat_view)
        self.assertEqual(app.overlay.clear_calls, 1)
        self.assertEqual(app.mini_overlay.clear_calls, 1)
        self.assertEqual(app.overlay.completed, [])
        self.assertEqual(app.mini_overlay.completed, [])
        self.assertEqual(app.overlay.transcript_updates[-1], "Listening for context...")
        self.assertEqual(app.overlay.mode_updates[-1], "general")
        self.assertEqual(app.mini_overlay.mode_updates[-1], "general")
        self.assertEqual(app.mini_overlay.ready_calls, 1)
        self.assertTrue(app.state.is_capturing)

    def test_start_new_session_surfaces_audio_unavailable_state(self):
        app = self._build_app()
        app.audio.ensure_session_ready_result = False

        OpenAssistApp.start_new_session(app)

        self.assertIn("Audio capture unavailable", app.overlay.transcript_updates[-1])

    def test_run_warms_runtime_without_starting_audio_capture(self):
        app = self._build_app()
        app._async_thread = SimpleNamespace(start=lambda: None)
        app.hotkeys = SimpleNamespace(start=lambda: None)
        app._background_warmup = lambda: None
        app._show_initial_window = lambda: None
        app.qt_app = SimpleNamespace(exec=lambda: 0)

        with patch("core.app.QTimer.singleShot", side_effect=lambda *_args: None):
            self.assertEqual(OpenAssistApp.run(app), 0)

        self.assertEqual(app.audio.start_calls, 0)

    def test_history_sync_uses_dict_entries_for_mini_overlay(self):
        """_sync_history_ui should push state to both HUDs when session is active."""
        app = self._build_app(mini_mode=True)
        app.session_active = True  # Guard: only syncs when session is live

        OpenAssistApp._sync_history_ui(app)

        self.assertEqual(len(app.overlay.history_updates), 1)
        self.assertEqual(len(app.mini_overlay.history_updates), 1)
        # At latest entry (idx=0, total=1) on_complete is NOT called again by
        # _sync_history_ui — the caller (_on_response_complete) already did it.
        # Here we verify the navigation state was still pushed.
        self.assertEqual(app.mini_overlay.history_updates[0][0], 0)  # index
        self.assertEqual(app.mini_overlay.history_updates[0][1], 1)  # total

    def test_history_sync_restores_active_response_draft_when_returning_to_latest(self):
        app = self._build_app(mini_mode=True)
        app.session_active = True
        app.history.state = (
            1,
            2,
            {"query": "older", "response": "older response", "provider": "groq"},
        )
        app.overlay.begin_active_response("current query")
        app.overlay.capture_active_response_chunk("draft chunk")
        app.mini_overlay.begin_active_response("current query")
        app.mini_overlay.capture_active_response_chunk("draft chunk")

        OpenAssistApp._sync_history_ui(app)

        self.assertEqual(app.overlay.history_updates, [])
        self.assertEqual(app.mini_overlay.history_updates, [])
        self.assertEqual(app.overlay.restore_calls, 1)
        self.assertEqual(app.mini_overlay.restore_calls, 1)
        self.assertFalse(app._history_navigation_active)

    def test_history_sync_does_nothing_when_session_inactive(self):
        """_sync_history_ui must not update the UI when no session is active.

        This prevents preloaded prior-session history (GAP4) from showing up
        in the HUDs on startup or when the user is on the standby screen.
        """
        app = self._build_app(mini_mode=True)
        # session_active is False by default in _build_app
        self.assertFalse(app.session_active)

        OpenAssistApp._sync_history_ui(app)

        # Both overlays must remain untouched
        self.assertEqual(app.overlay.history_updates, [])
        self.assertEqual(app.mini_overlay.history_updates, [])
        self.assertEqual(app.mini_overlay.completed, [])

    def test_response_complete_resets_turn_local_query_and_detector_buffer(self):
        app = self._build_app()
        app.session_active = True
        app._last_query = "what is react"
        app._last_query_time = 123.0
        app._pending_request_metadata = {"speech_to_transcript_ms": 250}
        resets = []
        app.ai = SimpleNamespace(
            detector=SimpleNamespace(
                reset_turn_state=lambda reason="": resets.append(reason)
            ),
            _providers={},
        )
        app.history.entries.append(
            {
                "query": "what is react",
                "response": "React is a UI library.",
                "provider": "groq",
                "latency": 250,
                "metadata": {},
            }
        )

        OpenAssistApp._on_response_complete(app, "React is a UI library.")

        self.assertEqual(app._last_query, "")
        self.assertEqual(app._last_query_time, 0.0)
        self.assertIsNone(app._pending_request_metadata)
        self.assertEqual(resets, ["response-complete"])

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
        self.assertEqual(app.audio.clear_calls, 0)
        self.assertEqual(app.audio.stop_calls, 1)
        self.assertEqual(app._last_query, "")
        self.assertEqual(app.overlay.indices, [0])
        self.assertEqual(app.overlay.current_widget, app.overlay.standby_view)
        self.assertEqual(app.overlay.clear_calls, 1)
        self.assertEqual(app.mini_overlay.clear_calls, 1)
        self.assertEqual(app.overlay.completed, [])
        self.assertEqual(app.mini_overlay.completed, [])
        self.assertEqual(app.overlay.transcript_updates[-1], "Ready...")
        self.assertFalse(app.state.is_capturing)

    def test_end_session_does_not_dispatch_leftover_query(self):
        app = self._build_app()
        app.session_active = True
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp.end_session(app)

        self.assertEqual(dispatched, [])

    def test_toggle_mini_mode_no_history_shown_when_session_inactive(self):
        """Toggling to mini-mode while no session is active must NOT show stale history.

        The session_active guard in _sync_history_ui prevents GAP4 preloaded
        entries from bleeding into the HUD after a restart.
        """
        app = self._build_app(mini_mode=False)
        self.assertFalse(app.session_active)  # no active session

        with patch.object(app, "_refresh_window_invariants"):
            OpenAssistApp.toggle_mini_mode(app)
            self.assertTrue(app.mini_mode)

            # _sync_history_ui must have been a no-op
            self.assertEqual(app.overlay.history_updates, [])
            self.assertEqual(app.mini_overlay.history_updates, [])
            self.assertEqual(app.mini_overlay.completed, [])

    def test_toggle_mini_mode_shows_active_response_when_session_live(self):
        """Toggling to mini-mode during an active session syncs the current response."""
        app = self._build_app(mini_mode=False)
        app.session_active = True  # session is running

        with patch.object(app, "_refresh_window_invariants"):
            OpenAssistApp.toggle_mini_mode(app)
            self.assertTrue(app.mini_mode)

            # Both overlays should have received the history state
            self.assertEqual(len(app.overlay.history_updates), 1)
            self.assertEqual(len(app.mini_overlay.history_updates), 1)

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

    def test_standard_transcription_always_runs_regardless_of_config(self):
        app = self._build_app()
        app.session_active = True
        pushed = []
        generated = []
        detector_calls = []
        app.nexus = SimpleNamespace(
            push=lambda source, value: pushed.append((source, value)),
            get_snapshot=lambda: {},
        )
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
        self.assertEqual(app.overlay.transcript_updates[-1], "Why?")


    def test_auto_mode_keeps_setup_speech_then_dispatches_fixture_question(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.auto_mode.enabled", True)
        dispatched = []
        app.nexus = SimpleNamespace(push=lambda source, value: None, get_snapshot=lambda: {})
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))
        app.audio = SimpleNamespace(get_last_transcription_metrics=lambda: {})
        app.ai.detector = SimpleNamespace(
            question_prefixes=["what ", "how ", "why ", "can ", "could "],
            question_patterns=["what is", "how do", "can you", "could you"],
            detect_with_confidence=lambda text, source="audio": SimpleNamespace(
                triggered=False,
                confidence=0.0,
                detected_text="",
                should_auto_respond=lambda: False,
            ),
        )

        OpenAssistApp._on_transcription(
            app,
            "Alright, let's talk about scaling. Imagine we have a monolithic application backed by a single relational database.",
        )
        self.assertEqual(dispatched, [])

        OpenAssistApp._on_transcription(
            app,
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0][1], "speech")
        self.assertTrue(dispatched[0][2]["auto_answer"])
        self.assertIn("alleviate that bottleneck", dispatched[0][0])
        self.assertTrue(app._pending_request_metadata["preserve_session_context"])

    def test_auto_mode_speculative_interim_dispatches_stable_complete_question(self):
        from ai.auto_answer_controller import handle_auto_interim_transcription

        app = self._build_app()
        app.session_active = True
        app.config.set("ai.auto_mode.enabled", True)
        app.config.set("ai.auto_mode.speculative_interim.enabled", True)
        app.config.set("ai.auto_mode.speculative_interim.stability_ms", 1)
        app.config.set("ai.auto_mode.speculative_interim.delay_ms", 0)
        app._auto_answer_context = "We are discussing React hooks and shared state."
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        interim = "Could you walk me through how you would decide between using a custom hook versus a higher-order component?"
        handle_auto_interim_transcription(app, interim, "session_test")
        app._auto_interim_stable_at = time.time() - 1

        with patch("PyQt6.QtCore.QTimer.singleShot", side_effect=lambda _delay, cb: cb()):
            handle_auto_interim_transcription(app, interim, "session_test")

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0][1], "speech")
        self.assertTrue(dispatched[0][2]["auto_answer"])
        self.assertTrue(dispatched[0][2]["auto_speculative"])
        self.assertIn("custom hook", dispatched[0][0])

    def test_auto_mode_speculative_interim_rejects_clipped_partial_question(self):
        from ai.auto_answer_controller import handle_auto_interim_transcription

        app = self._build_app()
        app.session_active = True
        app.config.set("ai.auto_mode.enabled", True)
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        handle_auto_interim_transcription(
            app,
            "What are some strategies you would consider to a..",
            "session_test",
        )

        self.assertEqual(dispatched, [])

    def test_auto_mode_speculative_interim_rejects_short_question_even_with_context(self):
        from ai.auto_answer_controller import handle_auto_interim_transcription

        app = self._build_app()
        app.session_active = True
        app.config.set("ai.auto_mode.enabled", True)
        app._auto_answer_context = (
            "Let's pivot to some CSS basics. "
            "A lot of developers get confused between CSS Grid and Flexbox."
        )
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        handle_auto_interim_transcription(app, "Can you explain the price?", "session_test")

        self.assertEqual(dispatched, [])

    def test_auto_mode_speculative_interim_rejects_contextual_tail_fragment(self):
        from ai.auto_answer_controller import handle_auto_interim_transcription

        app = self._build_app()
        app.session_active = True
        app.config.set("ai.auto_mode.enabled", True)
        app._auto_answer_context = "When building a secure REST API authentication is critical."
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        handle_auto_interim_transcription(
            app,
            "in the browser's local storage instead of an HTT.",
            "session_test",
        )

        self.assertEqual(dispatched, [])

    def test_toggle_auto_mode_enables_auto_answer_only(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.auto_mode.enabled", False)

        OpenAssistApp.toggle_auto_mode(app)

        self.assertTrue(app.config.get("ai.auto_mode.enabled"))

    def test_non_question_audio_does_not_prime_request_metadata(self):
        app = self._build_app()
        app.session_active = True
        app._pending_request_metadata = None
        app.audio = SimpleNamespace(
            get_last_transcription_metrics=lambda: {"speech_to_transcript_ms": 321}
        )
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

        OpenAssistApp._on_transcription(app, "React context overview")

        self.assertIsNone(app._pending_request_metadata)

    def test_complete_audio_question_falls_back_when_detector_misses(self):
        app = self._build_app()
        app.session_active = True
        generated = []
        app.generate_response = lambda *args, **kwargs: generated.append((args, kwargs))
        app.nexus = SimpleNamespace(push=lambda source, value: None, get_snapshot=lambda: {})
        app.audio = SimpleNamespace(
            get_last_transcription_metrics=lambda: {"speech_to_transcript_ms": 321}
        )
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

        OpenAssistApp._on_transcription(app, "difference between setTimeout and setInterval")

        self.assertEqual(len(generated), 1)
        self.assertEqual(
            generated[0][0],
            ("difference between setTimeout and setInterval", "speech", {"audio": "difference between setTimeout and setInterval"}),
        )

    def test_incomplete_audio_question_carries_forward_into_generic_followup(self):
        app = self._build_app()
        app.session_active = True
        generated = []
        app.generate_response = lambda *args, **kwargs: generated.append((args, kwargs))
        app.nexus = SimpleNamespace(push=lambda source, value: None, get_snapshot=lambda: {})
        app.audio = SimpleNamespace(get_last_transcription_metrics=lambda: {})

        def _detect(text, source="audio"):
            if text == "what is async, await,":
                return SimpleNamespace(
                    triggered=True,
                    confidence=1.0,
                    detected_text="what is async",
                    should_auto_respond=lambda: False,
                )
            return SimpleNamespace(
                triggered=True,
                confidence=1.0,
                detected_text=text,
                should_auto_respond=lambda: True,
            )

        app.ai.detector = SimpleNamespace(
            question_prefixes=["what ", "how ", "why ", "can "],
            question_patterns=["what is", "how do", "can you"],
            detect_with_confidence=_detect,
        )

        with patch("core.app.time.time", side_effect=[100.0] * 12 + [103.0] * 12):
            OpenAssistApp._on_transcription(app, "what is async, await,")
            OpenAssistApp._on_transcription(app, "can you explain?")

        self.assertEqual(len(generated), 1)
        self.assertEqual(
            generated[0][0],
            ("can you explain async, await?", "speech", {"audio": "can you explain async, await?"}),
        )

    def test_duplicate_system_audio_final_transcript_is_ignored(self):
        app = self._build_app()
        app.session_active = True
        app.audio = SimpleNamespace(
            capture_mode="system",
            get_last_transcription_metrics=lambda: {},
        )
        app.nexus = SimpleNamespace(push=lambda source, value: None, get_snapshot=lambda: {})
        app.ai.detector = SimpleNamespace(
            question_prefixes=["what ", "how ", "why "],
            question_patterns=["what is", "how do"],
            detect_with_confidence=lambda text, source="audio": SimpleNamespace(
                triggered=True,
                confidence=1.0,
                detected_text=text,
                should_auto_respond=lambda: True,
            ),
        )
        generated = []
        app.generate_response = lambda *args, **kwargs: generated.append((args, kwargs))

        with patch("core.app.time.time", side_effect=[100.0] * 8 + [101.5] * 8):
            OpenAssistApp._on_transcription(app, "What is Context API?")
            OpenAssistApp._on_transcription(app, "What is Context API?")

        self.assertEqual(len(generated), 1)

    def test_show_initial_window_always_shows_hud_when_onboarding_is_complete(self):
        app = self._build_app()
        app.config.set("onboarding.completed", True)

        OpenAssistApp._show_initial_window(app)

        self.assertEqual(app.overlay.show_calls, 1)
        self.assertEqual(app.overlay.raise_calls, 1)

    def test_show_initial_window_forces_onboarding(self):
        app = self._build_app()
        app.config.set("onboarding.completed", False)

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

        with patch.object(app, "_refresh_window_invariants") as refresh:
            result = OpenAssistApp._show_active_overlay(app)

        self.assertIs(result, app.overlay)
        refresh.assert_called_once_with(app.overlay)
        self.assertEqual(app.overlay.show_calls, 1)
        self.assertEqual(app.overlay.raise_calls, 1)
        self.assertEqual(app.overlay.activate_calls, 0)

    def test_present_window_only_focuses_when_requested(self):
        app = self._build_app()

        with patch.object(app, "_refresh_window_invariants") as refresh:
            OpenAssistApp._present_window(app, app.overlay, focus=False)
            OpenAssistApp._present_window(app, app.overlay, focus=True)

        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(app.overlay.show_calls, 2)
        self.assertEqual(app.overlay.raise_calls, 2)
        self.assertEqual(app.overlay.activate_calls, 1)

    def test_toggle_click_through_reapplies_window_invariants(self):
        app = self._build_app()

        with patch.object(app, "_refresh_window_invariants") as refresh:
            OpenAssistApp.toggle_click_through(app)

        refresh.assert_called_once_with()

    def test_refresh_window_invariants_reapplies_both_windows(self):
        app = self._build_app()

        with patch.object(app, "_apply_window_effects") as apply_effects:
            OpenAssistApp._refresh_window_invariants(app)

        self.assertEqual(apply_effects.call_args_list, [unittest.mock.call(app.overlay), unittest.mock.call(app.mini_overlay)])

    def test_apply_window_effects_hides_from_taskbar_and_reapplies_stealth(self):
        app = self._build_app()
        app.state.is_stealth = True
        app.config.set("app.opacity", 0.94)
        app.config.set("stealth.low_opacity", 0.75)
        app.stealth = SimpleNamespace(apply_to_window=Mock())

        with patch("core.app.WindowUtils.hide_from_taskbar") as hide_from_taskbar, patch(
            "core.app.WindowUtils.ensure_topmost"
        ) as ensure_topmost:
            OpenAssistApp._apply_window_effects(app, app.overlay)

        hide_from_taskbar.assert_called_once_with(app.overlay)
        ensure_topmost.assert_called_once_with(app.overlay)
        app.stealth.apply_to_window.assert_called_once_with(app.overlay, True)

    def test_check_screen_share_protection_reinforces_stealth_and_updates_state(self):
        app = self._build_app()
        app.ensure_stealth = Mock()
        app._set_screen_share_state = Mock()
        app._screen_share_active = False

        with patch("core.app.ProcessUtils.is_screen_sharing_active", return_value=True):
            OpenAssistApp._check_screen_share_protection(app)

        app.ensure_stealth.assert_called_once_with()
        app._set_screen_share_state.assert_called_once_with(True)

    def test_check_screen_share_protection_no_state_change_when_detection_is_same(self):
        app = self._build_app()
        app.ensure_stealth = Mock()
        app._set_screen_share_state = Mock()
        app._screen_share_active = False

        with patch("core.app.ProcessUtils.is_screen_sharing_active", return_value=False):
            OpenAssistApp._check_screen_share_protection(app)

        app.ensure_stealth.assert_not_called()
        app._set_screen_share_state.assert_not_called()

    def test_refresh_topmost_window_reapplies_full_invariants_for_visible_windows(self):
        app = self._build_app()
        app.overlay.visible = True
        app.mini_overlay.show_calls = 0

        with patch.object(app, "_refresh_window_invariants") as refresh:
            OpenAssistApp._refresh_topmost_window(app)

        self.assertEqual(
            refresh.call_args_list,
            [unittest.mock.call(app.overlay)],
        )

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

    def test_response_complete_sanitizes_standard_speech_meta_response(self):
        app = self._build_app()
        app.history.entries.append(
            {
                "query": "what is react router",
                "response": "placeholder",
                "provider": "groq",
                "latency": 1234,
                "metadata": {"request_metadata": {"origin": "speech"}},
            }
        )

        OpenAssistApp._on_response_complete(
            app,
            "My approach is to explain it clearly. React Router handles client-side routing.",
        )

        self.assertEqual(
            app.overlay.completed[-1][0],
            "React Router handles client-side routing.",
        )
        self.assertEqual(
            app.history.entries[-1]["response"],
            "React Router handles client-side routing.",
        )

    def test_response_complete_logs_compact_waterfall_summary(self):
        app = self._build_app()
        app.session_active = True
        app._current_turn_id = "session_1:utt_3"
        app.history.entries.append(
            {
                "query": "what is react",
                "response": "React is a UI library.",
                "provider": "groq",
                "latency": 1234,
                "metadata": {
                    "request_metadata": {
                        "utterance_id": "session_1:utt_3",
                        "audio_duration_ms": 820.0,
                        "transcribe_only_ms": 310.0,
                        "speech_to_transcript_ms": 1130.0,
                        "vad_backend": "webrtc",
                        "chunks": 2,
                    },
                    "stage_timings": {
                        "request_to_first_token_ms": 540.0,
                        "request_to_complete_ms": 1400.0,
                    },
                },
            }
        )

        with self.assertLogs("core.app", level="INFO") as captured:
            OpenAssistApp._on_response_complete(app, "React is a UI library.")

        self.assertTrue(
            any(
                "WATERFALL SUMMARY | speech=820ms | asr=310ms | llm_ttfb=540ms | stream=860ms | total=2530ms | utterance=session_1:utt_3 | vad=webrtc | chunks=2 | provider=groq"
                in line
                for line in captured.output
            )
        )

    def test_response_complete_tolerates_missing_response_start_time(self):
        app = self._build_app()
        app.session_active = True
        app._current_response_start_time = None

        OpenAssistApp._on_response_complete(app, "done")

        self.assertEqual(app.overlay.completed[-1][0], "done")

    def test_generate_response_sets_response_start_time_for_manual_turns(self):
        app = self._build_app()
        app.session_active = True
        app._ai_lock_ready = SimpleNamespace(wait=lambda timeout=2: True)
        app.ai = SimpleNamespace(
            detector=SimpleNamespace(learn_from_query=lambda q: None),
            demote_to_background=lambda: None,
        )
        app.simulator = SimpleNamespace(get_foreground_window=lambda: None)
        app.overlay.response_area.setHtml = lambda html: None
        app._pending_request_metadata = None
        async def _fake_process_ai(*args, **kwargs):
            return None
        app._process_ai = _fake_process_ai
        app.loop = object()
        scheduled = []

        with patch("core.app.asyncio.run_coroutine_threadsafe", side_effect=lambda coro, loop: scheduled.append((coro, loop))):
            OpenAssistApp.generate_response(app, "what is hoisting ?", "manual", {"audio": ""})
        for coro, _loop in scheduled:
            try:
                coro.close()
            except Exception:
                pass

        self.assertIsNotNone(app._current_response_start_time)


    def test_reset_benchmark_fixture_runtime_clears_auto_mode_turn_state(self):
        app = self._build_app()
        app._auto_answer_context = "stale context from previous fixture"
        app._auto_interim_pending_query = "half-heard question"
        app._pending_incomplete_audio_query = "incomplete carry-forward"
        app.overlay.response_area = SimpleNamespace(clear=lambda: None)

        OpenAssistApp.reset_benchmark_fixture_runtime(app)

        self.assertEqual(app._auto_answer_context, "")
        self.assertEqual(app._auto_interim_pending_query, "")
        self.assertEqual(app._pending_incomplete_audio_query, "")

    def test_reset_benchmark_fixture_runtime_clears_pending_audio_followup_state(self):
        app = self._build_app()
        app._pending_incomplete_audio_query = "Can you explain closures?"
        app._pending_incomplete_audio_at = 123.0
        app.overlay.response_area = SimpleNamespace(clear=lambda: None)

        OpenAssistApp.reset_benchmark_fixture_runtime(app)

        self.assertEqual(app._pending_incomplete_audio_query, "")
        self.assertEqual(app._pending_incomplete_audio_at, 0.0)


    def test_end_session_persists_last_exchange_before_history_rollover(self):
        app = self._build_app()
        app.session_active = True
        app.history.entries.append(
            {
                "query": "what changed",
                "response": "the queue backed up",
                "provider": "groq",
                "mode": "general",
                "latency": 42,
                "metadata": {},
            }
        )
        stored = []
        app.memory = SimpleNamespace(
            store=lambda session_id, q, r, mode="general": stored.append((q, r, mode))
        )

        OpenAssistApp.end_session(app)

        self.assertEqual(stored, [("what changed", "the queue backed up", "general")])

    def test_end_session_does_not_persist_benchmark_exchange_to_memory(self):
        app = self._build_app()
        app.session_active = True
        app.history.entries.append(
            {
                "query": "fixture question",
                "response": "fixture answer",
                "provider": "groq",
                "mode": "general",
                "latency": 42,
                "metadata": {
                    "request_metadata": {
                        "benchmark_isolated": True,
                    }
                },
            }
        )
        stored = []
        app.memory = SimpleNamespace(
            store=lambda session_id, q, r, mode="general": stored.append((q, r, mode))
        )

        OpenAssistApp.end_session(app)

        self.assertEqual(stored, [])


    def test_benchmark_isolated_process_ai_does_not_inject_screen_or_memory(self):
        app = self._build_app()
        app._ai_lock = asyncio.Lock()
        app._generation_epoch = 7
        app.screen = SimpleNamespace(last_img_hash="stale-screen-hash")
        app.context_builder = SimpleNamespace(build=Mock(side_effect=AssertionError("screen context should be skipped")))
        app.context_pruner = SimpleNamespace(prune=Mock(side_effect=AssertionError("screen pruning should be skipped")))
        app.memory = SimpleNamespace(
            is_ready=lambda: True,
            query=Mock(side_effect=AssertionError("memory should be skipped")),
        )
        app.nexus = SimpleNamespace(
            push=Mock(side_effect=AssertionError("screen should not be pushed")),
            get_snapshot=Mock(return_value={"latest_ocr": "stale OCR", "full_audio_history": "old audio"}),
        )
        captured = {}

        async def _capture_generate_response(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        app.ai = SimpleNamespace(generate_response=_capture_generate_response)

        asyncio.run(
            OpenAssistApp._process_ai(
                app,
                "What are some strategies you would consider to alleviate that bottleneck?",
                "speech",
                {"audio": "What are some strategies you would consider to alleviate that bottleneck?"},
                7,
                {"benchmark_isolated": True},
            )
        )

        self.assertEqual(captured["kwargs"]["screen_context"], "")
        self.assertEqual(captured["kwargs"]["screen_hash"], "")
        self.assertEqual(captured["args"][1], {})
        req_meta = captured["kwargs"]["request_metadata"]
        self.assertTrue(req_meta["suppress_memory_context"])
        self.assertTrue(req_meta["suppress_history_context"])
        self.assertTrue(req_meta["suppress_rag_context"])
        self.assertTrue(req_meta["suppress_response_cache"])

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

    def test_ensure_stealth_enforces_stealth_state_and_window_opacity(self):
        app = self._build_app()
        app.state.is_stealth = False
        app.config.set("stealth.low_opacity", 0.75)
        app.config.set("app.opacity", 0.94)

        OpenAssistApp.ensure_stealth(app)

        self.assertTrue(app.state.is_stealth)
        self.assertEqual(app.overlay.opacity_updates[-1], 0.75)
        self.assertEqual(app.mini_overlay.opacity_updates[-1], 0.75)

    def test_screen_share_reinforces_stealth_without_hiding_on_strong_platform(self):
        app = self._build_app(mini_mode=False)
        app.overlay.visible = True
        app.state.is_stealth = False
        app.ensure_stealth = lambda: OpenAssistApp.ensure_stealth(app)

        OpenAssistApp._set_screen_share_state(app, True)

        self.assertTrue(app._screen_share_active)
        self.assertTrue(app.overlay.isVisible())
        self.assertIsNone(app._screen_share_hidden_window)

    def test_screen_share_hides_and_restores_hud_on_limited_platform(self):
        app = self._build_app(mini_mode=False)
        app.overlay.visible = True
        app.ensure_stealth = lambda: OpenAssistApp.ensure_stealth(app)
        app.stealth.should_hide_for_screen_share = lambda: True

        OpenAssistApp._set_screen_share_state(app, True)
        self.assertFalse(app.overlay.isVisible())
        self.assertEqual(app._screen_share_hidden_window, "overlay")

        OpenAssistApp._set_screen_share_state(app, False)
        self.assertTrue(app.overlay.isVisible())
        self.assertIsNone(app._screen_share_hidden_window)

        OpenAssistApp.ensure_stealth(app)

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
        self.assertEqual(audio_starts, [])
        self.assertEqual(hotkey_starts, [True])
        self.assertEqual(warmups, [True])
        self.assertEqual(app.overlay.onboarding_calls, 1)

    def test_end_session_clears_mini_overlay_state(self):
        """end_session() must reset Mini-HUD to blank/ready state.

        P4: Ensures the user sees a clean Mini-HUD when returning to standby
        so there is no confusion about whether a response is from a past session.
        """
        app = self._build_app(mini_mode=True)
        app.session_active = True

        OpenAssistApp.end_session(app)

        # Mini overlay must be cleared without a fake completion event.
        self.assertEqual(app.mini_overlay.clear_calls, 1)
        self.assertEqual(app.mini_overlay.completed, [])
        self.assertGreaterEqual(app.mini_overlay.ready_calls, 1)
        self.assertFalse(app.session_active)

    def test_shutdown_with_active_session_calls_end_session(self):
        """Shutting down while a session is active must gracefully end it.

        P4: Prevents the Mini-HUD from persisting state to disk as an active
        session, which would cause stale history to appear on next launch.
        """
        app = self._build_app()
        app.session_active = True
        app._stop_background_tasks = lambda: None
        app.audio = SimpleNamespace(stop=lambda: None, clear=lambda: None)
        app.hotkeys = SimpleNamespace(stop=lambda: None, reset_state=lambda: None)
        app.rag = SimpleNamespace(stop=lambda: None)
        app.loop = SimpleNamespace(is_running=lambda: False)
        app._async_thread = SimpleNamespace(
            is_alive=lambda: False, join=lambda timeout=None: None
        )
        # Wire end_session onto the namespace so shutdown() can call self.end_session()
        end_calls = []
        original_end_session = OpenAssistApp.end_session

        def _fake_end_session():
            end_calls.append(True)
            original_end_session(app)

        app.end_session = _fake_end_session

        OpenAssistApp.shutdown(app)

        self.assertEqual(end_calls, [True])
        self.assertFalse(app.session_active)


    def test_history_prev_in_mini_mode_calls_on_complete_for_non_latest(self):
        """Ctrl+[ in Mini-HUD navigates to a prev entry and updates the display.

        _sync_history_ui must call mini_overlay.on_complete() only for
        non-latest entries to avoid a redundant re-render when at the tail.
        """
        app = self._build_app(mini_mode=True)
        app.session_active = True

        # Give history 2 entries and position at the second (latest)
        entry1 = {"query": "q1", "response": "r1", "provider": "groq", "mode": "general",
                  "latency": 0.0, "timestamp": 1.0, "metadata": {}}
        entry2 = {"query": "q2", "response": "r2", "provider": "groq", "mode": "general",
                  "latency": 0.0, "timestamp": 2.0, "metadata": {}}
        app.history.state = (1, 2, entry2)  # at latest (idx=1, total=2)

        # Navigate backwards — should call on_complete for the non-latest entry
        def move_prev_and_update():
            # Simulate move_prev moving to idx=0
            app.history.state = (0, 2, entry1)
            OpenAssistApp._sync_history_ui(app)

        move_prev_and_update()

        # on_complete must have been called with the prev entry's content
        self.assertIn(("r1", "q1"), app.mini_overlay.completed)

if __name__ == "__main__":
    unittest.main()
