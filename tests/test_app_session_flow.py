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
        self.live_mode_updates = []
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

    def update_live_mode_state(self, enabled, connected, reconnecting=False, fallback=False):
        self.live_mode_updates.append(
            {
                "enabled": enabled,
                "connected": connected,
                "reconnecting": reconnecting,
                "fallback": fallback,
            }
        )


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
        legacy_live_markers = (
            "test_live_",
            "_live_",
            "live_turn",
            "live_audio",
            "live_status",
            "live_transcript",
            "live_standard",
            "stop_live_mode",
            "promote_live",
            "remember_live",
            "should_replace_live",
        )
        name = self._testMethodName
        if (
            name != "test_auto_mode_does_not_start_legacy_live_session"
            and any(marker in name for marker in legacy_live_markers)
        ):
            self.skipTest("Legacy Gemini Live Mode has been removed.")

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
            live_session=SimpleNamespace(
                start=lambda *args, **kwargs: True,
                stop=lambda: None,
                end_audio_turn=lambda: None,
                send_audio_chunk=lambda *args, **kwargs: None,
                is_connected=False,
            ),
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
        app._live_turn_pending = False
        app._live_turn_waiting_for_transcript = False
        app._live_current_query = ""
        app._live_turn_ended_at = 0.0
        app._queued_live_query = ""
        app._queued_live_turn_ended_at = 0.0
        app._pending_live_query = ""
        app._last_live_transcript = ""
        app._live_recent_query_hint = ""
        app._live_assembled_query = ""
        app._live_native_input_buffer = ""
        app._live_native_query_text = ""
        app._live_native_transcript_seen_at = 0.0
        app._live_response_query_snapshot = ""
        app._live_empty_query_timeout_retries = 0
        app._live_mode_active = False
        app._live_mode_reconnecting = False
        app._live_fallback_active = False
        app._live_suppressed_completion_query = ""
        app._live_model_speaking = False
        app._live_turn_timer = SimpleNamespace(stop=lambda: None, start=lambda *_args, **_kwargs: None)
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
        app._should_dispatch_live_fallback_query = lambda text: OpenAssistApp._should_dispatch_live_fallback_query(app, text)
        app._looks_like_clipped_query_fragment = lambda text: OpenAssistApp._looks_like_clipped_query_fragment(text)
        app._normalized_live_query_words = lambda text: OpenAssistApp._normalized_live_query_words(text)
        app._stitch_live_interim_query = lambda existing, candidate: OpenAssistApp._stitch_live_interim_query(app, existing, candidate)
        app._append_live_transcript_fragment = lambda existing, fragment: OpenAssistApp._append_live_transcript_fragment(existing, fragment)
        app._live_query_fragments_overlap = lambda existing, candidate: OpenAssistApp._live_query_fragments_overlap(app, existing, candidate)
        app._merge_live_query_text = lambda existing, candidate: OpenAssistApp._merge_live_query_text(app, existing, candidate)
        app._live_query_label_score = lambda text: OpenAssistApp._live_query_label_score(app, text)
        app._select_best_live_query_label = lambda *parts: OpenAssistApp._select_best_live_query_label(app, *parts)
        app._is_stable_live_query_candidate = lambda text: OpenAssistApp._is_stable_live_query_candidate(app, text)
        app._should_promote_live_query_candidate = lambda candidate, existing: OpenAssistApp._should_promote_live_query_candidate(app, candidate, existing)
        app._remember_live_query_hint = lambda text: OpenAssistApp._remember_live_query_hint(app, text)
        app._remember_live_assembled_query = lambda text: OpenAssistApp._remember_live_assembled_query(app, text)
        app._can_enrich_pending_live_turn = lambda: OpenAssistApp._can_enrich_pending_live_turn(app)
        app._can_attach_to_pending_live_turn = lambda: OpenAssistApp._can_attach_to_pending_live_turn(app)
        app._can_attach_to_queued_live_turn = lambda: OpenAssistApp._can_attach_to_queued_live_turn(app)
        app._should_replace_live_fallback_query = lambda candidate, existing: OpenAssistApp._should_replace_live_fallback_query(app, candidate, existing)
        app._should_ignore_live_fallback_transcript = lambda text: OpenAssistApp._should_ignore_live_fallback_transcript(app, text)
        app._is_live_micro_utterance = lambda text: OpenAssistApp._is_live_micro_utterance(app, text)
        app._maybe_coalesce_queued_live_turn_into_pending = lambda candidate: OpenAssistApp._maybe_coalesce_queued_live_turn_into_pending(app, candidate)
        app._carry_forward_incomplete_audio_query = lambda text: OpenAssistApp._carry_forward_incomplete_audio_query(app, text)
        app.generate_response = lambda query, origin, ctx: None
        app._reset_turn_local_state = lambda reason="": OpenAssistApp._reset_turn_local_state(app, reason)
        app._log_turn_waterfall_summary = lambda provider=None, request_metadata=None, stage_timings=None: OpenAssistApp._log_turn_waterfall_summary(
            app,
            provider=provider,
            request_metadata=request_metadata,
            stage_timings=stage_timings,
        )
        app._live_mode_requested = lambda: OpenAssistApp._live_mode_requested(app)
        app._auto_mode_requested = lambda: OpenAssistApp._auto_mode_requested(app)
        app._live_mode_connected = lambda: OpenAssistApp._live_mode_connected(app)
        app._live_audio_passthrough_enabled = lambda: OpenAssistApp._live_audio_passthrough_enabled(app)
        app._live_turn_owns_audio = lambda: OpenAssistApp._live_turn_owns_audio(app)
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

    def test_end_session_does_not_dispatch_leftover_live_query(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.live_session.is_connected = True
        app._live_current_query = "Can you explain the primary differences between CSS Grid and Flexbox?"
        app._live_assembled_query = app._live_current_query
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

    def test_standard_transcription_still_runs_when_live_is_disabled(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", False)
        app._live_mode_active = True
        app._live_turn_pending = True
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

    def test_standard_transcription_runs_when_live_requested_but_not_connected(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "native")
        app._live_mode_active = True
        app._live_turn_pending = False
        app._live_mode_connected = lambda: False
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
        app.config.set("ai.live_mode.enabled", False)
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

    def test_live_error_promotes_pending_turn_to_standard_mode(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "native")
        app._live_mode_active = True
        app._live_mode_reconnecting = False
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._pending_live_query = "what changed"
        app._last_live_transcript = "older transcript"
        timer_stops = []
        app._live_turn_timer = SimpleNamespace(stop=lambda: timer_stops.append("stop"))
        generated = []
        app.generate_response = lambda q, s="manual", c=None: generated.append((q, s, c))
        app._promote_live_turn_to_standard = lambda reason: OpenAssistApp._promote_live_turn_to_standard(app, reason)

        OpenAssistApp._on_live_error(app, "socket closed")

        self.assertEqual(generated, [("what changed", "speech", {"audio": "what changed"})])
        self.assertFalse(app._live_turn_pending)
        self.assertTrue(app._live_fallback_active)
        self.assertEqual(app.overlay.live_mode_updates[-1]["fallback"], True)
        self.assertIn("standard audio pipeline", app.overlay.transcript_updates[-1].lower())
        self.assertTrue(timer_stops)

    def test_live_turn_complete_uses_pending_query_snapshot(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_mode_reconnecting = False
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._pending_live_query = "current live question"
        app._last_live_transcript = "previous question"
        timer_stops = []
        app._live_turn_timer = SimpleNamespace(stop=lambda: timer_stops.append("stop"))

        OpenAssistApp._on_live_turn_complete(app, "live answer")

        self.assertEqual(app.history.entries[-1]["query"], "current live question")
        self.assertEqual(app.history.entries[-1]["provider"], "gemini-live")
        self.assertEqual(app.overlay.completed[-1], ("live answer", "current live question"))
        self.assertFalse(app._live_turn_pending)
        self.assertEqual(app._pending_live_query, "")
        self.assertTrue(timer_stops)

    def test_live_turn_complete_falls_back_to_attached_transcript_snapshot(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_mode_reconnecting = False
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._live_current_query = ""
        app._pending_live_query = ""
        app._last_live_transcript = "What are loadable components?"
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(app, "live answer")

        self.assertEqual(app.history.entries[-1]["query"], "What are loadable components?")

    def test_live_turn_complete_recovers_when_streaming_but_pending_flag_was_cleared(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_current_query = "What are dictionaries in Python?"
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(app, "A dictionary stores key-value pairs.")

        self.assertEqual(app.history.entries[-1]["query"], "What are dictionaries in Python?")
        self.assertEqual(app.history.entries[-1]["provider"], "gemini-live")
        self.assertEqual(app.overlay.completed[-1], ("A dictionary stores key-value pairs.", "What are dictionaries in Python?"))

    def test_live_turn_complete_drops_non_answer_placeholder_without_history(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._live_current_query = ""
        app._pending_live_query = "CSS basics."
        app._last_live_transcript = ""
        app.live_session.is_connected = True
        app._record_live_outcome = lambda outcome: setattr(app, "_last_outcome", outcome)
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Okay, I'm switching gears now and focusing on CSS basics.",
        )

        self.assertEqual(app.history.entries, [])
        self.assertEqual(app.overlay.completed, [])
        self.assertFalse(app._live_turn_pending)
        self.assertEqual(app.overlay.transcript_updates[-1], "Live Listening...")

    def test_live_turn_complete_ignores_setup_speech_answer_without_history(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._pending_live_query = "Let's pivot to some CSS basics."
        app._last_live_transcript = "Let's pivot to some CSS basics."
        app.live_session.is_connected = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Sure, let's switch gears and talk about CSS basics.",
        )

        self.assertEqual(app.history.entries, [])
        self.assertEqual(app.overlay.completed, [])
        self.assertEqual(app.overlay.transcript_updates[-1], "Live Listening...")

    def test_live_turn_complete_prefers_recent_query_hint_over_short_fragment(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_current_query = "important"
        app._live_recent_query_hint = "What makes the borrow checker important?"
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "The borrow checker enforces ownership and borrowing rules at compile time.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "What makes the borrow checker important?",
        )

    def test_live_turn_complete_prefers_question_like_label_over_stale_suffix_fragment(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._pending_live_query = "thought experiment."
        app._last_live_transcript = "thought experiment."
        app._live_recent_query_hint = "What eviction policies might keep content fresh?"
        app._live_current_query = "ent fresh."
        app.overlay._current_query = "ent fresh."
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Use a CDN plus Redis, then prefer TTL and active invalidation for freshness.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "What eviction policies might keep content fresh?",
        )

    def test_live_turn_complete_uses_response_query_snapshot_after_pending_state_drift(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_response_query_snapshot = "What are props in React?"
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Props are immutable values passed from a parent component to a child component.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "What are props in React?",
        )

    def test_live_turn_complete_prefers_assembled_interim_query_over_last_fragment(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_assembled_query = "What if there were no closures in JavaScript?"
        app._live_current_query = "no closures?"
        app._live_recent_query_hint = "no closures?"
        app.overlay._current_query = "no closures?"
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Without closures, functions could not retain access to their lexical scope after returning.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "What if there were no closures in JavaScript?",
        )

    def test_live_turn_complete_prefers_native_query_text_over_fragmentary_fallback_label(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_native_query_text = "What are some key API design principles for a robust developer experience?"
        app._live_current_query = "are our mobile append team."
        app._live_recent_query_hint = "are our mobile append team."
        app.overlay._current_query = "are our mobile append team."
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Strong API design uses consistent resource naming, explicit versioning, clear error contracts, and predictable pagination for developers.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "What are some key API design principles for a robust developer experience?",
        )

    def test_live_turn_complete_drops_declarative_preamble_before_question(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_response_query_snapshot = (
            "Let's pivot to some CSS basics. A lot of developers get confused between CSS Grid and Flexbox. "
            "Can you explain the primary differences between the two?"
        )
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Grid is two-dimensional, while Flexbox is one-dimensional.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "Can you explain the primary differences between the two?",
        )

    def test_live_turn_complete_keeps_only_question_sentences_from_multi_part_prompt(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_response_query_snapshot = (
            "When building a secure REST API authentication is critical. "
            "Could you explain how JWT tokens work? "
            "What are the potential security risks if you store them in browser local storage?"
        )
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "JWTs are signed tokens, and local storage raises XSS exposure risks.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "Could you explain how JWT tokens work? What are the potential security risks if you store them in browser local storage?",
        )

    def test_live_turn_complete_ignores_scaling_setup_speech(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_response_query_snapshot = (
            "All right Let's talk about scaling. imagine we have monolithic application "
            "ba cked by single relational database that 's starting to s lowdown under heavy traffic."
        )
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Here are a few ways to scale that system.",
        )

        self.assertEqual(app.history.entries, [])

    def test_live_turn_complete_ignores_garbled_react_setup_speech(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_response_query_snapshot = (
            "Let's put Tom's-Pix Lepibusis aces A lot of developers can use Tweety as their lexicon. "
            "I developed GetFupid and SESCrenfluxbox. QElaine, Prairie Institute and explain the "
            "primary difference we knew and give example a layout where would finish river xbox Evan, "
            "I'm gonna allow you to use the flowers. So Somo So moving on to the next topic. I was"
        )
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Let's move to the next topic.",
        )

        self.assertEqual(app.history.entries, [])

    def test_live_turn_completion_query_repairs_split_words_in_question_label(self):
        app = self._build_app()
        app._live_response_query_snapshot = "Can you explain the primary differen ces between the two?"

        from ai.live_turn_state import select_live_turn_completion_query
        self.assertEqual(
            select_live_turn_completion_query(app),
            "Can you explain the primary differences between the two?",
        )

    def test_live_turn_completion_query_repairs_consider_split_in_scaling_prompt(self):
        app = self._build_app()
        app._live_response_query_snapshot = "What are some strategies you would con sider to alleviate that bottleneck?"

        from ai.live_turn_state import select_live_turn_completion_query
        self.assertEqual(
            select_live_turn_completion_query(app),
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_live_turn_completion_query_repairs_latest_scaling_fragment_duplication(self):
        app = self._build_app()
        app._live_response_query_snapshot = "What are some stra.. tegi.. What are some strategies you would?"

        from ai.live_turn_state import select_live_turn_completion_query
        self.assertEqual(
            select_live_turn_completion_query(app),
            "What are some strategies you would?",
        )

    def test_live_turn_completion_query_sanitizes_fragmented_query_label(self):
        app = self._build_app()
        app._live_response_query_snapshot = "What. are. props. in. React?"

        from ai.live_turn_state import select_live_turn_completion_query
        self.assertEqual(
            select_live_turn_completion_query(app),
            "What are props in React?",
        )

    def test_remember_live_assembled_query_stitches_interim_fragments(self):
        app = self._build_app()
        app._live_assembled_query = ""

        OpenAssistApp._remember_live_assembled_query(app, "CSS basics.")
        OpenAssistApp._remember_live_assembled_query(app, "and flex box.")
        OpenAssistApp._remember_live_assembled_query(app, "ces between the two?")

        self.assertEqual(
            app._live_assembled_query,
            "CSS basics. and flex box. ces between the two?",
        )

    def _replay_live_transcript_chunks(self, chunks):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_pending = False
        app._queued_live_query = ""
        app._queued_live_turn_ended_at = 0.0

        for chunk in chunks:
            OpenAssistApp._on_live_transcript_update(app, chunk)

        return app

    def test_live_transcript_chunk_replay_repairs_css_benchmark_stream(self):
        app = self._replay_live_transcript_chunks(
            [
                "Let's pivot to some",
                "CSS basics.",
                "A lot of de",
                "velop",
                "ers",
                "get",
                "con",
                "fused",
                "bet",
                "ween",
                "C",
                "SS",
                "gri",
                "d",
                "and flex box.",
                "Can",
                "you",
                "ex",
                "plain",
                "the",
                "pri",
                "mary",
                "di",
                "fferen",
                "ces between the two?",
            ]
        )

        self.assertEqual(
            app._live_current_query,
            "Can you explain the primary differences between the two?",
        )
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "Can you explain the primary differences between the two?",
        )

    def test_live_transcript_chunk_replay_repairs_jwt_benchmark_stream(self):
        app = self._replay_live_transcript_chunks(
            [
                "When building a secure REST API authentication is",
                "critical.",
                "Could",
                "you",
                "ex",
                "plain",
                "how",
                "J",
                "S",
                "ON",
                "web tokens work?",
                "what",
                "the",
                "pote",
                "nti",
                "al",
                "se",
                "curity",
                "ris",
                "ks",
                "if",
                "you",
                "sto",
                "re a",
                "J",
                "W",
                "T",
                "in",
                "the",
                "bro",
                "wser",
                "lo",
                "cal",
                "stor",
                "age",
                "instead",
                "of",
                "an",
                "h",
                "http-only cookie?",
            ]
        )

        self.assertEqual(
            app._live_current_query,
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_live_transcript_chunk_replay_repairs_react_hooks_benchmark_stream(self):
        app = self._replay_live_transcript_chunks(
            [
                "So moving on to the next topic.",
                "I was looking at your",
                "resu",
                "me",
                "and I see you",
                "'veusedreact",
                "quite a bit.",
                "Could",
                "you",
                "walk",
                "me",
                "through",
                "how",
                "you",
                "would",
                "cidebet",
                "weenu",
                "sing",
                "a cu",
                "stomhook",
                "versus",
                "a",
                "hi",
                "gher",
                "or dercomp",
                "onent",
                "for",
                "sha",
                "ringstate",
                "ful",
                "logic?",
            ]
        )

        self.assertEqual(
            app._live_current_query,
            "Could you walk me through how you would decide between using a custom hook versus a higher-order component for sharing stateful logic?",
        )
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "Could you walk me through how you would decide between using a custom hook versus a higher-order component for sharing stateful logic?",
        )

    def test_live_transcript_chunk_replay_repairs_api_design_benchmark_stream(self):
        app = self._replay_live_transcript_chunks(
            [
                "What",
                "are",
                "some",
                "of",
                "the",
                "keypri",
                "ncip",
                "les",
                "you",
                "would",
                "follow",
                "to",
                "en",
                "sure",
                "the",
                "A",
                "PI",
                "is",
                "ro",
                "bust",
                ",",
                "ver",
                "su",
                "inable",
                "and",
                "pro",
                "vi",
                "des",
                "a",
                "good",
                "de",
                "velo",
                "per",
                "ex",
                "perience",
                "for",
                "the",
                "fron",
                "t",
                "end team?",
            ]
        )

        self.assertEqual(
            app._live_current_query,
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides a good developer experience for the frontend team?",
        )
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides a good developer experience for the frontend team?",
        )

    def test_live_transcript_update_promotes_assembled_query_for_fragmented_native_transcript(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_pending = False
        app._queued_live_query = ""
        app._queued_live_turn_ended_at = 0.0

        OpenAssistApp._on_live_transcript_update(app, "CSS basics.")
        OpenAssistApp._on_live_transcript_update(app, "and flex box.")
        OpenAssistApp._on_live_transcript_update(app, "ces between the two?")

        self.assertEqual(
            app._live_assembled_query,
            "CSS basics. and Flexbox. ces between the two?",
        )
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "CSS basics. and Flexbox. ces between the two?",
        )

    def test_live_transcript_update_sanitizes_fragmented_native_question_before_overlay(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_pending = False

        OpenAssistApp._on_live_transcript_update(
            app,
            "Can. you. ex. plain. the. pri. mary. di. fferen. ces between the two?",
        )

        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "Can you explain the primary differences between the two?",
        )

    def test_live_transcript_update_assembles_native_token_stream_into_full_query(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_pending = False

        for token in [
            "What",
            "are",
            "some",
            "strategies",
            "you",
            "would",
            "consider",
            "to",
            "alleviate that bottleneck?",
        ]:
            OpenAssistApp._on_live_transcript_update(app, token)

        self.assertEqual(
            app._live_native_query_text,
            "What are some strategies you would consider to alleviate that bottleneck?",
        )
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_live_transcript_update_preserves_single_character_native_shards(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_pending = False

        for token in [
            "A lot of de",
            "velop",
            "ers",
            "get",
            "con",
            "fused",
            "bet",
            "ween",
            "C",
            "SS",
            "gri",
            "d",
            "and flex box.",
            "Can",
            "you",
            "ex",
            "plain",
            "the",
            "pri",
            "mary",
            "di",
            "fferen",
            "ces between the two?",
        ]:
            OpenAssistApp._on_live_transcript_update(app, token)

        self.assertIn("C SS", app._live_native_input_buffer)
        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "Can you explain the primary differences between the two?",
        )

    def test_live_standard_final_fragment_does_not_overwrite_better_native_query(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = True
        app._pending_live_query = "What are some strategies you would consider to alleviate that bottleneck?"
        app._live_native_query_text = "What are some strategies you would consider to alleviate that bottleneck?"
        app._live_current_query = "What are some strategies you would consider to alleviate that bottleneck?"
        app._live_fallback_active = False
        app._live_model_speaking = False
        app.audio.get_last_transcription_metrics = lambda: None
        app.nexus.push = lambda *_args, **_kwargs: None

        OpenAssistApp._on_transcription(app, "about scaling. alleviate that bottleneck?")

        self.assertEqual(
            app._live_current_query,
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_live_standard_final_setup_speech_does_not_pollute_retroactive_live_query(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "native")
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_ended_at = time.time()
        app.audio.get_last_transcription_metrics = lambda: None
        app.nexus.push = lambda *_args, **_kwargs: None
        timer_starts = []
        app._live_turn_timer = SimpleNamespace(
            start=lambda ms: timer_starts.append(ms),
            stop=lambda: None,
        )
        end_calls = []
        app.live_session.end_audio_turn = lambda: end_calls.append(True)

        OpenAssistApp._on_transcription(app, "A lot of development")
        OpenAssistApp._on_transcription(app, "Can you explain the primary differences between the two?")

        self.assertEqual(app._pending_incomplete_audio_query, "Can you explain the primary differences between the two?")
        self.assertEqual(
            app._pending_live_query,
            "Can you explain the primary differences between the two?",
        )
        self.assertEqual(end_calls, [True])
        self.assertEqual(len(timer_starts), 1)

    def test_live_standard_final_clipped_query_does_not_flush_live_turn_early(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_ended_at = time.time()
        app.audio.get_last_transcription_metrics = lambda: None
        app.nexus.push = lambda *_args, **_kwargs: None
        app._live_turn_timer = SimpleNamespace(start=lambda *_args: None, stop=lambda: None)
        end_calls = []
        app.live_session.end_audio_turn = lambda: end_calls.append(True)

        OpenAssistApp._on_transcription(app, "Could you walk me through-")

        self.assertFalse(app._live_turn_pending)
        self.assertEqual(end_calls, [])

    def test_live_standard_final_short_query_waits_for_pending_stt_context(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "hybrid_fast")
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_ended_at = time.time()
        app.audio.get_last_transcription_metrics = lambda: None
        app.audio.has_pending_transcription_jobs = lambda: True
        app.nexus.push = lambda *_args, **_kwargs: None
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp._on_transcription(app, "Could you explain how JSON Web Tokens work?")

        self.assertEqual(dispatched, [])
        self.assertEqual(
            app._pending_incomplete_audio_query,
            "Could you explain how JWT tokens work?",
        )

    def test_live_standard_final_jwt_followup_clause_dispatches_merged_query(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "hybrid_fast")
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_ended_at = time.time()
        app._pending_incomplete_audio_query = "Could you explain how JWT tokens work?"
        app._live_current_query = "Could you explain how JWT tokens work?"
        app._live_recent_query_hint = "Could you explain how JWT tokens work?"
        app.audio.get_last_transcription_metrics = lambda: None
        app.audio.has_pending_transcription_jobs = lambda: False
        app.nexus.push = lambda *_args, **_kwargs: None
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp._on_transcription(
            app,
            "and what the potential security risks are if you store a JWT in the browser's local storage instead of an HTTP-only cookie.",
        )

        self.assertEqual(
            dispatched[0][0],
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_live_transcript_update_ignores_fragmented_clipped_shard(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_pending = False

        OpenAssistApp._on_live_transcript_update(app, "Le. t'")

        self.assertEqual(app.overlay.transcript_updates, [])

    def test_live_transcript_update_does_not_replace_cleaner_label_with_weaker_shard(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_last_surface_text = "Can you explain the primary differences between the two?"

        OpenAssistApp._on_live_transcript_update(app, "where")

        self.assertEqual(app.overlay.transcript_updates, [])
        self.assertEqual(
            app._live_last_surface_text,
            "Can you explain the primary differences between the two?",
        )

    def test_live_interim_transcription_does_not_override_recent_native_live_query(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_assembled_query = "What is state in React?"
        app._live_current_query = "What is state in React?"
        app._live_native_transcript_seen_at = time.time()

        OpenAssistApp._on_interim_transcription(
            app,
            "Could. how. would. onent. teful logic?",
        )

        self.assertEqual(app._live_assembled_query, "What is state in React?")
        self.assertEqual(app.overlay.transcript_updates, [])

    def test_live_transcript_update_prefers_richer_assembled_query_over_latest_native_shard(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = True
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_native_query_text = "Can you explain the primary differences between the two?"
        app._live_assembled_query = (
            "Can you explain the primary differences between the two? "
            "give an example of a layout where you would definitely choose grid over Flexbox?"
        )
        app._pending_live_query = app._live_assembled_query

        OpenAssistApp._on_live_transcript_update(
            app,
            "give an example of a layout where you wouldde finitelychoosegrid over Flexbox?",
        )

        self.assertEqual(
            app._live_current_query,
            "Can you explain the primary differences between the two? give an example of a layout where you would definitely choose grid over Flexbox?",
        )

    def test_live_turn_complete_non_answer_with_pending_drift_dispatches_standard_fallback(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = False
        app._live_current_query = ""
        app._live_recent_query_hint = "What are the differences between CSS Grid and Flexbox?"
        app.overlay._is_streaming = True
        app.audio = SimpleNamespace(set_standard_transcription_suspended=lambda *args, **kwargs: None)
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp._on_live_turn_complete(
            app,
            "Okay, I'm switching gears now and focusing on CSS basics.",
        )

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(
            dispatched[0][0],
            "What are the differences between CSS Grid and Flexbox?",
        )
        self.assertEqual(dispatched[0][1], "speech")

    def test_live_status_connected_suspends_standard_asr(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        suspends = []
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": suspends.append((enabled, reason))
        )
        app.live_session = SimpleNamespace(is_connected=True)

        OpenAssistApp._on_live_status_changed(app, "Live Listening...")

        self.assertEqual(suspends[-1], (True, "live listening..."))
        self.assertFalse(app._live_model_speaking)

    def test_live_status_responding_marks_model_as_speaking(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": None
        )
        app.live_session = SimpleNamespace(is_connected=True)

        OpenAssistApp._on_live_status_changed(app, "Live Responding...")

        self.assertTrue(app._live_model_speaking)

    def test_live_status_responding_does_not_clear_standard_fallback_takeover(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_fallback_active = True
        app._live_turn_pending = False
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": None
        )
        app.live_session = SimpleNamespace(is_connected=True)

        OpenAssistApp._on_live_status_changed(app, "Live Responding...")

        self.assertTrue(app._live_fallback_active)

    def test_live_text_delta_ignores_stale_stream_without_turn_context(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_turn_pending = False
        app._pending_live_query = ""
        app._live_current_query = ""
        app._live_response_query_snapshot = ""

        OpenAssistApp._on_live_text_delta(app, "stale chunk")

        self.assertEqual(app.overlay.appended, [])
        self.assertEqual(app.mini_overlay.appended, [])

    def test_live_text_delta_streams_into_overlays(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_fallback_active = False
        app._pending_live_query = "current live question"
        app._last_live_transcript = ""
        app._live_current_query = ""

        OpenAssistApp._on_live_text_delta(app, "live chunk")

        self.assertEqual(app.overlay._current_query, "current live question")
        self.assertEqual(app.overlay.appended, ["live chunk"])
        self.assertEqual(app.mini_overlay.appended, ["live chunk"])

    def test_live_interim_transcription_is_sanitized_before_overlay_update(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False
        app._live_current_query = ""

        OpenAssistApp._on_interim_transcription(
            app,
            "Can. you. ex. plain. the. pri. mary. di. fferen. ces between the two?",
        )

        self.assertEqual(
            app.overlay.transcript_updates[-1],
            "Can you explain the primary differences between the two?",
        )

    def test_live_interim_transcription_suppresses_clipped_fragment_noise(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False

        OpenAssistApp._on_interim_transcription(app, "where")

        self.assertEqual(app.overlay.transcript_updates, [])

    def test_stop_live_mode_resumes_standard_asr(self):
        app = self._build_app()
        resumed = []
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": resumed.append((enabled, reason))
        )
        app.live_session = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._stop_live_mode(app)

        self.assertEqual(resumed[-1], (False, "live-stopped"))

    def test_live_audio_turn_end_keeps_turn_open_for_setup_speech(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.live_session.is_connected = True
        app._live_mode_active = True
        app._live_current_query = "Let's pivot to some CSS basics."
        app._last_live_transcript = "Let's pivot to some CSS basics."
        app._live_turn_pending = False
        timer_starts = []
        app._live_turn_timer = SimpleNamespace(start=lambda ms: timer_starts.append(ms))
        end_calls = []
        app.live_session.end_audio_turn = lambda: end_calls.append(True)

        OpenAssistApp._on_live_audio_turn_end(app)

        self.assertFalse(app._live_turn_pending)
        self.assertEqual(end_calls, [])
        self.assertEqual(timer_starts, [])

    def test_live_audio_turn_end_dispatches_when_accumulated_query_is_actionable(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "native")
        app.live_session.is_connected = True
        app._live_mode_active = True
        app._live_current_query = (
            "Could you explain how JWT tokens work? What are the potential security risks "
            "if you store a JWT in the browser local storage instead of an http-only cookie?"
        )
        app._last_live_transcript = app._live_current_query
        app._live_turn_pending = False
        timer_starts = []
        app._live_turn_timer = SimpleNamespace(start=lambda ms: timer_starts.append(ms))
        end_calls = []
        app.live_session.end_audio_turn = lambda: end_calls.append(True)

        OpenAssistApp._on_live_audio_turn_end(app)

        self.assertTrue(app._live_turn_pending)
        self.assertEqual(end_calls, [True])
        self.assertEqual(len(timer_starts), 1)

    def test_live_audio_turn_end_hybrid_fast_dispatches_standard_without_waiting_for_native_answer(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "hybrid_fast")
        app.live_session.is_connected = True
        app._live_mode_active = True
        app._live_current_query = "What are some strategies you would consider to alleviate that bottleneck?"
        app._last_live_transcript = app._live_current_query
        timer_starts = []
        app._live_turn_timer = SimpleNamespace(start=lambda ms: timer_starts.append(ms))
        end_calls = []
        app.live_session.end_audio_turn = lambda: end_calls.append(True)
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp._on_live_audio_turn_end(app)

        self.assertFalse(app._live_turn_pending)
        self.assertEqual(end_calls, [])
        self.assertEqual(timer_starts, [])
        self.assertEqual(
            dispatched,
            [
                (
                    "What are some strategies you would consider to alleviate that bottleneck?",
                    "speech",
                    {
                        "audio": "What are some strategies you would consider to alleviate that bottleneck?",
                        "live_hybrid_fast": True,
                    },
                )
            ],
        )

    def test_live_transcript_update_refreshes_completion_snapshot_while_turn_pending(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_turn_pending = True
        app._pending_live_query = "where you would definitely choose grid over Flexbox?"
        app._live_response_query_snapshot = app._pending_live_query

        OpenAssistApp._on_live_transcript_update(
            app,
            "Can you explain the primary differences between the two? give an example of a layout where you would definitely choose grid over Flexbox?",
        )

        self.assertEqual(
            app._live_response_query_snapshot,
            "Can you explain the primary differences between the two? give an example of a layout where you would definitely choose grid over Flexbox?",
        )
        self.assertEqual(app._pending_live_query, app._live_response_query_snapshot)

    def test_live_status_refreshing_does_not_promote_standard_fallback(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        resumed = []
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": resumed.append((enabled, reason))
        )
        app._live_turn_pending = True
        app._pending_live_query = "What is React?"

        OpenAssistApp._on_live_status_changed(app, "Refreshing Live Mode...")

        self.assertTrue(app._live_mode_refreshing)
        self.assertTrue(app._live_turn_pending)
        self.assertEqual(app._pending_live_query, "What is React?")
        self.assertEqual(resumed[-1], (True, "refreshing live mode..."))

    def test_live_status_refreshing_suppresses_transcript_banner_churn(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": None
        )

        OpenAssistApp._on_live_status_changed(app, "Refreshing Live Mode...")

        self.assertEqual(app.overlay.transcript_updates, [])

    def test_live_status_connected_after_refresh_suppresses_connected_banner(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_refreshing = True
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": None
        )
        app.live_session = SimpleNamespace(is_connected=True)

        OpenAssistApp._on_live_status_changed(app, "Live Mode Connected")

        self.assertEqual(app.overlay.transcript_updates, [])
        self.assertFalse(app._live_mode_refreshing)

    def test_live_status_reconnecting_after_refresh_keeps_standard_fallback_dormant(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_refreshing = True
        app._live_turn_pending = True
        app._pending_live_query = "What is React?"
        suspends = []
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": suspends.append((enabled, reason))
        )
        promoted = []
        app._promote_live_turn_to_standard = lambda reason: promoted.append(reason)

        OpenAssistApp._on_live_status_changed(app, "Reconnecting Live Mode...")

        self.assertTrue(app._live_mode_refreshing)
        self.assertTrue(app._live_turn_pending)
        self.assertEqual(app._pending_live_query, "What is React?")
        self.assertEqual(promoted, [])
        self.assertEqual(suspends[-1], (True, "reconnecting live mode..."))

    def test_stop_live_mode_routes_only_sanitized_question_from_leftover_query(self):
        app = self._build_app()
        app.session_active = True
        app._live_turn_pending = False
        app._pending_live_query = (
            "All right Let's talk about scaling. imagine we have monolithic application "
            "ba cked by single relational database that's starting to s lowdown under heavy traffic. "
            "What are some strategies you would consider to alleviate that bottleneck?"
        )
        app._last_live_transcript = app._pending_live_query
        routed = []
        app.generate_response = lambda q, origin, ctx: routed.append((q, origin, ctx))
        app.audio = SimpleNamespace(
            set_standard_transcription_suspended=lambda enabled, reason="": None
        )
        app.live_session = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._stop_live_mode(app, dispatch_leftover=True)

        self.assertEqual(
            routed,
            [(
                "What are some strategies you would consider to alleviate that bottleneck?",
                "speech",
                {"audio": "What are some strategies you would consider to alleviate that bottleneck?"},
            )],
        )

    def test_live_requested_but_unavailable_marks_standard_fallback(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.live_session.start = lambda *args, **kwargs: False
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._start_live_mode(app)

        self.assertFalse(app._live_mode_active)
        self.assertTrue(app._live_fallback_active)
        self.assertEqual(app.overlay.live_mode_updates[-1]["fallback"], True)
        self.assertIn("standard audio pipeline", app.overlay.transcript_updates[-1].lower())

    def test_live_turn_complete_ignores_stale_completion_after_fixture_reset(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_turn_pending = False
        app._live_fallback_active = False
        app._live_current_query = ""
        app._pending_live_query = ""
        app._live_response_query_snapshot = ""
        app._live_native_query_text = ""
        app._live_assembled_query = ""

        OpenAssistApp._on_live_turn_complete(app, "This answer arrived too late.")

        self.assertEqual(app.history.entries, [])
        self.assertEqual(app.overlay.completed, [])

    def test_live_turn_complete_suppresses_late_live_answer_after_standard_fallback_dispatch(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_turn_pending = False
        app._live_response_query_snapshot = (
            "Could you explain how JWT tokens work? What are the potential security risks "
            "if you store a JWT in the browser local storage instead of an http-only cookie?"
        )
        app._live_suppressed_completion_query = app._live_response_query_snapshot
        app._live_fallback_active = True
        app._live_model_speaking = True
        app._last_live_transcript = app._live_response_query_snapshot

        OpenAssistApp._on_live_turn_complete(app, "Late Gemini live answer.")

        self.assertEqual(app.history.entries, [])
        self.assertEqual(app.overlay.completed, [])
        self.assertFalse(app._live_fallback_active)
        self.assertFalse(app._live_model_speaking)
        self.assertEqual(app._live_response_query_snapshot, "")
        self.assertEqual(app._last_live_transcript, app._live_suppressed_completion_query)

    def test_reset_benchmark_fixture_runtime_clears_recent_live_query_hint(self):
        app = self._build_app()
        app._live_recent_query_hint = "What are the differences between CSS Grid and Flexbox?"
        app.overlay.response_area = SimpleNamespace(clear=lambda: None)

        OpenAssistApp.reset_benchmark_fixture_runtime(app)

        self.assertEqual(app._live_recent_query_hint, "")

    def test_reset_benchmark_fixture_runtime_clears_pending_audio_followup_state(self):
        app = self._build_app()
        app._pending_incomplete_audio_query = "Can you explain closures?"
        app._pending_incomplete_audio_at = 123.0
        app.overlay.response_area = SimpleNamespace(clear=lambda: None)

        OpenAssistApp.reset_benchmark_fixture_runtime(app)

        self.assertEqual(app._pending_incomplete_audio_query, "")
        self.assertEqual(app._pending_incomplete_audio_at, 0.0)

    def test_reset_benchmark_fixture_runtime_preserves_live_reconnect_flags(self):
        app = self._build_app()
        app._live_mode_reconnecting = True
        app._live_mode_refreshing = True
        app.overlay.response_area = SimpleNamespace(clear=lambda: None)

        OpenAssistApp.reset_benchmark_fixture_runtime(app)

        self.assertTrue(app._live_mode_reconnecting)
        self.assertTrue(app._live_mode_refreshing)

    def test_live_turn_owns_audio_during_refresh_gap(self):
        app = self._build_app()
        app.config.set("ai.live_mode.enabled", True)
        app.live_session.is_connected = False
        app.live_session.is_running = True
        app._live_mode_refreshing = True
        app._live_turn_pending = False

        self.assertTrue(OpenAssistApp._live_turn_owns_audio(app))

    def test_live_audio_chunk_queues_during_initial_connect_gap(self):
        app = self._build_app()
        app.config.set("ai.live_mode.enabled", True)
        app.live_session.is_connected = False
        app.live_session.is_running = True
        sent = []
        shadowed = []
        app.live_session.send_audio_chunk = lambda pcm, sr: sent.append((pcm, sr))
        app.audio.push_to_live_shadow_buffer = lambda pcm, sr: shadowed.append((pcm, sr))

        OpenAssistApp._on_live_audio_chunk(app, b"pcm", 16000)

        self.assertEqual(sent, [(b"pcm", 16000)])
        self.assertEqual(shadowed, [(b"pcm", 16000)])

    def test_live_standard_final_dispatch_sanitizes_scaling_setup_preamble(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app.config.set("ai.live_mode.answer_strategy", "hybrid_fast")
        app._live_mode_active = True
        app.live_session.is_connected = True
        app._live_turn_pending = False
        app._live_fallback_active = False
        app._live_model_speaking = False
        app._live_turn_ended_at = time.time()
        app._live_current_query = "Let's talk about scaling. What are some strategies you would consider to alleviate that bottleneck?"
        app.audio.get_last_transcription_metrics = lambda: None
        app.audio.has_pending_transcription_jobs = lambda: False
        app.nexus.push = lambda *_args, **_kwargs: None
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp._on_transcription(app, "What are some strategies you would consider to alleviate that bottleneck?")

        self.assertEqual(
            dispatched[0][0],
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_promote_live_turn_to_standard_sanitizes_fragmented_query_before_dispatch(self):
        app = self._build_app()
        app.session_active = True
        app._live_turn_pending = True
        app._pending_live_query = "How. do. J. W. T. tokens work?"
        app._live_response_query_snapshot = "How. do. J. W. T. tokens work?"
        app.audio = SimpleNamespace(set_standard_transcription_suspended=lambda *args, **kwargs: None)
        dispatched = []
        app.generate_response = lambda query, origin, ctx: dispatched.append((query, origin, ctx))

        OpenAssistApp._promote_live_turn_to_standard(app, "test-timeout")

        self.assertEqual(dispatched[0][0], "How do JWT tokens work?")

    def test_live_turn_complete_repairs_latest_benchmark_jwt_query_pollution(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._pending_live_query = (
            "When building is secure REST API authentication is critical. and we b to kens work and "
            "What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?"
        )
        app._live_response_query_snapshot = app._pending_live_query
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "JWTs are signed tokens, and browser local storage increases XSS exposure compared with http-only cookies.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "Could you explain how JWT tokens work? What are the potential security risks if you store a JWT in the browser local storage instead of an http-only cookie?",
        )

    def test_live_turn_complete_recovers_scaling_tail_fragment_into_question(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._pending_live_query = (
            "single relational database that 's starting to slowdown under heavy traffic. alleviate that bottleneck."
        )
        app._live_response_query_snapshot = app._pending_live_query
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "You could add read replicas, caching, query optimization, and partitioning to reduce load on the database.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "What are some strategies you would consider to alleviate that bottleneck?",
        )

    def test_live_turn_complete_recovers_css_malformed_tail_into_actionable_question(self):
        app = self._build_app()
        app.session_active = True
        app.config.set("ai.live_mode.enabled", True)
        app._live_mode_active = True
        app._live_fallback_active = False
        app._live_turn_pending = True
        app._live_response_query_snapshot = "between CSS grid and Flexbox."
        app._live_native_query_text = "between CSS grid and Flexbox."
        app._live_assembled_query = (
            "between CSS grid. and Flexbox. ween the two and give an example of a layout where "
            "you would definitely choose grid overflex box.."
        )
        app.overlay._is_streaming = True
        app._live_turn_timer = SimpleNamespace(stop=lambda: None)

        OpenAssistApp._on_live_turn_complete(
            app,
            "Grid handles two-dimensional layouts, while Flexbox is better for one-dimensional alignment.",
        )

        self.assertEqual(
            app.history.entries[-1]["query"],
            "Can you explain the primary differences between the two? give an example of a layout where you would definitely choose grid over Flexbox?",
        )

    def test_live_turn_completion_query_merges_api_setup_with_followup_clause(self):
        app = self._build_app()
        app._live_response_query_snapshot = "a new public-facing PI for our mobile app."
        app._live_assembled_query = (
            "a new public-facing PI for our mobile app. the PI is robust, maintainable and pro vides "
            "good developer experience for the frontend team."
        )
        app._live_current_query = "you wouldfo llow to ensure the API is robust and provides a good developer experience"

        from ai.live_turn_state import select_live_turn_completion_query

        self.assertEqual(
            select_live_turn_completion_query(app),
            "What are some of the key principles you would follow to ensure the API is robust, maintainable and provides a good developer experience for the frontend team?",
        )

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

    def test_should_replace_live_query_rejects_clipped_fragment_over_richer_text(self):
        app = self._build_app()

        should_replace = OpenAssistApp._should_replace_live_fallback_query(
            app,
            "web tokens work?",
            "When should I use JWT tokens?",
        )

        self.assertFalse(should_replace)

    def test_live_fallback_dispatch_rejects_when_building_setup_statement(self):
        app = self._build_app()
        app.ai.detector.question_prefixes.append("when ")

        self.assertFalse(
            OpenAssistApp._should_dispatch_live_fallback_query(
                app,
                "when building a secure REST API authentication is critical",
            )
        )
        self.assertFalse(
            OpenAssistApp._looks_question_like_transcript(
                app,
                "when building a secure REST API authentication is critical",
            )
        )

    def test_merge_live_query_text_stitches_scaling_suffix_fragment_into_topic(self):
        app = self._build_app()

        merged = OpenAssistApp._merge_live_query_text(
            app,
            "about scaling.",
            "alleviate that bottleneck?",
        )

        self.assertEqual(merged, "about scaling. alleviate that bottleneck?")

    def test_live_query_keyword_extraction_includes_inflight_live_buffers(self):
        app = self._build_app()
        app._live_current_query = "JWT tokens"
        app._live_assembled_query = "database scaling"
        app._pending_live_query = "component stateful logic"
        app._live_recent_query_hint = "auth flow"
        app._last_live_transcript = ""
        app.history = SimpleNamespace(get_last=lambda n: [])

        keywords = OpenAssistApp._get_session_asr_keywords(app)

        lowered = {k.lower() for k in keywords}
        self.assertIn("jwt", lowered)
        self.assertIn("scaling", lowered)
        self.assertIn("stateful", lowered)
        self.assertIn("auth", lowered)

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

    def test_shutdown_stops_live_mode_without_dispatching_leftover_query(self):
        app = self._build_app()
        app.session_active = False
        app._stop_background_tasks = lambda: None
        app.audio = SimpleNamespace(stop=lambda: None, clear=lambda: None)
        app.hotkeys = SimpleNamespace(stop=lambda: None, reset_state=lambda: None)
        app.rag = SimpleNamespace(stop=lambda: None)
        app.loop = SimpleNamespace(is_running=lambda: False)
        app._async_thread = SimpleNamespace(
            is_alive=lambda: False, join=lambda timeout=None: None
        )
        stop_calls = []
        app._stop_live_mode = lambda dispatch_leftover=True: stop_calls.append(dispatch_leftover)

        OpenAssistApp.shutdown(app)

        self.assertEqual(stop_calls, [False])

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
