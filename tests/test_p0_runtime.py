import asyncio
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.history import ResponseHistory
from ai.engine import AIEngine
from capture.audio import AudioCapture
from capture.screen import ScreenCapture
from core.state import AppState


class ConfigStub:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def get(self, path, default=None):
        return self.settings.get(path, default)

    def set(self, path, value):
        self.settings[path] = value


class HistoryStub:
    def __init__(self):
        self.entries = []

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


class ParallelStub:
    async def generate(self, system, user, task="general", tier=None):
        return "parallel answer"


class ProviderStub:
    enabled = True
    name = "groq"


class AudioCaptureLifecycleTests(unittest.TestCase):
    def test_toggle_returns_mute_state_and_updates_pause(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))

        muted = audio.toggle()
        self.assertTrue(muted)
        self.assertTrue(audio._muted)
        self.assertTrue(audio._paused)

        muted = audio.toggle()
        self.assertFalse(muted)
        self.assertFalse(audio._muted)
        self.assertFalse(audio._paused)

    def test_stop_clears_running_flag_and_queue(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._running = True
        audio.q.put_nowait(b"frame")

        audio.stop()

        self.assertFalse(audio._running)
        self.assertTrue(audio.q.empty())

    def test_restart_updates_mode_and_restarts_when_running(self):
        audio = AudioCapture(
            ConfigStub({"capture.audio.mode": "mic", "capture.audio.enabled": True})
        )
        audio._running = True
        audio.last_mode = "system"

        start_calls = []
        with patch.object(audio, "start", side_effect=lambda: start_calls.append("start")):
            audio.restart()

        self.assertEqual(audio.capture_mode, "mic")
        self.assertEqual(audio.last_mode, "mic")
        self.assertEqual(start_calls, ["start"])


class AIEngineParallelOrderingTests(unittest.TestCase):
    def test_parallel_generation_emits_completion_without_duplicate_chunks(self):
        config = ConfigStub(
            {
                "ai.mode": "general",
                "ai.parallel.enabled": True,
                "ai.use_complexity_routing": True,
            }
        )
        history = HistoryStub()
        engine = AIEngine(config, history, rag=None)
        engine._parallel = ParallelStub()
        engine._providers = {"groq": ProviderStub()}
        engine._active_provider_id = "groq"

        chunks = []
        completed = []
        errors = []
        engine.response_chunk.connect(chunks.append)
        engine.response_complete.connect(completed.append)
        engine.error_occurred.connect(errors.append)

        asyncio.run(
            engine.generate_response(
                "hello",
                {"recent_audio": "", "full_audio_history": "", "latest_ocr": ""},
            )
        )

        self.assertEqual(errors, [])
        self.assertEqual(chunks, [])
        self.assertEqual(completed, ["parallel answer"])
        self.assertEqual(history.entries[-1]["provider"], "parallel")


class AppStateIsolationTests(unittest.TestCase):
    def test_app_state_instances_are_independent(self):
        first = AppState(ConfigStub({"ai.mode": "general"}))
        second = AppState(ConfigStub({"ai.mode": "coding"}))

        first.mode = "meeting"

        self.assertEqual(first.mode, "meeting")
        self.assertEqual(second.mode, "coding")


class AIEngineProviderSelectionTests(unittest.TestCase):
    def test_select_provider_uses_router_with_complexity_preferences(self):
        config = ConfigStub({"ai.mode": "general"})
        engine = AIEngine(config, HistoryStub(), rag=None)

        selected = []

        class RouterStub:
            def select(
                self,
                task="general",
                prefer_speed=False,
                prefer_quality=False,
                tier=None,
                exclude=None,
                preferred=None,
            ):
                selected.append(
                    {
                        "task": task,
                        "prefer_speed": prefer_speed,
                        "prefer_quality": prefer_quality,
                        "preferred": preferred,
                    }
                )
                provider = ProviderStub()
                provider.name = "gemini"
                return provider, "balanced"

            @staticmethod
            def _tier_for_task(task):
                return "balanced"

        engine._router = RouterStub()

        provider, tier = engine._select_provider(
            "general",
            "reasoning",
            engine._preferred_providers_for_complexity("reasoning"),
        )

        self.assertEqual(provider.name, "gemini")
        self.assertEqual(tier, "balanced")
        self.assertEqual(selected[0]["task"], "general")
        self.assertFalse(selected[0]["prefer_speed"])
        self.assertTrue(selected[0]["prefer_quality"])
        self.assertEqual(selected[0]["preferred"][0], "gemini")


class ResponseHistoryReadOnlyTests(unittest.TestCase):
    def test_read_session_does_not_mutate_active_session(self):
        history_dir = Path("openassist") / "data" / "test_history_runtime"
        if history_dir.exists():
            shutil.rmtree(history_dir, ignore_errors=True)

        with patch(
            "ai.history.time.time",
            side_effect=[1000, 1000, 1000, 1000, 1001, 1001, 1001, 1001, 1002, 1002],
        ):
            history = ResponseHistory(history_dir=str(history_dir))
            history.start_new_session()
            first_session_id = history.current_session_id
            history.add("q1", "r1", provider="groq")

            history.start_new_session()
            active_session_id = history.current_session_id
            history.add("q2", "r2", provider="groq")

            entries = history.read_session(first_session_id)

        self.assertEqual(history.current_session_id, active_session_id)
        self.assertEqual(history.entries[-1].query, "q2")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].query, "q1")

        shutil.rmtree(history_dir, ignore_errors=True)


class ScreenCaptureContextTests(unittest.TestCase):
    def test_capture_context_uses_pipeline_result(self):
        capture = ScreenCapture.__new__(ScreenCapture)
        capture._last_text = "old text"

        async def fake_capture():
            return "new text"

        capture.capture = fake_capture

        result = asyncio.run(ScreenCapture.capture_context(capture))

        self.assertEqual(result, "new text")

    def test_capture_context_falls_back_to_last_text_when_debounced(self):
        capture = ScreenCapture.__new__(ScreenCapture)
        capture._last_text = "cached text"

        async def fake_capture():
            return None

        capture.capture = fake_capture

        result = asyncio.run(ScreenCapture.capture_context(capture))

        self.assertEqual(result, "cached text")


if __name__ == "__main__":
    unittest.main()
