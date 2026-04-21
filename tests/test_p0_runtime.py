import asyncio
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.history import ResponseHistory
from ai.engine import AIEngine
from capture.audio import AudioCapture
from capture.screen import ScreenCapture
from core.config import Config
from core.hotkeys import HotkeyManager, NativeHotkeyThread
from core.state import AppState
from utils.platform_utils import WindowUtils


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
        self.screen_analyses = []

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

    def add_screen_analysis(self, query, response, provider, metadata=None):
        self.screen_analyses.append(
            {
                "prompt": query,
                "response": response,
                "provider": provider,
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

    def test_find_system_audio_source_prefers_stereo_mix_over_output_loopback(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))

        fake_devices = [
            {
                "name": "Speakers (Realtek(R) Audio)",
                "hostapi": 2,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "Stereo Mix (Realtek(R) Audio)",
                "hostapi": 2,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
        ]
        fake_hostapis = [{"name": "Windows WASAPI"}]

        with patch("sounddevice.query_devices", return_value=fake_devices), patch(
            "sounddevice.query_hostapis", return_value=fake_hostapis
        ):
            self.assertEqual(
                audio._find_system_audio_source(),
                (1, "Stereo Mix (Realtek(R) Audio)", False),
            )

    def test_resample_to_target_rate_matches_configured_rate(self):
        audio = AudioCapture(ConfigStub({"capture.audio.sample_rate": 16000}))
        samples = np.ones((480, 2), dtype=np.float32)

        resampled = audio._resample_to_target_rate(samples, 48000)

        self.assertEqual(resampled.shape, (160, 2))


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

    def test_vision_fallback_clears_partial_stream_before_retry_success(self):
        config = ConfigStub(
            {"ai.mode": "general", "ai.vision.allow_paid_fallback": True}
        )
        history = HistoryStub()
        engine = AIEngine(config, history, rag=None)

        class FailingVisionProvider:
            enabled = True
            name = "gemini"

            @staticmethod
            def check_rate():
                return True

            @staticmethod
            def supports_vision():
                return True

            @staticmethod
            def supports_vision_stream():
                return True

            async def analyze_image_stream(self, system, user, image_bytes, mime_type="image/png"):
                yield "partial"
                raise Exception("boom")

        class SuccessVisionProvider:
            enabled = True
            name = "openai"

            @staticmethod
            def check_rate():
                return True

            @staticmethod
            def supports_vision():
                return True

            @staticmethod
            def supports_vision_stream():
                return False

            async def analyze_image(self, system, user, image_bytes, mime_type="image/png"):
                return "final answer"

        engine._providers = {
            "gemini": FailingVisionProvider(),
            "openai": SuccessVisionProvider(),
        }

        chunks = []
        completed = []
        engine.response_chunk.connect(chunks.append)
        engine.response_complete.connect(completed.append)

        asyncio.run(
            engine.analyze_image_response(
                "analyze",
                b"img",
                {"latest_ocr": "", "full_audio_history": ""},
            )
        )

        self.assertEqual(chunks, ["partial"])
        self.assertEqual(completed, ["", "final answer"])
        self.assertEqual(history.entries[-1]["provider"], "openai")


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

    def test_read_session_bundle_returns_entries_and_screen_analyses(self):
        history_dir = Path("openassist") / "data" / "test_history_bundle_runtime"
        if history_dir.exists():
            shutil.rmtree(history_dir, ignore_errors=True)

        history = ResponseHistory(history_dir=str(history_dir))
        history.start_new_session()
        session_id = history.current_session_id
        history.add("q1", "r1", provider="groq")
        history.add_screen_analysis("analyze", "screen result", provider="gemini")

        bundle = history.read_session_bundle(session_id)

        self.assertEqual(len(bundle["entries"]), 1)
        self.assertEqual(bundle["entries"][0].query, "q1")
        self.assertEqual(len(bundle["screen_analyses"]), 1)
        self.assertEqual(bundle["screen_analyses"][0]["provider"], "gemini")

        shutil.rmtree(history_dir, ignore_errors=True)


class ConfigResetTests(unittest.TestCase):
    def test_reset_all_clears_secrets_and_restores_first_run_state(self):
        class FakeSecureStorage:
            def __init__(self, filepath="data/settings.enc"):
                self.data = {"api_key_groq": "secret"}

            def get_api_key(self, provider):
                return self.data.get(f"api_key_{provider}", "")

            def set_api_key(self, provider, key):
                self.data[f"api_key_{provider}"] = key

            def clear_all(self):
                self.data.clear()

        config_path = Path("openassist") / "data" / "test_reset_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("onboarding:\n  completed: true\n", encoding="utf-8")

        with patch("core.config.SecureStorage", FakeSecureStorage):
            config = Config(str(config_path))
            config.set("ai.mode", "coding")
            config.reset_all()

            self.assertFalse(config.get("onboarding.completed", True))
            self.assertEqual(config.get("ai.vision.allow_paid_fallback", None), False)
            self.assertEqual(config.secrets.get_api_key("groq"), "")

        if config_path.exists():
            config_path.unlink()


class HotkeyMatchingTests(unittest.TestCase):
    def test_exact_modifier_match_prevents_scroll_triggering_move(self):
        manager = HotkeyManager.__new__(HotkeyManager)
        manager.active_keys = {
            "key.ctrl": 1.0,
            "key.shift": 1.0,
            "key.up": 1.0,
        }

        move_req = HotkeyManager._parse_key_pynput(manager, "ctrl+up")
        scroll_req = HotkeyManager._parse_key_pynput(manager, "ctrl+shift+up")

        self.assertFalse(HotkeyManager._is_exact_hotkey_match(manager, move_req))
        self.assertTrue(HotkeyManager._is_exact_hotkey_match(manager, scroll_req))

    def test_toggle_hotkey_native_registration_uses_no_repeat(self):
        manager = HotkeyManager.__new__(HotkeyManager)
        manager.config = ConfigStub({"hotkeys": {"toggle": "ctrl+\\"}})
        manager._id_to_action = {}

        class FakeUser32:
            def __init__(self):
                self.register_calls = []

            @staticmethod
            def UnregisterHotKey(hwnd, hotkey_id):
                return 1

            def RegisterHotKey(self, hwnd, hotkey_id, mods, vk):
                self.register_calls.append((hotkey_id, mods, vk))
                return 1

            @staticmethod
            def GetMessageW(msg, hwnd, min_filter, max_filter):
                return 0

            @staticmethod
            def TranslateMessage(msg):
                return 1

            @staticmethod
            def DispatchMessageW(msg):
                return 1

        fake_user32 = FakeUser32()
        fake_kernel32 = Mock(GetCurrentThreadId=Mock(return_value=321))
        thread = NativeHotkeyThread(manager)

        with patch("ctypes.windll", Mock(user32=fake_user32, kernel32=fake_kernel32), create=True):
            thread.run()

        self.assertEqual(len(fake_user32.register_calls), 1)
        _, mods, vk = fake_user32.register_calls[0]
        self.assertEqual(vk, 0xDC)
        self.assertTrue(mods & NativeHotkeyThread.MOD_NOREPEAT)


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


class WindowUtilsTests(unittest.TestCase):
    def test_hide_from_taskbar_marks_window_once_on_windows(self):
        class FakeWindow:
            def winId(self):
                return 101

        class FakeUser32:
            def __init__(self):
                self.set_window_long_calls = 0
                self.set_window_pos_calls = 0

            @staticmethod
            def GetAncestor(hwnd, flag):
                return hwnd

            @staticmethod
            def GetWindowLongW(hwnd, index):
                return 0x00040000

            def SetWindowLongW(self, hwnd, index, style):
                self.set_window_long_calls += 1
                return style

            def SetWindowPos(self, hwnd, insert_after, x, y, cx, cy, flags):
                self.set_window_pos_calls += 1
                return 1

        fake_user32 = FakeUser32()
        fake_window = FakeWindow()

        with patch("utils.platform_utils.PlatformInfo.IS_WINDOWS", True), patch(
            "ctypes.windll", Mock(user32=fake_user32), create=True
        ):
            self.assertTrue(WindowUtils.hide_from_taskbar(fake_window))
            self.assertTrue(fake_window._openassist_taskbar_hidden)
            self.assertEqual(fake_user32.set_window_long_calls, 1)
            self.assertEqual(fake_user32.set_window_pos_calls, 1)

            self.assertTrue(WindowUtils.hide_from_taskbar(fake_window))
            self.assertEqual(fake_user32.set_window_long_calls, 1)
            self.assertEqual(fake_user32.set_window_pos_calls, 1)


if __name__ == "__main__":
    unittest.main()
