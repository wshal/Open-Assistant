import asyncio
import shutil
import sys
import unittest
import os
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
import json
from itertools import count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.history import ResponseHistory
from ai.engine import AIEngine
from capture.audio import AudioCapture
from capture.screen import ScreenCapture
from core.config import Config
from core.hotkeys import HotkeyManager, NativeHotkeyThread
from core.state import AppState
from utils.platform_utils import WindowUtils, ProcessUtils
from ai.providers.ollama_provider import OllamaProvider
from ai.prompts import PromptBuilder
from stealth.anti_detect import StealthManager
from ui.settings_view import ProviderTestWorker, SettingsView
from PyQt6.QtWidgets import QApplication


class ConfigStub:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.saved = 0

    def get(self, path, default=None):
        return self.settings.get(path, default)

    def set(self, path, value):
        self.settings[path] = value

    def save(self):
        self.saved += 1

    def get_api_key(self, provider):
        return self.settings.get(f"api_key.{provider}", "")

    def set_api_key(self, provider, value):
        self.settings[f"api_key.{provider}"] = value

    def validate_key_for_ui(self, provider, key):
        if provider == "ollama":
            return True, "OK"
        if key and len(key) >= 10:
            return True, "OK"
        return False, "Invalid Key"


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

    def test_webrtc_vad_helper_is_safe_and_frames_correctly(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.sample_rate": 16000,
                    "capture.audio.vad.frame_ms": 20,
                }
            )
        )

        calls = []

        class _FakeVAD:
            def is_speech(self, frame_bytes, sample_rate):
                calls.append((len(frame_bytes), sample_rate))
                return True

        audio._vad = _FakeVAD()
        block = np.zeros((audio.block_size,), dtype=np.float32)
        self.assertTrue(audio._webrtc_vad_has_speech(block))
        self.assertTrue(calls, "expected at least one VAD frame call")
        self.assertEqual(calls[0][1], 16000)
        self.assertEqual(calls[0][0], int(16000 * 0.02) * 2)

    def test_detect_speech_falls_back_to_webrtc(self):
        audio = AudioCapture(ConfigStub({}))
        audio._vad = Mock()
        audio._webrtc_vad_has_speech = lambda block: True

        has_speech, backend = audio._detect_speech(np.zeros((audio.block_size,), dtype=np.float32), 0.0)

        self.assertTrue(has_speech)
        self.assertEqual(backend, "webrtc")

    def test_detect_speech_falls_back_to_rms_when_webrtc_unavailable(self):
        audio = AudioCapture(ConfigStub({}))
        audio._vad = None

        has_speech, backend = audio._detect_speech(
            np.ones((audio.block_size,), dtype=np.float32) * 0.01, 0.01
        )

        self.assertTrue(has_speech)
        self.assertEqual(backend, "rms")

    def test_required_silence_blocks_shortens_short_utterance(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.vad.short_utterance_max_s": 2.8,
                    "capture.audio.vad.short_silence_ms": 500,
                }
            )
        )

        with patch("capture.audio.time.time", return_value=2.0):
            short_blocks = audio._required_silence_blocks(0.0)
        with patch("capture.audio.time.time", return_value=4.5):
            long_blocks = audio._required_silence_blocks(0.0)

        self.assertEqual(short_blocks, audio._short_silence_blocks)
        self.assertEqual(long_blocks, audio.silence_blocks)

    def test_required_silence_blocks_shortens_after_mid_utterance_slice(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.vad.short_silence_ms": 500,
                    "capture.audio.vad.post_chunk_silence_ms": 500,
                    "capture.audio.vad.short_utterance_max_s": 2.8,
                }
            )
        )

        # While the new chunk is still short (< short_utterance_max_s), the
        # aggressive post-chunk tail should apply.
        with patch("capture.audio.time.time", return_value=1.0):
            blocks_short = audio._required_silence_blocks(
                0.5,  # speech_started_at → elapsed = 0.5s (short)
                had_mid_utterance_slice=True,
            )
        self.assertEqual(blocks_short, audio._post_chunk_silence_blocks)
        # Document: 500ms config floors to 400ms effective given 200ms blocks
        self.assertEqual(blocks_short * audio.block_ms, 400)

        # Once the new chunk has run long (> short_utterance_max_s), the window
        # must revert to the full silence_blocks — NOT stay on the short tail.
        # This is the time-gate fix that prevents false early cuts on multi-clause prompts.
        with patch("capture.audio.time.time", return_value=5.0):
            blocks_long = audio._required_silence_blocks(
                0.5,  # speech_started_at → elapsed = 4.5s (long)
                had_mid_utterance_slice=True,
            )
        self.assertEqual(
            blocks_long,
            audio.silence_blocks,
            "Post-chunk window must revert to silence_blocks for long new chunks "
            "to prevent false early cuts on multi-clause prompts",
        )

    def test_emit_accumulated_records_vad_metrics(self):
        audio = AudioCapture(ConfigStub())
        audio._session_transcript_parts = ["what is react"]

        with patch("capture.audio.time.time", side_effect=[2.0]):
            audio._emit_accumulated(
                speech_started_at=1.0,
                provider="local",
                transcribe_started_at=1.5,
                vad_meta={"vad_backend": "webrtc", "end_silence_ms": 500},
            )

        metrics = audio.get_last_transcription_metrics()
        self.assertEqual(metrics["vad_backend"], "webrtc")
        self.assertEqual(metrics["end_silence_ms"], 500)
        self.assertEqual(metrics["audio_duration_ms"], 500.0)
        self.assertFalse(metrics["chunk_aware_eos"])

    def test_local_transcribe_uses_benchmarked_decoder_settings(self):
        audio = AudioCapture(ConfigStub())
        calls = {}

        class _FakeSegment:
            text = "what is react"

        class _FakeModel:
            def transcribe(self, samples, **kwargs):
                calls["samples_len"] = len(samples)
                calls["kwargs"] = kwargs
                return ([_FakeSegment()], None)

        emitted = []
        audio.model = _FakeModel()
        audio._model_loaded = True
        audio.transcription_ready.connect(emitted.append)

        buffer = [np.ones((audio.block_size, 1), dtype=np.float32) * 0.01 for _ in range(5)]
        audio._transcribe_local(buffer, speech_started_at=1.0, is_final=True)

        self.assertEqual(emitted, ["what is react"])
        self.assertEqual(calls["kwargs"]["beam_size"], 3)  # beam=3 is the production default after sweep
        self.assertFalse(calls["kwargs"]["condition_on_previous_text"])
        self.assertFalse(calls["kwargs"]["vad_filter"])
        # initial_prompt at call-time equals the base prompt (context injector updates
        # _whisper_initial_prompt only AFTER emit, for the next utterance).
        self.assertEqual(
            calls["kwargs"]["initial_prompt"],
            audio._whisper_base_prompt,
        )

    def test_interim_transcribe_uses_same_decoder_bias_without_vad_filter(self):
        audio = AudioCapture(ConfigStub())
        calls = {}

        class _FakeSegment:
            text = "partial react question"

        class _FakeModel:
            def transcribe(self, samples, **kwargs):
                calls["samples_len"] = len(samples)
                calls["kwargs"] = kwargs
                return ([_FakeSegment()], None)

        emitted = []
        audio.model = _FakeModel()
        audio._model_loaded = True
        audio.interim_transcription_ready.connect(emitted.append)
        audio._interim_epoch = 3

        buffer = [np.ones((audio.block_size, 1), dtype=np.float32) * 0.01 for _ in range(4)]
        audio._transcribe_interim(buffer, speech_started_at=1.0, epoch=3)

        self.assertEqual(emitted, ["partial react question"])
        self.assertEqual(calls["kwargs"]["beam_size"], 3)  # beam=3 is the production default after sweep
        self.assertFalse(calls["kwargs"]["condition_on_previous_text"])
        self.assertFalse(calls["kwargs"]["vad_filter"])
        self.assertEqual(
            calls["kwargs"]["initial_prompt"],
            audio._whisper_initial_prompt,
        )


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
        # P0.4 FIX: Parallel path now emits chunks for streaming UI feedback.
        # The full response is streamed word-by-word before response_complete fires.
        self.assertTrue(len(chunks) > 0, "Expected chunks to be emitted in parallel mode")
        self.assertEqual("".join(chunks).strip(), "parallel answer")
        self.assertEqual(completed, ["parallel answer"])
        self.assertEqual(history.entries[-1]["provider"], "parallel")

    def test_clean_final_response_strips_audio_meta_preface(self):
        engine = AIEngine(ConfigStub({"ai.mode": "general"}), HistoryStub(), rag=None)

        cleaned = engine._clean_final_response(
            "Based on the audio context, I'm assuming the ASR error was something like "
            "\"React\" instead of \"ReactJS.\" React is a JavaScript library for building UIs."
        )

        self.assertEqual(
            cleaned,
            "React is a JavaScript library for building UIs.",
        )

    def test_effective_speech_query_prefers_refined_question(self):
        engine = AIEngine(ConfigStub({"ai.mode": "general"}), HistoryStub(), rag=None)

        effective = engine._effective_speech_query(
            "What are the ways to post performance in React?",
            "What are the ways to boost performance in React?",
        )

        self.assertEqual(
            effective,
            "What are the ways to boost performance in React?",
        )


class PromptBuilderSpeechTests(unittest.TestCase):
    def test_general_knowledge_classifier_handles_noisy_polite_speech(self):
        self.assertTrue(
            PromptBuilder()._is_general_knowledge_query(
                "Couldyou care to explain what is React?"
            )
        )

    def test_speech_general_knowledge_prompt_avoids_audio_meta_instructions(self):
        prompt = PromptBuilder().user(
            query="What is React?",
            audio="What is React?",
            origin="speech",
            mode="general",
            nexus={"active_window": "Editor", "history_depth_secs": 60},
        )

        self.assertNotIn("(Origin: Audio. Fix ASR errors.)", prompt)
        self.assertNotIn("[AUDIO]", prompt)
        self.assertNotIn("[ENVIRONMENT]", prompt)
        self.assertIn("silently correct them before answering", prompt)
        self.assertIn("Do not mention audio context, ASR", prompt)

    def test_speech_live_context_prompt_disallows_direct_screen_access_claims(self):
        prompt = PromptBuilder().user(
            query="How can you see the screen?",
            audio="How can you see the screen?",
            origin="speech",
            mode="general",
            nexus={"active_window": "Editor", "history_depth_secs": 60},
            screen="visible code",
        )

        self.assertIn("captured screen/OCR context", prompt)

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
        self.assertEqual(selected[0]["preferred"][0], "groq")


class ResponseHistoryReadOnlyTests(unittest.TestCase):
    def test_read_session_does_not_mutate_active_session(self):
        history_dir = Path("openassist") / "data" / "test_history_runtime"
        if history_dir.exists():
            shutil.rmtree(history_dir, ignore_errors=True)

        with patch(
            "ai.history.time.time",
            side_effect=count(1000),
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


class OllamaProviderTests(unittest.TestCase):
    def test_blank_model_uses_first_available_local_model(self):
        provider = OllamaProvider(ConfigStub({}))
        provider._available_models = ["qwen2.5:7b", "llama3.2:latest"]

        picked = provider._pick_available_model("")

        self.assertEqual(picked, "llama3.2:latest")

    def test_check_availability_resolves_non_empty_model_when_config_missing(self):
        provider = OllamaProvider(ConfigStub({}))

        payload = {"models": [{"name": "llama3.2:latest"}]}

        class FakeResponse:
            status = 200

            async def json(self):
                return payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with patch("aiohttp.ClientSession", return_value=FakeSession()):
            ok = asyncio.run(provider.check_availability())

        self.assertTrue(ok)
        self.assertTrue(provider.is_ready)
        self.assertEqual(provider.get_model(), "llama3.2:latest")


class PromptBuilderTests(unittest.TestCase):
    def test_manual_general_knowledge_query_suppresses_unrelated_live_context(self):
        builder = PromptBuilder()

        prompt = builder.user(
            query="what is react ?",
            screen="def main():\n    print('python app')",
            audio="we are debugging a python bug",
            origin="manual",
            nexus={"active_window": "VS Code", "history_depth_secs": 60},
        )

        self.assertNotIn("[SCREEN]", prompt)
        self.assertNotIn("[AUDIO]", prompt)
        self.assertNotIn("[ENVIRONMENT]", prompt)
        self.assertIn("general knowledge", prompt)

    def test_manual_contextual_query_keeps_live_context(self):
        builder = PromptBuilder()

        prompt = builder.user(
            query="what does this Python function do?",
            screen="def add(a, b):\n    return a + b",
            audio="",
            origin="manual",
            nexus={"active_window": "VS Code", "history_depth_secs": 60},
        )

        self.assertIn("[SCREEN]", prompt)
        self.assertIn("[ENVIRONMENT]", prompt)

    def test_interview_mode_prioritizes_audio_before_screen(self):
        builder = PromptBuilder()

        prompt = builder.user(
            query="How should I answer this?",
            screen="Tell me about a time you handled conflict.",
            audio="The interviewer just asked about conflict resolution.",
            mode="interview",
            origin="manual",
            nexus={"active_window": "Zoom", "history_depth_secs": 60},
        )

        self.assertLess(prompt.index("[AUDIO]"), prompt.index("[SCREEN]"))


class ModeAwareProviderPreferenceTests(unittest.TestCase):
    def test_general_mode_prefers_fast_general_models(self):
        engine = AIEngine(ConfigStub({"ai.mode": "general"}), HistoryStub(), rag=None)

        preferred = engine._preferred_providers_for_complexity("moderate", "general")

        self.assertEqual(preferred[:3], ["groq", "cerebras", "together"])

    def test_meeting_mode_prefers_fast_general_models(self):
        engine = AIEngine(ConfigStub({"ai.mode": "meeting"}), HistoryStub(), rag=None)

        preferred = engine._preferred_providers_for_complexity("simple", "meeting")

        self.assertEqual(preferred[:3], ["groq", "cerebras", "together"])

    def test_interview_mode_keeps_audio_first_but_quality_for_harder_queries(self):
        engine = AIEngine(ConfigStub({"ai.mode": "interview"}), HistoryStub(), rag=None)

        preferred = engine._preferred_providers_for_complexity("complex", "interview")

        self.assertEqual(preferred[:2], ["gemini", "groq"])


class ProviderTestWorkerTests(unittest.TestCase):
    def test_ollama_test_does_not_require_api_key(self):
        config = ConfigStub({})
        worker = ProviderTestWorker("ollama", config)
        results = []
        worker.result_ready.connect(
            lambda pid, ok, msg, details: results.append((pid, ok, msg, details))
        )

        payload = {"models": [{"name": "llama3.2:latest"}]}

        class FakeResponse:
            status = 200

            async def json(self):
                return payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with patch("aiohttp.ClientSession", return_value=FakeSession()):
            worker.run()

        self.assertEqual(results[0][0], "ollama")
        self.assertTrue(results[0][1])
        self.assertIn("Connected", results[0][2])
        self.assertEqual(results[0][3]["models"], ["llama3.2:latest"])

    def test_cloud_provider_test_rejects_invalid_key_before_request(self):
        config = ConfigStub({})
        worker = ProviderTestWorker("groq", config)
        results = []
        worker.result_ready.connect(
            lambda pid, ok, msg, details: results.append((pid, ok, msg, details))
        )

        worker.run()

        self.assertEqual(results, [("groq", False, "Invalid Key", None)])

    def test_recommended_ollama_model_prefers_coder_for_coding_mode(self):
        models = ["llama3.2:latest", "qwen2.5-coder:7b", "mistral:7b"]

        picked = SettingsView._recommended_ollama_model(models, "coding")

        self.assertEqual(picked, "qwen2.5-coder:7b")


class SettingsViewSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_display_and_system_widgets_resync_from_config_on_reopen(self):
        config = ConfigStub(
            {
                "app.opacity": 0.91,
                "stealth.low_opacity": 0.73,
                "app.gaze_fade.enabled": True,
                "app.gaze_fade.margin": 40,
                "app.gaze_fade.target_opacity": 0.20,
                "app.focus_on_show": False,
                "capture.audio.mode": "system",
                "ai.mode": "general",
            }
        )

        settings = SettingsView(config, app=None)
        self.addCleanup(settings.deleteLater)

        config.set("app.opacity", 0.88)
        config.set("stealth.low_opacity", 0.79)
        config.set("app.gaze_fade.enabled", False)
        config.set("app.gaze_fade.margin", 80)
        config.set("app.gaze_fade.target_opacity", 0.25)
        config.set("app.focus_on_show", True)

        settings._sync_ui_from_config()

        self.assertEqual(settings.hud_opacity_slider.value(), 88)
        self.assertEqual(settings.hud_opacity_value.text(), "88%")
        self.assertEqual(settings.stealth_opacity_slider.value(), 79)
        self.assertEqual(settings.stealth_opacity_value.text(), "79%")
        self.assertFalse(settings.chk_gaze.isChecked())
        self.assertEqual(settings.margin_slider.currentIndex(), 5)
        self.assertEqual(settings.opacity_slider.currentIndex(), 4)
        self.assertTrue(settings.chk_focus_on_show.isChecked())

    def test_system_tab_is_last_and_removed_tabs_do_not_reappear(self):
        settings = SettingsView(ConfigStub({"capture.audio.mode": "system"}), app=None)
        self.addCleanup(settings.deleteLater)

        labels = [settings.tabs.tabText(i) for i in range(settings.tabs.count())]

        self.assertNotIn("GHOST", labels)
        self.assertEqual(labels[-1], "SYSTEM")

    def test_system_tab_settings_are_saved_after_move(self):
        config = ConfigStub(
            {
                "capture.audio.mode": "system",
                "ai.mode": "general",
                "app.focus_on_show": False,
            }
        )
        settings = SettingsView(config, app=None)
        self.addCleanup(settings.deleteLater)

        settings.chk_focus_on_show.setChecked(True)

        settings._save_all()

        self.assertTrue(config.get("app.focus_on_show"))
        self.assertEqual(config.saved, 1)

    def test_system_tab_shows_stealth_status_from_app_manager(self):
        app = SimpleNamespace(
            stealth=SimpleNamespace(
                get_status=lambda: {
                    "state": "fallback",
                    "platform": "strong",
                    "message": "Monitor-only capture fallback active",
                    "last_error": 87,
                    "last_affinity": 0x00000001,
                }
            )
        )
        settings = SettingsView(
            ConfigStub({"capture.audio.mode": "system", "ai.mode": "general"}),
            app=app,
        )
        self.addCleanup(settings.deleteLater)

        settings._sync_stealth_status()

        self.assertIn("Fallback protection on Strong", settings._stealth_status_lbl.text())
        self.assertIn("error 87", settings._stealth_status_detail.text())


class WindowUtilsTests(unittest.TestCase):
    def test_ensure_topmost_refreshes_hwnd_on_windows(self):
        class FakeWindow:
            def winId(self):
                return 101

        class FakeUser32:
            def __init__(self):
                self.calls = []

            def SetWindowPos(self, hwnd, insert_after, x, y, cx, cy, flags):
                self.calls.append((hwnd, insert_after, flags))
                return 1

        fake_user32 = FakeUser32()
        fake_window = FakeWindow()

        with patch("utils.platform_utils.PlatformInfo.IS_WINDOWS", True), patch(
            "ctypes.windll", Mock(user32=fake_user32), create=True
        ):
            self.assertTrue(WindowUtils.ensure_topmost(fake_window))
            self.assertEqual(len(fake_user32.calls), 1)
            hwnd, insert_after, _ = fake_user32.calls[0]
            self.assertEqual(hwnd, 101)
            self.assertEqual(insert_after, -1)

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


class ProcessUtilsTests(unittest.TestCase):
    def test_linux_screen_share_detection_uses_process_list(self):
        fake_proc = SimpleNamespace(
            info={"name": "obs", "cmdline": ["obs"]}
        )

        with patch("utils.platform_utils.PlatformInfo.IS_WINDOWS", False), patch(
            "utils.platform_utils.PlatformInfo.IS_MAC", False
        ), patch("utils.platform_utils.PlatformInfo.IS_LINUX", True), patch(
            "psutil.process_iter", return_value=[fake_proc]
        ):
            self.assertTrue(ProcessUtils.is_screen_sharing_active())

    def test_linux_browser_without_share_markers_is_ignored(self):
        fake_proc = SimpleNamespace(
            info={"name": "google-chrome", "cmdline": ["chrome", "https://example.com"]}
        )

        with patch("utils.platform_utils.PlatformInfo.IS_WINDOWS", False), patch(
            "utils.platform_utils.PlatformInfo.IS_MAC", False
        ), patch("utils.platform_utils.PlatformInfo.IS_LINUX", True), patch(
            "psutil.process_iter", return_value=[fake_proc]
        ), patch("subprocess.run", return_value=SimpleNamespace(returncode=1)):
            self.assertFalse(ProcessUtils.is_screen_sharing_active())


class StealthManagerTests(unittest.TestCase):
    def test_windows_prefers_exclude_from_capture(self):
        class FakeWindow:
            def winId(self):
                return 101

        class FakeUser32:
            def __init__(self):
                self.calls = []

            @staticmethod
            def GetAncestor(hwnd, flag):
                return hwnd

            def SetWindowDisplayAffinity(self, hwnd, affinity):
                self.calls.append((hwnd, affinity))
                return 1 if affinity == 0x00000011 else 0

        fake_user32 = FakeUser32()
        manager = StealthManager(ConfigStub({"stealth.enabled": True}))

        with patch("ctypes.windll", Mock(user32=fake_user32), create=True):
            manager.apply_to_window(FakeWindow(), True)

        self.assertEqual(fake_user32.calls, [(101, 0x00000011)])
        self.assertEqual(manager._last_affinity_state[101], (True, 0x00000011))
        self.assertEqual(manager.get_status()["state"], "protected")

    def test_windows_falls_back_to_monitor_affinity(self):
        class FakeWindow:
            def winId(self):
                return 101

        class FakeUser32:
            def __init__(self):
                self.calls = []

            @staticmethod
            def GetAncestor(hwnd, flag):
                return hwnd

            def SetWindowDisplayAffinity(self, hwnd, affinity):
                self.calls.append((hwnd, affinity))
                return 1 if affinity == 0x00000001 else 0

        fake_user32 = FakeUser32()
        fake_kernel32 = Mock(GetLastError=Mock(return_value=87))
        manager = StealthManager(ConfigStub({"stealth.enabled": True}))

        with patch(
            "ctypes.windll",
            Mock(user32=fake_user32, kernel32=fake_kernel32),
            create=True,
        ):
            manager.apply_to_window(FakeWindow(), True)

        self.assertEqual(
            fake_user32.calls,
            [(101, 0x00000011), (101, 0x00000001)],
        )
        self.assertEqual(manager._last_affinity_state[101], (True, 0x00000001))
        self.assertEqual(manager.get_status()["state"], "fallback")

    def test_windows_can_clear_affinity(self):
        class FakeWindow:
            def winId(self):
                return 101

        class FakeUser32:
            @staticmethod
            def GetAncestor(hwnd, flag):
                return hwnd

            def SetWindowDisplayAffinity(self, hwnd, affinity):
                return 1 if affinity == 0x00000000 else 0

        manager = StealthManager(ConfigStub({"stealth.enabled": True}))

        with patch("ctypes.windll", Mock(user32=FakeUser32()), create=True):
            manager.apply_to_window(FakeWindow(), False)

        self.assertEqual(manager._last_affinity_state[101], (False, 0x00000000))
        self.assertEqual(manager.get_status()["state"], "unprotected")

    def test_linux_status_reports_limited_protection(self):
        manager = StealthManager(ConfigStub({"stealth.enabled": True}))

        with patch("sys.platform", "linux"):
            manager.apply_to_window(Mock(), True)

        self.assertEqual(manager.get_status()["state"], "limited")
        self.assertEqual(manager.get_status()["platform"], "limited")


if __name__ == "__main__":
    unittest.main()
