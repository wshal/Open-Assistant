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
from ai.live_session import LiveSessionManager
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
    default_tier = "balanced"

    def check_rate(self):
        return True

    def has_model(self, tier=None):
        return True


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

    def test_restart_does_not_skip_when_pipeline_is_unhealthy(self):
        audio = AudioCapture(
            ConfigStub({"capture.audio.mode": "system", "capture.audio.enabled": True})
        )
        audio._running = True
        audio.last_mode = "system"
        audio._capture_thread = Mock(is_alive=lambda: False)
        audio._process_thread = Mock(is_alive=lambda: False)
        close_calls = []
        start_calls = []
        audio._close_streams = lambda: close_calls.append("close")
        audio._drain_queue = lambda: None

        with patch("capture.audio.time.sleep", return_value=None), patch.object(
            audio, "start", side_effect=lambda: start_calls.append("start")
        ):
            audio.restart()

        self.assertEqual(close_calls, ["close"])
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
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "mic"}))
        audio._vad = Mock()
        audio._webrtc_vad_has_speech = lambda block: True

        has_speech, backend = audio._detect_speech(np.zeros((audio.block_size,), dtype=np.float32), 0.0)

        self.assertTrue(has_speech)
        self.assertEqual(backend, "webrtc")

    def test_detect_speech_falls_back_to_rms_when_webrtc_unavailable(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "mic"}))
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
                    "capture.audio.mode": "mic",
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

    def test_submit_final_transcription_skips_superseded_short_system_job(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio.capture_mode = "system"
        jobs = []
        transcribed = []
        audio._submit_transcription_job = lambda fn, *args: jobs.append((fn, args)) or True
        audio._transcribe = lambda buffer, speech_started_at=None, is_final=True, vad_meta=None: transcribed.append(
            (list(buffer), (vad_meta or {}).get("utterance_id"))
        )

        audio._submit_final_transcription(
            [np.zeros((audio.block_size,), dtype=np.float32)],
            vad_meta={"utterance_id": "utt_1", "speech_duration": 0.8, "chunk_aware_eos": False, "peak_rms": 0.01},
        )
        audio._submit_final_transcription(
            [np.ones((audio.block_size,), dtype=np.float32)],
            vad_meta={"utterance_id": "utt_2", "speech_duration": 0.9, "chunk_aware_eos": False, "peak_rms": 0.02},
        )

        jobs[0][0](*jobs[0][1])
        jobs[1][0](*jobs[1][1])

        self.assertEqual(len(transcribed), 1)
        self.assertEqual(transcribed[0][1], "utt_2")

    def test_submit_final_transcription_skips_when_standard_asr_is_suspended(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._standard_transcription_suspended = True
        submit_calls = []
        audio._submit_transcription_job = lambda *args, **kwargs: submit_calls.append(True) or True

        queued = audio._submit_final_transcription(
            [np.zeros((audio.block_size,), dtype=np.float32)],
            vad_meta={"utterance_id": "utt_live"},
        )

        self.assertFalse(queued)
        self.assertEqual(submit_calls, [])

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

        with patch("capture.audio.time.time", side_effect=[2.0, 2.0, 2.0, 2.0]):
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
        self.assertEqual(metrics["transcribe_only_ms"], 500.0)
        self.assertFalse(metrics["chunk_aware_eos"])

    def test_emit_accumulated_preserves_chunked_utterance_timing_on_short_tail_flush(self):
        audio = AudioCapture(ConfigStub())
        audio._session_transcript_parts = ["what is usememo"]

        with patch("capture.audio.time.time", side_effect=[15.0, 15.0, 15.0, 15.0]):
            audio._emit_accumulated(
                speech_started_at=13.5,
                provider="local",
                transcribe_started_at=None,
                vad_meta={
                    "vad_backend": "webrtc",
                    "end_silence_ms": 400,
                    "chunk_aware_eos": True,
                    "utterance_started_at": 10.0,
                    "speech_finalized_at": 14.0,
                },
            )

        metrics = audio.get_last_transcription_metrics()
        self.assertEqual(metrics["speech_started_at"], 10.0)
        self.assertEqual(metrics["speech_finalized_at"], 14.0)
        self.assertEqual(metrics["audio_duration_ms"], 4000.0)
        self.assertEqual(metrics["transcribe_only_ms"], 1000.0)
        self.assertEqual(metrics["speech_to_transcript_ms"], 5000.0)
        self.assertTrue(metrics["chunk_aware_eos"])

    def test_system_mode_noise_gate_rejects_low_rms_false_positives_while_speaking(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.vad.system_noise_gate_enabled": True,
                    "capture.audio.vad.system_continue_floor_multiplier": 1.3,
                    "capture.audio.vad.system_continue_min_rms": 0.002,
                }
            )
        )
        audio._dynamic_rms_floor = 0.001

        self.assertFalse(audio._passes_speech_rms_gate(0.001, is_speaking=True))
        self.assertTrue(audio._passes_speech_rms_gate(0.008, is_speaking=True))

    def test_system_mode_requires_confirmed_speech_onset_blocks(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.vad.system_noise_gate_enabled": True,
                    "capture.audio.vad.system_start_confirm_blocks": 3,
                }
            )
        )
        self.assertEqual(audio._required_start_confirm_blocks(), 3)

        audio.capture_mode = "mic"
        self.assertEqual(audio._required_start_confirm_blocks(), 1)

    def test_ultra_short_system_utterance_uses_longer_eos_tail(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.vad.ultra_short_utterance_max_s": 0.35,
                    "capture.audio.vad.ultra_short_silence_ms": 900,
                    "capture.audio.vad.short_silence_ms": 500,
                }
            )
        )

        with patch("capture.audio.time.time", return_value=10.2):
            blocks = audio._required_silence_blocks(10.0, had_mid_utterance_slice=False)

        self.assertEqual(blocks, audio._ultra_short_silence_blocks)

    def test_short_phrase_utterance_uses_phrase_eos_tail(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.vad.phrase_utterance_max_s": 0.9,
                    "capture.audio.vad.phrase_silence_ms": 1200,
                    "capture.audio.vad.short_silence_ms": 500,
                }
            )
        )

        with patch("capture.audio.time.time", return_value=10.6):
            blocks = audio._required_silence_blocks(10.0, had_mid_utterance_slice=False)

        self.assertEqual(blocks, audio._phrase_silence_blocks)

    def test_queue_pressure_drops_weak_short_system_finals(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.vad.system_queue_pressure_drop_enabled": True,
                    "capture.audio.vad.system_queue_pressure_max_pending": 3,
                    "capture.audio.vad.system_queue_pressure_max_speech_ms": 600,
                    "capture.audio.vad.system_queue_pressure_max_voiced_blocks": 3,
                    "capture.audio.vad.system_queue_pressure_max_peak_rms": 0.02,
                }
            )
        )
        audio._final_decode_pending = 3

        self.assertTrue(
            audio._should_drop_queued_final_under_pressure(
                speech_duration=0.18,
                voiced_blocks=2,
                peak_rms=0.006,
                vad_meta={"chunk_aware_eos": False},
                buffer_len=8,
            )
        )
        self.assertFalse(
            audio._should_drop_queued_final_under_pressure(
                speech_duration=1.2,
                voiced_blocks=10,
                peak_rms=0.12,
                vad_meta={"chunk_aware_eos": False},
                buffer_len=40,
            )
        )

    def test_burst_followup_guard_drops_short_weak_system_followup(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.vad.system_followup_guard_enabled": True,
                    "capture.audio.vad.system_followup_guard_window_ms": 1800,
                    "capture.audio.vad.system_followup_guard_max_speech_ms": 700,
                    "capture.audio.vad.system_followup_guard_max_peak_rms": 0.03,
                    "capture.audio.vad.system_followup_guard_max_voiced_blocks": 18,
                }
            )
        )
        audio._final_decode_pending = 1

        with patch("capture.audio.time.time", return_value=11.0):
            audio._last_final_submit_at = 10.0
            self.assertTrue(
                audio._should_drop_burst_followup_system_utterance(
                    speech_duration=0.41,
                    voiced_blocks=17,
                    peak_rms=0.008,
                    vad_meta={"chunk_aware_eos": False},
                )
            )

    def test_burst_followup_guard_keeps_real_later_or_stronger_followup(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.vad.system_followup_guard_enabled": True,
                    "capture.audio.vad.system_followup_guard_window_ms": 1800,
                    "capture.audio.vad.system_followup_guard_max_speech_ms": 700,
                    "capture.audio.vad.system_followup_guard_max_peak_rms": 0.03,
                    "capture.audio.vad.system_followup_guard_max_voiced_blocks": 18,
                }
            )
        )
        audio._final_decode_pending = 1

        with patch("capture.audio.time.time", return_value=13.0):
            audio._last_final_submit_at = 10.0
            self.assertFalse(
                audio._should_drop_burst_followup_system_utterance(
                    speech_duration=0.41,
                    voiced_blocks=17,
                    peak_rms=0.008,
                    vad_meta={"chunk_aware_eos": False},
                )
            )

        with patch("capture.audio.time.time", return_value=11.0):
            audio._last_final_submit_at = 10.0
            self.assertFalse(
                audio._should_drop_burst_followup_system_utterance(
                    speech_duration=1.1,
                    voiced_blocks=24,
                    peak_rms=0.09,
                    vad_meta={"chunk_aware_eos": False},
                )
            )

    def test_set_trace_context_rearms_ambient_calibration_for_new_session(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._ambient_calib_remaining = 0
        audio._ambient_rms_samples = [0.1, 0.2]
        audio._dynamic_rms_floor = 0.05
        audio._session_transcript_parts = ["stale"]
        audio.q.put(np.ones((audio.block_size,), dtype=np.float32))

        audio.set_trace_context("session_test", 123.0)

        self.assertEqual(audio._ambient_calib_remaining, audio._ambient_calib_blocks)
        self.assertEqual(audio._ambient_rms_samples, [])
        self.assertEqual(audio._dynamic_rms_floor, 0.0)
        self.assertEqual(audio._session_transcript_parts, [])
        self.assertTrue(audio.q.empty())

    def test_enqueue_audio_frames_discards_pre_session_audio(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._running = True
        audio._paused = False
        audio._state.is_capturing = False
        audio._capture_chunk_buffer = np.ones((audio.block_size // 2, 1), dtype=np.float32)

        audio._enqueue_audio_frames(
            np.ones((audio.block_size, 1), dtype=np.float32) * 0.02,
            "system-audio",
        )

        self.assertTrue(audio.q.empty())
        self.assertEqual(audio._capture_chunk_buffer.shape[0], 0)

    def test_enqueue_audio_frames_rechunks_callback_audio_to_block_size(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._running = True
        audio._paused = False
        audio._state.is_capturing = True

        chunk = np.ones((412, 1), dtype=np.float32) * 0.02
        for _ in range(8):
            audio._enqueue_audio_frames(chunk, "system-audio")

        queued = []
        while not audio.q.empty():
            queued.append(audio.q.get_nowait())

        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].shape[0], audio.block_size)
        self.assertEqual(audio._capture_chunk_buffer.shape[0], (412 * 8) - audio.block_size)

    def test_ambient_calibration_skips_initial_speech_blocks(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._ambient_calib_remaining = 1
        audio._detect_speech = lambda data, rms: (True, "webrtc")
        audio._emit_live_audio_chunk = lambda block: None
        audio._required_start_confirm_blocks = lambda: 1
        audio._max_utterance_exceeded = lambda started_at: False
        block = np.ones((audio.block_size, 1), dtype=np.float32) * 0.03

        class _StopQueue:
            def __init__(self, owner, first_block):
                self.owner = owner
                self.first_block = first_block
                self.calls = 0

            def get(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    return self.first_block
                self.owner._running = False
                raise queue.Empty()

        audio.q = _StopQueue(audio, block)
        audio._running = True

        audio._process_loop()

        self.assertEqual(audio._ambient_calib_remaining, 1)
        self.assertEqual(audio._ambient_rms_samples, [])
        self.assertEqual(audio._dynamic_rms_floor, 0.0)

    def test_stale_final_jobs_are_skipped_after_new_session_generation(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "mic"}))
        submitted = []

        class _FakePool:
            def submit(self, fn, *args):
                submitted.append((fn, args))

        audio._transcribe_pool = _FakePool()
        calls = []
        audio._transcribe = lambda *args, **kwargs: calls.append((args, kwargs))

        audio._submit_final_transcription(
            ["old"],
            1.0,
            {"utterance_id": "utt_old", "vad_backend": "webrtc"},
        )
        audio.set_trace_context("session_test", 10.0)
        audio._submit_final_transcription(
            ["new"],
            2.0,
            {"utterance_id": "utt_new", "vad_backend": "webrtc"},
        )

        self.assertEqual(len(submitted), 2)

        submitted[0][0](*submitted[0][1])
        self.assertEqual(calls, [])

        submitted[1][0](*submitted[1][1])
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[0], ["new"])
        self.assertEqual(kwargs["vad_meta"]["utterance_id"], "utt_new")

    def test_local_transcribe_uses_benchmarked_decoder_settings(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "mic"}))
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
        self.assertEqual(calls["kwargs"]["beam_size"], 1)  # interim path uses lower beam for lower latency
        self.assertFalse(calls["kwargs"]["condition_on_previous_text"])
        self.assertFalse(calls["kwargs"]["vad_filter"])
        self.assertEqual(
            calls["kwargs"]["initial_prompt"],
            audio._whisper_initial_prompt,
        )

    def test_system_mode_final_transcribe_uses_latency_beam(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        calls = {}

        class _FakeSegment:
            text = "what is react"

        class _FakeModel:
            def transcribe(self, samples, **kwargs):
                calls["kwargs"] = kwargs
                return ([_FakeSegment()], None)

        emitted = []
        audio.model = _FakeModel()
        audio._model_loaded = True
        audio.transcription_ready.connect(emitted.append)

        buffer = [np.ones((audio.block_size, 1), dtype=np.float32) * 0.01 for _ in range(5)]
        audio._transcribe_local(buffer, speech_started_at=1.0, is_final=True)

        self.assertEqual(emitted, ["what is react"])
        self.assertEqual(calls["kwargs"]["beam_size"], audio._system_beam_size)

    def test_emit_live_audio_chunk_converts_float32_to_pcm16(self):
        audio = AudioCapture(
            ConfigStub({"ai.live_mode.enabled": True})
        )
        chunks = []
        audio.live_audio_chunk.connect(lambda data, sr: chunks.append((data, sr)))

        audio._emit_live_audio_chunk(np.array([[0.0], [0.5], [-0.5]], dtype=np.float32))

        self.assertEqual(len(chunks), 1)
        payload, sr = chunks[0]
        self.assertEqual(sr, 16000)
        samples = np.frombuffer(payload, dtype=np.int16)
        self.assertEqual(samples.tolist(), [0, 16383, -16383])

    def test_emit_live_audio_chunk_is_silent_when_live_mode_disabled(self):
        audio = AudioCapture(ConfigStub({"ai.live_mode.enabled": False}))
        chunks = []
        audio.live_audio_chunk.connect(lambda data, sr: chunks.append((data, sr)))

        audio._emit_live_audio_chunk(np.array([[0.0], [0.5]], dtype=np.float32))

        self.assertEqual(chunks, [])

    def test_clear_resets_chunk_accumulator_and_metrics(self):
        audio = AudioCapture(ConfigStub({}))
        audio.transcripts.append("what is react")
        audio._session_transcript_parts = ["stale chunk"]
        audio._recent_transcripts = [{"text": "useReducer", "terms": ["useReducer"], "topics": {"react_hooks"}, "age": 0}]
        audio._whisper_initial_prompt = "useReducer. " + audio._whisper_base_prompt
        audio._last_transcription_metrics = {"speech_to_transcript_ms": 1234}
        before_epoch = audio._interim_epoch

        audio.clear()

        self.assertEqual(list(audio.transcripts), [])
        self.assertEqual(audio._session_transcript_parts, [])
        self.assertEqual(audio._recent_transcripts, [])
        self.assertEqual(audio._whisper_initial_prompt, audio._whisper_base_prompt)
        self.assertEqual(audio._last_transcription_metrics, {})
        self.assertEqual(audio._interim_epoch, before_epoch + 1)

    def test_system_audio_final_jobs_skip_when_superseded_by_newer_turn(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))
        audio._capture_generation = 7
        audio._final_submission_generation = 11
        queued = []

        def _capture_submit(fn):
            queued.append(fn)
            return True

        transcribed = []
        audio._submit_transcription_job = _capture_submit
        audio._transcribe = lambda buffer, speech_started_at, is_final, vad_meta=None: transcribed.append(
            ((vad_meta or {}).get("utterance_id", ""), is_final)
        )

        audio._submit_final_transcription(
            [np.ones((audio.block_size,), dtype=np.float32)],
            vad_meta={"utterance_id": "utt_old", "speech_duration": 1.0, "voiced_blocks": 6, "peak_rms": 0.08},
        )
        audio._submit_final_transcription(
            [np.ones((audio.block_size,), dtype=np.float32)],
            vad_meta={"utterance_id": "utt_new", "speech_duration": 1.2, "voiced_blocks": 7, "peak_rms": 0.09},
        )

        self.assertEqual(len(queued), 2)
        queued[0]()
        queued[1]()

        # P2: Multi-query fix — different utterances should NOT supersede each other
        # even in system mode. This allows "Q1 [pause] Q2" to both process.
        self.assertEqual(transcribed, [("utt_old", True), ("utt_new", True)])
        self.assertEqual(audio._final_decode_pending, 0)

    def test_max_utterance_exceeded_uses_configured_limit(self):
        audio = AudioCapture(ConfigStub({"capture.audio.vad.max_utterance_s": 6.0}))

        with patch("capture.audio.time.time", return_value=10.0):
            self.assertFalse(audio._max_utterance_exceeded(5.5))
            self.assertTrue(audio._max_utterance_exceeded(3.5))

    def test_process_loop_forces_final_turn_during_continuous_speech(self):
        audio = AudioCapture(ConfigStub({"ai.live_mode.enabled": False}))
        audio._ambient_calib_remaining = 0
        audio._detect_speech = lambda data, rms: (True, "webrtc")
        audio._emit_live_audio_chunk = lambda block: None
        submitted = []

        def _capture_submit(buffer, speech_started_at=None, vad_meta=None):
            submitted.append((buffer, speech_started_at, vad_meta))
            audio._running = False
            return True

        audio._submit_final_transcription = _capture_submit
        audio._required_start_confirm_blocks = lambda: 1
        audio._max_utterance_exceeded = Mock(side_effect=[True])
        audio.q = Mock()
        audio.q.get.return_value = np.ones((audio.block_size,), dtype=np.float32) * 0.02
        audio._running = True

        audio._process_loop()

        self.assertEqual(len(submitted), 1)
        buffer, speech_started_at, vad_meta = submitted[0]
        self.assertEqual(len(buffer), 1)
        self.assertIsNotNone(speech_started_at)
        self.assertEqual(vad_meta["vad_backend"], "webrtc")
        self.assertEqual(vad_meta["end_silence_ms"], 0)

    def test_chunked_short_tail_flushes_accumulated_final_instead_of_dropping(self):
        audio = AudioCapture(ConfigStub({"ai.live_mode.enabled": False}))
        submitted = []

        audio._submit_final_transcription = lambda buffer, speech_started_at=None, vad_meta=None: submitted.append(
            (buffer, speech_started_at, vad_meta)
        ) or True

        speech_buffer = [np.ones((audio.block_size,), dtype=np.float32) * 0.001]
        speech_started_at = 10.0
        utterance_started_at = 8.0
        utterance_vad_backend = "webrtc"
        had_mid_utterance_slice = True
        utterance_id = "utt_chunked"
        required_silence_blocks = 2
        speech_duration = 0.03
        utterance_peak_rms = 0.0
        utterance_voiced_blocks = 0

        if audio._should_drop_final_utterance(
            speech_duration=speech_duration,
            voiced_blocks=utterance_voiced_blocks,
            peak_rms=utterance_peak_rms,
            utterance_id=utterance_id,
        ):
            if had_mid_utterance_slice:
                audio._submit_final_transcription(
                    list(speech_buffer),
                    speech_started_at,
                    {
                        "utterance_id": utterance_id,
                        "vad_backend": utterance_vad_backend,
                        "end_silence_ms": required_silence_blocks * audio.block_ms,
                        "chunk_aware_eos": had_mid_utterance_slice,
                    },
                )

        self.assertEqual(len(submitted), 1)
        buffer, submitted_started_at, vad_meta = submitted[0]
        self.assertEqual(len(buffer), 1)
        self.assertEqual(submitted_started_at, speech_started_at)
        self.assertEqual(vad_meta["utterance_id"], utterance_id)
        self.assertTrue(vad_meta["chunk_aware_eos"])

    def test_ensure_session_ready_starts_pipeline_when_not_running(self):
        audio = AudioCapture(ConfigStub({"capture.audio.enabled": True}))
        starts = []

        def _fake_start():
            starts.append("start")
            audio._running = True
            audio._capture_thread = Mock(is_alive=lambda: True)
            audio._process_thread = Mock(is_alive=lambda: True)
            audio._active_streams = [object()]

        with patch.object(audio, "start", side_effect=_fake_start):
            self.assertTrue(audio.ensure_session_ready())

        self.assertEqual(starts, ["start"])

    def test_ensure_session_ready_keeps_healthy_pipeline_running(self):
        audio = AudioCapture(ConfigStub({"capture.audio.enabled": True}))
        audio._running = True

        with patch.object(audio, "_capture_workers_healthy_locked", return_value=True), \
             patch.object(audio, "_ensure_whisper_loaded_async") as preload, \
             patch.object(audio, "_drain_queue") as drain, \
             patch.object(audio, "restart") as restart:
            self.assertTrue(audio.ensure_session_ready())

        preload.assert_called_once()
        drain.assert_called_once()
        restart.assert_not_called()

    def test_queued_final_utterances_all_execute_in_order(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "mic"}))
        submitted = []

        class _FakePool:
            def submit(self, fn, *args):
                submitted.append((fn, args))

        audio._transcribe_pool = _FakePool()
        calls = []
        audio._transcribe = lambda *args, **kwargs: calls.append((args, kwargs))

        audio._submit_final_transcription(
            ["old"],
            1.0,
            {"utterance_id": "utt_1", "vad_backend": "webrtc"},
        )
        audio._submit_final_transcription(
            ["new"],
            2.0,
            {"utterance_id": "utt_2", "vad_backend": "webrtc"},
        )

        self.assertEqual(len(submitted), 2)

        submitted[0][0](*submitted[0][1])
        self.assertEqual(len(calls), 1)
        first_args, first_kwargs = calls[0]
        self.assertEqual(first_args[0], ["old"])
        self.assertEqual(first_kwargs["vad_meta"]["utterance_id"], "utt_1")

        submitted[1][0](*submitted[1][1])
        self.assertEqual(len(calls), 2)
        args, kwargs = calls[1]
        self.assertEqual(args[0], ["new"])
        self.assertEqual(args[1], 2.0)
        self.assertTrue(args[2])
        self.assertEqual(kwargs["vad_meta"]["utterance_id"], "utt_2")

    def test_process_loop_requires_brief_silence_before_follow_up_turn(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "ai.live_mode.enabled": False,
                    "capture.audio.vad.inter_turn_start_silence_ms": 400,
                }
            )
        )
        audio._ambient_calib_remaining = 0
        audio._emit_live_audio_chunk = lambda block: None
        audio._required_start_confirm_blocks = lambda: 1
        audio._required_silence_blocks = lambda *args, **kwargs: 1
        audio._should_drop_final_utterance = lambda **kwargs: False
        audio._max_utterance_exceeded = lambda started_at: False
        audio._detect_speech = lambda data, rms: (rms > 0.005, "webrtc")

        speech = np.ones((audio.block_size,), dtype=np.float32) * 0.02
        silence = np.zeros((audio.block_size,), dtype=np.float32)
        blocks = [
            speech,   # first utterance
            silence,  # finalize first utterance
            speech,   # immediate retrigger should be ignored
            silence,  # build required inter-turn quiet gap
            silence,  # build required inter-turn quiet gap
            speech,   # second utterance now allowed
            silence,  # finalize second utterance
        ]

        submitted = []

        def _capture_submit(buffer, speech_started_at=None, vad_meta=None):
            submitted.append(vad_meta or {})
            if len(submitted) >= 2:
                audio._running = False
            return True

        audio._submit_final_transcription = _capture_submit
        audio.q = Mock()
        audio.q.get.side_effect = blocks + [Exception("stop")]
        audio._running = True

        audio._process_loop()

        self.assertEqual(len(submitted), 2)
        self.assertEqual(submitted[0]["utterance_id"], "utt_1")
        self.assertEqual(submitted[1]["utterance_id"], "utt_2")

    def test_ultra_short_low_energy_final_is_dropped(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.vad.min_final_speech_ms": 180,
                    "capture.audio.vad.min_final_voiced_blocks": 2,
                    "capture.audio.vad.min_final_peak_rms": 0.003,
                }
            )
        )

        self.assertTrue(
            audio._should_drop_final_utterance(
                speech_duration=0.13,
                voiced_blocks=1,
                peak_rms=0.00199,
                utterance_id="utt_noise",
            )
        )
        self.assertFalse(
            audio._should_drop_final_utterance(
                speech_duration=0.25,
                voiced_blocks=3,
                peak_rms=0.02,
                utterance_id="utt_real",
            )
        )

    def test_system_audio_uses_longer_short_pause_window_by_default(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))

        with patch("capture.audio.time.time", return_value=11.2):
            blocks = audio._required_silence_blocks(10.0)

        self.assertEqual(blocks, audio._system_short_silence_blocks)

    def test_system_ultra_short_final_requires_more_than_duration_alone(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))

        self.assertTrue(
            audio._should_drop_final_utterance(
                speech_duration=0.20,
                voiced_blocks=0,
                peak_rms=0.01,
                utterance_id="utt_noise",
            )
        )

    def test_system_mode_disables_chunking_by_default(self):
        audio = AudioCapture(ConfigStub({"capture.audio.mode": "system"}))

        self.assertFalse(audio._chunking_active())

    def test_interim_can_run_with_one_pending_final(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.interim.max_pending_finals": 1,
                }
            )
        )
        audio._final_decode_pending = 1

        self.assertTrue(audio._can_run_interim_with_pending_finals())

    def test_interim_stops_when_pending_finals_exceed_limit(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.mode": "system",
                    "capture.audio.interim.max_pending_finals": 1,
                }
            )
        )
        audio._final_decode_pending = 2

        self.assertFalse(audio._can_run_interim_with_pending_finals())

    def test_local_final_stays_local_even_when_free_groq_is_available(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.transcription_provider": "local",
                    "capture.audio.prefer_free_cloud": True,
                    "api_key.groq": "gsk_test_key_1234567890",
                }
            )
        )
        audio._whisper_device = "cpu"

        with patch("capture.audio.time.time", return_value=12.0):
            provider = audio._effective_transcription_provider(
                is_final=True,
                speech_started_at=10.5,
            )

        self.assertEqual(provider, "local")

    def test_auth_failure_blocks_free_cloud_stt_for_rest_of_session(self):
        audio = AudioCapture(
            ConfigStub(
                {
                    "capture.audio.transcription_provider": "local",
                    "capture.audio.prefer_free_cloud": True,
                    "api_key.groq": "gsk_test_key_1234567890",
                }
            )
        )
        audio._whisper_device = "cpu"
        audio._cloud_stt_session_blocked = True

        with patch("capture.audio.time.time", return_value=12.0):
            provider = audio._effective_transcription_provider(
                is_final=True,
                speech_started_at=10.5,
            )

        self.assertEqual(provider, "local")

    def test_auth_failure_blocks_same_groq_key_across_future_sessions(self):
        cfg = ConfigStub(
            {
                "capture.audio.transcription_provider": "local",
                "capture.audio.prefer_free_cloud": True,
                "api_key.groq": "gsk_test_key_1234567890",
            }
        )
        audio = AudioCapture(cfg)
        audio._whisper_device = "cpu"
        audio._cloud_stt_failed_key = "gsk_test_key_1234567890"

        with patch("capture.audio.time.time", return_value=12.0):
            provider = audio._effective_transcription_provider(
                is_final=True,
                speech_started_at=10.5,
            )

        self.assertEqual(provider, "local")


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
        history.get_last = lambda n: []
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


class LiveSessionManagerTests(unittest.TestCase):
    def test_is_enabled_requires_flag_and_gemini_key(self):
        cfg = ConfigStub({"ai.live_mode.enabled": True, "api_key.gemini": "AIza123456789012345678901234567890"})
        manager = LiveSessionManager(cfg)

        self.assertTrue(manager.is_enabled())

        disabled = LiveSessionManager(ConfigStub({"api_key.gemini": "AIza123456789012345678901234567890"}))
        self.assertFalse(disabled.is_enabled())

    def test_extracts_text_transcript_and_turn_complete_from_message(self):
        msg = SimpleNamespace(
            text="Hello ",
            server_content=SimpleNamespace(
                input_transcription=SimpleNamespace(text="what is react"),
                turn_complete=True,
            ),
        )

        self.assertEqual(LiveSessionManager._extract_output_text(msg), "Hello ")
        self.assertEqual(LiveSessionManager._extract_input_transcript(msg), "what is react")
        self.assertTrue(LiveSessionManager._is_turn_complete(msg))

    def test_extracts_model_turn_text_from_parts(self):
        msg = SimpleNamespace(
            serverContent=SimpleNamespace(
                modelTurn=SimpleNamespace(
                    parts=[SimpleNamespace(text="Alpha "), SimpleNamespace(text="Beta")]
                )
            )
        )

        self.assertEqual(LiveSessionManager._extract_output_text(msg), "Alpha Beta")

    def test_does_not_fallback_to_top_level_text_for_non_text_model_parts(self):
        msg = SimpleNamespace(
            text="Warning-backed helper text",
            server_content=SimpleNamespace(
                model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=b"...")])
            ),
        )

        self.assertEqual(LiveSessionManager._extract_output_text(msg), "")

    def test_extracts_output_audio_transcription_text(self):
        msg = SimpleNamespace(
            server_content=SimpleNamespace(
                output_transcription=SimpleNamespace(text="Here is the spoken reply")
            )
        )

        self.assertEqual(
            LiveSessionManager._extract_output_text(msg),
            "Here is the spoken reply",
        )

    def test_candidate_models_migrates_legacy_live_name(self):
        self.assertEqual(
            LiveSessionManager._candidate_models("gemini-live-2.5-flash-preview"),
            ["gemini-2.5-flash-native-audio-preview-12-2025"],
        )

    def test_formats_model_compat_errors_with_clear_message(self):
        msg = LiveSessionManager._format_error_message(
            Exception("1008 None. models/gemini-live-2.5-flash-preview is not found for API version v1beta")
        )

        self.assertIn("no longer supported", msg)

    def test_normal_close_error_detection(self):
        self.assertTrue(LiveSessionManager._is_normal_close_error(Exception("1000 None.")))
        self.assertFalse(LiveSessionManager._is_normal_close_error(Exception("1007 None.")))

    def test_build_live_system_prompt_forbids_reasoning_narration(self):
        prompt = LiveSessionManager._build_live_system_prompt("Base prompt")

        self.assertIn("Never narrate your reasoning", prompt)
        self.assertIn("Give only the final answer", prompt)
        self.assertIn("Never include planning headers", prompt)

    def test_clean_live_response_strips_planning_preface(self):
        cleaned = LiveSessionManager._clean_live_response(
            "Defining Asynchronous Concepts\n"
            "I'm now formulating concise definitions for event loop and async.\n"
            "The event loop keeps the app responsive by processing queued tasks when the call stack is free."
        )

        self.assertEqual(
            cleaned,
            "The event loop keeps the app responsive by processing queued tasks when the call stack is free.",
        )

    def test_clean_live_response_drops_meta_paragraph_and_keeps_answer_paragraph(self):
        cleaned = LiveSessionManager._clean_live_response(
            "Defining the Event Loop\n"
            "I'm now focusing on the user's request for an explanation of the event loop. "
            "My goal is to craft a concise definition.\n\n"
            "The event loop is a mechanism in JavaScript that monitors the call stack and callback queue, "
            "allowing non-blocking asynchronous work."
        )

        self.assertEqual(
            cleaned,
            "The event loop is a mechanism in JavaScript that monitors the call stack and callback queue, "
            "allowing non-blocking asynchronous work.",
        )

    def test_clean_live_response_strips_meta_sentences_before_answer(self):
        cleaned = LiveSessionManager._clean_live_response(
            "Clarifying Type vs. Interface. I'm focusing on the core distinction now. "
            "An interface is generally preferred for object shapes, while a type alias is more flexible."
        )

        self.assertEqual(
            cleaned,
            "An interface is generally preferred for object shapes, while a type alias is more flexible.",
        )

    def test_updates_session_resumption_handle_from_message(self):
        manager = LiveSessionManager(ConfigStub({}))
        msg = SimpleNamespace(
            session_resumption_update=SimpleNamespace(
                resumable=True,
                new_handle="resume-token-123",
            )
        )

        manager._update_session_resumption(msg)

        self.assertEqual(manager._resume_handle, "resume-token-123")


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
        history.get_last = lambda n: []
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

    def test_select_provider_honors_strict_preferred_order_for_speech(self):
        config = ConfigStub({"ai.mode": "general"})
        engine = AIEngine(config, HistoryStub(), rag=None)
        groq = SimpleNamespace(
            name="groq",
            enabled=True,
            check_rate=lambda: True,
            has_model=lambda tier=None: True,
            default_tier="balanced",
        )
        gemini = SimpleNamespace(
            name="gemini",
            enabled=True,
            check_rate=lambda: True,
            has_model=lambda tier=None: True,
            default_tier="balanced",
        )
        ollama = SimpleNamespace(
            name="ollama",
            enabled=True,
            check_rate=lambda: True,
            has_model=lambda tier=None: True,
            default_tier="balanced",
        )
        engine._providers = {"groq": groq, "gemini": gemini, "ollama": ollama}

        provider, tier = engine._select_provider(
            "general",
            "simple",
            ["groq", "gemini", "ollama"],
            strict_preferred=True,
        )

        self.assertEqual(provider.name, "groq")
        self.assertEqual(tier, "fast")

    def test_failed_provider_is_filtered_for_current_session(self):
        config = ConfigStub({"ai.mode": "general"})
        engine = AIEngine(config, HistoryStub(), rag=None)
        engine._providers = {
            "groq": SimpleNamespace(enabled=True),
            "gemini": SimpleNamespace(enabled=True),
            "ollama": SimpleNamespace(enabled=True),
        }
        engine._mark_provider_failed_for_session("groq")

        filtered = engine._filter_provider_preferences(["groq", "gemini", "ollama"])

        self.assertEqual(filtered, ["gemini", "ollama"])


class AIEngineFallbackTests(unittest.TestCase):
    def test_generate_response_retries_ollama_with_extended_timeout_before_error(self):
        config = ConfigStub(
            {
                "ai.mode": "general",
                "ai.text.preferred_providers": ["groq", "gemini", "ollama"],
                "ai.text.first_token_timeout_ms_by_provider": {
                    "groq": 10,
                    "gemini": 10,
                    "ollama": 10,
                },
                "ai.text.local_first_token_retry_timeout_ms": 50,
            }
        )
        history = HistoryStub()
        history.get_last = lambda n: []
        engine = AIEngine(config, history, rag=None)

        class _FailingProvider:
            enabled = True
            default_tier = "balanced"

            def __init__(self, name):
                self.name = name

            @staticmethod
            def check_rate():
                return True

            @staticmethod
            def has_model(tier=None):
                return True

            async def generate_stream(self, system, user, tier=None):
                await asyncio.sleep(0.02)
                yield "late"

        class _RetryingOllamaProvider(_FailingProvider):
            def __init__(self):
                super().__init__("ollama")
                self.calls = 0

            async def generate_stream(self, system, user, tier=None):
                self.calls += 1
                if self.calls == 1:
                    await asyncio.sleep(0.02)
                    yield "late"
                    return
                yield "local "
                yield "answer"

        groq = _FailingProvider("groq")
        gemini = _FailingProvider("gemini")
        ollama = _RetryingOllamaProvider()
        engine._providers = {"groq": groq, "gemini": gemini, "ollama": ollama}
        engine._active_provider_id = "groq"
        engine._select_provider = lambda *args, **kwargs: (groq, "balanced")

        completed = []
        errors = []
        engine.response_complete.connect(completed.append)
        engine.error_occurred.connect(errors.append)

        asyncio.run(
            engine.generate_response(
                "what is prop drilling?",
                {"recent_audio": "", "full_audio_history": "", "latest_ocr": ""},
                origin="manual",
            )
        )

        self.assertEqual(errors, [])
        self.assertEqual(completed, ["local answer"])
        self.assertEqual(ollama.calls, 2)
        self.assertEqual(history.entries[-1]["provider"], "ollama")
        self.assertEqual(history.entries[-1]["metadata"]["providers_tried"], ["groq", "gemini", "ollama"])

    def test_generate_response_emits_error_after_ollama_retry_is_exhausted(self):
        config = ConfigStub(
            {
                "ai.mode": "general",
                "ai.text.preferred_providers": ["groq", "gemini", "ollama"],
                "ai.text.first_token_timeout_ms_by_provider": {
                    "groq": 10,
                    "gemini": 10,
                    "ollama": 10,
                },
                "ai.text.local_first_token_retry_timeout_ms": 20,
            }
        )
        history = HistoryStub()
        history.get_last = lambda n: []
        engine = AIEngine(config, history, rag=None)

        class _AlwaysTimingOutProvider:
            enabled = True
            default_tier = "balanced"

            def __init__(self, name):
                self.name = name

            @staticmethod
            def check_rate():
                return True

            @staticmethod
            def has_model(tier=None):
                return True

            async def generate_stream(self, system, user, tier=None):
                await asyncio.sleep(0.03)
                yield "late"

        groq = _AlwaysTimingOutProvider("groq")
        gemini = _AlwaysTimingOutProvider("gemini")
        ollama = _AlwaysTimingOutProvider("ollama")
        engine._providers = {"groq": groq, "gemini": gemini, "ollama": ollama}
        engine._active_provider_id = "groq"
        engine._select_provider = lambda *args, **kwargs: (groq, "balanced")

        errors = []
        completed = []
        engine.error_occurred.connect(errors.append)
        engine.response_complete.connect(completed.append)

        asyncio.run(
            engine.generate_response(
                "what is closure?",
                {"recent_audio": "", "full_audio_history": "", "latest_ocr": ""},
                origin="manual",
            )
        )

        self.assertEqual(completed, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("All providers failed.", errors[0])
        self.assertEqual(history.entries, [])


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
                "ai.live_mode.enabled": True,
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
        self.assertTrue(settings.chk_live_mode.isChecked())

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
                "ai.live_mode.enabled": False,
                "app.focus_on_show": False,
            }
        )
        settings = SettingsView(config, app=None)
        self.addCleanup(settings.deleteLater)

        settings.chk_focus_on_show.setChecked(True)
        settings.chk_live_mode.setChecked(True)

        settings._save_all()

        self.assertTrue(config.get("app.focus_on_show"))
        self.assertTrue(config.get("ai.live_mode.enabled"))
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
