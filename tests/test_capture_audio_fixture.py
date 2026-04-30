import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.capture_audio_fixture import (
    analyze_audio_samples,
    build_metadata,
    estimate_utterance_end_ms,
    describe_input_device,
    resolve_input_device,
    resample_audio,
    resolve_input_sample_rate,
    slugify_name,
)


class CaptureAudioFixtureTests(unittest.TestCase):
    def test_slugify_name_normalizes_user_input(self):
        self.assertEqual(slugify_name("React: What is React? 01"), "react_what_is_react_01")
        self.assertEqual(slugify_name("   "), "audio_fixture")

    def test_estimate_utterance_end_ms_finds_last_speech_window(self):
        sample_rate = 16000
        silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
        tone = np.ones(int(sample_rate * 0.6), dtype=np.float32) * 0.05
        samples = np.concatenate([silence, tone, silence])

        utterance_end_ms = estimate_utterance_end_ms(samples, sample_rate, window_ms=50, threshold=0.01)

        self.assertGreaterEqual(utterance_end_ms, 750.0)
        self.assertLessEqual(utterance_end_ms, 850.0)

    def test_build_metadata_uses_expected_keys(self):
        payload = build_metadata(
            transcript="what is react",
            utterance_end_ms=1800.0,
            mode="interview",
            tags=["react", "frontend"],
            notes="clean recording",
            expected_segments=1,
        )

        self.assertEqual(payload["expected_transcript"], "what is react")
        self.assertEqual(payload["expected_utterance_end_ms"], 1800.0)
        self.assertEqual(payload["expected_segments"], 1)
        self.assertEqual(payload["mode"], "interview")

    def test_analyze_audio_samples_flags_low_volume(self):
        quiet = np.ones(16000, dtype=np.float32) * 0.0005
        loud = np.ones(16000, dtype=np.float32) * 0.05

        quiet_stats = analyze_audio_samples(quiet)
        loud_stats = analyze_audio_samples(loud)

        self.assertTrue(quiet_stats["low_volume_flag"])
        self.assertFalse(loud_stats["low_volume_flag"])

    def test_resample_audio_changes_length_for_new_rate(self):
        samples = np.ones(48000, dtype=np.float32)

        resampled = resample_audio(samples, 48000, 16000)

        self.assertEqual(len(resampled), 16000)

    def test_resolve_input_sample_rate_uses_device_default_when_available(self):
        from unittest.mock import patch

        with patch(
            "scripts.capture_audio_fixture.sd.query_devices",
            return_value={"default_samplerate": 48000.0},
        ):
            sample_rate = resolve_input_sample_rate(16000, device=11)

        self.assertEqual(sample_rate, 48000)

    def test_resolve_input_device_uses_default_input_when_available(self):
        from unittest.mock import patch
        from types import SimpleNamespace

        with patch("scripts.capture_audio_fixture.sd.default", SimpleNamespace(device=(3, 7))):
            device = resolve_input_device()

        self.assertEqual(device, 3)

    def test_describe_input_device_formats_device_info(self):
        from unittest.mock import patch

        with patch(
            "scripts.capture_audio_fixture.sd.query_devices",
            return_value={"name": "USB Mic", "default_samplerate": 48000.0, "max_input_channels": 1},
        ):
            label = describe_input_device(2)

        self.assertIn("USB Mic", label)


if __name__ == "__main__":
    unittest.main()
