import json
import shutil
import sys
import unittest
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = ROOT / "scratch" / "tests"

from benchmarks.audio_asr_benchmark import (
    AudioProfile,
    _required_silence_blocks as benchmark_required_silence_blocks,
    benchmark_fixture,
    compare_fixture_results,
    load_audio_fixture,
    run_benchmark,
    run_comparison,
    run_decoder_sweep,
    run_matrix_sweep,
    run_model_sweep,
)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _tone(duration_s: float, sample_rate: int = 16000, amplitude: float = 0.08) -> np.ndarray:
    count = int(duration_s * sample_rate)
    t = np.arange(count, dtype=np.float32) / float(sample_rate)
    return (np.sin(2 * np.pi * 220.0 * t) * amplitude).astype(np.float32)


def _silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


@contextmanager
def _workspace_tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_TMP_ROOT / f"tmp_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class AudioASRBenchmarkTests(unittest.TestCase):
    def test_load_audio_fixture_supports_expected_metadata_keys(self):
        with _workspace_tempdir() as root:
            audio_path = root / "react_question.wav"
            _write_wav(audio_path, np.concatenate([_silence(0.2), _tone(0.6), _silence(0.6)]))
            meta_path = audio_path.with_suffix(".wav.json")
            meta_path.write_text(
                json.dumps(
                    {
                        "expected_transcript": "what is react",
                        "expected_utterance_end_ms": 1300,
                        "expected_segments": 1,
                        "tags": ["react", "frontend"],
                    }
                ),
                encoding="utf-8",
            )

            fixture = load_audio_fixture(audio_path)

            self.assertEqual(fixture.transcript, "what is react")
            self.assertEqual(fixture.utterance_end_ms, 1300.0)
            self.assertEqual(fixture.expected_segments, 1)
            self.assertEqual(fixture.tags, ["react", "frontend"])

    def test_benchmark_required_silence_blocks_shortens_after_slice(self):
        audio = type(
            "AudioStub",
            (),
            {
                "silence_blocks": 4,
                "_short_silence_blocks": 2,
                "_post_chunk_silence_blocks": 2,
                "_short_utterance_max_s": 2.8,
                "_benchmark_had_mid_utterance_slice": True,
            },
        )()

        # Within short_utterance_max_s → post-chunk shortening applies
        self.assertEqual(benchmark_required_silence_blocks(audio, 1.5), 2)
        # Past short_utterance_max_s → reverts to full silence_blocks (guards multi-clause)
        self.assertEqual(benchmark_required_silence_blocks(audio, 4.2), 4)

    def test_benchmark_fixture_reports_segmentation_slices(self):
        with _workspace_tempdir() as root:
            audio_path = root / "hooks_question.wav"
            samples = np.concatenate(
                [
                    _silence(0.4),
                    _tone(2.2),
                    _silence(0.2),
                    _tone(0.6),
                    _silence(1.0),
                ]
            )
            _write_wav(audio_path, samples)
            audio_path.with_suffix(".wav.json").write_text(
                json.dumps(
                    {
                        "expected_transcript": "explain hooks in react",
                        "expected_utterance_end_ms": 3800,
                        "expected_segments": 1,
                    }
                ),
                encoding="utf-8",
            )

            fixture = load_audio_fixture(audio_path)
            profile = AudioProfile(
                name="rms_default",
                vad_backend="rms",
                silence_ms=900,
                short_silence_ms=500,
                short_utterance_max_s=2.8,
                chunking_enabled=True,
            )

            result = benchmark_fixture(
                fixture,
                profile,
                transcribe_fn=lambda audio, samples, fixture: (fixture.transcript, 14.0),
            )

            self.assertEqual(result.segment_count, 2)
            self.assertEqual(result.micro_pause_slices, 1)
            self.assertEqual(result.hard_cap_slices, 0)
            self.assertTrue(result.unexpected_segments)
            self.assertEqual(result.wer_normalized, 0.0)
            self.assertEqual(result.vad_backend, "rms")
            self.assertGreater(result.input_rms, 0.0)
            self.assertFalse(result.low_volume_flag)
            self.assertFalse(result.invalid_utterance_end_flag)

    def test_run_benchmark_and_comparison_emit_reports(self):
        with _workspace_tempdir() as root:
            fixtures_dir = root / "audio_ground_truth"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            audio_path = fixtures_dir / "react_intro.wav"
            _write_wav(audio_path, np.concatenate([_silence(0.2), _tone(0.8), _silence(0.8)]))
            audio_path.with_suffix(".wav.json").write_text(
                json.dumps(
                    {
                        "expected_transcript": "what is react",
                        "expected_utterance_end_ms": 1800,
                        "expected_segments": 1,
                    }
                ),
                encoding="utf-8",
            )

            base_profile = AudioProfile(name="base", vad_backend="rms", language="en")
            candidate_profile = AudioProfile(name="candidate", vad_backend="rms", language="fr")
            report_path = root / "report.json"
            compare_path = root / "compare.json"

            def base_transcribe(audio, samples, fixture):
                return "what is rack", 15.0

            def compare_transcribe(audio, samples, fixture):
                if audio._language == "fr":
                    return fixture.transcript, 25.0
                return "what is rack", 15.0

            report = run_benchmark(
                fixtures_dir,
                report_path,
                base_profile,
                transcribe_fn=base_transcribe,
            )
            comparison = run_comparison(
                fixtures_dir,
                compare_path,
                base_profile,
                candidate_profile,
                transcribe_fn=compare_transcribe,
            )

            self.assertTrue(report_path.exists())
            self.assertEqual(report["fixtures"][0]["filename"], "react_intro.wav")
            self.assertIn("validation", report)
            self.assertEqual(report["overall"]["invalid_utterance_end_count"], 0)
            self.assertTrue(compare_path.exists())
            self.assertEqual(
                comparison["comparisons"][0]["recommended_profile"],
                "candidate",
            )

    def test_run_model_sweep_emits_single_report(self):
        with _workspace_tempdir() as root:
            fixtures_dir = root / "audio_ground_truth"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            audio_path = fixtures_dir / "react_intro.wav"
            _write_wav(audio_path, np.concatenate([_silence(0.2), _tone(0.8), _silence(0.8)]))
            audio_path.with_suffix(".wav.json").write_text(
                json.dumps(
                    {
                        "expected_transcript": "what is react",
                        "expected_utterance_end_ms": 1800,
                        "expected_segments": 1,
                    }
                ),
                encoding="utf-8",
            )

            profile = AudioProfile(name="base_profile", vad_backend="rms", language="en")
            sweep_path = root / "model_sweep.json"

            def transcribe_by_model(audio, samples, fixture):
                if audio._model_name == "small.en":
                    return fixture.transcript, 30.0
                if audio._model_name == "base.en":
                    return "what is rack", 15.0
                return "what rack", 8.0

            sweep = run_model_sweep(
                fixtures_dir,
                sweep_path,
                profile,
                ["tiny.en", "base.en", "small.en"],
                transcribe_fn=transcribe_by_model,
            )

            self.assertTrue(sweep_path.exists())
            self.assertEqual(
                [item["whisper_model"] for item in sweep["models"]],
                ["tiny.en", "base.en", "small.en"],
            )
            self.assertEqual(sweep["summary"]["best_wer_model"], "small.en")
            self.assertEqual(sweep["summary"]["fastest_p50_model"], "tiny.en")
            self.assertEqual(
                sweep["summary"]["models_ranked_by_wer"][0]["whisper_model"],
                "small.en",
            )
            self.assertEqual(
                sweep["summary"]["fixture_winners"][0]["winning_model"],
                "small.en",
            )
            self.assertEqual(
                sweep["summary"]["worst_fixtures"][0]["worst_model"],
                "tiny.en",
            )
            self.assertIn("validation_summary", sweep["summary"])

    def test_run_decoder_sweep_emits_ranked_profiles(self):
        with _workspace_tempdir() as root:
            fixtures_dir = root / "audio_ground_truth"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            audio_path = fixtures_dir / "react_intro.wav"
            _write_wav(audio_path, np.concatenate([_silence(0.2), _tone(0.8), _silence(0.8)]))
            audio_path.with_suffix(".wav.json").write_text(
                json.dumps(
                    {
                        "expected_transcript": "what is react",
                        "expected_utterance_end_ms": 1800,
                        "expected_segments": 1,
                    }
                ),
                encoding="utf-8",
            )

            profile = AudioProfile(name="base_profile", vad_backend="rms", language="en", whisper_model="base.en")
            sweep_path = root / "decoder_sweep.json"

            def transcribe_by_decoder(audio, samples, fixture):
                beam = getattr(audio, "_benchmark_beam_size", 5)
                no_prev = not getattr(audio, "_benchmark_condition_on_previous_text", True)
                use_prompt = getattr(audio, "_benchmark_use_initial_prompt", True)
                if beam == 3 and no_prev and use_prompt:
                    return fixture.transcript, 18.0
                if beam == 1:
                    return "what rack", 8.0
                return "what is rack", 12.0

            sweep = run_decoder_sweep(
                fixtures_dir,
                sweep_path,
                profile,
                transcribe_fn=transcribe_by_decoder,
            )

            self.assertTrue(sweep_path.exists())
            self.assertEqual(sweep["summary"]["best_wer_profile"], "base_profile__decoder_beam3_no_prev")
            self.assertEqual(
                sweep["summary"]["profiles_ranked_by_wer"][0]["decoder_settings"]["beam_size"],
                3,
            )
            self.assertEqual(
                sweep["summary"]["fixture_winners"][0]["winning_profile"],
                "base_profile__decoder_beam3_no_prev",
            )

    def test_run_matrix_sweep_emits_best_profile_and_best_by_model(self):
        with _workspace_tempdir() as root:
            fixtures_dir = root / "audio_ground_truth"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            audio_path = fixtures_dir / "react_intro.wav"
            _write_wav(audio_path, np.concatenate([_silence(0.2), _tone(0.8), _silence(0.8)]))
            audio_path.with_suffix(".wav.json").write_text(
                json.dumps(
                    {
                        "expected_transcript": "what is react",
                        "expected_utterance_end_ms": 1800,
                        "expected_segments": 1,
                    }
                ),
                encoding="utf-8",
            )

            profile = AudioProfile(name="base_profile", vad_backend="rms", language="en", whisper_model="base.en")
            sweep_path = root / "matrix_sweep.json"

            def transcribe_matrix(audio, samples, fixture):
                model = audio._model_name
                beam = getattr(audio, "_benchmark_beam_size", 5)
                no_prev = not getattr(audio, "_benchmark_condition_on_previous_text", True)
                if model == "small.en" and beam == 3 and no_prev:
                    return fixture.transcript, 18.0
                if model == "base.en" and beam == 3 and no_prev:
                    return "what is rack", 12.0
                return "what rack", 8.0

            sweep = run_matrix_sweep(
                fixtures_dir,
                sweep_path,
                profile,
                ["base.en", "small.en"],
                transcribe_fn=transcribe_matrix,
            )

            self.assertTrue(sweep_path.exists())
            self.assertEqual(sweep["summary"]["best_profile"], "base_profile__small.en__decoder_beam3_no_prev")
            self.assertEqual(sweep["summary"]["best_by_model"]["small.en"]["profile"], "base_profile__small.en__decoder_beam3_no_prev")
            self.assertEqual(sweep["summary"]["fixture_winners"][0]["winning_model"], "small.en")

    def test_low_volume_and_invalid_utterance_end_are_flagged(self):
        with _workspace_tempdir() as root:
            fixtures_dir = root / "audio_ground_truth"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            audio_path = fixtures_dir / "quiet_long.wav"
            _write_wav(audio_path, np.concatenate([_silence(0.4), _tone(0.8, amplitude=0.001), _silence(0.8)]))
            audio_path.with_suffix(".wav.json").write_text(
                json.dumps(
                    {
                        "expected_transcript": "explain hooks in react",
                        "expected_utterance_end_ms": 0,
                        "expected_segments": 1,
                    }
                ),
                encoding="utf-8",
            )

            profile = AudioProfile(name="base", vad_backend="rms", language="en")
            report_path = root / "report.json"

            report = run_benchmark(
                fixtures_dir,
                report_path,
                profile,
                transcribe_fn=lambda audio, samples, fixture: ("", 12.0),
            )

            fixture = report["fixtures"][0]
            self.assertTrue(fixture["low_volume_flag"])
            self.assertTrue(fixture["invalid_utterance_end_flag"])
            self.assertEqual(report["overall"]["low_volume_fixture_count"], 1)
            self.assertEqual(report["overall"]["invalid_utterance_end_count"], 1)
            self.assertEqual(report["overall"]["empty_transcript_count"], 1)
            self.assertEqual(
                report["validation"]["invalid_utterance_end_fixtures"],
                ["quiet_long.wav"],
            )

    def test_compare_fixture_results_honors_segmentation_guardrail(self):
        base = {
            "filename": "react.wav",
            "profile": "base",
            "wer_normalized": 0.2,
            "transcribe_only_ms": 20.0,
            "endpoint_delta_ms": 50.0,
            "unexpected_segments": True,
        }
        candidate = {
            "filename": "react.wav",
            "profile": "candidate",
            "wer_normalized": 0.1,
            "transcribe_only_ms": 40.0,
            "endpoint_delta_ms": 40.0,
            "unexpected_segments": False,
        }

        result = compare_fixture_results(base, candidate)

        self.assertEqual(result["recommended_profile"], "candidate")
        self.assertEqual(result["reason"], "accuracy_gain")


if __name__ == "__main__":
    unittest.main()
