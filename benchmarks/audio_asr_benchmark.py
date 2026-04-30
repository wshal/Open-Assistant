"""Benchmark local audio segmentation and transcription against speech fixtures."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.capture_benchmark import (
    character_error_rate,
    percentile,
)
from capture.audio import AudioCapture
from utils.text_utils import normalize_transcript


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "audio_ground_truth"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "audio_asr_report.json"
DEFAULT_COMPARE_OUTPUT = ROOT / "benchmarks" / "audio_asr_compare.json"
DEFAULT_SWEEP_OUTPUT = ROOT / "benchmarks" / "audio_asr_model_sweep.json"
DEFAULT_DECODER_SWEEP_OUTPUT = ROOT / "benchmarks" / "audio_asr_decoder_sweep.json"
DEFAULT_MATRIX_SWEEP_OUTPUT = ROOT / "benchmarks" / "audio_asr_matrix_sweep.json"


@dataclass
class AudioFixture:
    filename: str
    audio_path: Path
    transcript: str
    utterance_end_ms: float | None
    expected_segments: int | None
    mode: str
    tags: list[str]
    notes: str


@dataclass
class AudioProfile:
    name: str
    silence_ms: int = 900
    short_utterance_max_s: float = 2.8
    short_silence_ms: int = 500
    chunking_enabled: bool = True
    vad_backend: str = "webrtc"  # "webrtc" or "rms"
    whisper_model: str = "base.en"
    language: str = "en"
    beam_size: int = 5
    condition_on_previous_text: bool = True
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500
    use_initial_prompt: bool = True


@dataclass
class AudioFixtureResult:
    filename: str
    profile: str
    transcript_raw: str
    transcript_normalized: str
    wer_raw: float
    wer_normalized: float
    cer_raw: float
    cer_normalized: float
    transcribe_only_ms: float
    speech_end_detected_ms: float
    endpoint_delta_ms: float | None
    false_early_cut: bool
    false_late_cut: bool
    unexpected_segments: bool
    segment_count: int
    micro_pause_slices: int
    hard_cap_slices: int
    audio_duration_ms: float
    vad_backend: str
    input_rms: float
    input_peak: float
    silence_percentage: float
    low_volume_flag: bool
    invalid_utterance_end_flag: bool
    all_models_empty_candidate: bool = False


def decoder_settings_dict(profile: AudioProfile) -> dict:
    return {
        "beam_size": profile.beam_size,
        "condition_on_previous_text": profile.condition_on_previous_text,
        "vad_filter": profile.vad_filter,
        "vad_min_silence_duration_ms": profile.vad_min_silence_duration_ms,
        "use_initial_prompt": profile.use_initial_prompt,
    }


def create_audio_capture(profile: AudioProfile) -> AudioCapture:
    audio = AudioCapture(_BenchmarkConfig(profile))
    audio.set_vad_silence_ms(profile.silence_ms)
    if profile.vad_backend == "rms":
        audio._vad = None
        audio._vad_backend_name = "rms"
    return audio


class _BenchmarkConfig:
    def __init__(self, profile: AudioProfile):
        self._profile = profile

    def get(self, path: str, default=None):
        if path == "capture.audio.sample_rate":
            return 16000
        if path == "capture.audio.mode":
            return "mic"
        if path == "capture.audio.enabled":
            return True
        if path == "capture.audio.whisper_model":
            return self._profile.whisper_model
        if path == "capture.audio.language":
            return self._profile.language
        if path == "capture.audio.chunking.enabled":
            return self._profile.chunking_enabled
        if path == "capture.audio.chunking.min_chunk_s":
            return 2.0
        if path == "capture.audio.chunking.max_chunk_s":
            return 4.0
        if path == "capture.audio.transcription_provider":
            return "local"
        if path == "capture.audio.ambient_calibration_ms":
            return 0
        if path == "capture.audio.interim.enabled":
            return False
        if path == "capture.audio.vad.frame_ms":
            return 20
        if path == "capture.audio.vad.mode":
            return 2
        if path == "capture.audio.vad.short_utterance_max_s":
            return self._profile.short_utterance_max_s
        if path == "capture.audio.vad.short_silence_ms":
            return self._profile.short_silence_ms
        if path == "performance.max_history":
            return 50
        return default


def load_wav_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported for fixtures: {path}")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != target_sr:
        target_len = max(1, int(round(len(samples) * target_sr / sample_rate)))
        src_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
        samples = np.interp(dst_x, src_x, samples).astype(np.float32)
        sample_rate = target_sr
    return samples.astype(np.float32), sample_rate


def analyze_audio_samples(samples: np.ndarray) -> dict:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "silence_percentage": 100.0,
            "low_volume_flag": True,
        }
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    silence_percentage = float(np.mean(np.abs(mono) < 0.0015) * 100.0)
    return {
        "rms": rms,
        "peak": peak,
        "silence_percentage": silence_percentage,
        "low_volume_flag": rms < 0.003 or peak < 0.02,
    }


def load_audio_fixture(audio_path: Path) -> AudioFixture:
    meta_path = audio_path.with_suffix(audio_path.suffix + ".json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    transcript = str(
        payload.get("expected_transcript", payload.get("transcript", ""))
    ).strip()
    utterance_end_ms = payload.get(
        "expected_utterance_end_ms", payload.get("utterance_end_ms")
    )
    return AudioFixture(
        filename=audio_path.name,
        audio_path=audio_path,
        transcript=transcript,
        utterance_end_ms=(float(utterance_end_ms) if utterance_end_ms is not None else None),
        expected_segments=(
            int(payload["expected_segments"]) if payload.get("expected_segments") is not None else None
        ),
        mode=str(payload.get("mode", "general") or "general"),
        tags=list(payload.get("tags", []) or []),
        notes=str(payload.get("notes", "") or ""),
    )


def iter_audio_fixtures(fixtures_dir: Path) -> Iterable[AudioFixture]:
    for path in sorted(fixtures_dir.glob("*.wav")):
        meta_path = path.with_suffix(path.suffix + ".json")
        if meta_path.exists():
            yield load_audio_fixture(path)


def normalize_eval_text(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    return " ".join(value.split())


def evaluation_word_error_rate(expected: str, actual: str) -> float:
    from benchmarks.capture_benchmark import levenshtein_distance

    expected_words = normalize_eval_text(expected).split()
    actual_words = normalize_eval_text(actual).split()
    if not expected_words:
        return 0.0 if not actual_words else 1.0
    return levenshtein_distance(expected_words, actual_words) / len(expected_words)


def evaluation_character_error_rate(expected: str, actual: str) -> float:
    from benchmarks.capture_benchmark import levenshtein_distance

    expected_chars = list(normalize_eval_text(expected))
    actual_chars = list(normalize_eval_text(actual))
    if not expected_chars:
        return 0.0 if not actual_chars else 1.0
    return levenshtein_distance(expected_chars, actual_chars) / len(expected_chars)


def _required_silence_blocks(audio: AudioCapture, elapsed_s: float) -> int:
    """Mirror of AudioCapture._required_silence_blocks for offline simulation.

    Post-chunk shortening is time-gated: we only apply the aggressive tail
    when the *new* chunk is still within short_utterance_max_s.  This matches
    the production fix and allows the harness to detect false-early-cut
    regressions caused by the sticky flag on multi-clause prompts.
    """
    had_slice = getattr(audio, "_benchmark_had_mid_utterance_slice", False)
    if had_slice and elapsed_s <= audio._short_utterance_max_s:
        return min(audio.silence_blocks, getattr(audio, "_post_chunk_silence_blocks", audio._short_silence_blocks))
    if elapsed_s <= audio._short_utterance_max_s:
        return min(audio.silence_blocks, audio._short_silence_blocks)
    return audio.silence_blocks


def simulate_segmentation(audio: AudioCapture, samples: np.ndarray, sample_rate: int) -> dict:
    if sample_rate != audio.sr:
        raise ValueError("fixture sample rate must match benchmark sample rate")

    block_size = audio.block_size
    is_speaking = False
    silence_count = 0
    speech_started_at_s: float | None = None
    backend_used = audio._vad_backend_name
    block_start_s = 0.0
    last_detected_ms = (len(samples) / float(sample_rate)) * 1000.0
    segments: list[tuple[int, int]] = []
    current_segment_start = 0
    micro_pause_slices = 0
    hard_cap_slices = 0
    had_mid_utterance_slice = False

    for offset in range(0, len(samples), block_size):
        block = samples[offset : offset + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))

        rms = float(np.sqrt(np.mean(block**2)))
        has_speech, backend = audio._detect_speech(block, rms)
        block_end_s = (offset + block_size) / float(sample_rate)

        if has_speech:
            last_detected_ms = block_end_s * 1000.0

        if has_speech:
            silence_count = 0
            if not is_speaking:
                speech_started_at_s = block_start_s
                backend_used = backend
                current_segment_start = offset
                had_mid_utterance_slice = False
            is_speaking = True
        elif is_speaking:
            silence_count += 1
            elapsed_s = max(0.0, block_start_s - (speech_started_at_s or 0.0))
            audio._benchmark_had_mid_utterance_slice = had_mid_utterance_slice
            in_scan_window = (
                audio._chunking_enabled and audio._chunk_min_s <= elapsed_s < audio._chunk_max_s
            )
            hard_cap_hit = audio._chunking_enabled and elapsed_s >= audio._chunk_max_s
            required_blocks = _required_silence_blocks(audio, elapsed_s)

            if in_scan_window and silence_count == 1:
                segments.append((current_segment_start, offset))
                micro_pause_slices += 1
                speech_started_at_s = block_start_s
                current_segment_start = offset
                had_mid_utterance_slice = True
            elif hard_cap_hit:
                segments.append((current_segment_start, offset))
                hard_cap_slices += 1
                silence_count = 0
                speech_started_at_s = block_start_s
                current_segment_start = offset
                had_mid_utterance_slice = True
            elif silence_count >= required_blocks:
                segments.append((current_segment_start, min(offset + block_size, len(samples))))
                audio._benchmark_had_mid_utterance_slice = False
                return {
                    "speech_end_detected_ms": block_end_s * 1000.0,
                    "vad_backend": backend_used,
                    "segment_count": len(segments),
                    "micro_pause_slices": micro_pause_slices,
                    "hard_cap_slices": hard_cap_slices,
                    "segments": segments,
                }

        block_start_s = block_end_s

    if is_speaking:
        segments.append((current_segment_start, len(samples)))
    elif not segments:
        segments.append((0, len(samples)))
    audio._benchmark_had_mid_utterance_slice = False
    return {
        "speech_end_detected_ms": last_detected_ms,
        "vad_backend": backend_used,
        "segment_count": len(segments),
        "micro_pause_slices": micro_pause_slices,
        "hard_cap_slices": hard_cap_slices,
        "segments": segments,
    }


def transcribe_samples(
    audio: AudioCapture,
    samples: np.ndarray,
    fixture: AudioFixture,
) -> tuple[str, float]:
    audio._ensure_whisper_loaded()
    if not audio.model:
        raise RuntimeError("Whisper model is not available for audio benchmark")

    start = time.perf_counter()
    transcribe_kwargs = {
        "language": audio._language,
        "beam_size": getattr(audio, "_benchmark_beam_size", 5),
        "condition_on_previous_text": getattr(audio, "_benchmark_condition_on_previous_text", True),
        "vad_filter": getattr(audio, "_benchmark_vad_filter", True),
        "initial_prompt": (
            audio._whisper_initial_prompt
            if getattr(audio, "_benchmark_use_initial_prompt", True)
            else None
        ),
        # Prevent Whisper from silently dropping the first few low-energy words
        # (the leading clause of long questions like "can you explain why...").
        # suppress_blank=False ensures blank-start tokens don't skip real speech.
        "no_speech_threshold": 0.7,
        "suppress_blank": False,
    }
    if transcribe_kwargs["vad_filter"]:
        transcribe_kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=getattr(audio, "_benchmark_vad_min_silence_duration_ms", 500)
        )
    with audio._infer_lock:
        segments, _ = audio.model.transcribe(samples, **transcribe_kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # Use the same confidence-gated filter as production so benchmark WER reflects
    # what users actually receive, not what Whisper outputs before filtering.
    transcript = audio._filter_segments(segments)
    return transcript, elapsed_ms


def benchmark_fixture(
    fixture: AudioFixture,
    profile: AudioProfile,
    audio: AudioCapture | None = None,
    transcribe_fn: Callable[[AudioCapture, np.ndarray, AudioFixture], tuple[str, float]] | None = None,
) -> AudioFixtureResult:
    audio = audio or create_audio_capture(profile)

    samples, sample_rate = load_wav_mono(fixture.audio_path, target_sr=audio.sr)
    signal = analyze_audio_samples(samples)
    segmentation = simulate_segmentation(audio, samples, sample_rate)
    transcriber = transcribe_fn or transcribe_samples
    transcript_raw, transcribe_only_ms = transcriber(audio, samples, fixture)
    transcript_normalized = normalize_transcript(transcript_raw)

    expected = fixture.transcript
    invalid_utterance_end_flag = (
        fixture.utterance_end_ms is not None and fixture.utterance_end_ms <= 0.0
    )
    endpoint_delta_ms = (
        segmentation["speech_end_detected_ms"] - fixture.utterance_end_ms
        if fixture.utterance_end_ms is not None and fixture.utterance_end_ms > 0.0
        else None
    )
    false_early = endpoint_delta_ms is not None and endpoint_delta_ms < -120.0
    false_late = endpoint_delta_ms is not None and endpoint_delta_ms > 250.0
    expected_segments = fixture.expected_segments or 1
    unexpected_segments = segmentation["segment_count"] > expected_segments

    return AudioFixtureResult(
        filename=fixture.filename,
        profile=profile.name,
        transcript_raw=transcript_raw,
        transcript_normalized=transcript_normalized,
        wer_raw=evaluation_word_error_rate(expected, transcript_raw),
        wer_normalized=evaluation_word_error_rate(expected, transcript_normalized),
        cer_raw=evaluation_character_error_rate(expected, transcript_raw),
        cer_normalized=evaluation_character_error_rate(expected, transcript_normalized),
        transcribe_only_ms=transcribe_only_ms,
        speech_end_detected_ms=segmentation["speech_end_detected_ms"],
        endpoint_delta_ms=endpoint_delta_ms,
        false_early_cut=bool(false_early),
        false_late_cut=bool(false_late),
        unexpected_segments=unexpected_segments,
        segment_count=segmentation["segment_count"],
        micro_pause_slices=segmentation["micro_pause_slices"],
        hard_cap_slices=segmentation["hard_cap_slices"],
        audio_duration_ms=(len(samples) / float(sample_rate)) * 1000.0,
        vad_backend=segmentation["vad_backend"],
        input_rms=signal["rms"],
        input_peak=signal["peak"],
        silence_percentage=signal["silence_percentage"],
        low_volume_flag=bool(signal["low_volume_flag"]),
        invalid_utterance_end_flag=bool(invalid_utterance_end_flag),
    )


def summarize_results(profile: AudioProfile, results: list[AudioFixtureResult]) -> dict:
    wer_raw = [r.wer_raw for r in results]
    wer_norm = [r.wer_normalized for r in results]
    transcribe_ms = [r.transcribe_only_ms for r in results]
    endpoint_deltas = [r.endpoint_delta_ms for r in results if r.endpoint_delta_ms is not None]
    segment_counts = [r.segment_count for r in results]
    low_volume = [r.filename for r in results if r.low_volume_flag]
    invalid_utterance_end = [r.filename for r in results if r.invalid_utterance_end_flag]
    empty_transcripts = [r.filename for r in results if not (r.transcript_normalized or "").strip()]
    return {
        "profile": profile.name,
        "whisper_model": profile.whisper_model,
        "language": profile.language,
        "vad_backend": profile.vad_backend,
        "decoder_settings": decoder_settings_dict(profile),
        "timestamp": time.time(),
        "fixtures": [r.__dict__ for r in results],
        "overall": {
            "average_wer_raw": statistics.fmean(wer_raw) if wer_raw else 0.0,
            "average_wer_normalized": statistics.fmean(wer_norm) if wer_norm else 0.0,
            "transcribe_p50_ms": percentile(transcribe_ms, 0.50),
            "transcribe_p95_ms": percentile(transcribe_ms, 0.95),
            "endpoint_delta_p50_ms": percentile(endpoint_deltas, 0.50) if endpoint_deltas else 0.0,
            "endpoint_delta_p95_ms": percentile(endpoint_deltas, 0.95) if endpoint_deltas else 0.0,
            "false_early_cut_count": sum(1 for r in results if r.false_early_cut),
            "false_late_cut_count": sum(1 for r in results if r.false_late_cut),
            "unexpected_segment_count": sum(1 for r in results if r.unexpected_segments),
            "average_segments": statistics.fmean(segment_counts) if segment_counts else 0.0,
            "low_volume_fixture_count": len(low_volume),
            "invalid_utterance_end_count": len(invalid_utterance_end),
            "empty_transcript_count": len(empty_transcripts),
            "average_input_rms": statistics.fmean([r.input_rms for r in results]) if results else 0.0,
            "average_input_peak": statistics.fmean([r.input_peak for r in results]) if results else 0.0,
        },
        "validation": {
            "low_volume_fixtures": low_volume,
            "invalid_utterance_end_fixtures": invalid_utterance_end,
            "empty_transcript_fixtures": empty_transcripts,
        },
    }


def run_benchmark(
    fixtures_dir: Path,
    output_path: Path,
    profile: AudioProfile,
    transcribe_fn: Callable[[AudioCapture, np.ndarray, AudioFixture], tuple[str, float]] | None = None,
) -> dict:
    if not fixtures_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixtures_dir}")
    fixtures = list(iter_audio_fixtures(fixtures_dir))
    if not fixtures:
        raise FileNotFoundError(
            f"No .wav fixtures with matching .wav.json metadata found in {fixtures_dir}"
        )

    audio = create_audio_capture(profile)
    audio._benchmark_beam_size = profile.beam_size
    audio._benchmark_condition_on_previous_text = profile.condition_on_previous_text
    audio._benchmark_vad_filter = profile.vad_filter
    audio._benchmark_vad_min_silence_duration_ms = profile.vad_min_silence_duration_ms
    audio._benchmark_use_initial_prompt = profile.use_initial_prompt
    results = [
        benchmark_fixture(fixture, profile, audio=audio, transcribe_fn=transcribe_fn)
        for fixture in fixtures
    ]
    payload = summarize_results(profile, results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_model_sweep(models: list[dict]) -> dict:
    if not models:
        return {
            "best_wer_model": None,
            "fastest_p50_model": None,
            "fastest_p95_model": None,
            "lowest_endpoint_p50_model": None,
            "models_ranked_by_wer": [],
            "fixture_winners": [],
            "worst_fixtures": [],
        }

    by_wer = sorted(
        models,
        key=lambda item: (
            item["overall"]["average_wer_normalized"],
            item["overall"]["transcribe_p50_ms"],
        ),
    )
    by_p50 = sorted(models, key=lambda item: item["overall"]["transcribe_p50_ms"])
    by_p95 = sorted(models, key=lambda item: item["overall"]["transcribe_p95_ms"])
    by_endpoint = sorted(models, key=lambda item: abs(item["overall"]["endpoint_delta_p50_ms"]))

    fixture_names = sorted(
        {
            fixture["filename"]
            for model in models
            for fixture in model.get("fixtures", [])
        }
    )
    fixture_winners = []
    worst_fixture_rows = []
    all_model_empty_fixtures = []
    for fixture_name in fixture_names:
        candidates = []
        for model in models:
            fixture = next(
                (item for item in model.get("fixtures", []) if item["filename"] == fixture_name),
                None,
            )
            if fixture is None:
                continue
            candidates.append(
                {
                    "whisper_model": model["whisper_model"],
                    "wer_normalized": fixture["wer_normalized"],
                    "transcribe_only_ms": fixture["transcribe_only_ms"],
                    "endpoint_delta_ms": fixture.get("endpoint_delta_ms"),
                    "transcript_normalized": fixture.get("transcript_normalized", ""),
                    "false_late_cut": bool(fixture.get("false_late_cut")),
                }
            )
        if not candidates:
            continue
        if all(not (item["transcript_normalized"] or "").strip() for item in candidates):
            all_model_empty_fixtures.append(fixture_name)

        ranked = sorted(
            candidates,
            key=lambda item: (
                item["wer_normalized"],
                item["transcribe_only_ms"],
                abs(item.get("endpoint_delta_ms") or 0.0),
            ),
        )
        winner = ranked[0]
        fixture_winners.append(
            {
                "filename": fixture_name,
                "winning_model": winner["whisper_model"],
                "winning_wer_normalized": winner["wer_normalized"],
                "winning_transcribe_only_ms": winner["transcribe_only_ms"],
                "models": [
                    {
                        "whisper_model": item["whisper_model"],
                        "wer_normalized": item["wer_normalized"],
                        "transcribe_only_ms": item["transcribe_only_ms"],
                        "endpoint_delta_ms": item["endpoint_delta_ms"],
                        "false_late_cut": item["false_late_cut"],
                    }
                    for item in ranked
                ],
            }
        )

        worst = max(
            candidates,
            key=lambda item: (
                item["wer_normalized"],
                abs(item.get("endpoint_delta_ms") or 0.0),
                item["transcribe_only_ms"],
            ),
        )
        worst_fixture_rows.append(
            {
                "filename": fixture_name,
                "worst_model": worst["whisper_model"],
                "worst_wer_normalized": worst["wer_normalized"],
                "worst_transcribe_only_ms": worst["transcribe_only_ms"],
                "worst_endpoint_delta_ms": worst["endpoint_delta_ms"],
                "transcript_normalized": worst["transcript_normalized"],
            }
        )

    worst_fixtures = sorted(
        worst_fixture_rows,
        key=lambda item: (
            item["worst_wer_normalized"],
            abs(item.get("worst_endpoint_delta_ms") or 0.0),
            item["worst_transcribe_only_ms"],
        ),
        reverse=True,
    )[:5]

    return {
        "best_wer_model": by_wer[0]["whisper_model"],
        "fastest_p50_model": by_p50[0]["whisper_model"],
        "fastest_p95_model": by_p95[0]["whisper_model"],
        "lowest_endpoint_p50_model": by_endpoint[0]["whisper_model"],
        "models_ranked_by_wer": [
            {
                "whisper_model": item["whisper_model"],
                "average_wer_normalized": item["overall"]["average_wer_normalized"],
                "transcribe_p50_ms": item["overall"]["transcribe_p50_ms"],
                "endpoint_delta_p50_ms": item["overall"]["endpoint_delta_p50_ms"],
            }
            for item in by_wer
        ],
        "fixture_winners": fixture_winners,
        "worst_fixtures": worst_fixtures,
        "all_model_empty_fixtures": all_model_empty_fixtures,
        "validation_summary": {
            "low_volume_fixture_counts": {
                item["whisper_model"]: item.get("overall", {}).get("low_volume_fixture_count", 0)
                for item in models
            },
            "invalid_utterance_end_counts": {
                item["whisper_model"]: item.get("overall", {}).get("invalid_utterance_end_count", 0)
                for item in models
            },
            "empty_transcript_counts": {
                item["whisper_model"]: item.get("overall", {}).get("empty_transcript_count", 0)
                for item in models
            },
        },
    }


def summarize_decoder_sweep(runs: list[dict]) -> dict:
    if not runs:
        return {
            "best_wer_profile": None,
            "fastest_p50_profile": None,
            "lowest_endpoint_p50_profile": None,
            "profiles_ranked_by_wer": [],
            "fixture_winners": [],
            "worst_fixtures": [],
        }

    by_wer = sorted(
        runs,
        key=lambda item: (
            item["overall"]["average_wer_normalized"],
            item["overall"]["transcribe_p50_ms"],
        ),
    )
    by_p50 = sorted(runs, key=lambda item: item["overall"]["transcribe_p50_ms"])
    by_endpoint = sorted(runs, key=lambda item: abs(item["overall"]["endpoint_delta_p50_ms"]))

    fixture_names = sorted(
        {
            fixture["filename"]
            for run in runs
            for fixture in run.get("fixtures", [])
        }
    )
    fixture_winners = []
    worst_fixture_rows = []
    for fixture_name in fixture_names:
        candidates = []
        for run in runs:
            fixture = next(
                (item for item in run.get("fixtures", []) if item["filename"] == fixture_name),
                None,
            )
            if fixture is None:
                continue
            candidates.append(
                {
                    "profile": run["profile"],
                    "decoder_settings": run.get("decoder_settings", {}),
                    "wer_normalized": fixture["wer_normalized"],
                    "transcribe_only_ms": fixture["transcribe_only_ms"],
                    "endpoint_delta_ms": fixture.get("endpoint_delta_ms"),
                    "transcript_normalized": fixture.get("transcript_normalized", ""),
                }
            )
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (
                item["wer_normalized"],
                item["transcribe_only_ms"],
                abs(item.get("endpoint_delta_ms") or 0.0),
            ),
        )
        winner = ranked[0]
        fixture_winners.append(
            {
                "filename": fixture_name,
                "winning_profile": winner["profile"],
                "winning_decoder_settings": winner["decoder_settings"],
                "winning_wer_normalized": winner["wer_normalized"],
                "winning_transcribe_only_ms": winner["transcribe_only_ms"],
            }
        )
        worst = max(
            candidates,
            key=lambda item: (
                item["wer_normalized"],
                abs(item.get("endpoint_delta_ms") or 0.0),
                item["transcribe_only_ms"],
            ),
        )
        worst_fixture_rows.append(
            {
                "filename": fixture_name,
                "worst_profile": worst["profile"],
                "worst_decoder_settings": worst["decoder_settings"],
                "worst_wer_normalized": worst["wer_normalized"],
                "worst_transcribe_only_ms": worst["transcribe_only_ms"],
                "worst_endpoint_delta_ms": worst["endpoint_delta_ms"],
                "transcript_normalized": worst["transcript_normalized"],
            }
        )

    return {
        "best_wer_profile": by_wer[0]["profile"],
        "fastest_p50_profile": by_p50[0]["profile"],
        "lowest_endpoint_p50_profile": by_endpoint[0]["profile"],
        "profiles_ranked_by_wer": [
            {
                "profile": item["profile"],
                "decoder_settings": item.get("decoder_settings", {}),
                "average_wer_normalized": item["overall"]["average_wer_normalized"],
                "transcribe_p50_ms": item["overall"]["transcribe_p50_ms"],
                "endpoint_delta_p50_ms": item["overall"]["endpoint_delta_p50_ms"],
            }
            for item in by_wer
        ],
        "fixture_winners": fixture_winners,
        "worst_fixtures": sorted(
            worst_fixture_rows,
            key=lambda item: (
                item["worst_wer_normalized"],
                abs(item.get("worst_endpoint_delta_ms") or 0.0),
                item["worst_transcribe_only_ms"],
            ),
            reverse=True,
        )[:5],
    }


def decoder_profiles(base_profile: AudioProfile) -> list[AudioProfile]:
    return [
        replace(
            base_profile,
            name=f"{base_profile.name}__decoder_default",
            beam_size=5,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_min_silence_duration_ms=500,
            use_initial_prompt=True,
        ),
        replace(
            base_profile,
            name=f"{base_profile.name}__decoder_no_prev_text",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_min_silence_duration_ms=500,
            use_initial_prompt=True,
        ),
        replace(
            base_profile,
            name=f"{base_profile.name}__decoder_no_vad_filter",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
            vad_min_silence_duration_ms=500,
            use_initial_prompt=True,
        ),
        replace(
            base_profile,
            name=f"{base_profile.name}__decoder_beam1_no_prev",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
            vad_min_silence_duration_ms=500,
            use_initial_prompt=True,
        ),
        replace(
            base_profile,
            name=f"{base_profile.name}__decoder_beam3_no_prev",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=False,
            vad_min_silence_duration_ms=500,
            use_initial_prompt=True,
        ),
        replace(
            base_profile,
            name=f"{base_profile.name}__decoder_beam3_no_prompt",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=False,
            vad_min_silence_duration_ms=500,
            use_initial_prompt=False,
        ),
    ]


def run_decoder_sweep(
    fixtures_dir: Path,
    output_path: Path,
    base_profile: AudioProfile,
    transcribe_fn: Callable[[AudioCapture, np.ndarray, AudioFixture], tuple[str, float]] | None = None,
) -> dict:
    payloads: list[dict] = []
    for profile in decoder_profiles(base_profile):
        payloads.append(
            run_benchmark(
                fixtures_dir,
                output_path.parent / f"_tmp_{profile.name}.json",
                profile,
                transcribe_fn=transcribe_fn,
            )
        )

    payload = {
        "timestamp": time.time(),
        "base_profile": base_profile.name,
        "whisper_model": base_profile.whisper_model,
        "runs": payloads,
        "summary": summarize_decoder_sweep(payloads),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in payloads:
        tmp = output_path.parent / f"_tmp_{item['profile']}.json"
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return payload


def summarize_matrix_sweep(runs: list[dict]) -> dict:
    if not runs:
        return {
            "best_profile": None,
            "fastest_p50_profile": None,
            "lowest_endpoint_p50_profile": None,
            "profiles_ranked_by_wer": [],
            "best_by_model": {},
            "fixture_winners": [],
            "worst_fixtures": [],
        }

    by_wer = sorted(
        runs,
        key=lambda item: (
            item["overall"]["average_wer_normalized"],
            item["overall"]["transcribe_p50_ms"],
        ),
    )
    by_p50 = sorted(runs, key=lambda item: item["overall"]["transcribe_p50_ms"])
    by_endpoint = sorted(runs, key=lambda item: abs(item["overall"]["endpoint_delta_p50_ms"]))

    best_by_model: dict[str, dict] = {}
    for item in by_wer:
        model = item.get("whisper_model", "")
        if model and model not in best_by_model:
            best_by_model[model] = {
                "profile": item["profile"],
                "decoder_settings": item.get("decoder_settings", {}),
                "average_wer_normalized": item["overall"]["average_wer_normalized"],
                "transcribe_p50_ms": item["overall"]["transcribe_p50_ms"],
                "endpoint_delta_p50_ms": item["overall"]["endpoint_delta_p50_ms"],
            }

    fixture_names = sorted(
        {
            fixture["filename"]
            for run in runs
            for fixture in run.get("fixtures", [])
        }
    )
    fixture_winners = []
    worst_fixture_rows = []
    for fixture_name in fixture_names:
        candidates = []
        for run in runs:
            fixture = next(
                (item for item in run.get("fixtures", []) if item["filename"] == fixture_name),
                None,
            )
            if fixture is None:
                continue
            candidates.append(
                {
                    "profile": run["profile"],
                    "whisper_model": run.get("whisper_model", ""),
                    "decoder_settings": run.get("decoder_settings", {}),
                    "wer_normalized": fixture["wer_normalized"],
                    "transcribe_only_ms": fixture["transcribe_only_ms"],
                    "endpoint_delta_ms": fixture.get("endpoint_delta_ms"),
                    "transcript_normalized": fixture.get("transcript_normalized", ""),
                }
            )
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (
                item["wer_normalized"],
                item["transcribe_only_ms"],
                abs(item.get("endpoint_delta_ms") or 0.0),
            ),
        )
        winner = ranked[0]
        fixture_winners.append(
            {
                "filename": fixture_name,
                "winning_profile": winner["profile"],
                "winning_model": winner["whisper_model"],
                "winning_decoder_settings": winner["decoder_settings"],
                "winning_wer_normalized": winner["wer_normalized"],
                "winning_transcribe_only_ms": winner["transcribe_only_ms"],
            }
        )
        worst = max(
            candidates,
            key=lambda item: (
                item["wer_normalized"],
                abs(item.get("endpoint_delta_ms") or 0.0),
                item["transcribe_only_ms"],
            ),
        )
        worst_fixture_rows.append(
            {
                "filename": fixture_name,
                "worst_profile": worst["profile"],
                "worst_model": worst["whisper_model"],
                "worst_decoder_settings": worst["decoder_settings"],
                "worst_wer_normalized": worst["wer_normalized"],
                "worst_transcribe_only_ms": worst["transcribe_only_ms"],
                "worst_endpoint_delta_ms": worst["endpoint_delta_ms"],
                "transcript_normalized": worst["transcript_normalized"],
            }
        )

    return {
        "best_profile": by_wer[0]["profile"],
        "fastest_p50_profile": by_p50[0]["profile"],
        "lowest_endpoint_p50_profile": by_endpoint[0]["profile"],
        "profiles_ranked_by_wer": [
            {
                "profile": item["profile"],
                "whisper_model": item.get("whisper_model", ""),
                "decoder_settings": item.get("decoder_settings", {}),
                "average_wer_normalized": item["overall"]["average_wer_normalized"],
                "transcribe_p50_ms": item["overall"]["transcribe_p50_ms"],
                "endpoint_delta_p50_ms": item["overall"]["endpoint_delta_p50_ms"],
            }
            for item in by_wer
        ],
        "best_by_model": best_by_model,
        "fixture_winners": fixture_winners,
        "worst_fixtures": sorted(
            worst_fixture_rows,
            key=lambda item: (
                item["worst_wer_normalized"],
                abs(item.get("worst_endpoint_delta_ms") or 0.0),
                item["worst_transcribe_only_ms"],
            ),
            reverse=True,
        )[:5],
    }


def run_matrix_sweep(
    fixtures_dir: Path,
    output_path: Path,
    base_profile: AudioProfile,
    models: Sequence[str],
    transcribe_fn: Callable[[AudioCapture, np.ndarray, AudioFixture], tuple[str, float]] | None = None,
) -> dict:
    payloads: list[dict] = []
    for model_name in models:
        model_profile = replace(
            base_profile,
            name=f"{base_profile.name}__{model_name}",
            whisper_model=model_name,
        )
        for profile in decoder_profiles(model_profile):
            payloads.append(
                run_benchmark(
                    fixtures_dir,
                    output_path.parent / f"_tmp_{profile.name}.json",
                    profile,
                    transcribe_fn=transcribe_fn,
                )
            )

    payload = {
        "timestamp": time.time(),
        "base_profile": base_profile.name,
        "models": list(models),
        "runs": payloads,
        "summary": summarize_matrix_sweep(payloads),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in payloads:
        tmp = output_path.parent / f"_tmp_{item['profile']}.json"
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return payload


def run_model_sweep(
    fixtures_dir: Path,
    output_path: Path,
    profile: AudioProfile,
    models: Sequence[str],
    transcribe_fn: Callable[[AudioCapture, np.ndarray, AudioFixture], tuple[str, float]] | None = None,
) -> dict:
    payloads: list[dict] = []
    for model_name in models:
        model_profile = replace(
            profile,
            name=f"{profile.name}__{model_name}",
            whisper_model=model_name,
        )
        payloads.append(
            run_benchmark(
                fixtures_dir,
                output_path.parent / f"_tmp_{model_profile.name}.json",
                model_profile,
                transcribe_fn=transcribe_fn,
            )
        )

    payload = {
        "timestamp": time.time(),
        "base_profile": profile.name,
        "models": payloads,
        "summary": summarize_model_sweep(payloads),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in payloads:
        tmp = output_path.parent / f"_tmp_{item['profile']}.json"
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return payload


def compare_fixture_results(base: dict, candidate: dict) -> dict:
    wer_gain = base["wer_normalized"] - candidate["wer_normalized"]
    latency_penalty = candidate["transcribe_only_ms"] - base["transcribe_only_ms"]
    base_endpoint = abs(base.get("endpoint_delta_ms") or 0.0)
    candidate_endpoint = abs(candidate.get("endpoint_delta_ms") or 0.0)
    endpoint_gain = base_endpoint - candidate_endpoint
    base_segment_penalty = 1 if base.get("unexpected_segments") else 0
    candidate_segment_penalty = 1 if candidate.get("unexpected_segments") else 0
    segment_gain = base_segment_penalty - candidate_segment_penalty

    if (
        wer_gain > 0.08
        and latency_penalty < 120
        and endpoint_gain >= -120
        and segment_gain >= 0
    ):
        recommended = candidate["profile"]
        reason = "accuracy_gain"
    elif (
        wer_gain > 0.03
        and latency_penalty < 60
        and endpoint_gain >= -80
        and segment_gain >= 0
    ):
        recommended = candidate["profile"]
        reason = "balanced_gain"
    elif segment_gain > 0 and latency_penalty < 80 and endpoint_gain >= -120:
        recommended = candidate["profile"]
        reason = "segmentation_gain"
    elif endpoint_gain > 120 and latency_penalty < 80:
        recommended = candidate["profile"]
        reason = "endpoint_gain"
    else:
        recommended = base["profile"]
        reason = "latency_guardrail"

    return {
        "filename": base["filename"],
        "base_profile": base["profile"],
        "candidate_profile": candidate["profile"],
        "base_wer_normalized": base["wer_normalized"],
        "candidate_wer_normalized": candidate["wer_normalized"],
        "base_transcribe_only_ms": base["transcribe_only_ms"],
        "candidate_transcribe_only_ms": candidate["transcribe_only_ms"],
        "base_endpoint_delta_ms": base.get("endpoint_delta_ms"),
        "candidate_endpoint_delta_ms": candidate.get("endpoint_delta_ms"),
        "base_unexpected_segments": bool(base.get("unexpected_segments")),
        "candidate_unexpected_segments": bool(candidate.get("unexpected_segments")),
        "wer_gain": wer_gain,
        "latency_penalty_ms": latency_penalty,
        "endpoint_gain_ms": endpoint_gain,
        "segment_gain": segment_gain,
        "recommended_profile": recommended,
        "reason": reason,
    }


def run_comparison(
    fixtures_dir: Path,
    output_path: Path,
    base_profile: AudioProfile,
    candidate_profile: AudioProfile,
    transcribe_fn: Callable[[AudioCapture, np.ndarray, AudioFixture], tuple[str, float]] | None = None,
) -> dict:
    base_payload = run_benchmark(fixtures_dir, output_path.parent / f"_tmp_{base_profile.name}.json", base_profile, transcribe_fn=transcribe_fn)
    candidate_payload = run_benchmark(fixtures_dir, output_path.parent / f"_tmp_{candidate_profile.name}.json", candidate_profile, transcribe_fn=transcribe_fn)

    base_by_file = {item["filename"]: item for item in base_payload["fixtures"]}
    candidate_by_file = {item["filename"]: item for item in candidate_payload["fixtures"]}
    comparisons = [
        compare_fixture_results(base_by_file[name], candidate_by_file[name])
        for name in sorted(base_by_file.keys())
        if name in candidate_by_file
    ]

    recommend_counts: dict[str, int] = {}
    for item in comparisons:
        recommend_counts[item["recommended_profile"]] = recommend_counts.get(item["recommended_profile"], 0) + 1

    payload = {
        "timestamp": time.time(),
        "base": base_payload,
        "candidate": candidate_payload,
        "comparisons": comparisons,
        "summary": {
            "recommended_counts": recommend_counts,
            "candidate_better_wer_count": sum(1 for item in comparisons if item["wer_gain"] > 0),
            "candidate_better_endpoint_count": sum(1 for item in comparisons if item["endpoint_gain_ms"] > 0),
            "candidate_better_segmentation_count": sum(1 for item in comparisons if item["segment_gain"] > 0),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for tmp in (
        output_path.parent / f"_tmp_{base_profile.name}.json",
        output_path.parent / f"_tmp_{candidate_profile.name}.json",
    ):
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return payload


def default_profiles() -> dict[str, AudioProfile]:
    return {
        "webrtc_default": AudioProfile(name="webrtc_default", silence_ms=900, short_utterance_max_s=2.8, short_silence_ms=500, vad_backend="webrtc"),
        "webrtc_fast_endpoint": AudioProfile(name="webrtc_fast_endpoint", silence_ms=700, short_utterance_max_s=2.6, short_silence_ms=400, vad_backend="webrtc"),
        "rms_fallback": AudioProfile(name="rms_fallback", silence_ms=900, short_utterance_max_s=2.8, short_silence_ms=500, vad_backend="rms"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", default="webrtc_default")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--base-profile", default="webrtc_default")
    parser.add_argument("--candidate-profile", default="webrtc_fast_endpoint")
    parser.add_argument(
        "--sweep-models",
        default="",
        help="Comma-separated Whisper models to benchmark in one run, e.g. tiny.en,base.en,small.en",
    )
    parser.add_argument(
        "--sweep-decoders",
        action="store_true",
        help="Benchmark a fixed set of decoder settings for the selected profile/model.",
    )
    parser.add_argument(
        "--sweep-matrix",
        default="",
        help="Comma-separated Whisper models to benchmark against the decoder sweep matrix in one run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profiles = default_profiles()
    if args.sweep_matrix:
        output_path = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_MATRIX_SWEEP_OUTPUT
        model_names = [item.strip() for item in args.sweep_matrix.split(",") if item.strip()]
        if not model_names:
            raise ValueError("At least one model must be provided to --sweep-matrix")
        payload = run_matrix_sweep(
            args.fixtures,
            output_path,
            profiles[args.profile],
            model_names,
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"\nSaved matrix sweep report to {output_path}")
        return 0
    if args.sweep_decoders:
        output_path = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_DECODER_SWEEP_OUTPUT
        payload = run_decoder_sweep(
            args.fixtures,
            output_path,
            profiles[args.profile],
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"\nSaved decoder sweep report to {output_path}")
        return 0
    if args.sweep_models:
        output_path = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_SWEEP_OUTPUT
        model_names = [item.strip() for item in args.sweep_models.split(",") if item.strip()]
        if not model_names:
            raise ValueError("At least one model must be provided to --sweep-models")
        payload = run_model_sweep(
            args.fixtures,
            output_path,
            profiles[args.profile],
            model_names,
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"\nSaved model sweep report to {output_path}")
        return 0
    if args.compare:
        output_path = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_COMPARE_OUTPUT
        payload = run_comparison(
            args.fixtures,
            output_path,
            profiles[args.base_profile],
            profiles[args.candidate_profile],
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"\nSaved comparison report to {output_path}")
        return 0

    payload = run_benchmark(
        args.fixtures,
        args.output,
        profiles[args.profile],
    )
    print(json.dumps(payload["overall"], indent=2))
    print(f"\nSaved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
