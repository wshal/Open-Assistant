"""Quick single-WAV benchmark to confirm local STT pipeline is working.

Usage:
    python -m benchmarks.single_wav_test
    python -m benchmarks.single_wav_test --wav tests/fixtures/audio_ground_truth/react_what_is_react_01.wav
    python -m benchmarks.single_wav_test --model small.en
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.audio_asr_benchmark import (
    AudioProfile,
    _BenchmarkConfig,
    create_audio_capture,
    evaluation_word_error_rate,
    load_audio_fixture,
    load_wav_mono,
    transcribe_samples,
)


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "audio_ground_truth"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run STT benchmark on a single WAV file")
    parser.add_argument(
        "--wav",
        type=Path,
        default=FIXTURE_DIR / "react_what_is_react_01.wav",
        help="Path to a .wav fixture file (must have a matching .wav.json metadata file)",
    )
    parser.add_argument(
        "--model",
        default="small.en",
        help="Whisper model to use (default: small.en)",
    )
    parser.add_argument(
        "--profile",
        default="webrtc_default",
        choices=["webrtc_default", "webrtc_fast_endpoint", "rms_fallback"],
    )
    args = parser.parse_args()

    wav_path = args.wav
    if not wav_path.exists():
        print(f"[ERROR] WAV file not found: {wav_path}")
        return 1

    meta_path = wav_path.with_suffix(wav_path.suffix + ".json")
    if not meta_path.exists():
        print(f"[ERROR] Metadata file not found: {meta_path}")
        return 1

    print(f"\n{'='*60}")
    print(f"  Single-WAV STT Benchmark")
    print(f"{'='*60}")
    print(f"  File   : {wav_path.name}")
    print(f"  Model  : {args.model}")
    print(f"  Profile: {args.profile}")
    print(f"{'='*60}\n")

    fixture = load_audio_fixture(wav_path)
    print(f"Expected transcript : {fixture.transcript!r}")

    profile = AudioProfile(name=args.profile, whisper_model=args.model)
    audio = create_audio_capture(profile)

    samples, sr = load_wav_mono(wav_path, target_sr=audio.sr)
    duration_ms = len(samples) / sr * 1000.0
    print(f"Audio duration      : {duration_ms:.0f}ms")

    print("\nLoading Whisper model and transcribing...")
    t0 = time.perf_counter()
    transcript, transcribe_ms = transcribe_samples(audio, samples, fixture)
    total_ms = (time.perf_counter() - t0) * 1000.0

    wer = evaluation_word_error_rate(fixture.transcript, transcript)

    print(f"\nTranscript          : {transcript!r}")
    print(f"WER                 : {wer:.3f}  ({wer*100:.1f}%)")
    print(f"Transcribe time     : {transcribe_ms:.0f}ms")
    print(f"Total time          : {total_ms:.0f}ms  (includes model load)")

    if wer == 0.0:
        print("\n[PASS] Perfect transcription - WER=0.000")
    elif wer < 0.15:
        print(f"\n[PASS] Good transcription - WER={wer:.3f}")
    elif wer < 0.30:
        print(f"\n[WARN] Acceptable transcription - WER={wer:.3f}")
    else:
        print(f"\n[FAIL] Poor transcription - WER={wer:.3f}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
