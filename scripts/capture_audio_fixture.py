"""Record a benchmark-ready audio fixture and matching metadata file.

Usage:
  python scripts/capture_audio_fixture.py --name react_what_is_react_01 --transcript "what is react"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - import failure is handled at runtime
    sd = None


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "audio_ground_truth"


def analyze_audio_samples(samples: np.ndarray) -> dict:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "silence_percentage": 100.0,
            "low_volume_flag": True,
        }
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    silence_percentage = float(np.mean(np.abs(audio) < 0.0015) * 100.0)
    return {
        "rms": rms,
        "peak": peak,
        "silence_percentage": silence_percentage,
        "low_volume_flag": rms < 0.003 or peak < 0.02,
    }


def slugify_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "audio_fixture"


def estimate_utterance_end_ms(
    samples: np.ndarray,
    sample_rate: int,
    window_ms: int = 50,
    threshold: float = 0.01,
) -> float:
    if samples.size == 0:
        return 0.0
    window_size = max(1, int(sample_rate * (window_ms / 1000.0)))
    last_speech_index = 0
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    for start in range(0, len(mono), window_size):
        window = mono[start : start + window_size]
        if window.size == 0:
            continue
        rms = float(np.sqrt(np.mean(window**2)))
        if rms >= threshold:
            last_speech_index = min(len(mono), start + window.size)
    return round((last_speech_index / float(sample_rate)) * 1000.0, 1)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def resolve_input_sample_rate(requested_sample_rate: int, device: int | None = None) -> int:
    if sd is None:
        raise RuntimeError("sounddevice is not installed")
    try:
        info = sd.query_devices(device, "input")
    except Exception:
        return int(requested_sample_rate)

    default_rate = info.get("default_samplerate") if isinstance(info, dict) else None
    if default_rate:
        return int(round(float(default_rate)))
    return int(requested_sample_rate)


def resolve_input_device(device: int | None = None) -> int | None:
    if sd is None:
        raise RuntimeError("sounddevice is not installed")
    if device is not None:
        return int(device)
    default = sd.default.device
    if isinstance(default, (list, tuple)) and default:
        default_input = default[0]
        if default_input is not None and int(default_input) >= 0:
            return int(default_input)
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for idx, info in enumerate(devices):
        if float(info.get("max_input_channels", 0) or 0) > 0:
            return idx
    return None


def describe_input_device(device: int | None) -> str:
    if sd is None:
        return "sounddevice unavailable"
    try:
        info = sd.query_devices(device, "input")
    except Exception as exc:
        return f"unavailable input device ({exc})"
    if isinstance(info, dict):
        name = str(info.get("name", f"Device {device}"))
        rate = info.get("default_samplerate")
        channels = info.get("max_input_channels")
        return f"{name} (channels={channels}, default_sr={rate})"
    return str(info)


def resample_audio(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32)
    target_len = max(1, int(round(len(audio) * dst_rate / float(src_rate))))
    src_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def record_samples(
    sample_rate: int,
    device: int | None = None,
    duration_s: float | None = None,
) -> tuple[np.ndarray, int, int | None]:
    if sd is None:
        raise RuntimeError("sounddevice is not installed")

    chunks: list[np.ndarray] = []
    resolved_device = resolve_input_device(device=device)
    actual_sample_rate = resolve_input_sample_rate(sample_rate, device=resolved_device)

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", flush=True)
        chunks.append(indata.copy())

    print("Press Enter to start recording.")
    input()
    device_label = describe_input_device(resolved_device)
    print(f"Input device: {device_label}")
    time.sleep(0.2)
    try:
        with sd.InputStream(
            samplerate=actual_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(actual_sample_rate * 0.2),
            device=resolved_device,
            callback=callback,
        ):
            if duration_s is not None and duration_s > 0:
                print(
                    f"Recording for {duration_s:.1f}s... speak naturally now. "
                    f"(device sample rate: {actual_sample_rate} Hz)"
                )
                sd.sleep(int(duration_s * 1000))
            else:
                print(
                    f"Recording... ask one question, then press Enter to stop. "
                    f"(device sample rate: {actual_sample_rate} Hz)"
                )
                input()
    except Exception as exc:
        raise RuntimeError(
            f"Could not open input device {resolved_device!r} at {actual_sample_rate} Hz: {exc}"
        ) from exc

    if not chunks:
        return np.zeros(0, dtype=np.float32), actual_sample_rate, resolved_device
    return (
        np.concatenate(chunks, axis=0).reshape(-1).astype(np.float32),
        actual_sample_rate,
        resolved_device,
    )


def build_metadata(
    transcript: str,
    utterance_end_ms: float,
    mode: str,
    tags: list[str],
    notes: str,
    expected_segments: int = 1,
) -> dict:
    return {
        "expected_transcript": transcript.strip(),
        "expected_utterance_end_ms": utterance_end_ms,
        "expected_segments": int(expected_segments),
        "mode": mode.strip() or "general",
        "tags": tags,
        "notes": notes.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Base fixture name, without extension.")
    parser.add_argument("--transcript", help="Expected transcript. If omitted, you'll be prompted.")
    parser.add_argument("--mode", default="general", help="Fixture mode label, e.g. interview or coding.")
    parser.add_argument("--tags", default="", help="Comma-separated tags.")
    parser.add_argument("--notes", default="", help="Optional notes stored in metadata.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate.")
    parser.add_argument("--device", type=int, help="Optional sounddevice input device index.")
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Recording duration in seconds. Use 0 to stop manually with Enter.",
    )
    parser.add_argument(
        "--force-low-volume",
        action="store_true",
        help="Save the fixture even if the recording looks too quiet.",
    )
    parser.add_argument(
        "--expected-segments",
        type=int,
        default=1,
        help="Expected chunk count for this utterance.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_devices:
        if sd is None:
            print("sounddevice is not installed")
            return 1
        devices = sd.query_devices()
        default_input = None
        if isinstance(sd.default.device, (list, tuple)) and sd.default.device:
            default_input = sd.default.device[0]
        for idx, info in enumerate(devices):
            max_inputs = int(info.get("max_input_channels", 0) or 0)
            if max_inputs <= 0:
                continue
            marker = " (default)" if default_input is not None and idx == int(default_input) else ""
            print(
                f"[{idx}] {info.get('name', f'Device {idx}')}{marker} "
                f"- inputs={max_inputs}, default_sr={info.get('default_samplerate')}"
            )
        return 0

    if not args.name:
        print("error: --name is required unless --list-devices is used")
        return 1

    fixture_name = slugify_name(args.name)
    transcript = (args.transcript or "").strip()
    if not transcript:
        transcript = input("Expected transcript: ").strip()
    if not transcript:
        print("Expected transcript is required.")
        return 1

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = FIXTURE_DIR / f"{fixture_name}.wav"
    json_path = FIXTURE_DIR / f"{fixture_name}.wav.json"

    try:
        samples, recorded_sample_rate, resolved_device = record_samples(
            sample_rate=args.sample_rate,
            device=args.device,
            duration_s=(None if args.duration == 0 else float(args.duration)),
        )
    except RuntimeError as exc:
        print(exc)
        return 1
    if samples.size == 0:
        print("No audio captured.")
        return 1

    signal = analyze_audio_samples(samples)
    utterance_end_ms = estimate_utterance_end_ms(samples, recorded_sample_rate)
    if signal["low_volume_flag"] and not args.force_low_volume:
        print("Recording looks too quiet and was not saved.")
        print(
            f"RMS={signal['rms']:.6f}, peak={signal['peak']:.6f}, "
            f"silence={signal['silence_percentage']:.1f}%"
        )
        print("Try a different input device with `--list-devices` and re-record closer/louder.")
        print("If you really want to keep it anyway, rerun with `--force-low-volume`.")
        return 1
    if utterance_end_ms <= 0.0 and not args.force_low_volume:
        print("Recording did not contain a detectable speech region and was not saved.")
        print(
            f"RMS={signal['rms']:.6f}, peak={signal['peak']:.6f}, "
            f"silence={signal['silence_percentage']:.1f}%"
        )
        print("Try a different device or speak louder, then record again.")
        return 1

    output_sample_rate = int(args.sample_rate)
    output_samples = resample_audio(samples, recorded_sample_rate, output_sample_rate)
    write_wav(wav_path, output_samples, output_sample_rate)
    metadata = build_metadata(
        transcript=transcript,
        utterance_end_ms=utterance_end_ms,
        mode=args.mode,
        tags=tags,
        notes=args.notes,
        expected_segments=args.expected_segments,
    )
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved audio fixture: {wav_path}")
    print(f"Saved metadata: {json_path}")
    print(f"Input device used: {describe_input_device(resolved_device)}")
    print(f"Estimated utterance end: {utterance_end_ms:.1f}ms")
    print(
        f"Audio stats: RMS={signal['rms']:.6f}, peak={signal['peak']:.6f}, "
        f"silence={signal['silence_percentage']:.1f}%"
    )
    if recorded_sample_rate != output_sample_rate:
        print(
            f"Recorded at {recorded_sample_rate} Hz and saved fixture at {output_sample_rate} Hz"
        )
    print("")
    print("Next step:")
    print("python benchmarks/audio_asr_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
