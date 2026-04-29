"""Benchmark the OCR capture path against fixture ground truth."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.ocr import OCREngine
from utils.telemetry import telemetry


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ocr_ground_truth"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "baseline_ocr.json"
DEFAULT_COMPARE_OUTPUT = ROOT / "benchmarks" / "ocr_compare.json"


@dataclass
class FixtureResult:
    filename: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cer: float
    wer: float
    truth_len: int
    ext_len: int


class _BenchmarkConfig:
    def __init__(self, preferred_engine: str):
        self._preferred_engine = preferred_engine

    def get(self, path: str, default=None):
        if path == "capture.screen.ocr_engine":
            return self._preferred_engine
        if path == "capture.screen.ocr_editor_recrop":
            return False
        return default


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").replace("\r", "\n").split())


def levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(expected: str, actual: str) -> float:
    expected_chars = list(normalize_text(expected))
    actual_chars = list(normalize_text(actual))
    if not expected_chars:
        return 0.0 if not actual_chars else 1.0
    return levenshtein_distance(expected_chars, actual_chars) / len(expected_chars)


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = normalize_text(expected).split()
    actual_words = normalize_text(actual).split()
    if not expected_words:
        return 0.0 if not actual_words else 1.0
    return levenshtein_distance(expected_words, actual_words) / len(expected_words)


def iter_fixture_images(fixtures_dir: Path) -> Iterable[Path]:
    for path in sorted(fixtures_dir.glob("*.png")):
        truth_path = path.with_suffix(path.suffix + ".txt")
        if truth_path.exists():
            yield path


def summarize_results(engine_name: str, iterations: int, results: list[FixtureResult]) -> dict:
    all_p50s = [result.latency_p50_ms for result in results]
    all_p95s = [result.latency_p95_ms for result in results]
    cer_values = [result.cer for result in results]
    return {
        "engine": engine_name,
        "timestamp": time.time(),
        "iterations": iterations,
        "fixtures": [result.__dict__ for result in results],
        "overall": {
            "latency_p50_ms": percentile(all_p50s, 0.50),
            "latency_p95_ms": percentile(all_p95s, 0.95),
            "average_cer": statistics.fmean(cer_values) if cer_values else 0.0,
        },
    }


async def benchmark_fixture(ocr: OCREngine, image_path: Path, iterations: int) -> FixtureResult:
    truth_path = image_path.with_suffix(image_path.suffix + ".txt")
    truth_text = truth_path.read_text(encoding="utf-8").strip()
    latencies_ms: list[float] = []
    extracted_text = ""

    for _ in range(iterations):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        t0 = time.perf_counter()
        text, _ = await ocr.extract(image)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if text:
            extracted_text = text

    extracted_text = extracted_text or ""
    cer = character_error_rate(truth_text, extracted_text)
    result = FixtureResult(
        filename=image_path.name,
        latency_p50_ms=percentile(latencies_ms, 0.50),
        latency_p95_ms=percentile(latencies_ms, 0.95),
        latency_p99_ms=percentile(latencies_ms, 0.99),
        cer=cer,
        wer=word_error_rate(truth_text, extracted_text),
        truth_len=len(truth_text),
        ext_len=len(extracted_text),
    )
    # Record per-fixture CER to telemetry (engine-aware)
    telemetry.record_ocr_cer(cer, engine=ocr.name)
    return result


async def run_benchmark(
    fixtures_dir: Path,
    output_path: Path,
    iterations: int,
    preferred_engine: str,
) -> dict:
    if not fixtures_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixtures_dir}")

    fixture_paths = list(iter_fixture_images(fixtures_dir))
    if not fixture_paths:
        raise FileNotFoundError(
            f"No PNG fixtures with matching .txt ground truth found in {fixtures_dir}"
        )

    benchmark_config = _BenchmarkConfig(preferred_engine)
    ocr = OCREngine(benchmark_config)
    ocr._ensure_loaded()
    if not ocr._loaded:
        raise RuntimeError(
            "No OCR engine could be loaded. Install/configure a supported backend first."
        )

    results: list[FixtureResult] = []
    for fixture_path in fixture_paths:
        results.append(await benchmark_fixture(ocr, fixture_path, iterations))

    payload = summarize_results(ocr.name, iterations, results)
    payload["requested_engine"] = OCREngine._canonical_engine_name(preferred_engine)
    payload["available_backends"] = [name for name, _ in ocr._backends]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def compare_fixture_results(base: dict, candidate: dict) -> dict:
    cer_gain = base["cer"] - candidate["cer"]
    latency_penalty = candidate["latency_p50_ms"] - base["latency_p50_ms"]
    accuracy_winner = candidate["engine"] if candidate["cer"] < base["cer"] else base["engine"]
    latency_winner = candidate["engine"] if candidate["latency_p50_ms"] < base["latency_p50_ms"] else base["engine"]

    if cer_gain > 0.08 and latency_penalty < 80:
        recommended = candidate["engine"]
        reason = "accuracy_gain"
    elif cer_gain > 0.03 and latency_penalty < 35:
        recommended = candidate["engine"]
        reason = "balanced_gain"
    else:
        recommended = base["engine"]
        reason = "latency_guardrail"

    return {
        "filename": base["filename"],
        "base_engine": base["engine"],
        "candidate_engine": candidate["engine"],
        "base_latency_p50_ms": base["latency_p50_ms"],
        "candidate_latency_p50_ms": candidate["latency_p50_ms"],
        "base_cer": base["cer"],
        "candidate_cer": candidate["cer"],
        "cer_gain": cer_gain,
        "latency_penalty_ms": latency_penalty,
        "accuracy_winner": accuracy_winner,
        "latency_winner": latency_winner,
            "recommended_engine": recommended,
            "reason": reason,
    }


async def run_comparison(
    fixtures_dir: Path,
    output_path: Path,
    iterations: int,
    base_engine: str,
    candidate_engine: str,
) -> dict:
    # Logic issue 5 fix: use distinct paths for each leg so the candidate
    # result does not overwrite the base result on disk.
    base_out = output_path.parent / f"_tmp_base_{base_engine}.json"
    candidate_out = output_path.parent / f"_tmp_candidate_{candidate_engine}.json"
    base_payload = await run_benchmark(fixtures_dir, base_out, iterations, base_engine)
    candidate_payload = await run_benchmark(fixtures_dir, candidate_out, iterations, candidate_engine)

    base_by_file = {
        fixture["filename"]: {**fixture, "engine": base_payload["requested_engine"]}
        for fixture in base_payload["fixtures"]
    }
    candidate_by_file = {
        fixture["filename"]: {**fixture, "engine": candidate_payload["requested_engine"]}
        for fixture in candidate_payload["fixtures"]
    }

    comparisons = [
        compare_fixture_results(base_by_file[name], candidate_by_file[name])
        for name in sorted(base_by_file.keys())
        if name in candidate_by_file
    ]

    recommend_counts: dict[str, int] = {}
    for item in comparisons:
        recommend_counts[item["recommended_engine"]] = recommend_counts.get(item["recommended_engine"], 0) + 1

    payload = {
        "timestamp": time.time(),
        "iterations": iterations,
        "base": base_payload,
        "candidate": candidate_payload,
        "comparisons": comparisons,
        "summary": {
            "recommended_counts": recommend_counts,
            "candidate_better_accuracy_count": sum(1 for item in comparisons if item["accuracy_winner"] == candidate_payload["requested_engine"]),
            "candidate_faster_count": sum(1 for item in comparisons if item["latency_winner"] == candidate_payload["requested_engine"]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Clean up the per-leg temp files
    for tmp in (base_out, candidate_out):
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=FIXTURE_DIR,
        help="Directory containing .png fixtures and .png.txt ground truth files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the benchmark JSON report.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of OCR runs per fixture.",
    )
    parser.add_argument(
        "--engine",
        default="auto",
        help="Preferred OCR engine hint passed to OCREngine.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run a side-by-side backend comparison instead of a single-engine benchmark.",
    )
    parser.add_argument(
        "--base-engine",
        default="windows",
        help="Base engine for comparison mode.",
    )
    parser.add_argument(
        "--candidate-engine",
        default="paddle",
        help="Candidate engine for comparison mode.",
    )
    return parser


def print_summary(payload: dict) -> None:
    print(f"OCR engine: {payload['requested_engine']} (resolved primary: {payload['engine']})")
    print(f"Iterations per fixture: {payload['iterations']}")
    print("")
    for fixture in payload["fixtures"]:
        print(
            f"{fixture['filename']}: "
            f"p50={fixture['latency_p50_ms']:.1f}ms "
            f"p95={fixture['latency_p95_ms']:.1f}ms "
            f"p99={fixture['latency_p99_ms']:.1f}ms "
            f"CER={fixture['cer']:.3f} "
            f"WER={fixture['wer']:.3f}"
        )
    print("")
    overall = payload["overall"]
    print(
        "Overall: "
        f"p50={overall['latency_p50_ms']:.1f}ms "
        f"p95={overall['latency_p95_ms']:.1f}ms "
        f"avg_CER={overall['average_cer']:.3f}"
    )


def print_comparison_summary(payload: dict) -> None:
    base = payload["base"]
    candidate = payload["candidate"]
    print(
        f"Compare: {base['requested_engine']} vs {candidate['requested_engine']} "
        f"(resolved primaries: {base['engine']} vs {candidate['engine']})"
    )
    print(f"Iterations per fixture: {payload['iterations']}")
    print("")
    for item in payload["comparisons"]:
        print(
            f"{item['filename']}: "
            f"{item['base_engine']} CER={item['base_cer']:.3f} p50={item['base_latency_p50_ms']:.1f}ms | "
            f"{item['candidate_engine']} CER={item['candidate_cer']:.3f} p50={item['candidate_latency_p50_ms']:.1f}ms | "
            f"recommend={item['recommended_engine']} ({item['reason']})"
        )
    print("")
    summary = payload["summary"]
    print(
        "Summary: "
        f"candidate_better_accuracy={summary['candidate_better_accuracy_count']} "
        f"candidate_faster={summary['candidate_faster_count']} "
        f"recommended={summary['recommended_counts']}"
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be >= 1")

    if args.compare:
        output_path = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_COMPARE_OUTPUT
        payload = asyncio.run(
            run_comparison(
                fixtures_dir=args.fixtures,
                output_path=output_path,
                iterations=args.iterations,
                base_engine=args.base_engine,
                candidate_engine=args.candidate_engine,
            )
        )
        print_comparison_summary(payload)
        print(f"\nSaved comparison report to {output_path}")
        return 0

    payload = asyncio.run(
        run_benchmark(
            fixtures_dir=args.fixtures,
            output_path=args.output,
            iterations=args.iterations,
            preferred_engine=args.engine,
        )
    )
    print_summary(payload)
    print(f"\nSaved report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
