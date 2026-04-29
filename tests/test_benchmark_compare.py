import unittest

from benchmarks.capture_benchmark import compare_fixture_results, summarize_results, FixtureResult


class BenchmarkCompareTests(unittest.TestCase):
    def test_compare_prefers_candidate_when_accuracy_gain_outweighs_latency(self):
        result = compare_fixture_results(
            {
                "filename": "react.png",
                "engine": "winrt",
                "latency_p50_ms": 20.0,
                "cer": 0.50,
            },
            {
                "filename": "react.png",
                "engine": "paddle",
                "latency_p50_ms": 45.0,
                "cer": 0.30,
            },
        )

        self.assertEqual(result["recommended_engine"], "paddle")
        self.assertEqual(result["reason"], "accuracy_gain")

    def test_compare_prefers_base_when_latency_penalty_is_too_high(self):
        result = compare_fixture_results(
            {
                "filename": "terminal.png",
                "engine": "winrt",
                "latency_p50_ms": 15.0,
                "cer": 0.20,
            },
            {
                "filename": "terminal.png",
                "engine": "paddle",
                "latency_p50_ms": 2200.0,
                "cer": 0.18,
            },
        )

        self.assertEqual(result["recommended_engine"], "winrt")
        self.assertEqual(result["reason"], "latency_guardrail")

    def test_summarize_results_builds_overall_metrics(self):
        payload = summarize_results(
            "winrt",
            3,
            [
                FixtureResult("a.png", 10.0, 12.0, 13.0, 0.10, 0.20, 100, 90),
                FixtureResult("b.png", 20.0, 25.0, 30.0, 0.20, 0.30, 100, 80),
            ],
        )

        self.assertEqual(payload["engine"], "winrt")
        self.assertEqual(payload["iterations"], 3)
        self.assertEqual(payload["overall"]["latency_p50_ms"], 10.0)
        self.assertAlmostEqual(payload["overall"]["average_cer"], 0.15)


if __name__ == "__main__":
    unittest.main()
