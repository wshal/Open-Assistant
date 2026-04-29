"""
Tests for OCR engine and telemetry (simplified single-backend version).

Validates:
  - Single backend (WinRT) loading
  - Editor crop retry logic
  - Heuristics: syntax density, noise penalty, code detection
  - Telemetry counters and summary
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

from capture.ocr import OCRResult, OCREngine
from capture.screen import ScreenCapture, _WindowState
from utils.telemetry import Telemetry


class ConfigStub:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def get(self, path, default=None):
        return self.settings.get(path, default)


class OCREngineSingleBackendTests(unittest.TestCase):
    """Tests for single-backend OCR with optional crop retry."""

    def test_requested_engine_normalized_to_winrt(self):
        engine = OCREngine(ConfigStub({"capture.screen.ocr_engine": "windows"}))
        self.assertEqual(engine.name, "winrt")

    def test_single_backend_order_is_only_requested_engine(self):
        engine = OCREngine(ConfigStub())
        # No multi-backend order; just verify the name is winrt
        self.assertEqual(engine.name, "winrt")

    def test_editor_crop_is_suggested_for_dark_window_with_titlebar_noise(self):
        engine = OCREngine(ConfigStub({"capture.screen.ocr_editor_recrop": True}))
        crop = engine._suggest_editor_crop_box(
            Image.new("RGB", (800, 300), "#0f172a"),
            "VS Code - Open Assist\n1\n2\n3\nconst ready = true;",
        )
        self.assertEqual(crop, (70, 60, 800, 300))

    def test_noise_penalty_prefers_cropped_text_without_window_chrome(self):
        engine = OCREngine(ConfigStub())
        noisy = "VS Code - Open Assist\n1\n2\n3\nconst ready = true;"
        clean = "const ready = true;\nreturn ready;"
        self.assertGreater(engine._result_score(clean, True), engine._result_score(noisy, True))

    def test_image_looks_code_like_dark_background(self):
        img = Image.new("RGB", (800, 600), "#0f172a")
        self.assertTrue(OCREngine._image_looks_code_like(img))

    def test_syntax_density_calculation(self):
        text = "{}[]()<>=;._/\\'\"`-"
        density = OCREngine._syntax_density(text)
        self.assertAlmostEqual(density, 1.0, places=1)

    def test_code_keyword_hits(self):
        text = "const x = async function() { return await fetch(); };"
        hits = OCREngine._code_keyword_hits(text)
        self.assertGreaterEqual(hits, 3)


class TelemetryTests(unittest.TestCase):
    def test_summary_includes_ocr_roi_and_eviction_metrics(self):
        telemetry = Telemetry()
        telemetry.record_cache_hit(1)
        telemetry.record_cache_miss()
        telemetry.record_screen_ocr(18.5, engine="winrt")
        telemetry.record_ocr_backend("winrt", outcome="success")
        telemetry.record_roi(640, 480, source="active_window")
        telemetry.record_cache_eviction("screen_window_state")

        summary = telemetry.summary()

        self.assertEqual(summary["cache"]["hits_t1"], 1)
        self.assertEqual(summary["cache"]["misses"], 1)
        self.assertEqual(summary["ocr"]["backend_success"]["winrt"], 1)
        # Fallback and failure counters should be empty (no fallback chain)
        self.assertEqual(summary["ocr"]["backend_fallback_success"], {})
        self.assertEqual(summary["ocr"]["backend_failures"], {})
        self.assertEqual(summary["roi"]["sources"]["active_window"], 1)
        self.assertEqual(summary["cache"]["evictions"]["screen_window_state"], 1)


class ScreenCaptureTelemetryTests(unittest.TestCase):
    def test_window_state_eviction_records_telemetry(self):
        capture = ScreenCapture.__new__(ScreenCapture)
        capture._window_states = {
            "a": _WindowState(last_seen=1.0),
            "b": _WindowState(last_seen=2.0),
            "c": _WindowState(last_seen=3.0),
        }
        capture._window_cache_size = 2
        capture._active_window_key = "c"

        with patch("capture.screen.telemetry.record_cache_eviction") as record_eviction:
            ScreenCapture._evict_window_states_if_needed(capture)

        self.assertEqual(len(capture._window_states), 2)
        record_eviction.assert_called_once_with("screen_window_state")


if __name__ == "__main__":
    unittest.main()
