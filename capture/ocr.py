"""
Simplified single-backend OCR engine.

Design:
  - Primary engine only (WinRT on Windows). No fallback chain.
  - Optional editor crop retry: if image looks like code and has window chrome,
    crop out the title bar and re-run OCR on the cropped region.
  - Full telemetry integration for latency and backend usage.
"""

from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)


Box = Tuple[int, int, int, int]


@dataclass
class OCRResult:
    text: Optional[str]
    boxes: List[Box]
    backend: str


class OCREngine:
    """Single-backend OCR engine with optional editor-crop retry."""

    def __init__(self, config):
        self.config = config
        self.engine = None
        self.name = self._canonical_engine_name(
            config.get("capture.screen.ocr_engine", "windows")
        )
        self.enable_editor_recrop = bool(
            config.get("capture.screen.ocr_editor_recrop", False)
        )
        self._editor_top_crop_px = int(
            config.get("capture.screen.ocr_editor_top_crop_px", 60) or 60
        )
        self._editor_left_crop_px = int(
            config.get("capture.screen.ocr_editor_left_crop_px", 70) or 70
        )
        self.last_backend_used = ""
        self._loaded = False
        self._lock = threading.Lock()
        self._backends: list[tuple[str, Any]] = []  # Populated by _ensure_loaded()

    # ── Engine name normalization ──────────────────────────────────────────

    @staticmethod
    def _canonical_engine_name(name: Optional[str]) -> str:
        raw = str(name or "windows").strip().lower()
        aliases = {
            "auto": "winrt",
            "windows": "winrt",
            "windows_native": "winrt",
            "winrt": "winrt",
        }
        return aliases.get(raw, raw)

    # ── Backend loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the configured OCR backend (WinRT only)."""
        with self._lock:
            if self._loaded:
                return
            backend = self._try_load_winrt()
            if backend is None:
                logger.error("[OCR] WinRT backend unavailable — OCR disabled")
                self._loaded = False
                return
            self.engine = backend
            self._backends = [(self.name, backend)]
            self._loaded = True
            logger.info("[OCR] Windows OCR ready as primary backend")

    def _try_load_winrt(self):
        try:
            import ctypes
            # Ensure COM is initialized on this thread (required for WinRT in thread pools)
            ctypes.windll.ole32.CoInitialize(0)
            
            import winrt.windows.media.ocr as ocr
            from winrt.windows.globalization import Language

            if not ocr.OcrEngine.is_language_supported(Language("en-US")):
                logger.warning("[OCR] English language pack not available for WinRT")
                return None
            engine = ocr.OcrEngine.try_create_from_user_profile_languages()
            if engine is not None:
                logger.info("[OCR] Windows OCR ready")
            return engine
        except Exception as exc:
            logger.error("[OCR] WinRT load failed: %s", exc)
            return None

    # ── Public API ──────────────────────────────────────────────────────────

    async def extract(self, img: Image.Image) -> Tuple[Optional[str], List[Box]]:
        """Extract text and bounding boxes from an image."""
        self._ensure_loaded()
        if not self._loaded or self.engine is None:
            logger.warning("[OCR] extract() called but no OCR backend is loaded")
            return None, []

        from utils.telemetry import telemetry as _tel

        start = asyncio.get_running_loop().time()
        try:
            result = await self._run_backend(self.name, self.engine, img)

            # Optional crop-retry on same backend for editor windows.
            # Must run BEFORE the empty-text bail so that a blank primary pass
            # can still recover text from the cropped content region. (Bug 2 fix)
            if self.enable_editor_recrop:
                recropped = await self._maybe_retry_with_cropped_image(img, result)
                if recropped is not None:
                    result = recropped

            if not self._result_has_text(result):
                _tel.record_ocr_backend(result.backend, outcome="failure")
                return None, []

            self.last_backend_used = result.backend
            _tel.record_ocr_backend(result.backend, outcome="success")
            return result.text, result.boxes

        except Exception as exc:
            logger.debug("[OCR] Runtime error: %s", exc)
            return None, []
        finally:
            elapsed_ms = (asyncio.get_running_loop().time() - start) * 1000.0
            _tel.record_screen_ocr(elapsed_ms, engine=self.last_backend_used or self.name)

    async def _run_backend(self, backend_name: str, backend: Any, img: Image.Image) -> OCRResult:
        if backend_name == "winrt":
            text, boxes = await self._extract_winrt(backend, img)
        else:
            text, boxes = None, []
        return OCRResult(text=text, boxes=boxes, backend=backend_name)

    @staticmethod
    def _result_has_text(result: OCRResult) -> bool:
        return bool((result.text or "").strip())

    # ── Editor crop retry ────────────────────────────────────────────────────

    async def _maybe_retry_with_cropped_image(
        self,
        img: Image.Image,
        primary: OCRResult,
    ) -> Optional[OCRResult]:
        crop_box = self._suggest_editor_crop_box(img, primary.text or "")
        if crop_box is None:
            return None

        cropped = img.crop(crop_box)
        retry = await self._run_backend(self.name, self.engine, cropped)
        if not self._result_has_text(retry):
            return None

        offset_x, offset_y = crop_box[0], crop_box[1]
        retry.boxes = [
            (left + offset_x, top + offset_y, right + offset_x, bottom + offset_y)
            for left, top, right, bottom in retry.boxes
        ]

        # Compare quality scores; return better result
        prefer_code = self._image_looks_code_like(img)
        primary_score = self._result_score(primary.text or "", prefer_code)
        retry_score = self._result_score(retry.text or "", prefer_code)
        return retry if retry_score >= primary_score else None

    def _suggest_editor_crop_box(
        self,
        img: Image.Image,
        text: str,
    ) -> Optional[tuple[int, int, int, int]]:
        if not self._image_looks_code_like(img):
            return None
        if not self.enable_editor_recrop:
            return None
        if img.width < 420 or img.height < 180:
            return None

        left = min(max(0, self._editor_left_crop_px), max(0, img.width - 80))
        top = min(max(0, self._editor_top_crop_px), max(0, img.height - 80))
        # Bug 8 fix: use `or` — a single non-zero axis (e.g. title-bar only) is a valid crop
        if left <= 0 and top <= 0:
            return None
        if img.width - left < 160 or img.height - top < 80:
            return None
        return (left, top, img.width, img.height)


    @staticmethod
    def _image_looks_code_like(img: Image.Image) -> bool:
        try:
            tiny = img.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            pixels = list(tiny.tobytes())
            if not pixels:
                return False
            mean_luma = sum(pixels) / len(pixels)
            dark_ratio = sum(1 for px in pixels if px < 90) / len(pixels)
            return mean_luma < 120 or dark_ratio > 0.35
        except Exception:
            return False

    @staticmethod
    def _syntax_density(text: str) -> float:
        if not text:
            return 0.0
        syntax_chars = "{}[]()<>:=;._/\\'\"`-"
        count = sum(1 for ch in text if ch in syntax_chars)
        return count / max(len(text), 1)

    @staticmethod
    def _noise_penalty(text: str) -> int:
        if not text:
            return 0
        penalty = 0
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:4]:
            lowered = line.lower()
            if "vs code" in lowered or "powershell" in lowered or "open assist" in lowered:
                penalty += 20
            if line.isdigit():
                penalty += 8
        penalty += sum(8 for line in lines[:6] if line.isdigit())
        return penalty

    @staticmethod
    def _timestamp_like_hits(text: str) -> int:
        hits = 0
        for token in text.split():
            if len(token) >= 8 and token.count("-") >= 2:
                hits += 1
        return hits

    @staticmethod
    def _code_keyword_hits(text: str) -> int:
        keywords = (
            "const", "function", "return", "className", "useState", "interface", "type",
            "SELECT", "FROM", "async", "await", "def ", "var", "let", "import", "export",
            "class", "if", "else", "for", "while", "switch", "case", "break", "continue",
            "try", "catch", "throw", "new", "this", "super", "extends", "get", "set"
        )
        return sum(1 for token in keywords if token in text)

    def _result_score(self, text: str, prefer_code: bool) -> float:
        normalized = text.strip()
        if not normalized:
            return 0.0
        lines = [line for line in normalized.splitlines() if line.strip()]
        keyword_hits = self._code_keyword_hits(normalized)
        score = len(normalized) + (len(lines) * 12) + (keyword_hits * 16)
        if prefer_code:
            score += self._syntax_density(normalized) * 800
        score -= self._noise_penalty(normalized)
        return score

    # ── WinRT extract ────────────────────────────────────────────────────────

    async def _extract_winrt(self, engine, img: Image.Image) -> Tuple[Optional[str], List[Box]]:
        try:
            import winrt.windows.graphics.imaging as imaging
            import winrt.windows.storage.streams as streams

            byte_io = io.BytesIO()
            img.save(byte_io, format="PNG")
            byte_io.seek(0)
            data = byte_io.getvalue()

            stream = streams.InMemoryRandomAccessStream()
            writer = streams.DataWriter(stream)
            writer.write_bytes(data)
            await writer.store_async()
            stream.seek(0)

            decoder = await imaging.BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()

            result = await engine.recognize_async(bitmap)
            if not result or not result.lines:
                return None, []

            text = "\n".join(line.text for line in result.lines)
            boxes: list[Box] = []
            for line in result.lines:
                for word in line.words:
                    b = word.bounding_rect
                    boxes.append((int(b.x), int(b.y), int(b.x + b.width), int(b.y + b.height)))
            return text, boxes
        except Exception as exc:
            logger.warning("[OCR] WinRT OCR failed: %s", exc)
            return None, []
