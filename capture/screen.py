"""
Screen capture with change detection — v4.1 (Layer 3 Hardened).
RESTORED: Dynamic Monitor Selection.
FIXED: Active region tracking for Smart Crop (Vision Focus).
"""

import io
import time
from typing import Optional, Tuple
from PIL import Image
import mss
import ctypes
from PyQt6.QtCore import QObject, pyqtSignal
from capture.ocr import OCREngine
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ScreenCapture(QObject):
    text_captured = pyqtSignal(str)

    def __init__(self, config, ocr: OCREngine):
        super().__init__()
        self.config = config
        self.ocr = ocr
        self._last_text = ""
        self._last_time = 0.0
        # Prefer the explicit screen capture interval setting (P1.7). Fall back
        # to the legacy debounce key for backwards compatibility.
        interval_ms = config.get("capture.screen.interval_ms", None)
        if interval_ms is None:
            interval_ms = config.get("performance.debounce_ms", 400)
        self._debounce = float(interval_ms) / 1000.0
        self._threshold = config.get("capture.screen.change_threshold", 0.15)

        # Image quality affects capture resolution AND JPEG compression (P2.2)
        quality = config.get("capture.screen.quality", "medium")
        quality_widths = {"low": 1024, "medium": 1920, "high": 2560}
        quality_jpeg = {"low": 40, "medium": 72, "high": 95}
        self._max_w = quality_widths.get(quality, 1920)
        self._jpeg_quality = quality_jpeg.get(quality, 72)

        # Separate (lower) defaults for manual screenshot analysis to reduce latency.
        analysis_quality = config.get("capture.screen.analysis_quality", quality)
        self._analysis_max_w = quality_widths.get(analysis_quality, self._max_w)
        self._analysis_jpeg_quality = quality_jpeg.get(analysis_quality, self._jpeg_quality)

        self._smart_crop_enabled = config.get("capture.screen.smart_crop", True)
        self._enabled = config.get("capture.screen.enabled", True)

        # RESTORATION: Active Text Region Tracking
        self._last_crop_box = None  # (left, top, right, bottom)
        self._crop_hits = 0
        self._crop_misses = 0

    def initialize(self):
        """Standardized warm-up hook called by the controller.

        OCR engine is lazy-loaded on the first extract() call via _ensure_loaded().
        We do NOT force-load it here so startup is non-blocking.
        The deferred warmup thread calls this to pre-load OCR in the background.
        """
        self.ocr._ensure_loaded()
        logger.info("  ✅ Screen capture pipeline ready")


    async def capture(self) -> Optional[str]:
        """Periodic capture — respects debounce interval and Smart Crop."""
        if not self._enabled:
            return None
        now = time.time()
        if now - self._last_time < self._debounce:
            return None
        self._last_time = now

        try:
            full_img = self._screenshot()
            if full_img is None:
                return None

            # phase 1: Smart Crop Focus
            target_img = full_img
            offset_x, offset_y = 0, 0
            if self._smart_crop_enabled and self._last_crop_box:
                # We crop with 5% padding for movement safety
                target_img = full_img.crop(self._last_crop_box)
                offset_x, offset_y = self._last_crop_box[0], self._last_crop_box[1]
                logger.debug("Vision: Focused capture using Smart Crop")

            # phase 2: Extraction
            text, boxes = await self.ocr.extract(target_img)

            # phase 3: Hysteresis Logic (Update the focus zone)
            if text and len(text.strip()) > 10:
                self._update_crop_box(boxes, offset_x, offset_y, full_img.size)
                self._crop_hits += 1
                self._crop_misses = 0
            else:
                self._crop_misses += 1
                if self._crop_misses > 3:  # Reset if no text for 3 frames
                    self._last_crop_box = None
                    self._crop_hits = 0

            # phase 4: Change Detection
            if text and self._has_changed(text):
                self._last_text = text
                self.text_captured.emit(text)
                return text
        except Exception as e:
            logger.debug(f"Screen capture error: {e}")
        return None

    async def capture_now(self, emit_signal: bool = True) -> str:
        """Force immediate capture regardless of debounce."""
        if not self._enabled:
            return ""
        img = self._screenshot()
        if img:
            text, _ = await self.ocr.extract(img)
            if text:
                self._last_text = text
                if emit_signal:
                    self.text_captured.emit(text)
                return text
        return ""

    async def capture_context(self) -> str:
        text = await self.capture()
        if text is not None:
            return text
        return self._last_text

    async def capture_snapshot(self) -> Tuple[bytes, str]:
        """Capture one screenshot and derive OCR text from the same image."""
        if not self._enabled:
            return b"", ""

        img = self._screenshot()
        if img is None:
            return b"", ""

        text, _ = await self.ocr.extract(img)
        if text:
            self._last_text = text

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), text or ""

    async def capture_image_bytes(self, for_analysis: bool = False) -> bytes:
        """Capture one screenshot as JPEG bytes without OCR.

        When `for_analysis=True`, uses a lower default quality and (if available)
        Smart-Crop focus region to reduce latency for vision providers.
        """
        if not self._enabled:
            return b""

        img = self._screenshot()
        if img is None:
            return b""

        if for_analysis and self._smart_crop_enabled and self._last_crop_box:
            try:
                img = img.crop(self._last_crop_box)
            except Exception:
                pass

        # Downscale again for analysis if configured smaller than the default capture tier.
        if for_analysis and img.width > self._analysis_max_w:
            ratio = self._analysis_max_w / img.width
            new_h = int(img.height * ratio)
            img = img.resize((self._analysis_max_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        # P2.2: Use JPEG with quality tier for faster network transfer and less memory
        q = self._analysis_jpeg_quality if for_analysis else self._jpeg_quality
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        return buf.getvalue()

    async def extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        """Run OCR over an existing screenshot blob."""
        if not image_bytes:
            return ""

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            text, _ = await self.ocr.extract(img)
            if text:
                self._last_text = text
            return text or ""
        except Exception as e:
            logger.debug(f"Snapshot OCR error: {e}")
            return ""

    def _update_crop_box(self, boxes, offset_x, offset_y, screen_size):
        """Logic to calculate target region based on active text hits."""
        if not boxes:
            return

        # 1. Translate local box hits to full screen coordinates
        abs_boxes = [
            (b[0] + offset_x, b[1] + offset_y, b[2] + offset_x, b[3] + offset_y)
            for b in boxes
        ]

        # 2. Find the bounding hull of all text
        min_x = min(b[0] for b in abs_boxes)
        min_y = min(b[1] for b in abs_boxes)
        max_x = max(b[2] for b in abs_boxes)
        max_y = max(b[3] for b in abs_boxes)

        # 3. Add 10% padding for visual context and movement
        pad_w = int((max_x - min_x) * 0.1)
        pad_h = int((max_y - min_y) * 0.1)

        # 4. Clamp to screen bounds
        sw, sh = screen_size
        final_box = (
            max(0, min_x - pad_w),
            max(0, min_y - pad_h),
            min(sw, max_x + pad_w),
            min(sh, max_y + pad_h),
        )

        # 5. Stability check: Only crop if the region is significant but not the whole screen
        # We want to focus on editors/interviews, not just a single word.
        if (final_box[2] - final_box[0]) > 50 and (final_box[3] - final_box[1]) > 50:
            self._last_crop_box = final_box

    def _screenshot(self) -> Optional[Image.Image]:
        """Capture a screenshot as a PIL image.

        Important: Avoid calling Qt GUI APIs here. This method can be executed from
        the background asyncio thread (e.g., Analyze Screen), and Qt widgets are
        not thread-safe. We instead select the monitor by cursor position.
        """
        try:
            sct = mss.mss()
            monitors = getattr(sct, "monitors", []) or []
            if len(monitors) < 2:
                # mss uses index 0 as the virtual "all monitors" region.
                target_monitor = monitors[0] if monitors else None
            else:
                target_monitor = monitors[1]  # fallback: primary

            # Prefer the monitor under the cursor (thread-safe via Win32).
            try:
                class _POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                pt = _POINT()
                if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                    cx, cy = int(pt.x), int(pt.y)
                    for m in monitors[1:]:
                        left, top = int(m.get("left", 0)), int(m.get("top", 0))
                        width, height = int(m.get("width", 0)), int(m.get("height", 0))
                        if left <= cx < (left + width) and top <= cy < (top + height):
                            target_monitor = m
                            break
            except Exception:
                # Cursor-based selection is best-effort; primary fallback remains.
                pass

            if not target_monitor:
                return None

            raw = sct.grab(target_monitor)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

            if img.width > self._max_w:
                ratio = self._max_w / img.width
                new_h = int(img.height * ratio)
                img = img.resize((self._max_w, new_h), Image.LANCZOS)

            return img
        except Exception as e:
            logger.debug(f"Screenshot error: {e}")
            return None

    def _has_changed(self, new_text: str) -> bool:
        """Character-level Jaccard similarity for code-edit sensitivity."""
        if not self._last_text:
            return True
        if not new_text:
            return False

        def get_grams(t):
            t = t.lower()
            return set(t[i : i + 3] for i in range(len(t) - 2))

        old_grams = get_grams(self._last_text)
        new_grams = get_grams(new_text)
        union = old_grams | new_grams
        if not union:
            return False

        similarity = len(old_grams & new_grams) / len(union)
        return similarity < (1.0 - self._threshold)
