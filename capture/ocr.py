"""
Multi-backend OCR - OpenAssist v4.1 (Midnight Hardened).
RESTORATION: Implemented native Windows OCR (WinRT) fallback.
"""

import asyncio
import io
import threading
from typing import List, Optional, Tuple

from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)


class OCREngine:
    def __init__(self, config):
        self.engine = None
        self.name = config.get("capture.screen.ocr_engine", "auto")
        self._loaded = False
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        with self._lock:
            if self._loaded:
                return

            # 1. Preferred: Windows Native WinRT OCR (High Speed, Sub-100ms)
            try:
                import winrt.windows.media.ocr as ocr
                from winrt.windows.globalization import Language
                
                if ocr.OcrEngine.is_language_supported(Language("en-US")):
                    self.engine = ocr.OcrEngine.try_create_from_user_profile_languages()
                    self.name = "winrt"
                    self._loaded = True
                    logger.info("  ✅ Windows Native OCR ready (Primary) — sub-100ms path active")
                    return
            except Exception as e:
                logger.debug(f"WinRT load failed: {e}")

            # 2. Fallback: EasyOCR (High Accuracy, slower)
            # P2.2 Lite: passes raw numpy arrays directly — skips any OpenCV encoding step
            try:
                import easyocr
                self.engine = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.name = "easyocr"
                self._loaded = True
                logger.info(
                    "  ✅ EasyOCR ready (Fallback) — P2.2 Lite: PIL→numpy direct path active"
                )
                return
            except ImportError:
                pass

            logger.warning("  No OCR engine detected. Run: pip install easyocr")

    async def extract(self, img: Image.Image) -> Tuple[Optional[str], List[Tuple[int, int, int, int]]]:
        """EXTRACT: Extracts text and bounding boxes (Vision Focus)."""
        self._ensure_loaded()
        if not self._loaded:
            logger.warning("[OCR] extract() called but no OCR engine loaded")
            return None, []

        import time as _time
        t0 = _time.perf_counter()

        try:
            if self.name == "winrt" and self.engine:
                result = await self._extract_winrt(img)
                elapsed = (_time.perf_counter() - t0) * 1000
                logger.debug(f"[OCR] WinRT extraction: {elapsed:.1f}ms, text_len={len(result[0] or '')}")
                return result
            
            elif self.name == "easyocr" and self.engine:
                import numpy as np
                # P2.2 Lite: PIL → numpy array directly — no OpenCV or JPEG encode/decode roundtrip
                arr = np.array(img)
                logger.debug(
                    f"[OCR] P2.2 Lite: PIL→numpy direct path, shape={arr.shape}, dtype={arr.dtype}"
                )
                result = await asyncio.to_thread(self._extract_sync_easyocr, arr)
                elapsed = (_time.perf_counter() - t0) * 1000
                logger.debug(f"[OCR] EasyOCR extraction: {elapsed:.1f}ms, text_len={len(result[0] or '')}")
                return result
                
        except Exception as e:
            logger.debug(f"OCR Runtime Error ({self.name}): {e}")
        return None, []

    def _extract_sync_easyocr(self, arr):
        result = self.engine.readtext(arr)
        if not result: return None, []
        text = "\n".join(r[1] for r in result)
        boxes = []
        for res in result:
            pts = res[0]
            boxes.append((int(pts[0][0]), int(pts[0][1]), int(pts[2][0]), int(pts[2][1])))
        return text, boxes

    async def _extract_winrt(self, img: Image.Image) -> Tuple[Optional[str], List[Tuple[int, int, int, int]]]:
        """NATIVE RESTORATION: Windows.Media.Ocr logic with box tracking."""
        try:
            import winrt.windows.graphics.imaging as imaging
            import winrt.windows.storage.streams as streams
            import io

            byte_io = io.BytesIO()
            img.save(byte_io, format='PNG')
            byte_io.seek(0)
            data = byte_io.getvalue()
            logger.debug(
                f"[OCR] WinRT: encoding PIL image → PNG bytes ({len(data)} bytes) "
                "for WinRT IInputStream"
            )
            stream = streams.InMemoryRandomAccessStream()
            writer = streams.DataWriter(stream)
            writer.write_bytes(data)
            await writer.store_async()
            stream.seek(0)
            
            decoder = await imaging.BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            
            result = await self.engine.recognize_async(bitmap)
            if not result or not result.lines:
                logger.debug("[OCR] WinRT: no text lines detected")
                return None, []
            
            text = "\n".join(line.text for line in result.lines)
            boxes = []
            for line in result.lines:
                for word in line.words:
                    b = word.bounding_rect
                    boxes.append((int(b.x), int(b.y), int(b.x + b.width), int(b.y + b.height)))
            return text, boxes
        except Exception as e:
            logger.warning(f"WinRT OCR Failed: {e}")
        return None, []
