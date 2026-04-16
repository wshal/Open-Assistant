"""
Multi-backend OCR - OpenAssist v4.1 (Midnight Hardened).
RESTORATION: Implemented native Windows OCR (WinRT) fallback.
"""

from typing import Optional
import threading
import asyncio
from PIL import Image
from utils.logger import setup_logger
import numpy as np
import io

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

            # 1. Preferred: EasyOCR (High Accuracy)
            try:
                import easyocr
                self.engine = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.name = "easyocr"
                self._loaded = True
                logger.info("  ✅ EasyOCR ready (lazy loaded)")
                return
            except ImportError:
                pass

            # 2. Fallback: Windows Native WinRT OCR
            try:
                import winrt.windows.media.ocr as ocr
                import winrt.windows.graphics.imaging as imaging
                import winrt.windows.storage.streams as streams
                
                # Check for English support
                from winrt.windows.globalization import Language
                if ocr.OcrEngine.is_language_supported(Language("en-US")):
                    self.engine = ocr.OcrEngine.try_create_from_user_profile_languages()
                    self.name = "winrt"
                    self._loaded = True
                    logger.info("  ✅ Windows Native OCR ready (fallback)")
                    return
            except ImportError:
                pass

            logger.warning("  No OCR engine detected. Run: pip install easyocr")

    def extract(self, img: Image.Image) -> Tuple[Optional[str], List[Tuple[int, int, int, int]]]:
        """EXTRACT: Extracts text and bounding boxes (Vision Focus)."""
        self._ensure_loaded()
        if not self._loaded: return None, []

        try:
            if self.name == "easyocr" and self.engine:
                arr = np.array(img)
                result = self.engine.readtext(arr)
                if not result: return None, []
                text = "\n".join(r[1] for r in result)
                # Box: [ [x,y], [x,y], [x,y], [x,y] ] -> (min_x, min_y, max_x, max_y)
                boxes = []
                for res in result:
                    pts = res[0]
                    boxes.append((int(pts[0][0]), int(pts[0][1]), int(pts[2][0]), int(pts[2][1])))
                return text, boxes
            
            elif self.name == "winrt" and self.engine:
                return asyncio.run(self._extract_winrt(img))
                
        except Exception as e:
            logger.debug(f"OCR Runtime Error ({self.name}): {e}")
        return None, []

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
            stream = streams.InMemoryRandomAccessStream()
            writer = streams.DataWriter(stream)
            writer.write_bytes(data)
            await writer.store_async()
            stream.seek(0)
            
            decoder = await imaging.BitmapDecoder.create_async(stream)
            bitmap = await decoder.get_software_bitmap_async()
            
            result = await self.engine.recognize_async(bitmap)
            if not result or not result.lines: return None, []
            
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
