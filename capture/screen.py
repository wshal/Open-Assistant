"""
Screen capture with change detection — v4.1 (Layer 3 Hardened).
RESTORED: Dynamic Monitor Selection.
FIXED: Active region tracking for Smart Crop (Vision Focus).
"""

import asyncio
import io
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image
import mss
import ctypes
from PyQt6.QtCore import QObject, pyqtSignal
from capture.ocr import OCREngine
from utils.logger import setup_logger
from utils.platform_utils import ProcessUtils
from utils.telemetry import telemetry

logger = setup_logger(__name__)


@dataclass
class _WindowState:
    img_hash: str = ""
    text: str = ""
    last_ocr_time: float = 0.0
    last_seen: float = 0.0


class ScreenCapture(QObject):
    text_captured = pyqtSignal(str)

    def __init__(self, config, ocr: OCREngine):
        super().__init__()
        self.config = config
        self.ocr = ocr
        self._last_text = ""
        self._last_img_hash = ""
        self._last_ocr_time = 0.0
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

        # P1.1: Cache OCR/hash per active window (title-based by default).
        self._window_states: dict[str, _WindowState] = {}
        self._active_window_key: str = ""
        self._window_cache_size = int(config.get("capture.screen.window_cache_size", 8) or 8)
        self._key_by_window = bool(config.get("capture.screen.key_by_window", True))

        # P1.1: Stability gate for perceptual hash skipping.
        self._hash_threshold_bits = int(config.get("capture.screen.hash_threshold_bits", 2) or 2)
        stable_ttl_ms = config.get("capture.screen.stable_ttl_ms", 2000)
        self._stable_ttl_s = float(stable_ttl_ms) / 1000.0

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

    @property
    def last_img_hash(self) -> str:
        return self._last_img_hash

    def _get_window_key(self) -> str:
        if not getattr(self, "_key_by_window", True):
            return "global"
        try:
            title = ProcessUtils.get_active_window_title() or ""
        except Exception:
            title = ""
        title = (title or "").strip()
        return title[:200] if title else "global"

    def _evict_window_states_if_needed(self) -> None:
        states = getattr(self, "_window_states", None)
        if not isinstance(states, dict):
            return
        try:
            max_items = int(getattr(self, "_window_cache_size", 0) or 0)
        except Exception:
            max_items = 0
        if max_items <= 0 or len(states) <= max_items:
            return
        active = getattr(self, "_active_window_key", "")
        victims = sorted(
            (k for k in states.keys() if k != active),
            key=lambda k: states.get(k, _WindowState()).last_seen,
        )
        while len(states) > max_items and victims:
            states.pop(victims.pop(0), None)
            telemetry.record_cache_eviction("screen_window_state")

    def _sync_active_state(self) -> None:
        states = getattr(self, "_window_states", None)
        key = getattr(self, "_active_window_key", "")
        if not isinstance(states, dict) or not key:
            return
        states[key] = _WindowState(
            img_hash=getattr(self, "_last_img_hash", "") or "",
            text=getattr(self, "_last_text", "") or "",
            last_ocr_time=float(getattr(self, "_last_ocr_time", 0.0) or 0.0),
            last_seen=time.time(),
        )
        self._evict_window_states_if_needed()

    def _activate_window_state(self, window_key: str) -> None:
        # PyQt QObject instances constructed via __new__ (in tests) can raise if
        # we touch Qt-backed attributes before QObject.__init__.
        try:
            states = self.__dict__.get("_window_states", None)
        except Exception:
            states = None
        if not isinstance(states, dict):
            return
        window_key = window_key or "global"
        if window_key == getattr(self, "_active_window_key", ""):
            self._sync_active_state()
            return

        if getattr(self, "_active_window_key", ""):
            self._sync_active_state()

        state = states.get(window_key) or _WindowState()
        self._active_window_key = window_key
        self._last_img_hash = state.img_hash or ""
        self._last_text = state.text or ""
        self._last_ocr_time = float(state.last_ocr_time or 0.0)
        self._last_crop_box = None  # Reset crop box for new window
        self._sync_active_state()

    @staticmethod
    def _dhash(image: Image.Image, hash_size: int = 8) -> str:
        """Calculate a Difference Hash (dHash) for an image."""
        try:
            # Resize to (hash_size + 1) x hash_size, convert to grayscale
            img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            
            # Compare adjacent pixels
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left_pixel = pixels[row * (hash_size + 1) + col]
                    right_pixel = pixels[row * (hash_size + 1) + col + 1]
                    diff.append(left_pixel > right_pixel)
                    
            # Convert boolean array to hex string
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
                    decimal_value = 0
                    
            return ''.join(hex_string)
        except Exception as e:
            logger.error(f"Error computing dhash: {e}")
            return ""

    @staticmethod
    def _hash_distance(h1: str, h2: str) -> int:
        """Calculate the Hamming distance between two hex hashes."""
        if not h1 or not h2 or len(h1) != len(h2):
            return 999
        return sum(bin(int(c1, 16) ^ int(c2, 16)).count('1') for c1, c2 in zip(h1, h2))


    async def capture(self) -> Optional[str]:
        """Periodic capture — respects debounce interval and Smart Crop."""
        if not self._enabled:
            return None
        try:
            if isinstance(self.__dict__.get("_window_states", None), dict):
                self._activate_window_state(self._get_window_key())
        except Exception:
            pass
        now = time.time()
        if now - self._last_time < self._debounce:
            return None
        self._last_time = now

        try:
            full_img = await self._screenshot_async()
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

            # phase 1.5: Screen Diff (P1.1)
            # Compute perceptual hash to see if we can skip OCR entirely
            curr_hash = self._dhash(target_img)
            if curr_hash and self._last_img_hash:
                dist = self._hash_distance(curr_hash, self._last_img_hash)
                stable_recent = (now - float(getattr(self, "_last_ocr_time", 0.0) or 0.0)) < float(
                    getattr(self, "_stable_ttl_s", 0.0) or 0.0
                )
                if stable_recent and dist <= int(getattr(self, "_hash_threshold_bits", 2) or 2):
                    logger.debug(
                        f"[ScreenCapture P1.1] OCR SKIPPED — dhash distance={dist} ≤ "
                        f"{self._hash_threshold_bits} bits, screen visually unchanged "
                        f"(within {self._stable_ttl_s:.1f}s TTL)"
                    )
                    # Q11: Telemetry — track per-window TTL cache hits
                    try:
                        telemetry.record_screen_ocr(0.0, engine="cache")
                    except Exception:
                        pass
                    self._last_img_hash = curr_hash
                    self._sync_active_state()
                    return None  # No change, caller will use _last_text
                elif not stable_recent:
                    logger.debug(
                        f"[ScreenCapture P1.1] TTL expired — forcing OCR re-run "
                        f"(last_ocr={now - float(getattr(self, '_last_ocr_time', 0.0) or 0.0):.1f}s ago)"
                    )
                else:
                    logger.debug(
                        f"[ScreenCapture P1.1] OCR RUNNING — dhash distance={dist} > "
                        f"{self._hash_threshold_bits} bits (visual change detected)"
                    )

            self._last_img_hash = curr_hash or self._last_img_hash
            self._sync_active_state()

            # phase 2: Extraction
            text, boxes = await self.ocr.extract(target_img)
            self._last_ocr_time = now
            self._sync_active_state()

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
                self._sync_active_state()
                self.text_captured.emit(text)
                return text
        except Exception as e:
            logger.debug(f"Screen capture error: {e}")
        return None

    async def capture_now(self, emit_signal: bool = True) -> str:
        """Force immediate capture regardless of debounce."""
        if not self._enabled:
            return ""
        try:
            if isinstance(self.__dict__.get("_window_states", None), dict):
                self._activate_window_state(self._get_window_key())
        except Exception:
            pass
        img = await self._screenshot_async()
        if img:
            # Bug 7 fix: update _last_img_hash so the next periodic capture()
            # can correctly diff against this fresh frame instead of stale state.
            new_hash = self._dhash(img)
            if new_hash:
                self._last_img_hash = new_hash
            text, _ = await self.ocr.extract(img)
            if text:
                self._last_text = text
                self._last_ocr_time = time.time()
                self._sync_active_state()
                if emit_signal:
                    self.text_captured.emit(text)
                return text
        return ""

    async def capture_context(self) -> str:
        try:
            if isinstance(self.__dict__.get("_window_states", None), dict):
                self._activate_window_state(self._get_window_key())
        except Exception:
            pass
        text = await self.capture()
        if text is not None:
            return text
        return self._last_text

    async def capture_snapshot(self) -> Tuple[bytes, str]:
        """Capture one screenshot and derive OCR text from the same image."""
        if not self._enabled:
            return b"", ""
        try:
            if isinstance(self.__dict__.get("_window_states", None), dict):
                self._activate_window_state(self._get_window_key())
        except Exception:
            pass

        img = self._screenshot()
        if img is None:
            return b"", ""

        text, _ = await self.ocr.extract(img)
        if text:
            self._last_text = text
            self._last_ocr_time = time.time()
            self._sync_active_state()

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
            if isinstance(self.__dict__.get("_window_states", None), dict):
                self._activate_window_state(self._get_window_key())
        except Exception:
            pass

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            text, _ = await self.ocr.extract(img)
            if text:
                self._last_text = text
                self._last_ocr_time = time.time()
                self._sync_active_state()
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
            telemetry.record_roi(
                final_box[2] - final_box[0],
                final_box[3] - final_box[1],
                source="smart_crop_box",
            )

    async def _screenshot_async(self) -> Optional[Image.Image]:
        """Run _screenshot() in a thread executor so the async event loop
        is never blocked during the mss OS-level screen grab (~5-15 ms).

        Phase 2 async tuning: keeping the event loop free during I/O means
        other coroutines (OCR, AI streaming, audio) remain responsive.
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._screenshot)
        except Exception as e:
            logger.debug("Async screenshot error: %s", e)
            return None

    def _screenshot(self) -> Optional[Image.Image]:
        """Capture a screenshot as a PIL image (synchronous, runs in thread executor).

        Uses get_active_client_rect() (Phase 2) which strips the DWM shadow
        border and title bar, giving OCR only the true content pixels.
        Avoid calling Qt GUI APIs here — this runs off the main thread.
        """
        try:
            # Bug 3 fix: use context manager so OS handles are always released
            with mss.mss() as sct:
                # Phase 2 ROI: use client rect to exclude window chrome
                # (title bar ~30px, DWM shadow ~8px per side).
                # Falls back to outer window rect if DWM API is unavailable.
                active_rect = ProcessUtils.get_active_client_rect()
                if active_rect:
                    x, y, w, h = active_rect

                    screen_width, screen_height = ProcessUtils.get_primary_screen_size()

                    # Sanitize the window rect to clamp it within the primary display bounding box
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, screen_width - x)
                    h = min(h, screen_height - y)

                    # Only use the ROI if it's a valid size (e.g. greater than 50x50)
                    if w > 50 and h > 50:
                        bbox = {"top": y, "left": x, "width": w, "height": h}
                        telemetry.record_roi(w, h, source="active_window")
                        sct_img = sct.grab(bbox)
                        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                        return img

                # Fallback: Capture the entire screen containing the cursor
                monitors = getattr(sct, "monitors", []) or []
                if len(monitors) < 2:
                    target_monitor = monitors[0] if monitors else None
                else:
                    target_monitor = monitors[1]  # fallback: primary

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
                    pass

                if not target_monitor:
                    return None

                raw = sct.grab(target_monitor)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                telemetry.record_roi(img.width, img.height, source="monitor_capture")

                if img.width > self._max_w:
                    ratio = self._max_w / img.width
                    new_h = int(img.height * ratio)
                    img = img.resize((self._max_w, new_h), Image.LANCZOS)
                    telemetry.record_roi(img.width, img.height, source="downscaled_capture")

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
