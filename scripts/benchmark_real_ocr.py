"""
Capture the current active window, run OCR, and compute CER against ground truth.

Usage (manual):
  1. Open a code file in your editor (VS Code, etc.)
  2. Run: python scripts/benchmark_real_ocr.py --output real_capture.png
  3. The script saves the screenshot and prints OCR text.
  4. Compare with the actual file content to compute CER.

Optional: provide ground truth file directly:
  python scripts/benchmark_real_ocr.py --truth path/to/file.py --output capture.png
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from core.config import Config
from capture.ocr import OCREngine
from capture.screen import ScreenCapture, ProcessUtils
from utils.telemetry import telemetry
import time

def character_error_rate(a: str, b: str) -> float:
    """Simple CER using edit distance."""
    if a == b:
        return 0.0
    exp = list(" ".join((a or "").split()))
    act = list(" ".join((b or "").split()))
    if not exp:
        return 0.0 if not act else 1.0
    prev = list(range(len(act) + 1))
    for i, ca in enumerate(exp, 1):
        cur = [i]
        for j, cb in enumerate(act, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return round(prev[-1] / len(exp), 3) if exp else 0.0

async def main():
    parser = argparse.ArgumentParser(description="Capture active window and evaluate OCR accuracy")
    parser.add_argument("--output", "-o", default="real_capture.png", help="Output image path")
    parser.add_argument("--truth", "-t", help="Ground truth text file (optional)")
    parser.add_argument("--show-text", action="store_true", help="Print OCR text to stdout")
    args = parser.parse_args()

    # Initialize OCR
    cfg = Config("config.yaml")
    ocr = OCREngine(cfg)
    ocr._ensure_loaded()
    if not ocr._loaded:
        print("ERROR: OCR engine not loaded. Ensure WinRT is available.")
        sys.exit(1)

    print(f"[RealOCR] Engine: {ocr.name}")
    print(f"[RealOCR] Capturing active window...")

    # Get active window rect using the same utility as ScreenCapture
    try:
        rect = ProcessUtils.get_active_window_rect()
        if rect is None:
            print("ERROR: Could not get active window. Is a window focused?")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR getting window rect: {e}")
        sys.exit(1)

    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    print(f"[RealOCR] Active window: {left},{top} {width}×{height}")

    # Capture using mss (same as ScreenCapture)
    import mss
    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(args.output)
        print(f"[RealOCR] Saved screenshot → {args.output}")

    # Run OCR
    print("[RealOCR] Running OCR...")
    start = time.perf_counter()
    text, boxes = await ocr.extract(img)
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"[RealOCR] OCR completed in {elapsed:.1f}ms, extracted {len(boxes)} boxes")

    if args.show_text:
        print("\n=== OCR TEXT ===")
        print(text or "<empty>")
        print("===============\n")

    # Load ground truth if provided
    if args.truth:
        truth_path = Path(args.truth)
        if truth_path.exists():
            truth = truth_path.read_text(encoding="utf-8", errors="replace")
            cer = character_error_rate(truth, text or "")
            print(f"\n[RealOCR] Ground truth: {args.truth}")
            print(f"[RealOCR] CER: {cer:.3f}")
            telemetry.record_ocr_cer(cer, engine=ocr.name)
        else:
            print(f"WARNING: Ground truth file not found: {args.truth}")

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
