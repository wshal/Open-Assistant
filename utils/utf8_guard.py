"""
UTF-8 / mojibake guard (P3).

This does NOT rewrite files. It scans for common mojibake sequences that usually
indicate encoding drift (e.g. "—", "’", "Ã…").

Usage (PowerShell):
  python utils/utf8_guard.py
  python utils/utf8_guard.py --paths ai core ui utils
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_PATHS = ["ai", "core", "ui", "utils", "modes", "capture", "stealth"]

SUSPICIOUS = [
    "—",
    "–",
    "’",
    "“",
    "â€",
    "â€˜",
    "â€‹",
    "Ã",
]


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == "utf8_guard.py":
            continue
        if p.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml"}:
            continue
        if any(part in {"venv", ".git", "__pycache__", "data", "logs"} for part in p.parts):
            continue
        yield p


def scan_file(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return hits
    for i, line in enumerate(text.splitlines(), start=1):
        if any(s in line for s in SUSPICIOUS):
            hits.append((i, line.strip()[:200]))
    return hits


def main() -> int:
    # Avoid crashes on Windows consoles with non-UTF-8 code pages.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if suspicious sequences are found.",
    )
    args = ap.parse_args()

    roots = [Path(p) for p in (args.paths or [])]
    any_hits = False
    for root in roots:
        if not root.exists():
            continue
        for f in iter_files(root):
            hits = scan_file(f)
            if not hits:
                continue
            any_hits = True
            for line_no, preview in hits[:20]:
                print(f"{f}:{line_no}: {preview}")
            if len(hits) > 20:
                print(f"{f}: (+{len(hits) - 20} more)")

    if any_hits:
        print("\nFound suspicious mojibake sequences. Consider normalizing those files to UTF-8.")
        return 1 if args.strict else 0
    print("OK: no suspicious mojibake sequences found in scanned paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
