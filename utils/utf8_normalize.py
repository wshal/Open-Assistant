"""
UTF-8 normalization helper (P3).

Fixes common mojibake that appears when UTF-8 bytes were decoded as cp1252.

Default mode is dry-run (prints which files would change). Use --apply to
rewrite files in-place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_PATHS = ["."]
EXTS = {".py", ".md", ".txt", ".yaml", ".yml"}
SKIP_DIRS = {"venv", ".git", "__pycache__", "data", "logs"}


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in EXTS:
            continue
        yield p


def looks_mojibake(text: str) -> bool:
    return any(s in text for s in ["â€”", "â€“", "â€™", "â€œ", "â€", "â€˜", "Ã"])


def fix_mojibake_cp1252(text: str) -> str:
    # Common reverse transformation: text that is actually UTF-8 bytes decoded as cp1252.
    try:
        candidate = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        # Fall back to a conservative replacement map (doesn't risk losing chars).
        repl = {
            "â€”": "—",
            "â€“": "–",
            "â€™": "’",
            "â€œ": "“",
            "â€": "”",
            "â€˜": "‘",
            "â€¦": "…",
        }
        out = text
        for k, v in repl.items():
            out = out.replace(k, v)
        return out
    return candidate


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    changed = 0
    for root_s in args.paths or []:
        root = Path(root_s)
        if root.is_file():
            files = [root]
        else:
            files = list(iter_files(root))
        for f in files:
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not looks_mojibake(raw):
                continue
            fixed = fix_mojibake_cp1252(raw)
            if fixed == raw:
                continue
            changed += 1
            if args.apply:
                try:
                    f.write_text(fixed, encoding="utf-8", errors="strict")
                    print(f"fixed: {f}")
                except Exception as e:
                    print(f"failed: {f} ({e})")
            else:
                print(f"would-fix: {f}")

    if not changed:
        print("OK: no files require normalization.")
    else:
        print(f"{'Fixed' if args.apply else 'Found'} {changed} file(s) needing normalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

