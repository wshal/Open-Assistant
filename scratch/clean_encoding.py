import os
import re
from pathlib import Path

# Mapping of corrupted CP1252 sequences back to their intended UTF-8 characters
REPLACEMENTS = {
    "─": "─", "┬": "┬", "┘": "┘", "┴": "┴", "┐": "┐", "┌": "┌", "│": "│",
    "🚀": "🚀", "✅": "✅", "🧠": "🧠", "🔌": "🔌", "🧙": "🧙", "⚙️": "⚙️",
    "🎙️": "🎙️", "📺": "📺", "🗣️": "🗣️", "💡": "💡", "✓": "✓", "⏳": "⏳",
    "🤖": "🤖", "🎯": "🎯", "📋": "📋", "💻": "💻", "✍️": "✍️", "📍": "📍",
    "👻": "👻", "▶": "▶", "❌": "❌", "🔄": "🔄", "💬": "💬", "📋": "📋",
    "ℹ️": "ℹ️", "⚠️": "⚠️", "📚": "📚", "🔭": "🔭", "⌨️": "⌨️",
    "●": "●", "○": "○", "🗳️": "🗳️"
}

def clean_file(path):
    try:
        content = path.read_text(encoding='utf-8')
        original = content
        for bad, good in REPLACEMENTS.items():
            content = content.replace(bad, good)
        
        if content != original:
            path.write_text(content, encoding='utf-8')
            return True
    except Exception as e:
        print(f"Error cleaning {path}: {e}")
    return False

def main():
    root = Path(".")
    cleaned_count = 0
    for f in root.rglob("*.py"):
        if "venv" in str(f) or "__pycache__" in str(f):
            continue
        if clean_file(f):
            cleaned_count += 1
            print(f"Cleaned: {f}")
    
    # Also clean some specifically known non-py files
    for f in [root / "requirements.txt", root / "config.yaml"]:
        if f.exists() and clean_file(f):
            cleaned_count += 1
            print(f"Cleaned: {f}")

    print(f"\nTotal files cleaned: {cleaned_count}")

if __name__ == "__main__":
    main()
