"""Build standalone EXE using PyInstaller."""

import os
import sys
import subprocess
from pathlib import Path


def build():
    print("Building OpenAssist AI EXE...")

    spec_path = Path("OpenAssist-AI.spec")
    if spec_path.exists():
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm",
            "--clean",
            str(spec_path),
        ]
    else:
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--name", "OpenAssist-AI",
            "--onedir",  # Use --onefile for single EXE (slower startup)
            "--windowed",  # No console window
            "--add-data", f"config.yaml{os.pathsep}.",
            "--hidden-import", "google.genai",
            "--hidden-import", "groq",
            "--hidden-import", "mistralai",
            "--hidden-import", "cohere",
            "--hidden-import", "together",
            "--hidden-import", "anthropic",
            "--hidden-import", "ollama",
            "--hidden-import", "faster_whisper",
            "--hidden-import", "chromadb",
            "--hidden-import", "rapidocr_onnxruntime",
            "--hidden-import", "sounddevice",
            "--hidden-import", "pynput",
            "--hidden-import", "scipy",
            "--collect-all", "rapidocr_onnxruntime",
            "--collect-all", "faster_whisper",
            "--collect-all", "chromadb",
            "--exclude-module", "tkinter",
            "--exclude-module", "matplotlib",
            "--noconfirm",
            "--clean",
            "main.py"
        ]

    print(f"Running: {' '.join(cmd[:5])}...")
    subprocess.run(cmd, check=True)

    print("\nBuild complete!")
    print("Output: dist/OpenAssist-AI/")
    print("Run: dist/OpenAssist-AI/OpenAssist-AI.exe")


if __name__ == "__main__":
    build()
