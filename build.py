"""Build standalone EXE using PyInstaller."""

import os
import sys
import subprocess
import shutil
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
            "--exclude-module", "torch",
            "--exclude-module", "easyocr",
            "--exclude-module", "sentence_transformers",
            "--exclude-module", "cv2",
            "--noconfirm",
            "--clean",
            "main.py"
        ]

    print(f"Running: {' '.join(cmd[:5])}...")
    subprocess.run(cmd, check=True)

    dist_root = Path("dist/OpenAssist-AI")
    packaged_data_root = dist_root / "OpenAssist_Data"
    copied = []

    # Copy fastembed cached model to build directory if it exists
    src_cache = Path("data/cache/fastembed")
    if src_cache.exists():
        dst_cache = packaged_data_root / "data/cache/fastembed"
        dst_cache.parent.mkdir(parents=True, exist_ok=True)
        if dst_cache.exists():
            shutil.rmtree(dst_cache)
        try:
            shutil.copytree(src_cache, dst_cache, symlinks=False)
            print("[OK] Copied fastembed cached model to build directory.")
            copied.append("data/cache/fastembed")
        except Exception as e:
            print(f"[WARN] Warning: could not copy fastembed cache to build: {e}")

    if copied:
        print(f"[OK] Copied runtime data into packaged app: {', '.join(copied)}")

    print("\nBuild complete!")
    print("Output: dist/OpenAssist-AI/")
    print("Run: dist/OpenAssist-AI/OpenAssist-AI.exe")


if __name__ == "__main__":
    build()
