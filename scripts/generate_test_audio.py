import os
import sys
from pathlib import Path

# L-7: Guard the Windows-only SAPI import so importing this module on macOS or
# Linux (e.g. during ``pytest --collect-only``) doesn't blow up with an
# ImportError. Actual TTS generation still requires Windows; we exit early
# with a helpful message on other platforms.
if sys.platform == "win32":
    import win32com.client  # type: ignore
else:
    win32com = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "fixtures" / "auto_ground_truth"

QUERIES = [
    ("react_router", "What is React Router and how does it work?"),
    ("python_dict", "How do I merge two dictionaries in Python?"),
    ("rust_lifetime", "Explain lifetimes in Rust briefly."),
    ("general_greeting", "Hello there, can you help me with some coding?"),
    ("noisy_query", "Um, yeah, so I was wondering, what is a closure in JavaScript?"),
]

def generate_tts_wav(text, filename):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    filestream = win32com.client.Dispatch("SAPI.SpFileStream")
    
    # SSFMCreateForWrite = 3
    filestream.Open(str(filename), 3, False)
    
    speaker.AudioOutputStream = filestream
    speaker.Speak(text)
    filestream.Close()
    
if __name__ == "__main__":
    if sys.platform != "win32":
        print("generate_test_audio.py requires Windows SAPI; skipping on this platform.")
        sys.exit(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, text in QUERIES:
        out_path = OUT_DIR / f"{name}.wav"
        print(f"Generating {out_path.name} ...")
        generate_tts_wav(text, out_path)

    print("Done generating test fixtures!")
