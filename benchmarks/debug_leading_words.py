"""
Deep-dive: for the 4 fixtures still missing leading words,
run Whisper on the full audio AND on a pre-padded version,
and print timestamps + text for each segment to see exactly
what Whisper is doing at the start.
"""
import sys, os, wave, numpy as np, time
sys.path.insert(0, r'C:\Users\Vishal\Desktop\Open Assist')

from capture.audio import AudioCapture
from core.config import Config

cfg = Config()
cap = AudioCapture(cfg)
cap._ensure_whisper_loaded()

FIXTURE_DIR = r'C:\Users\Vishal\Desktop\Open Assist\tests\fixtures\audio_ground_truth'

TARGETS = {
    "js_closure_different_values_01.wav":           "what is happening in this javascript closure example and why does each function remember a different value",
    "react_context_api_vs_prop_drilling_01.wav":    "can you tell me what the context api does and when it is better than prop drilling",
    "screen_code_obvious_api_bug_01.wav":            "what do you see in the code on my screen and is there an obvious bug in the api call",
    "react_performance_rerenders_01.wav":            "how do i improve performance in this react app without causing unnecessary re-renders across the whole page",
}

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr

def run_whisper(audio, label):
    with cap._infer_lock:
        segs, info = cap.model.transcribe(
            audio,
            language="en",
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=cap._whisper_initial_prompt or None,
            no_speech_threshold=0.7,
            suppress_blank=False,
            word_timestamps=True,
        )
    print(f"  [{label}]")
    full = []
    for s in segs:
        print(f"    [{s.start:.2f}s-{s.end:.2f}s] no_speech={s.no_speech_prob:.3f} logprob={s.avg_logprob:.3f} | {s.text.strip()}")
        full.append(s.text.strip())
    print(f"  FULL: {' '.join(full)}")
    print()

for fname, expected in TARGETS.items():
    path = os.path.join(FIXTURE_DIR, fname)
    audio, sr = load_wav(path)
    print(f"\n{'='*72}")
    print(f"FIXTURE : {fname}")
    print(f"EXPECTED: {expected}")
    print(f"Duration: {len(audio)/sr*1000:.0f}ms  RMS={float(np.sqrt(np.mean(audio**2))):.4f}")
    print()

    # 1. Full audio as-is
    run_whisper(audio, "full audio no padding")

    # 2. Pad 400ms silence at start (simulate pre-roll giving more context)
    pad = np.zeros(int(sr * 0.4), dtype=np.float32)
    padded = np.concatenate([pad, audio])
    run_whisper(padded, "400ms silence prepended")

    # 3. First 1s only — is the opener even in there?
    first_1s = audio[:sr]
    run_whisper(first_1s, "first 1s only")
