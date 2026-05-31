# OpenAssist

OpenAssist is a local-first AI assistant for real-time work. It listens to system audio, reads screen context when needed, and routes questions through fast AI providers or local models.

The goal is simple: capture context from the environment you are already working in, understand the actual question, and return a useful answer quickly.

## Use Cases

- AI coding assistant for live debugging, architecture questions, and code review.
- Meeting copilot for system-audio transcription and follow-up answers.
- Interview assistant for technical and behavioral question practice.
- Screen-aware assistant for OCR, docs, terminals, and browser workflows.
- Local knowledge assistant with RAG over project files and documents.

## What It Does

- Turns spoken audio into actionable questions with Auto Mode.
- Captures screen/OCR context for code, docs, slides, terminals, and forms.
- Routes answers through Groq, Gemini, Cerebras, Together, Ollama, and other providers.
- Uses local RAG and semantic cache for project or knowledge-base context.
- Keeps session context so follow-up questions make sense.
- Supports local Whisper fallback when cloud transcription is unavailable.
- Runs as a desktop app with hotkeys, settings, and an optional overlay UI.

## Core Modes

| Mode | Purpose |
| --- | --- |
| Auto Mode | Continuously listens, extracts the real question from speech, and answers through the normal provider pipeline. |
| Standard Mode | Manual query flow using hotkeys, screen capture, clipboard, audio, or typed input. |
| Capture Modes | Presets for general use, interviews, coding, meetings, exams, and writing. |

## Quick Start

### Requirements

- Python 3.11+
- Windows 10/11 recommended
- API key for at least one cloud provider, or Ollama for local inference

### Install

```powershell
git clone <repo-url>
cd openassist

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` or add keys from the in-app setup wizard.

Common providers:

- Groq: fast text generation and Whisper transcription
- Gemini: text and vision fallback
- Cerebras / Together: low-latency cloud alternatives
- Ollama: local models, no API key

You can also edit `config.yaml` directly during development. Local config and runtime data are ignored by git.

### Run

```powershell
.\run.bat
```

or:

```powershell
python main.py
```

## Audio Capture

For system audio on Windows, OpenAssist prefers:

1. WASAPI loopback
2. Virtual audio cable
3. Stereo Mix, if explicitly enabled

Microphone fallback is opt-in because it is not a reliable substitute for meeting/system audio capture.

## Knowledge Base

Drop files into `knowledge/documents/` and the app can index them for retrieval. Supported content includes text, markdown, code, Q&A-style files, and PDFs where extraction is available.

You can also add documents with:

```powershell
python main.py --add-docs path\to\docs
```

## Useful Commands

Run the app:

```powershell
.\run.bat
```

Run tests:

```powershell
python -m pytest
```

Run Auto Mode benchmark:

```powershell
python benchmarks/auto_mode_benchmark.py --dir tests/fixtures/auto_ground_truth --out benchmarks/auto_mode_full.json
```

Run a focused test file:

```powershell
python -m pytest tests/test_text_utils.py -q
```

## Project Layout

```text
ai/          Provider routing, prompts, cache, memory, RAG, intent logic
capture/     Audio capture, speech transcription, screen/OCR capture
core/        App lifecycle, config, hotkeys, session state
knowledge/   Local documents and ingestion helpers
modes/       Mode profiles and behavior tuning
ui/          Desktop UI, settings, overlay, standby screen
utils/       Text cleanup, crypto, platform helpers, telemetry
tests/       Regression and behavior tests
benchmarks/  Fixture-driven latency and quality benchmarks
```

## Notes

- Auto Mode is the main real-time voice path.
- Generated benchmark JSON, logs, local cache, and learned runtime data are ignored.
- The app warms critical models before enabling Start Session.
- Cloud services can still vary in latency, so local fallback paths stay available.

## License

MIT
