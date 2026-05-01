# 🧠 OpenAssist

**The Ultimate Real-Time AI Partner for Vision and Audio Intelligence.**

OpenAssist is a premium AI-powered overlay assistant for Windows. It analyzes your screen and audio in real time to provide contextually-aware guidance, decision support, and instant knowledge retrieval — invisibly, during any live session.

---

## ✨ Features

### 👁️ Midnight HUD (Heads-Up Display)
- **Ghost Stealth Mode** — hardened protection against screen recorders (Zoom, Teams, OBS). Invisible in any professional meeting.
- **Smart Crop Vision** — WinRT-accelerated OCR with contextual smart-cropping that focuses on active code or text regions.
- **Neural Gaze Transparency** — the overlay fades when your mouse approaches, so you always see what's underneath.
- **Mini-HUD Mode** — compact floating widget for distraction-free use.

### 🎙️ Advanced Audio Pipeline
- **WASAPI Loopback** — capture system audio and microphone simultaneously for full meeting/interview transcription.
- **Faster-Whisper STT** — on-device speech-to-text with VAD-aware silence detection per mode.
- **Cloud-ASR Correction** — auxiliary AI models fix transcription errors before the main LLM sees them.

### 🧠 Intelligent Brain
- **Smart Router** — dynamically switches between high-speed cloud providers (Groq, Gemini, Cerebras, Together) and local inference (Ollama) based on query complexity.
- **Four-Tier Semantic Cache** — high-performance engine using local ONNX embeddings (`BAAI/bge-small-en-v1.5`) and Jaccard token overlap to autonomously resolve paraphrased questions with sub-20ms latency.
- **Actionable Queries** — native intent routing execution that can launch dev servers, run tests, and execute git commands automatically.
- **Predictive Prefetch** — anticipates documentation needs by scanning cursor and IDE context to warm up the RAG cache in the background.
- **RAG Engine** — built-in local vector database (ChromaDB) for instant indexing and retrieval from project directories or knowledge bases.
- **Conversation History** — maintains multi-turn context across a session and resolves follow-up queries automatically, with automatic contextual boosting.
- **Parallel Inference** — optional mode that fires multiple providers simultaneously and uses the fastest valid response.

### 📝 Session Context (Custom Instructions)
- Write per-session AI instructions once — persona, tech stack, tone, response style — and the AI follows them for every response in that session.
- **7 built-in presets**: Job Interview, Exam, Code Review, Meeting Copilot, Presentation, Negotiation, Practice.
- **Auto-suggest** — selecting a Capture Mode automatically loads the best-matching context preset (Interview → "Job Interview", Coding → "Code Review", etc.). Your manually written context is never overwritten.
- Presets persist across app restarts; context is cleared and saved when you end a session.

### ⚡ Capture Modes
Each mode tunes the full engine stack — audio VAD timing, provider routing, context weights, and AI persona — for the task at hand:

| Mode | Optimized for |
|---|---|
| **General** | Everyday Q&A, research, quick lookups |
| **Interview** | Real-time interview coaching, short precise answers |
| **Coding** | Code review, debugging, architecture questions |
| **Meeting** | Live transcription, action items, response suggestions |
| **Exam** | Direct answers from screen content, MCQ detection |
| **Writing** | Editing, structuring, speaker-friendly language |

### ⌨️ Keyboard Shortcuts
All actions are hotkey-driven so the HUD stays invisible:

| Action | Default Shortcut |
|---|---|
| Analyze Screen | `Ctrl+Enter` |
| Quick Answer | `Ctrl+Shift+Q` |
| Emergency Erase | `Ctrl+Shift+E` |
| History Prev / Next | `Ctrl+[` / `Ctrl+]` |
| Mini-HUD Mode | `Ctrl+Alt+N` |
| Rotate Mode | `Ctrl+Shift+M` |
| Ghost Stealth | `Ctrl+Shift+Z` |
| Show / Hide HUD | `Ctrl+\` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI/UX** | PyQt6 — custom Midnight-glass dark theme |
| **Vision** | WinRT Windows Native OCR / EasyOCR fallback |
| **Audio** | SoundDevice + Faster-Whisper (tiny/base) |
| **AI Cloud** | Groq · Gemini · Cerebras · Together AI |
| **AI Local** | Ollama (any pulled model) |
| **RAG & Cache** | ChromaDB + FastEmbed ONNX |
| **Config** | Encrypted YAML (`cryptography` / Fernet) |
| **Orchestration** | asyncio multi-model routing with smart fallback |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Windows 10/11** (WinRT OCR is Windows-only)
- *(Optional)* [Ollama](https://ollama.ai/) for local inference

### Installation
```bash
# Clone
git clone https://github.com/yourusername/openassist.git
cd openassist

# Virtual environment
python -m venv venv
venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

### Configuration
On first launch the **Setup Wizard** walks you through adding API keys. You can also edit them later via **Settings → AI ENGINES**.

Supported providers and where to get keys:
- [Groq](https://console.groq.com/) — fastest free-tier cloud inference
- [Google Gemini](https://aistudio.google.com/) — free vision + text
- [Cerebras](https://cloud.cerebras.ai/) — ultra-low latency
- [Together AI](https://api.together.xyz/) — wide model selection
- [Ollama](https://ollama.ai/) — fully local, no key required

### ⚠️ Troubleshooting Windows Defender Application Control (WDAC)
Because OpenAssist uses compiled local AI engines (Whisper via `ctranslate2` and RAG via `chromadb`), **Windows Smart App Control** may block these DLLs upon first launch, showing a `DLL load failed` error in the logs.
- **Fix 1 (Recommended):** Turn off "Smart App Control" in Windows settings if you are a developer running local Python AI scripts.
- **Fix 2:** Open Windows Security, go to "Virus & threat protection settings" → "Exclusions", and add your `venv` folder as an exclusion.

### Launch
```bash
python main.py
```

---

## 📁 Project Structure

```
openassist/
├── core/           # App controller, config, state, hotkeys
├── ai/             # Engine, prompt builder, router, providers, RAG, Semantic Cache
├── capture/        # Screen OCR, audio capture pipeline
├── modes/          # Capture mode profiles (Interview, Coding, etc.)
├── ui/             # HUD overlay, standby view, settings, mini-HUD
├── utils/          # Logger, context store, platform utilities, telemetry
├── stealth/        # Anti-screen-capture and input simulator
├── docs/           # Architecture deep-dives and technical reports
├── scripts/        # Utility and build scripts
├── tests/          # Unit test suite (291 tests)
└── data/           # Runtime data (encrypted, gitignored)
```

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests
```

All **291 tests** should pass. Tests cover session lifecycle, context store, mode switching, prompt injection, audio pipeline behavior, semantic caching, background async workers, and question detection.

---

## 🛡️ License

MIT — *Made with passion for developers who demand the best.*
