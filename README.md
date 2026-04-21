# 🧠 OpenAssist

**The Ultimate Real-Time AI Partner for Vision and Audio Intelligence.**

OpenAssist is a premium AI-powered assistant designed for high-performance context awareness. Whether you're in a high-stakes technical interview, a complex development sprint, or a marathon research session, OpenAssist analyzes your screen and audio in real-time to provide contextually-aware guidance, decision support, and instant knowledge retrieval.

---

## ✨ Features

### 👁️ Midnight HUD (Heads-Up Display)
*   **Smart Crop Vision**: WinRT-accelerated OCR with contextual smart-cropping that focuses on active code or text regions.
*   **Neural Gaze Transparency**: The overlay intelligently fades when your mouse approaches, allowing you to see what's directly underneath without interruption.
*   **Ghost Stealth Mode**: Hardened protection against screen recorders (Zoom, Teams, OBS). Use it confidently in any professional meeting.

### 🎙️ Advanced Audio Pipeline
*   **WASAPI Loopback**: Capture system audio and microphone streams simultaneously for full meeting transcription.
*   **Cloud-ASR Correction**: Uses fast, auxiliary AI models to fix transcription errors in real-time before the main LLM processes them.

### 🧠 Intelligent Brain
*   **Smart Router**: Dynamically switches between High-Speed Cloud providers (Groq, Gemini, Together) and Local Inference (Ollama) based on query complexity.
*   **RAG Engine**: Built-in local vector database (ChromaDB) for instant indexing of project directories and knowledge bases.
*   **Snap-Lock Typing**: A simulator that can type AI-generated answers directly into your active window at human-like speeds.

---

## 🛠️ Tech Stack

- **UI/UX**: PyQt6 with custom Midnight-glass styling.
- **Vision**: WinRT OCR Windows Native APIs / EasyOCR.
- **Audio**: SoundDevice / Faster-Whisper.
- **RAG**: ChromaDB / FastEmbed.
- **AI Orchestration**: Asyncio-driven multi-model routing.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) for local inference.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/openassist.git
cd openassist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Rename `config.yaml.example` (if provided) or create a `config.yaml` with your API keys:
- Groq / Cerebras / Gemini / Together AI.

### 4. Launch
```bash
python main.py
```

---

## 🛡️ License
[Your Choice of License] — *Made with passion for developers who demand the best.*
