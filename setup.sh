#!/bin/bash
set -e
C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; N='\033[0m'

echo -e "${C}"
echo "╔═════════════════════════════════════════╗"
echo "║    🧠 OpenAssist AI v1.0.0 — Setup      ║"
echo "║    Ultimate Free AI Assistant            ║"
echo "╚═════════════════════════════════════════╝"
echo -e "${N}"

python3 --version || { echo "Python 3.9+ required"; exit 1; }

echo -e "${Y}📦 Virtual environment...${N}"
python3 -m venv venv && source venv/bin/activate
pip install -U pip -q

echo -e "${Y}📦 Dependencies...${N}"
pip install -r requirements.txt -q

# System deps
case "$(uname -s)" in
    Linux*) sudo apt-get install -y -qq portaudio19-dev 2>/dev/null || true ;;
    Darwin*) brew install portaudio 2>/dev/null || true ;;
esac

# Ollama
echo -e "${Y}🦙 Ollama...${N}"
if ! command -v ollama &>/dev/null; then
    case "$(uname -s)" in
        Linux*)
            echo "Ollama not found. Installing via the official installer..."
            tmp_install="$(mktemp)"
            curl -fsSL https://ollama.com/install.sh -o "$tmp_install"
            sh "$tmp_install"
            rm -f "$tmp_install"
            ;;
        Darwin*)
            if command -v brew &>/dev/null; then
                echo "Ollama not found. Installing via Homebrew..."
                brew install ollama
            else
                echo -e "${Y}[!] Ollama is not installed and Homebrew was not found.${N}"
                echo -e "${Y}    Install Ollama manually from https://ollama.com${N}"
            fi
            ;;
        *)
            echo -e "${Y}[!] Ollama is not installed. Please download and install it manually from https://ollama.com${N}"
            ;;
    esac
fi

if command -v ollama &>/dev/null; then
    echo "Pulling model qwen2.5:7b..."
    ollama pull qwen2.5:7b
else
    echo -e "${Y}[!] Skipping model pull because Ollama is unavailable.${N}"
fi

mkdir -p data/vectordb knowledge/documents logs
[ ! -f .env ] && cp .env.example .env

echo -e "\n${G}╔═════════════════════════════════════════╗"
echo -e "║         ✅ Setup Complete!                ║"
echo -e "╚═════════════════════════════════════════╝${N}"
echo ""
echo "1. Add FREE API keys to .env:"
echo "   Groq:     https://console.groq.com/keys"
echo "   Gemini:   https://aistudio.google.com/apikey"
echo "   Cerebras: https://cloud.cerebras.ai/"
echo ""
echo "2. Start:  source venv/bin/activate && python main.py"
echo "3. Bench:  python main.py --benchmark"
echo "4. Modes:  python main.py --mode interview"
echo "5. Stealth: python main.py --stealth"
