#!/bin/bash
set -e
C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; N='\033[0m'

echo -e "${C}"
echo "╔═════════════════════════════════════════╗"
echo "║    🧠 OpenAssist AI v3.0 — Setup        ║"
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
command -v ollama &>/dev/null || (curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || true)
command -v ollama &>/dev/null && ollama pull llama3.2:3b 2>/dev/null &

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