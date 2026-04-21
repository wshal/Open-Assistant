"""App-wide constants."""

APP_NAME = "OpenAssist AI"
APP_VERSION = "4.0.0"
APP_ID = "com.openassist.ai"
SETTINGS_FILE = "data/settings.enc"
CONFIG_FILE = "config.yaml"
DB_DIR = "data/vectordb"
CACHE_DIR = "data/cache"
LOG_DIR = "logs"
DOCS_DIR = "knowledge/documents"

# Provider metadata — display names, URLs, free tier info
PROVIDERS = {
    "groq": {
        "name": "Groq",
        "icon": "⚡",
        "url": "https://console.groq.com/keys",
        "free": "30 RPM, 14.4K req/day",
        "speed": "1,300 tok/s",
        "env_key": "GROQ_API_KEY",
        "key_prefix": "gsk_",
        "description": "Ultra-fast inference. Best for speed.",
    },
    "cerebras": {
        "name": "Cerebras",
        "icon": "🚀",
        "url": "https://cloud.cerebras.ai/",
        "free": "30 RPM, 1K req/day",
        "speed": "2,100 tok/s",
        "env_key": "CEREBRAS_API_KEY",
        "key_prefix": "csk-",
        "description": "World's fastest inference engine.",
    },
    "sambanova": {
        "name": "SambaNova",
        "icon": "🔥",
        "url": "https://cloud.sambanova.ai/",
        "free": "20 RPM, Llama 405B FREE",
        "speed": "900 tok/s",
        "env_key": "SAMBANOVA_API_KEY",
        "key_prefix": "",
        "description": "Free access to Llama 405B & DeepSeek R1.",
    },
    "gemini": {
        "name": "Google Gemini",
        "icon": "💎",
        "url": "https://aistudio.google.com/apikey",
        "free": "15 RPM, 1M+ tokens/day",
        "speed": "200 tok/s",
        "env_key": "GEMINI_API_KEY",
        "key_prefix": "AIza",
        "description": "Best quality. Huge free tier (1M tokens/day).",
    },
    "together": {
        "name": "Together AI",
        "icon": "🤝",
        "url": "https://api.together.xyz/settings/api-keys",
        "free": "$5 free credit",
        "speed": "400 tok/s",
        "env_key": "TOGETHER_API_KEY",
        "key_prefix": "",
        "description": "Fast open models. $5 free credit on signup.",
    },
    "openrouter": {
        "name": "OpenRouter",
        "icon": "🔄",
        "url": "https://openrouter.ai/keys",
        "free": "Free models available",
        "speed": "300 tok/s",
        "env_key": "OPENROUTER_API_KEY",
        "key_prefix": "sk-or-",
        "description": "Gateway to free models (Llama, Qwen, DeepSeek).",
    },
    "mistral": {
        "name": "Mistral AI",
        "icon": "🌬️",
        "url": "https://console.mistral.ai/api-keys/",
        "free": "Generous free tier",
        "speed": "250 tok/s",
        "env_key": "MISTRAL_API_KEY",
        "key_prefix": "",
        "description": "Strong coding model (Codestral).",
    },
    "cohere": {
        "name": "Cohere",
        "icon": "🧪",
        "url": "https://dashboard.cohere.com/api-keys",
        "free": "1,000 req/month",
        "speed": "150 tok/s",
        "env_key": "COHERE_API_KEY",
        "key_prefix": "",
        "description": "Excellent for RAG and retrieval tasks.",
    },
    "hyperbolic": {
        "name": "Hyperbolic",
        "icon": "📌",
        "url": "https://app.hyperbolic.xyz/",
        "free": "$10 free credit",
        "speed": "300 tok/s",
        "env_key": "HYPERBOLIC_API_KEY",
        "key_prefix": "",
        "description": "DeepSeek V3/R1 with $10 free credit.",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "icon": "🦙",
        "url": "https://ollama.com",
        "free": "∞ (runs locally)",
        "speed": "Depends on GPU",
        "env_key": "OLLAMA_ENDPOINT",
        "key_prefix": "",
        "description": "Run AI locally. No internet needed.",
    },
    "openai": {
        "name": "OpenAI",
        "icon": "🟢",
        "url": "https://platform.openai.com/api-keys",
        "free": "Paid ($5+ credit)",
        "speed": "~200 tok/s",
        "env_key": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "description": "GPT-4o. Requires paid API credits.",
    },
    "anthropic": {
        "name": "Anthropic",
        "icon": "🟠",
        "url": "https://console.anthropic.com/settings/keys",
        "free": "Paid ($5+ credit)",
        "speed": "~150 tok/s",
        "env_key": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "description": "Claude. Requires paid API credits.",
    },
}

MODES = ["general", "interview", "meeting", "coding", "writing", "exam"]