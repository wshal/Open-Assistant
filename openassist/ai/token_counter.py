"""Token counting and budget management across providers.

Accurate counting prevents:
  - Exceeding context windows â API errors
  - Wasting money on paid providers
  - Sending truncated/useless context
"""

import re
from typing import Optional, Dict, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Approximate context window sizes per model family
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    # Groq
    "llama-3.1-8b-instant": 131072,
    "llama-3.3-70b-versatile": 131072,
    "deepseek-r1-distill-llama-70b": 131072,

    # Cerebras
    "llama3.1-8b": 8192,
    "llama-3.3-70b": 8192,

    # SambaNova
    "Meta-Llama-3.1-8B-Instruct": 8192,
    "Meta-Llama-3.1-70B-Instruct": 8192,
    "Meta-Llama-3.1-405B-Instruct": 8192,
    "DeepSeek-R1": 8192,

    # Gemini
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-pro-preview-06-05": 1048576,
    "gemini-2.5-flash-preview-04-17": 1048576,

    # Together
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": 131072,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": 131072,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 32768,
    "deepseek-ai/DeepSeek-R1": 65536,

    # OpenRouter
    "meta-llama/llama-3.1-8b-instruct:free": 131072,
    "qwen/qwen3-235b-a22b:free": 40960,
    "deepseek/deepseek-r1:free": 65536,

    # Mistral
    "open-mistral-nemo": 128000,
    "mistral-small-latest": 32000,
    "codestral-latest": 32000,

    # Cohere
    "command-r": 128000,
    "command-r-plus": 128000,

    # OpenAI
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "o3-mini": 200000,

    # Anthropic
    "claude-3-5-haiku-20241022": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,

    # Ollama (local)
    "llama3.2:3b": 8192,
    "llama3.1:8b": 8192,
    "qwen2.5-coder:7b": 32768,
    "deepseek-r1:8b": 8192,
}


class TokenCounter:
    """
    Fast approximate token counter.
    
    Uses character-based estimation with language-aware adjustments.
    Accurate to ~Â±10% vs tiktoken, but 100x faster and no dependencies.
    
    Rules of thumb:
      - English: ~4 chars per token
      - Code: ~3.5 chars per token (more symbols)
      - CJK: ~1.5 chars per token
      - Mixed: ~3.8 chars per token
    """

    # Ratio of chars-to-tokens by content type
    RATIO_ENGLISH = 4.0
    RATIO_CODE = 3.5
    RATIO_CJK = 1.5
    RATIO_MIXED = 3.8

    def __init__(self, config):
        self.config = config
        self.default_budget = config.get("performance.token_budget", 4096)

        # Try loading tiktoken for exact counting (optional)
        self._tiktoken_encoder = None
        try:
            import tiktoken
            self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            logger.debug("TokenCounter: Using tiktoken (exact)")
        except Exception as e:
            logger.debug(f"TokenCounter: Using fast estimation (Â±10%) | Reason: {e}")

    def count(self, text: str) -> int:
        """Count tokens in text. Uses tiktoken if available, else estimates."""
        if not text:
            return 0

        if self._tiktoken_encoder:
            return self._count_tiktoken(text)
        return self._count_estimate(text)

    def _count_tiktoken(self, text: str) -> int:
        """Exact count using tiktoken."""
        try:
            return len(self._tiktoken_encoder.encode(text))
        except Exception:
            return self._count_estimate(text)

    def _count_estimate(self, text: str) -> int:
        """Fast estimation based on character analysis."""
        if not text:
            return 0

        total_chars = len(text)

        # Detect content type
        code_ratio = self._code_density(text)
        cjk_ratio = self._cjk_density(text)

        # Weighted average ratio
        if cjk_ratio > 0.3:
            ratio = self.RATIO_CJK * cjk_ratio + self.RATIO_ENGLISH * (1 - cjk_ratio)
        elif code_ratio > 0.3:
            ratio = self.RATIO_CODE * code_ratio + self.RATIO_ENGLISH * (1 - code_ratio)
        else:
            ratio = self.RATIO_ENGLISH

        estimated = int(total_chars / ratio)

        # Account for special tokens (messages typically add 3-7 tokens overhead)
        return estimated + 4

    def count_messages(self, system: str, user: str) -> int:
        """Count tokens for a full message exchange."""
        # Each message has ~4 token overhead (role, delimiters)
        overhead = 8  # system msg overhead + user msg overhead
        return self.count(system) + self.count(user) + overhead

    def get_context_window(self, model: str) -> int:
        """Get the context window size for a model."""
        # Exact match
        if model in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model]

        # Fuzzy match (model name contains key)
        model_lower = model.lower()
        for key, window in MODEL_CONTEXT_WINDOWS.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return window

        # Defaults by model family
        if "gemini" in model_lower:
            return 1048576
        elif "claude" in model_lower:
            return 200000
        elif "gpt-4" in model_lower:
            return 128000
        elif "llama" in model_lower:
            return 131072
        elif "deepseek" in model_lower:
            return 65536
        elif "qwen" in model_lower:
            return 32768
        elif "mistral" in model_lower:
            return 32000
        elif "command" in model_lower:
            return 128000

        # Conservative default
        return 8192

    def budget_context(
        self,
        query: str,
        screen_text: str,
        audio_text: str,
        rag_text: str,
        clipboard_text: str,
        model: str,
        max_response_tokens: int = None
    ) -> Tuple[str, str, str, str]:
        """
        Trim context to fit within model's context window.
        
        Returns trimmed (screen_text, audio_text, rag_text, clipboard_text).
        
        Budget allocation:
          - Query: always full (small)
          - Response reserve: 25% of window
          - System prompt: ~500 tokens
          - Remaining split: 40% screen, 30% audio, 20% RAG, 10% clipboard
        """
        # Cap the window by our configured budget to avoid rate limits (TPM)
        window = min(self.get_context_window(model), self.default_budget)
        
        response_reserve = max_response_tokens or min(self.default_budget // 2, window // 4)
        system_reserve = 500

        query_tokens = self.count(query)
        available = window - response_reserve - system_reserve - query_tokens

        if available <= 0:
            logger.warning(f"Token budget exhausted: window={window}, query={query_tokens}")
            return ("", "", "", "")

        # Allocate proportionally
        screen_budget = int(available * 0.40)
        audio_budget = int(available * 0.30)
        rag_budget = int(available * 0.20)
        clipboard_budget = max(available - screen_budget - audio_budget - rag_budget, 0)

        # Trim each context to budget
        screen_trimmed = self._trim_to_tokens(screen_text, screen_budget)
        audio_trimmed = self._trim_to_tokens(audio_text, audio_budget, keep_end=True)
        rag_trimmed = self._trim_to_tokens(rag_text, rag_budget)
        clipboard_trimmed = self._trim_to_tokens(clipboard_text, clipboard_budget)

        actual = (
            self.count(screen_trimmed)
            + self.count(audio_trimmed)
            + self.count(rag_trimmed)
            + self.count(clipboard_trimmed)
        )
        logger.debug(
            f"Token budget: window={window}, reserved={response_reserve}, "
            f"context={actual}/{available} "
            f"(screen={self.count(screen_trimmed)}, "
            f"audio={self.count(audio_trimmed)}, "
            f"rag={self.count(rag_trimmed)}, "
            f"clipboard={self.count(clipboard_trimmed)})"
        )

        return (screen_trimmed, audio_trimmed, rag_trimmed, clipboard_trimmed)

    def _trim_to_tokens(self, text: str, max_tokens: int, keep_end: bool = False) -> str:
        """Trim text to fit within token budget."""
        if not text or max_tokens <= 0:
            return ""

        current_tokens = self.count(text)
        if current_tokens <= max_tokens:
            return text

        # Estimate character limit
        ratio = len(text) / max(current_tokens, 1)
        char_limit = int(max_tokens * ratio * 0.90)  # 10% safety margin

        if keep_end:
            # For audio/conversation: keep the most recent part
            return "..." + text[-char_limit:]
        else:
            # For screen/RAG: keep the beginning
            return text[:char_limit] + "..."

    @staticmethod
    def _code_density(text: str) -> float:
        """Estimate what fraction of text is code."""
        if not text:
            return 0.0
        code_chars = set('{}[]();<>=+-*/&|^~!@#$%\\`')
        code_count = sum(1 for c in text if c in code_chars)
        # Also check for common code patterns
        patterns = [r'\bdef\b', r'\bfunction\b', r'\bclass\b', r'\bimport\b',
                    r'\breturn\b', r'\bif\b.*:', r'\bfor\b.*:', r'//.*', r'/\*']
        pattern_hits = sum(1 for p in patterns if re.search(p, text[:2000]))
        density = (code_count / len(text)) + (pattern_hits * 0.05)
        return min(density, 1.0)

    @staticmethod
    def _cjk_density(text: str) -> float:
        """Estimate what fraction of text is CJK characters."""
        if not text:
            return 0.0
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or
                        '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or
                        '\uac00' <= c <= '\ud7af')
        return cjk_count / len(text)

    def get_usage_report(self) -> Dict:
        """Get token usage statistics."""
        return {
            "default_budget": self.default_budget,
            "engine": "tiktoken" if self._tiktoken_encoder else "estimation",
        }