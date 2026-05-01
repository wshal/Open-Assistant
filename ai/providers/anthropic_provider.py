"""Anthropic Claude â Claude Sonnet 4, Opus 4, Haiku. Paid API."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider.
    
    Models:
      - claude-3-5-haiku-20241022    (fast, cheap)
      - claude-sonnet-4-20250514     (balanced)
      - claude-opus-4-20250514       (strongest)
    
    Pricing: Paid only. No free tier.
    Requires: ANTHROPIC_API_KEY
    """

    def __init__(self, config):
        super().__init__("anthropic", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return

        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=key)

            # Validate key format
            if not key.startswith("sk-ant-"):
                logger.warning("  â ï¸ Anthropic key doesn't start with 'sk-ant-'")

            logger.info("  [OK] Anthropic ready (Claude)")
        except ImportError:
            logger.warning("  â Anthropic: pip install anthropic")
            self.enabled = False
        except Exception as e:
            logger.warning(f"  â Anthropic: {e}")
            self.enabled = False

    async def generate(
        self, system: str, user: str, tier: str = None
    ) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()

        try:
            message = await self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.7,
            )

            # Extract text from content blocks
            text = self._extract_text(message.content)

            # Token tracking
            input_tok = message.usage.input_tokens if message.usage else 0
            output_tok = message.usage.output_tokens if message.usage else 0
            total_tok = input_tok + output_tok
            self.stats.record(total_tok, time.time() - t0)

            return text
        except Exception as e:
            self.stats.errors += 1
            raise self._handle_error(e)

    async def generate_stream(
        self, system: str, user: str, tier: str = None
    ) -> AsyncGenerator[str, None]:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        total_tok = 0

        try:
            async with self.client.messages.stream(
                model=model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0.7,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        total_tok += 1
                        yield text

            self.stats.record(total_tok, time.time() - t0)
        except Exception as e:
            self.stats.errors += 1
            raise self._handle_error(e)

    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from Anthropic content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, 'text'):
                    parts.append(block.text)
                elif isinstance(block, dict) and 'text' in block:
                    parts.append(block['text'])
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _handle_error(error) -> Exception:
        """Wrap Anthropic-specific errors with helpful messages."""
        error_str = str(error).lower()

        if "authentication" in error_str or "api key" in error_str:
            return Exception("Anthropic: Invalid API key. Check your ANTHROPIC_API_KEY.")
        elif "rate_limit" in error_str:
            return Exception("Anthropic: Rate limit exceeded. Wait and retry.")
        elif "overloaded" in error_str:
            return Exception("Anthropic: API overloaded. Trying fallback provider...")
        elif "insufficient" in error_str or "credit" in error_str:
            return Exception("Anthropic: Insufficient credits. Add billing at console.anthropic.com.")
        elif "context_length" in error_str or "too long" in error_str:
            return Exception("Anthropic: Input too long. Reduce context or use a larger model.")

        return Exception(f"Anthropic: {error}")

    async def health_check(self) -> bool:
        """Verify API key and model access."""
        try:
            message = await self.client.messages.create(
                model=self.get_model("fast"),
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'ok'"}],
            )
            return bool(self._extract_text(message.content))
        except Exception as e:
            logger.debug(f"Anthropic health check failed: {e}")
            return False
