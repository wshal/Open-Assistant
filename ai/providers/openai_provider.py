"""OpenAI â GPT-4o, GPT-4o-mini, o3-mini. Paid API."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider using native SDK.
    
    Models:
      - gpt-4o-mini    (fast, cheap)
      - gpt-4o         (balanced)
      - o3-mini        (reasoning)
    
    Pricing: Paid. Some free trial credits for new accounts.
    Requires: OPENAI_API_KEY
    """

    def __init__(self, config):
        super().__init__("openai", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=key)
            logger.info("  â OpenAI ready (GPT-4o)")
        except ImportError:
            logger.warning("  â OpenAI: pip install openai")
            self.enabled = False
        except Exception as e:
            logger.warning(f"  â OpenAI: {e}")
            self.enabled = False

    async def generate(
        self, system: str, user: str, tier: str = None
    ) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()

        try:
            # o3-mini doesn't support system messages the same way
            messages = self._build_messages(system, user, model)

            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": self.max_tokens,
            }

            # o-series models don't support temperature
            if not model.startswith("o"):
                kwargs["temperature"] = 0.7

            response = await self.client.chat.completions.create(**kwargs)

            text = response.choices[0].message.content or ""
            tok = response.usage.total_tokens if response.usage else len(text) // 4
            self.stats.record(tok, time.time() - t0)
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
        tok = 0

        try:
            messages = self._build_messages(system, user, model)

            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            if not model.startswith("o"):
                kwargs["temperature"] = 0.7

            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    tok += 1
                    yield content

            self.stats.record(tok, time.time() - t0)

        except Exception as e:
            self.stats.errors += 1
            raise self._handle_error(e)

    @staticmethod
    def _build_messages(system: str, user: str, model: str) -> list:
        """Build message list, handling o-series model quirks."""
        if model.startswith("o"):
            # o-series: system message as developer message or prepend to user
            return [
                {"role": "user", "content": f"Instructions: {system}\n\n{user}"}
            ]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _handle_error(error) -> Exception:
        error_str = str(error).lower()
        if "api key" in error_str or "authentication" in error_str:
            return Exception("OpenAI: Invalid API key.")
        elif "rate_limit" in error_str or "429" in error_str:
            return Exception("OpenAI: Rate limit. Wait or upgrade plan.")
        elif "insufficient_quota" in error_str or "billing" in error_str:
            return Exception("OpenAI: No credits. Add billing at platform.openai.com.")
        elif "context_length" in error_str:
            return Exception("OpenAI: Input too long. Reduce context.")
        return Exception(f"OpenAI: {error}")