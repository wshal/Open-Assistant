"""Generic OpenAI-compatible provider factory."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAICompatProvider(BaseProvider):
    """Works with any OpenAI-compatible API: SambaNova, Together, OpenRouter, Hyperbolic."""

    def __init__(self, name: str, config, base_url: str, extra_headers: dict = None):
        super().__init__(name, config)
        key = self.pcfg.get("api_key", "")
        ep = self.pcfg.get("endpoint", base_url)
        if not key:
            self.enabled = False
            return
        try:
            from openai import AsyncOpenAI
            kwargs = {"api_key": key, "base_url": ep}
            if extra_headers:
                kwargs["default_headers"] = extra_headers
            self.client = AsyncOpenAI(**kwargs)
            logger.info(f"  â {name} ready")
        except Exception as e:
            logger.warning(f"  â {name}: {e}")
            self.enabled = False

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            r = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=0.7
            )
            text = r.choices[0].message.content
            tok = r.usage.total_tokens if r.usage else len(text) // 4
            self.stats.record(tok, time.time() - t0)
            return text
        except Exception as e:
            self.stats.errors += 1
            raise

    async def generate_stream(self, system: str, user: str, tier: str = None) -> AsyncGenerator[str, None]:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        tok = 0
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=0.7, stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    c = chunk.choices[0].delta.content
                    tok += 1
                    yield c
            self.stats.record(tok, time.time() - t0)
        except Exception as e:
            self.stats.errors += 1
            raise