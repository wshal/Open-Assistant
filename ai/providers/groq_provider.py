"""Groq â 1300 tok/s, 30 RPM free. Best for speed."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GroqProvider(BaseProvider):
    def __init__(self, config):
        super().__init__("groq", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return
        if not self.pcfg.get("model") and not self.pcfg.get("models"):
            self.pcfg["model"] = "llama-3.1-8b-instant"
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=key)
            logger.info("  [OK] Groq ready (1300 tok/s)")
        except Exception as e:
            logger.warning(f"  â Groq: {e}")
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
                c = chunk.choices[0].delta.content
                if c:
                    tok += 1
                    yield c
            self.stats.record(tok, time.time() - t0)
        except Exception as e:
            self.stats.errors += 1
            raise
