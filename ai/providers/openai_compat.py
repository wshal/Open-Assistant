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
            if not r.choices:
                raise Exception(f"{self.name}: empty choices list (model={model})")
            text = r.choices[0].message.content or ""
            tok = r.usage.total_tokens if r.usage else max(1, len(text) // 4)
            self.stats.record(tok, time.time() - t0)
            return text
        except Exception:
            self.stats.record_error()
            raise

    async def generate_stream(self, system: str, user: str, tier: str = None) -> AsyncGenerator[str, None]:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        accumulated_len = 0
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=0.7, stream=True
            )
            async for chunk in stream:
                # M51 FIX: null-check delta before accessing .content
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    c = delta.content if delta else None
                    if c:
                        accumulated_len += len(c)
                        yield c
            tok = max(1, accumulated_len // 4)
            self.stats.record(tok, time.time() - t0)
        except Exception:
            self.stats.record_error()
            raise

    async def health_check(self) -> bool:
        """Verify the API key by completing a tiny chat request."""
        try:
            model = self.get_model("fast")
            if not model:
                return False
            r = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            return bool(getattr(r, "choices", None))
        except Exception as e:
            logger.debug("%s health check failed: %s", self.name, e)
            return False
