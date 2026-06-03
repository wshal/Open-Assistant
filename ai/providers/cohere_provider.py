"""Cohere â Best for RAG, 1000 req/month free."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CohereProvider(BaseProvider):
    def __init__(self, config):
        super().__init__("cohere", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return
        try:
            import cohere
            self.client = cohere.AsyncClientV2(api_key=key)
            logger.info("  [OK] Cohere ready (best for RAG)")
        except Exception as e:
            logger.warning(f"  â Cohere: {e}")
            self.enabled = False

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            r = await self.client.chat(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=0.7
            )
            text = r.message.content[0].text
            tok = (r.usage.tokens.input_tokens + r.usage.tokens.output_tokens) if r.usage else len(text) // 4
            self.stats.record(tok, time.time() - t0)
            return text
        except Exception:
            self.stats.record_error()
            raise

    async def generate_stream(self, system: str, user: str, tier: str = None) -> AsyncGenerator[str, None]:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        tok = 0
        try:
            # M47 FIX: await the coroutine — was missing, causing TypeError
            stream = await self.client.chat_stream(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens, temperature=0.7
            )
            async for event in stream:
                if event.type == "content-delta":
                    c = event.delta.message.content.text
                    if c:
                        tok += 1
                        yield c
            self.stats.record(tok, time.time() - t0)
        except Exception:
            self.stats.record_error()
            raise

    async def health_check(self) -> bool:
        """Verify the key can complete a tiny chat request."""
        try:
            model = self.get_model("fast")
            if not model:
                return False
            r = await self.client.chat(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            content = getattr(getattr(r, "message", None), "content", None)
            if isinstance(content, list):
                for item in content:
                    if getattr(item, "text", ""):
                        return True
                return False
            if hasattr(content, "text"):
                return bool(getattr(content, "text", "").strip())
            return bool(str(content or "").strip())
        except Exception as e:
            logger.debug("Cohere health check failed: %s", e)
            return False
