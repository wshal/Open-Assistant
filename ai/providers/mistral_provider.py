"""Mistral - Great for coding (Codestral)."""

import time
from typing import AsyncGenerator

from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MistralProvider(BaseProvider):
    def __init__(self, config):
        super().__init__("mistral", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return

        try:
            # Mistral's SDK has used both import paths across versions. The
            # docs now show `from mistralai.client import Mistral`, while some
            # quickstarts still show the top-level import. Try both so the app
            # works across installed SDK revisions.
            try:
                from mistralai.client import Mistral
            except ImportError:
                from mistralai import Mistral

            self.client = Mistral(api_key=key)
            logger.info("  [OK] Mistral ready (Codestral)")
        except ImportError as e:
            logger.warning(
                "  [ERR] Mistral SDK import failed: %s. Install or upgrade with `pip install -U mistralai`.",
                e,
            )
            self.enabled = False
        except Exception as e:
            logger.warning(f"  [ERR] Mistral: {e}")
            self.enabled = False

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            r = await self.client.chat.complete_async(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=0.7,
            )
            if not r.choices:
                raise Exception(f"Mistral returned empty choices list (model={model})")
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
        tok = 0
        try:
            stream = await self.client.chat.stream_async(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=0.7,
            )
            async for event in stream:
                choices = getattr(getattr(event, "data", None), "choices", None) or []
                if not choices:
                    continue
                delta = choices[0].delta
                c = delta.content if delta else None
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
            r = await self.client.chat.complete_async(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            return bool(getattr(r, "choices", None))
        except Exception as e:
            logger.debug("Mistral health check failed: %s", e)
            return False
