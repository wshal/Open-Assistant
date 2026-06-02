"""Groq provider. Fast, quota-sensitive, and best for low-latency text."""

import asyncio
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

            # max_retries=0: surface 429 rate-limit errors immediately so the
            # engine fallback chain switches quickly instead of blocking on the
            # SDK's server-specified retry backoff.
            self.client = AsyncGroq(api_key=key, max_retries=0)
            logger.info("  [OK] Groq ready")
        except Exception as e:
            logger.warning(f"  [ERR] Groq: {e}")
            self.enabled = False

    async def warm_connection_async(self) -> None:
        """Pre-open the Groq TCP/TLS connection with a minimal request.

        Called opportunistically during standby warmup so the first real user
        prompt may hit an already-established socket rather than paying the
        full TCP + TLS handshake cost on the live path.

        Design rules:
          - Uses max_tokens=1 to keep the request tiny.
          - Does not call _pre_request(), so it does not advance local
            rate-limit timestamps. It is still a Groq API request and may count
            against server-side quota.
          - Silent: any failure is swallowed because a warm failure is harmless.
          - Never called from the hot path.
        """
        if not self.enabled:
            return
        try:
            model = self.pcfg.get("model") or "llama-3.1-8b-instant"
            await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            logger.debug("[Groq] Connection pre-warmed")
        except Exception:
            # Failure is acceptable; warmup is best-effort.
            pass

    async def keepalive_loop(self, idle_threshold_s: float = 25.0) -> None:
        """Background loop: re-warm the TCP connection after long idle gaps.

        This is intentionally config-gated by AIEngine because each ping is
        still a Groq API request. It runs until the owning asyncio loop cancels
        it.
        """
        sleep_s = max(10.0, idle_threshold_s / 2.0)
        while True:
            try:
                await asyncio.sleep(sleep_s)
                idle = self.idle_seconds()
                if idle_threshold_s <= idle < idle_threshold_s * 4:
                    logger.debug(
                        "[Groq] Keepalive ping - idle=%.1fs (threshold=%.1fs)",
                        idle,
                        idle_threshold_s,
                    )
                    await self.warm_connection_async()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0)

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            r = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=0.7,
            )
            if not r.choices:
                raise Exception(f"Groq returned empty choices list (model={model})")
            choice = r.choices[0]
            text = choice.message.content or ""
            if not text and getattr(choice, "finish_reason", None) == "content_filter":
                raise Exception("Groq: response blocked by content filter")
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
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=0.7,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                c = delta.content if delta else None
                if c:
                    tok += max(1, len(c) // 4)
                    yield c
            self.stats.record(tok, time.time() - t0)
        except Exception:
            self.stats.record_error()
            raise
