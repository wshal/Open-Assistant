"""Groq provider. Fast, quota-sensitive, and best for low-latency text."""

import asyncio
import time
from typing import AsyncGenerator

from ai.providers.base import BaseProvider
from core.constants import GROQ_DEFAULT_TEXT_MODEL, GROQ_DEPRECATED_TEXT_MODELS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GroqProvider(BaseProvider):
    def __init__(self, config):
        super().__init__("groq", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return
        configured_model = str(self.pcfg.get("model", "") or "").strip()
        if configured_model in GROQ_DEPRECATED_TEXT_MODELS:
            self.pcfg["model"] = GROQ_DEFAULT_TEXT_MODEL
        configured_models = self.pcfg.get("models")
        if isinstance(configured_models, dict):
            for tier, model in list(configured_models.items()):
                if str(model or "").strip() in GROQ_DEPRECATED_TEXT_MODELS:
                    configured_models[tier] = GROQ_DEFAULT_TEXT_MODEL
        if not self.pcfg.get("model") and not self.pcfg.get("models"):
            self.pcfg["model"] = GROQ_DEFAULT_TEXT_MODEL
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
        """Pre-open the shared Groq TCP/TLS connection without inference.

        Called opportunistically during standby warmup so the first real user
        prompt may hit an already-established socket rather than paying the
        full TCP + TLS handshake cost on the live path.

        Design rules:
          - Uses the SDK's underlying HTTP client against the API root rather
            than sending a chat completion. This establishes DNS/TCP/TLS and
            keeps the first real request fast without consuming model tokens.
          - Silent: any failure is swallowed because a warm failure is harmless.
          - Never called from the hot path.
        """
        if not self.enabled:
            return
        try:
            http_client = getattr(self.client, "_client", None)
            if http_client is None:
                return
            response = await http_client.get(
                "https://api.groq.com/",
                timeout=5.0,
            )
            logger.debug(
                "[Groq] HTTP connection pre-warmed (status=%s)",
                getattr(response, "status_code", "unknown"),
            )
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
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": self.max_tokens,
                "temperature": 0.7,
            }
            self._add_reasoning_effort(kwargs, model)
            r = await self.client.chat.completions.create(**kwargs)
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
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": self.max_tokens,
                "temperature": 0.7,
                "stream": True,
            }
            self._add_reasoning_effort(kwargs, model)
            stream = await self.client.chat.completions.create(**kwargs)
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

    async def health_check(self) -> bool:
        """Verify API access without spending chat completion quota."""
        try:
            response = await self.client.models.list()
            configured = self.get_model("fast") or GROQ_DEFAULT_TEXT_MODEL
            model_ids = {
                str(getattr(model, "id", "") or "")
                for model in (getattr(response, "data", None) or [])
            }
            return not model_ids or configured in model_ids
        except Exception as e:
            logger.debug("Groq health check failed: %s", e)
            return False

    def supports_non_billing_health_check(self) -> bool:
        """Allow startup validation without issuing an inference request."""
        return True

    def _add_reasoning_effort(self, kwargs: dict, model: str) -> None:
        """Apply GPT-OSS reasoning control without affecting other Groq models."""
        if str(model).startswith("openai/gpt-oss-"):
            effort = str(self.pcfg.get("reasoning_effort", "low") or "low").lower()
            if effort in {"low", "medium", "high"}:
                kwargs["reasoning_effort"] = effort
