"""Ollama - Local, offline, no API key. FIXED: No model pull during generation."""

import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Optional

import aiohttp

from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaProvider(BaseProvider):
    """
    Ollama local LLM provider.

    FIXED in v4.0:
    - Model availability check happens at init, NOT during generate()
    - If model isn't available, provider marks itself disabled
    - Model pull only happens via explicit pull_model() call
    - No more hanging during live queries
    """

    # Model states
    STATE_UNKNOWN = "unknown"
    STATE_CHECKING = "checking"
    STATE_READY = "ready"
    STATE_MISSING = "missing"
    STATE_PULLING = "pulling"
    STATE_UNAVAILABLE = "unavailable"

    def __init__(self, config):
        super().__init__("ollama", config)
        self.endpoint = (
            self.pcfg.get("endpoint")
            or self.config.get_api_key("ollama")
            or "http://localhost:11434"
        )
        if not self.endpoint.startswith("http"):
            logger.warning(
                f"  Ollama endpoint looks invalid: {self.endpoint!r}. "
                "Falling back to http://localhost:11434"
            )
            self.endpoint = "http://localhost:11434"

        self.enabled = True  # Assume available until proven otherwise
        self._state = self.STATE_UNKNOWN
        self._available_models = []
        self._connect_timeout = aiohttp.ClientTimeout(total=5)
        self._generate_timeout = aiohttp.ClientTimeout(total=180)
        self._resolved_model = ""
        self._resolved_vision_model = ""
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Reuse a single aiohttp session to avoid per-request TCP overhead."""
        if self._session and not self._session.closed:
            return self._session
        if self._connector is None or self._connector.closed:
            self._connector = aiohttp.TCPConnector(keepalive_timeout=30, limit=8)
        self._session = aiohttp.ClientSession(connector=self._connector)
        return self._session

    async def close(self):
        """Best-effort cleanup (optional)."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass

    def _get_configured_vision_model(self) -> str:
        """
        Read an explicitly configured vision model without falling back to the
        provider's general `model` setting.
        """
        models = self.pcfg.get("models", {}) if isinstance(self.pcfg, dict) else {}
        if isinstance(models, dict):
            return (models.get("vision") or "").strip()
        return ""

    def get_model(self, tier: str = None) -> str:
        # Prefer a dedicated vision model for screenshot analysis if available.
        if (tier or "").lower() == "vision":
            configured = self._get_configured_vision_model()
            if configured:
                return configured
            if self._resolved_vision_model:
                return self._resolved_vision_model

        model = super().get_model(tier)
        if model:
            return model
        if self._resolved_model:
            return self._resolved_model
        return "llama3.1:8b"

    def _pick_available_model(self, requested: str = "") -> str:
        if not self._available_models:
            return ""

        requested = (requested or "").strip()
        if requested:
            requested_base = requested.split(":")[0]
            for model in self._available_models:
                if model == requested or model.startswith(f"{requested}:"):
                    return model
            for model in self._available_models:
                if model.split(":")[0] == requested_base:
                    return model

        preferred_prefixes = (
            "llama3.1",
            "llama3.2",
            "qwen2.5",
            "qwen2",
            "mistral",
            "phi3",
            "gemma",
        )
        for prefix in preferred_prefixes:
            for model in self._available_models:
                if model.startswith(prefix):
                    return model

        return self._available_models[0]

    def _pick_available_vision_model(self, requested: str = "") -> str:
        """
        Pick a locally available vision-capable model.
        If `requested` is set, try to honor it; otherwise pick a fast/reasonable default.
        """
        if not self._available_models:
            return ""

        requested = (requested or "").strip()
        if requested:
            requested_base = requested.split(":")[0]
            for model in self._available_models:
                if model == requested or model.startswith(f"{requested}:"):
                    return model
            for model in self._available_models:
                if model.split(":")[0] == requested_base:
                    return model

        vision_models = [m for m in self._available_models if self._looks_like_vision_model(m)]
        if not vision_models:
            return ""

        # Prefer commonly fast + decent VL models first.
        preferred_prefixes = (
            "qwen3-vl",
            "qwen2.5-vl",
            "qwen2-vl",
            "llama3.2-vision",
            "minicpm-v",
            "moondream",
            "llava",
            "bakllava",
        )
        for prefix in preferred_prefixes:
            for model in vision_models:
                if model.startswith(prefix):
                    return model

        return vision_models[0]

    async def check_availability(self) -> bool:
        """
        Check if Ollama is running and the required model exists.
        Called during app initialization, NOT during generate().

        Returns True if ready to serve requests.
        """
        self._state = self.STATE_CHECKING

        try:
            session = await self._get_session()
            # Step 1: Check if Ollama server is running
            async with session.get(
                f"{self.endpoint}/api/tags", timeout=self._connect_timeout
            ) as resp:
                if resp.status != 200:
                    self._state = self.STATE_UNAVAILABLE
                    self.enabled = False
                    logger.warning("  Ollama: Server returned non-200")
                    return False

                data = await resp.json()
                self._available_models = [model["name"] for model in data.get("models", [])]

            # Step 2: Check if our target model is available
            configured_target = super().get_model()
            target = self._pick_available_model(configured_target)
            if target:
                self._resolved_model = target
                self._state = self.STATE_READY
                self.enabled = True
                logger.debug(f"  Ollama ready (model: {target})")

                # Also resolve an optional vision model if present.
                configured_vision = self._get_configured_vision_model()
                vtarget = self._pick_available_vision_model(configured_vision)
                if vtarget:
                    self._resolved_vision_model = vtarget
                    logger.info(f"  Ollama vision ready (model: {vtarget})")

                return True

            self._state = self.STATE_MISSING
            self.enabled = False
            available_str = ", ".join(self._available_models[:5]) or "none"
            logger.warning(
                f"  Ollama: Model '{configured_target or self.get_model()}' not found. "
                f"Available: {available_str}. "
                f"Run: ollama pull {configured_target or self.get_model()}"
            )
            return False

        except aiohttp.ClientConnectorError:
            self._state = self.STATE_UNAVAILABLE
            self.enabled = False
            logger.info("  Ollama: Not running (install: https://ollama.com)")
            return False
        except asyncio.TimeoutError:
            self._state = self.STATE_UNAVAILABLE
            self.enabled = False
            logger.warning("  Ollama: Connection timeout")
            return False
        except Exception as exc:
            self._state = self.STATE_UNAVAILABLE
            self.enabled = False
            logger.warning(f"  Ollama: {exc}")
            return False

    async def pull_model(self, model: str = None, progress_callback=None) -> bool:
        """
        Explicitly pull a model. Called from settings flows, NEVER from generate().

        Args:
            model: Model name to pull (default: configured model)
            progress_callback: Optional callable(status: str, percent: float)

        Returns True if pull succeeded.
        """
        model = model or self.get_model()
        self._state = self.STATE_PULLING
        logger.info(f"Pulling Ollama model: {model}...")

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.endpoint}/api/pull",
                json={"name": model},
                timeout=aiohttp.ClientTimeout(total=3600),
            ) as resp:
                last_status = ""
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    status = data.get("status", "")
                    if status == last_status:
                        continue

                    last_status = status
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    percent = (completed / total * 100) if total > 0 else 0

                    if progress_callback:
                        progress_callback(status, percent)

                    if "pulling" in status.lower():
                        logger.info(f"  Download {status}: {percent:.0f}%")

            # Verify the model is now available
            self._state = self.STATE_READY
            self.enabled = True
            self._available_models.append(model)
            logger.info(f"  Model '{model}' pulled successfully")
            return True

        except Exception as exc:
            self._state = self.STATE_MISSING
            self.enabled = False
            logger.error(f"  Pull failed: {exc}")
            return False

    def get_available_models(self) -> list:
        """Return list of locally available models."""
        return list(self._available_models)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == self.STATE_READY

    @staticmethod
    def _looks_like_vision_model(model: str) -> bool:
        model_lower = (model or "").lower()
        markers = [
            "vision",
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen3-vl",
            "llama3.2-vision",
            "phi3.5-vision",
            "granite3.2-vision",
        ]
        return any(marker in model_lower for marker in markers)

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        """Generate response. Lazy-checks availability if state is unknown."""
        if self._state == self.STATE_UNKNOWN:
            await self.check_availability()

        if not self.is_ready:
            raise Exception(
                f"Ollama model not available (state: {self._state}). "
                f"Pull it first: ollama pull {self.get_model(tier)}"
            )

        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()

        session = await self._get_session()
        async with session.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"num_ctx": 8192, "temperature": 0.7},
            },
            timeout=self._generate_timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {body[:200]}")

            data = await resp.json()
            text = data.get("message", {}).get("content", "")
            tok = data.get("eval_count", len(text) // 4)
            self.stats.record(tok, time.time() - t0)
            return text

    async def generate_stream(
        self, system: str, user: str, tier: str = None
    ) -> AsyncGenerator[str, None]:
        """Stream response. Lazy-checks availability if state is unknown."""
        if self._state == self.STATE_UNKNOWN:
            await self.check_availability()

        if not self.is_ready:
            raise Exception(
                f"Ollama model not available (state: {self._state}). "
                f"Pull it first: ollama pull {self.get_model(tier)}"
            )

        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        tok = 0

        session = await self._get_session()
        async with session.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "options": {"num_ctx": 8192, "temperature": 0.7},
            },
            timeout=self._generate_timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {body[:200]}")

            async for line in resp.content:
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    tok += 1
                    yield chunk

        self.stats.record(tok, time.time() - t0)

    async def health_check(self) -> bool:
        """Quick health check."""
        return await self.check_availability()

    def supports_vision(self) -> bool:
        # Allow a dedicated vision model without forcing the default text model
        # to be multimodal (e.g., keep a coder model for chat, use a VL model for screenshots).
        vision_model = self._get_configured_vision_model() or self._resolved_vision_model
        if vision_model:
            return self._looks_like_vision_model(vision_model)
        return self._looks_like_vision_model(self.get_model())

    async def analyze_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        tier: str = None,
    ) -> str:
        if self._state == self.STATE_UNKNOWN:
            await self.check_availability()

        if not self.is_ready:
            raise Exception(
                f"Ollama model not available (state: {self._state}). "
                f"Pull it first: ollama pull {self.get_model(tier)}"
            )

        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        session = await self._get_session()
        async with session.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": user,
                        "images": [image_b64],
                    },
                ],
                "stream": False,
                "options": {"num_ctx": 8192, "temperature": 0.4},
            },
            timeout=self._generate_timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {body[:200]}")

            data = await resp.json()
            text = data.get("message", {}).get("content", "")
            tok = data.get("eval_count", len(text) // 4)
            self.stats.record(tok, time.time() - t0)
            return text

    def supports_vision_stream(self) -> bool:
        return self.supports_vision()

    async def analyze_image_stream(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        tier: str = None,
    ) -> AsyncGenerator[str, None]:
        if self._state == self.STATE_UNKNOWN:
            await self.check_availability()

        if not self.is_ready:
            raise Exception(
                f"Ollama model not available (state: {self._state}). "
                f"Pull it first: ollama pull {self.get_model(tier)}"
            )

        if not self.supports_vision():
            raise Exception(f"Ollama model '{self.get_model(tier)}' does not appear to support vision.")

        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        tok = 0
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        session = await self._get_session()
        async with session.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": user,
                        "images": [image_b64],
                    },
                ],
                "stream": True,
                "options": {"num_ctx": 8192, "temperature": 0.4},
            },
            timeout=self._generate_timeout,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {body[:200]}")

            async for line in resp.content:
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    tok += 1
                    yield chunk

        self.stats.record(tok, time.time() - t0)
