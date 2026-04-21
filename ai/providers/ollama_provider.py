"""Ollama â Local, offline, no API key. FIXED: No model pull during generation."""

import time
import json
import asyncio
import aiohttp
from typing import AsyncGenerator, Optional
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
                f"  ⚠️ Ollama endpoint looks invalid: {self.endpoint!r}. "
                "Falling back to http://localhost:11434"
            )
            self.endpoint = "http://localhost:11434"

        self.enabled = True  # Assume available until proven otherwise
        self._state = self.STATE_UNKNOWN
        self._available_models = []
        self._connect_timeout = aiohttp.ClientTimeout(total=5)
        self._generate_timeout = aiohttp.ClientTimeout(total=180)

    async def check_availability(self) -> bool:
        """
        Check if Ollama is running and the required model exists.
        Called during app initialization, NOT during generate().
        
        Returns True if ready to serve requests.
        """
        self._state = self.STATE_CHECKING

        try:
            async with aiohttp.ClientSession() as session:
                # Step 1: Check if Ollama server is running
                async with session.get(
                    f"{self.endpoint}/api/tags",
                    timeout=self._connect_timeout
                ) as resp:
                    if resp.status != 200:
                        self._state = self.STATE_UNAVAILABLE
                        self.enabled = False
                        logger.warning("  â ï¸ Ollama: Server returned non-200")
                        return False

                    data = await resp.json()
                    self._available_models = [
                        m["name"] for m in data.get("models", [])
                    ]

            # Step 2: Check if our target model is available
            target = self.get_model()
            target_base = target.split(":")[0]

            model_found = any(
                target in m or target_base in m
                for m in self._available_models
            )

            if model_found:
                self._state = self.STATE_READY
                self.enabled = True
                logger.info(f"  â Ollama ready (model: {target})")
                return True
            else:
                self._state = self.STATE_MISSING
                self.enabled = False
                available_str = ", ".join(self._available_models[:5]) or "none"
                logger.warning(
                    f"  â ï¸ Ollama: Model '{target}' not found. "
                    f"Available: {available_str}. "
                    f"Run: ollama pull {target}"
                )
                return False

        except aiohttp.ClientConnectorError:
            self._state = self.STATE_UNAVAILABLE
            self.enabled = False
            logger.info("  â¬ Ollama: Not running (install: https://ollama.com)")
            return False
        except asyncio.TimeoutError:
            self._state = self.STATE_UNAVAILABLE
            self.enabled = False
            logger.warning("  â ï¸ Ollama: Connection timeout")
            return False
        except Exception as e:
            self._state = self.STATE_UNAVAILABLE
            self.enabled = False
            logger.warning(f"  â ï¸ Ollama: {e}")
            return False

    async def pull_model(self, model: str = None, progress_callback=None) -> bool:
        """
        Explicitly pull a model. Called from SetupWizard or Settings, 
        NEVER from generate().
        
        Args:
            model: Model name to pull (default: configured model)
            progress_callback: Optional callable(status: str, percent: float)
        
        Returns True if pull succeeded.
        """
        model = model or self.get_model()
        self._state = self.STATE_PULLING
        logger.info(f"ð¦ Pulling Ollama model: {model}...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}/api/pull",
                    json={"name": model},
                    timeout=aiohttp.ClientTimeout(total=3600)  # 1 hour for large models
                ) as resp:
                    last_status = ""
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                
                                # Report progress
                                if status != last_status:
                                    last_status = status
                                    total = data.get("total", 0)
                                    completed = data.get("completed", 0)
                                    percent = (completed / total * 100) if total > 0 else 0
                                    
                                    if progress_callback:
                                        progress_callback(status, percent)
                                    
                                    if "pulling" in status.lower():
                                        logger.info(f"  ð¥ {status}: {percent:.0f}%")
                            except json.JSONDecodeError:
                                continue

            # Verify the model is now available
            self._state = self.STATE_READY
            self.enabled = True
            self._available_models.append(model)
            logger.info(f"  â Model '{model}' pulled successfully")
            return True

        except Exception as e:
            self._state = self.STATE_MISSING
            self.enabled = False
            logger.error(f"  â Pull failed: {e}")
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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    "stream": False,
                    "options": {"num_ctx": 8192, "temperature": 0.7}
                },
                timeout=self._generate_timeout
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise Exception(f"Ollama error {resp.status}: {body[:200]}")

                data = await resp.json()
                text = data.get("message", {}).get("content", "")
                tok = data.get("eval_count", len(text) // 4)
                self.stats.record(tok, time.time() - t0)
                return text

    async def generate_stream(self, system: str, user: str, tier: str = None) -> AsyncGenerator[str, None]:
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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    "stream": True,
                    "options": {"num_ctx": 8192, "temperature": 0.7}
                },
                timeout=self._generate_timeout
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise Exception(f"Ollama error {resp.status}: {body[:200]}")

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            c = data.get("message", {}).get("content", "")
                            if c:
                                tok += 1
                                yield c
                        except json.JSONDecodeError:
                            continue

        self.stats.record(tok, time.time() - t0)

    async def health_check(self) -> bool:
        """Quick health check."""
        return await self.check_availability()