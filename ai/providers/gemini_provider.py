"""Gemini â Best quality free model, 1M+ tokens/day."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiProvider(BaseProvider):
    def __init__(self, config):
        super().__init__("gemini", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return
        try:
            from google import genai
            self.client = genai.Client(api_key=key)
            logger.info("  â Gemini ready (best quality free)")
        except Exception as e:
            logger.warning(f"  â Gemini: {e}")
            self.enabled = False

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            from google.genai import types
            r = self.client.models.generate_content(
                model=model, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                    temperature=0.7
                )
            )
            text = r.text
            tok = r.usage_metadata.total_token_count if r.usage_metadata else len(text) // 4
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
            from google.genai import types
            stream = self.client.models.generate_content_stream(
                model=model, contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                    temperature=0.7
                )
            )
            for chunk in stream:
                if chunk.text:
                    tok += len(chunk.text) // 4
                    yield chunk.text
            self.stats.record(tok, time.time() - t0)
        except Exception as e:
            self.stats.errors += 1
            raise

    def supports_vision(self) -> bool:
        return True

    async def analyze_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        tier: str = None,
    ) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            from google.genai import types

            contents = [
                types.Part.from_text(text=user),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ]
            r = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                    temperature=0.4,
                ),
            )
            text = r.text or ""
            tok = (
                r.usage_metadata.total_token_count
                if getattr(r, "usage_metadata", None)
                else len(text) // 4
            )
            self.stats.record(tok, time.time() - t0)
            return text
        except Exception as e:
            self.stats.errors += 1
            raise
