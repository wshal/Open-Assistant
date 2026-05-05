"""Gemini provider — uses gemma-3-27b-it by default (same API key, much higher free quota)."""

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

        # Default model when config.yaml doesn't specify one.
        # Use gemma-3-27b-it (same Gemini API key, open Gemma model):
        #   - gemini-2.5-flash: 20 req/day free — exhausted within 1-2 sessions
        #   - gemma-3-27b-it: much higher free quota, comparable quality & speed
        # This matches the cheating/ reference repo's model selection.
        if not self.pcfg.get("model") and not self.pcfg.get("models"):
            self.pcfg["model"] = "gemma-3-27b-it"
        try:
            from google import genai
            self.client = genai.Client(api_key=key)
            logger.info("  [OK] Gemini ready (best quality free)")
        except Exception as e:
            logger.warning(f"  [FAIL] Gemini: {e}")
            self.enabled = False

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _is_gemma_model(model: str) -> bool:
        """Gemma models don't support system_instruction in GenerateContentConfig."""
        return bool(model and model.startswith("gemma"))

    def _build_contents(self, model: str, system: str, user: str):
        """
        Build (contents, system_instruction_or_None) for the Gemini API.

        Gemini models support system_instruction in GenerateContentConfig.
        Gemma models raise 400 INVALID_ARGUMENT if system_instruction is set,
        so we inject the system prompt into the user content using the Gemma
        chat template format instead.
        """
        if self._is_gemma_model(model) and system:
            combined = (
                f"<start_of_turn>system\n{system}\n<end_of_turn>\n"
                f"<start_of_turn>user\n{user}\n<end_of_turn>"
            )
            return combined, None
        return user, (system or None)

    # -- Text generation --------------------------------------------------

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            from google.genai import types
            contents, sys_instr = self._build_contents(model, system, user)
            cfg_kwargs = dict(max_output_tokens=self.max_tokens, temperature=0.7)
            if sys_instr:
                cfg_kwargs["system_instruction"] = sys_instr
            r = await self.client.aio.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs)
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
            contents, sys_instr = self._build_contents(model, system, user)
            cfg_kwargs = dict(max_output_tokens=self.max_tokens, temperature=0.7)
            if sys_instr:
                cfg_kwargs["system_instruction"] = sys_instr
            stream = await self.client.aio.models.generate_content_stream(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs)
            )
            async for chunk in stream:
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
            _, sys_instr = self._build_contents(model, system, "")
            cfg_kwargs = dict(max_output_tokens=self.max_tokens, temperature=0.4)
            if sys_instr:
                cfg_kwargs["system_instruction"] = sys_instr

            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            text_part = types.Part.from_text(text=user)
            if self._is_gemma_model(model) and system:
                sys_part = types.Part.from_text(text=f"[System]: {system}\n\n")
                final_contents = [sys_part, text_part, image_part]
            else:
                final_contents = [text_part, image_part]

            r = await self.client.aio.models.generate_content(
                model=model,
                contents=final_contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
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

    def supports_vision_stream(self) -> bool:
        return True

    async def analyze_image_stream(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        tier: str = None,
    ) -> AsyncGenerator[str, None]:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        tok = 0
        try:
            from google.genai import types
            _, sys_instr = self._build_contents(model, system, "")
            cfg_kwargs = dict(max_output_tokens=self.max_tokens, temperature=0.4)
            if sys_instr:
                cfg_kwargs["system_instruction"] = sys_instr

            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            text_part = types.Part.from_text(text=user)
            if self._is_gemma_model(model) and system:
                sys_part = types.Part.from_text(text=f"[System]: {system}\n\n")
                final_contents = [sys_part, text_part, image_part]
            else:
                final_contents = [text_part, image_part]

            stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=final_contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            async for chunk in stream:
                if chunk.text:
                    tok += len(chunk.text) // 4
                    yield chunk.text
            self.stats.record(tok, time.time() - t0)
        except Exception as e:
            self.stats.errors += 1
            raise
