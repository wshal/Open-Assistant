"""Gemini provider — gemini-2.5-flash default, tiered model support, thinking budget."""

import time
from typing import AsyncGenerator
from ai.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Deprecated / broken models ────────────────────────────────────────────────
# These are silently replaced at startup so config.yaml drift can't break users.
_DEPRECATED_MODELS: set[str] = {
    # 2.0 family — permanently shut down June 1, 2026
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    # 1.5 family — deprecated
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    # Gemma via Gemini API — returns 404 on v1beta endpoint (SDK default)
    "gemma-3-27b-it",
    "gemma-3n-e4b-it",
}

# ── Model tier table (May 2026) ───────────────────────────────────────────────
# Concrete quotas vary by project tier; use AI Studio as the source of truth.
# These tiers keep the default on a stable low-latency model while allowing
# quality/premium opt-ins from config.yaml.
_DEFAULT_MODELS = {
    "fast":     "gemini-2.5-flash-lite",   # quick/interim queries
    "balanced": "gemini-2.5-flash",        # primary workhorse
    "quality":  "gemini-2.5-pro",          # vision / complex tasks
    "premium":  "gemini-3-flash-preview",  # current-generation preview opt-in
}
_RECOMMENDED_MODEL = "gemini-2.5-flash"


class GeminiProvider(BaseProvider):
    def __init__(self, config):
        super().__init__("gemini", config)
        key = self.pcfg.get("api_key", "")
        if not key:
            self.enabled = False
            return

        # ── Set model defaults if config doesn't specify ──────────────────────
        if not self.pcfg.get("model") and not self.pcfg.get("models"):
            self.pcfg["model"] = _RECOMMENDED_MODEL

        # ── Auto-correct any deprecated/broken model saved in config.yaml ─────
        current_model = self.pcfg.get("model", "")
        if current_model in _DEPRECATED_MODELS:
            logger.warning(
                f"[Gemini] Model '{current_model}' is deprecated/shut-down — "
                f"auto-upgrading to '{_RECOMMENDED_MODEL}'. "
                f"Update ai.providers.gemini.model in config.yaml to silence this."
            )
            self.pcfg["model"] = _RECOMMENDED_MODEL

        # Also fix any tiered models dict
        if self.pcfg.get("models"):
            for tier, m in list(self.pcfg["models"].items()):
                if m in _DEPRECATED_MODELS:
                    replacement = _DEFAULT_MODELS.get(tier, _RECOMMENDED_MODEL)
                    logger.warning(
                        f"[Gemini] Tiered model '{tier}={m}' is deprecated — "
                        f"upgrading to '{replacement}'."
                    )
                    self.pcfg["models"][tier] = replacement

        # ── Thinking budget: controls extended reasoning in 2.5 models ────────
        # Set to 0 to disable thinking (faster, cheaper).
        # Set to -1 to let the model decide (best quality, but slower).
        # Set to 1-24576 for a specific token budget.
        # Default: 0 (disabled) — keeps latency low for real-time use.
        self._thinking_budget = int(
            self.pcfg.get("thinking_budget", 0)
        )

        try:
            from google import genai
            self.client = genai.Client(api_key=key)
            active_model = self.pcfg.get("model") or _RECOMMENDED_MODEL
            logger.info(f"  [OK] Gemini ready → {active_model} (thinking_budget={self._thinking_budget})")
        except Exception as e:
            logger.warning(f"  [FAIL] Gemini: {e}")
            self.enabled = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_gemma_model(model: str) -> bool:
        """Gemma models don't support system_instruction in GenerateContentConfig."""
        return bool(model and model.startswith("gemma"))

    @staticmethod
    def _supports_thinking(model: str) -> bool:
        """Return True for models that support the thinking_budget parameter."""
        # Only 2.5+ models support thinking; older and lite variants do not.
        return bool(model) and (
            "gemini-2.5-flash" in model or
            "gemini-2.5-pro" in model or
            "gemini-3" in model
        ) and "lite" not in model

    def _build_contents(self, model: str, system: str, user: str):
        """
        Build (contents, system_instruction_or_None) for the Gemini API.

        Gemini models support system_instruction in GenerateContentConfig.
        Gemma models raise 400 INVALID_ARGUMENT if system_instruction is set,
        so we inject the system prompt into the user content instead.
        """
        if self._is_gemma_model(model) and system:
            combined = (
                f"<start_of_turn>system\n{system}\n<end_of_turn>\n"
                f"<start_of_turn>user\n{user}\n<end_of_turn>"
            )
            return combined, None
        return user, (system or None)

    def _build_config_kwargs(self, model: str, temperature: float = 0.7, types_module=None) -> dict:
        """Build GenerateContentConfig kwargs, including thinking budget where supported."""
        kwargs = dict(max_output_tokens=self.max_tokens, temperature=temperature)
        if self._thinking_budget != 0 and self._supports_thinking(model):
            # thinking_budget=-1 → dynamic (model decides); >0 → fixed token cap
            if types_module is not None and hasattr(types_module, "ThinkingConfig"):
                kwargs["thinking_config"] = types_module.ThinkingConfig(
                    thinking_budget=self._thinking_budget
                )
            else:
                kwargs["thinking_config"] = {"thinking_budget": self._thinking_budget}
        return kwargs

    # ── Text generation ───────────────────────────────────────────────────────

    async def generate(self, system: str, user: str, tier: str = None) -> str:
        self._pre_request()
        model = self.get_model(tier)
        t0 = time.time()
        try:
            from google.genai import types
            contents, sys_instr = self._build_contents(model, system, user)
            cfg_kwargs = self._build_config_kwargs(model, types_module=types)
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
            cfg_kwargs = self._build_config_kwargs(model, types_module=types)
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

    # ── Vision ────────────────────────────────────────────────────────────────

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
        # Vision prefers quality tier; fall back to balanced
        model = self.get_model(tier or "quality")
        t0 = time.time()
        try:
            from google.genai import types
            _, sys_instr = self._build_contents(model, system, "")
            cfg_kwargs = self._build_config_kwargs(model, temperature=0.4, types_module=types)
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
        model = self.get_model(tier or "quality")
        t0 = time.time()
        tok = 0
        try:
            from google.genai import types
            _, sys_instr = self._build_contents(model, system, "")
            cfg_kwargs = self._build_config_kwargs(model, temperature=0.4, types_module=types)
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
