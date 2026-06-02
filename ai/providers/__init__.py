"""Provider registry - discovers and initialises all AI providers from config."""

import time
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _is_health_check_custom(provider) -> bool:
    from ai.providers.base import BaseProvider
    return isinstance(provider, BaseProvider) and provider.supports_health_check()


def _provider_is_enabled(config, name: str) -> bool:
    try:
        enabled = config.get(f"ai.providers.{name}.enabled", True)
    except Exception:
        enabled = True
    return bool(enabled)


def _provider_has_key(config, name: str) -> bool:
    checker = getattr(config, "has_provider_key", None)
    if callable(checker):
        try:
            return bool(checker(name))
        except Exception:
            pass

    getter = getattr(config, "get_api_key", None)
    if callable(getter):
        try:
            return bool(str(getter(name) or "").strip())
        except Exception:
            pass

    try:
        raw = config.get(f"ai.providers.{name}.api_key", "")
    except Exception:
        raw = ""
    return bool(str(raw or "").strip()) or name == "ollama"


def init_providers(config) -> dict:
    """Instantiate every configured provider and return the enabled ones."""
    def _groq():
        from ai.providers.groq_provider import GroqProvider
        return GroqProvider(config)

    def _gemini():
        from ai.providers.gemini_provider import GeminiProvider
        return GeminiProvider(config)

    def _cerebras():
        from ai.providers.cerebras_provider import CerebrasProvider
        return CerebrasProvider(config)

    def _sambanova():
        from ai.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider("sambanova", config, "https://api.sambanova.ai/v1")

    def _together():
        from ai.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider("together", config, "https://api.together.xyz/v1")

    def _openrouter():
        from ai.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider("openrouter", config, "https://openrouter.ai/api/v1")

    def _hyperbolic():
        from ai.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider("hyperbolic", config, "https://api.hyperbolic.xyz/v1")

    def _mistral():
        from ai.providers.mistral_provider import MistralProvider
        return MistralProvider(config)

    def _cohere():
        from ai.providers.cohere_provider import CohereProvider
        return CohereProvider(config)

    def _ollama():
        from ai.providers.ollama_provider import OllamaProvider
        return OllamaProvider(config)

    candidate_factories = {
        "groq": _groq,
        "gemini": _gemini,
        "cerebras": _cerebras,
        "sambanova": _sambanova,
        "together": _together,
        "openrouter": _openrouter,
        "hyperbolic": _hyperbolic,
        "mistral": _mistral,
        "cohere": _cohere,
        "ollama": _ollama,
    }

    try:
        def _openai():
            from ai.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(config)

        candidate_factories["openai"] = _openai
    except ImportError:
        pass

    try:
        def _anthropic():
            from ai.providers.anthropic_provider import AnthropicProvider
            return AnthropicProvider(config)

        candidate_factories["anthropic"] = _anthropic
    except ImportError:
        pass

    providers = {}
    started_at = time.time()
    skipped = []

    # First, instantiate only providers that are both enabled and configured.
    for name, factory in candidate_factories.items():
        if not _provider_is_enabled(config, name):
            skipped.append(f"{name} (disabled)")
            continue
        if name != "ollama" and not _provider_has_key(config, name):
            skipped.append(f"{name} (no key)")
            continue
        try:
            prov = factory()
            if prov.enabled:
                providers[name] = prov
        except Exception as exc:
            logger.warning(f"  x {name} failed to load: {exc}")

    if skipped:
        logger.info("Skipping unconfigured providers: %s", ", ".join(skipped))
    if providers and bool(config.get("ai.providers.validate_on_init", True)):
        # Validation is now kicked off by AIEngine.warmup() on the app loop so
        # it runs in parallel with Whisper, OCR, and the rest of startup.
        # Keeping it out of this hot path removes a large synchronous stall.
        logger.info(
            "Provider validation deferred to background (%d provider(s) loaded)",
            len(providers),
        )

    active = [
        name
        for name, prov in providers.items()
        if getattr(prov, "health_state", lambda: "unknown")() != "down"
    ]
    if active:
        logger.info(f"Providers active: {', '.join(active)}")
    else:
        logger.warning("No providers active - add API keys in Settings")

    return providers
