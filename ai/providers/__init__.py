"""Provider registry - discovers and initialises all AI providers from config."""

import asyncio
from utils.logger import setup_logger

logger = setup_logger(__name__)


def _is_health_check_custom(provider) -> bool:
    from ai.providers.base import BaseProvider
    return isinstance(provider, BaseProvider) and provider.supports_health_check()


def init_providers(config) -> dict:
    """Instantiate every configured provider and return the enabled ones."""
    from ai.providers.groq_provider import GroqProvider
    from ai.providers.gemini_provider import GeminiProvider
    from ai.providers.cerebras_provider import CerebrasProvider
    from ai.providers.mistral_provider import MistralProvider
    from ai.providers.cohere_provider import CohereProvider
    from ai.providers.ollama_provider import OllamaProvider
    from ai.providers.openai_compat import OpenAICompatProvider

    candidates = {
        "groq":       lambda: GroqProvider(config),
        "gemini":     lambda: GeminiProvider(config),
        "cerebras":   lambda: CerebrasProvider(config),
        "sambanova":  lambda: OpenAICompatProvider("sambanova", config, "https://api.sambanova.ai/v1"),
        "together":   lambda: OpenAICompatProvider("together", config, "https://api.together.xyz/v1"),
        "openrouter": lambda: OpenAICompatProvider("openrouter", config, "https://openrouter.ai/api/v1"),
        "hyperbolic": lambda: OpenAICompatProvider("hyperbolic", config, "https://api.hyperbolic.xyz/v1"),
        "mistral":    lambda: MistralProvider(config),
        "cohere":     lambda: CohereProvider(config),
        "ollama":     lambda: OllamaProvider(config),
    }

    try:
        from ai.providers.openai_provider import OpenAIProvider
        candidates["openai"] = lambda: OpenAIProvider(config)
    except ImportError:
        pass

    try:
        from ai.providers.anthropic_provider import AnthropicProvider
        candidates["anthropic"] = lambda: AnthropicProvider(config)
    except ImportError:
        pass

    providers = {}
    validate = config.get("ai.providers.validate_on_init", False)
    timeout = config.get("ai.providers.health_check_timeout", 5)

    for name, factory in candidates.items():
        try:
            prov = factory()
            if prov.enabled and validate and _is_health_check_custom(prov):
                try:
                    ok = asyncio.run(asyncio.wait_for(prov.health_check(), timeout=timeout))
                    if not ok:
                        prov.enabled = False
                        logger.warning(f"  x {name} failed health check")
                except Exception as exc:
                    prov.enabled = False
                    logger.warning(f"  x {name} health check failed: {exc}")

            if prov.enabled:
                providers[name] = prov
        except Exception as exc:
            logger.warning(f"  x {name} failed to load: {exc}")

    active = list(providers.keys())
    if active:
        logger.info(f"Providers active: {', '.join(active)}")
    else:
        logger.warning("No providers active - add API keys in Settings")

    return providers
