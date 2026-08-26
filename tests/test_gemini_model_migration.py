import unittest

from core.config import Config
from ai.providers.gemini_provider import GeminiProvider


class GeminiModelMigrationTests(unittest.TestCase):
    def test_persisted_gemini_models_migrate_to_stable_flash(self):
        config = Config.__new__(Config)
        config._data = {
            "ai": {"providers": {"gemini": {
                "model": "gemini-2.5-flash",
                "models": {
                    "fast": "gemini-2.5-flash-lite",
                    "quality": "gemini-2.5-pro",
                },
            }}}
        }

        self.assertTrue(config._migrate_deprecated_models())
        gemini = config._data["ai"]["providers"]["gemini"]
        self.assertEqual(gemini["model"], "gemini-3.5-flash")
        self.assertEqual(gemini["models"]["fast"], "gemini-3.1-flash-lite")
        self.assertEqual(gemini["models"]["quality"], "gemini-3.5-flash")

    def test_gemini_3_tiers_keep_fast_and_vision_capable_defaults(self):
        self.assertEqual(GeminiProvider._supports_thinking("gemini-3.1-flash-lite"), True)
        self.assertEqual(GeminiProvider._supports_thinking("gemini-3.5-flash"), True)

    def test_zero_thinking_budget_maps_to_minimal_for_gemini_3(self):
        provider = GeminiProvider.__new__(GeminiProvider)
        provider.max_tokens = 4096
        provider._thinking_budget = 0
        from google.genai import types

        kwargs = provider._build_config_kwargs(
            "gemini-3.5-flash", types_module=types
        )
        self.assertEqual(kwargs["thinking_config"].thinking_level.value, "MINIMAL")
        self.assertNotIn("temperature", kwargs)


if __name__ == "__main__":
    unittest.main()
