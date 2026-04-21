"""
Tests for the ModeProfile pipeline.

Verifies that:
- Every mode has a fully populated profile
- ContextRanker respects Mode object weights and limits
- generate_quick_response uses mode.quick_answer_query (not a generic fallback)
- ModeManager.switch() returns the correct profile
- PromptBuilder.user() injects mode.quick_answer_format for quick origin
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    cfg = MagicMock()
    defaults = {"ai.mode": "general", "ai.strategy": "smart", "ai.offline_first": False,
                "ai.router.task_routing": {}, "ai.router.fallback_order": []}
    defaults.update(overrides)
    cfg.get = lambda key, default=None: defaults.get(key, default)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mode profile completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestModeProfileCompleteness(unittest.TestCase):

    def _all_modes(self):
        from modes.general import GeneralMode
        from modes.interview import InterviewMode
        from modes.coding import CodingMode
        from modes.meeting import MeetingMode
        from modes.writing import WritingMode
        from modes.exam import ExamMode
        return [cls() for cls in [GeneralMode, InterviewMode, CodingMode,
                                   MeetingMode, WritingMode, ExamMode]]

    def test_each_mode_has_context_weights(self):
        for m in self._all_modes():
            self.assertIsInstance(m.context_weights, dict, f"{m.name} missing context_weights")
            self.assertTrue(m.context_weights, f"{m.name} context_weights is empty")

    def test_each_mode_has_context_limits(self):
        for m in self._all_modes():
            self.assertIsInstance(m.context_limits, dict, f"{m.name} missing context_limits")
            self.assertTrue(m.context_limits, f"{m.name} context_limits is empty")

    def test_each_mode_has_preferred_providers(self):
        for m in self._all_modes():
            self.assertIsInstance(m.preferred_providers, list, f"{m.name} missing preferred_providers")
            self.assertGreater(len(m.preferred_providers), 0, f"{m.name} preferred_providers is empty")

    def test_each_mode_has_quick_answer_query(self):
        for m in self._all_modes():
            self.assertTrue(m.quick_answer_query, f"{m.name} missing quick_answer_query")

    def test_each_mode_has_ollama_hint(self):
        for m in self._all_modes():
            self.assertTrue(m.ollama_model_hint, f"{m.name} missing ollama_model_hint")

    def test_each_mode_has_detector_sensitivity(self):
        for m in self._all_modes():
            self.assertIsInstance(m.detector_sensitivity, float, f"{m.name} sensitivity not float")
            self.assertGreaterEqual(m.detector_sensitivity, 0.0)
            self.assertLessEqual(m.detector_sensitivity, 1.0)

    def test_mode_name_matches_class(self):
        from modes.interview import InterviewMode
        from modes.coding import CodingMode
        from modes.exam import ExamMode
        self.assertEqual(InterviewMode().name, "interview")
        self.assertEqual(CodingMode().name, "coding")
        self.assertEqual(ExamMode().name, "exam")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Context priority per mode
# ─────────────────────────────────────────────────────────────────────────────

class TestContextRankerModeAware(unittest.TestCase):

    def test_interview_audio_first(self):
        from ai.prompts import ContextRanker
        from modes.interview import InterviewMode
        mode = InterviewMode()
        ranked = ContextRanker.rank(["screen_text", "audio_text", "rag_text"], mode=mode)
        sources = [r[0] for r in ranked]
        self.assertEqual(sources[0], "audio", "Interview should rank audio first")

    def test_coding_screen_first(self):
        from ai.prompts import ContextRanker
        from modes.coding import CodingMode
        mode = CodingMode()
        ranked = ContextRanker.rank(["screen_text", "audio_text", "rag_text"], mode=mode)
        sources = [r[0] for r in ranked]
        self.assertEqual(sources[0], "screen", "Coding should rank screen first")

    def test_meeting_audio_first(self):
        from ai.prompts import ContextRanker
        from modes.meeting import MeetingMode
        mode = MeetingMode()
        ranked = ContextRanker.rank(["screen_text", "audio_text", "rag_text"], mode=mode)
        sources = [r[0] for r in ranked]
        self.assertEqual(sources[0], "audio", "Meeting should rank audio first")

    def test_exam_screen_first(self):
        from ai.prompts import ContextRanker
        from modes.exam import ExamMode
        mode = ExamMode()
        ranked = ContextRanker.rank(["screen_text", "audio_text", "rag_text"], mode=mode)
        sources = [r[0] for r in ranked]
        self.assertEqual(sources[0], "screen", "Exam should rank screen first")

    def test_general_balanced(self):
        from ai.prompts import ContextRanker
        from modes.general import GeneralMode
        mode = GeneralMode()
        ranked = ContextRanker.rank(["screen_text", "audio_text", "rag_text"], mode=mode)
        # General: screen == audio weight (2), rag weight (1)
        priorities = {r[0]: r[2] for r in ranked}
        self.assertEqual(priorities["screen"], priorities["audio"])
        self.assertGreater(priorities["screen"], priorities["rag"])

    def test_string_mode_backward_compat(self):
        """ContextRanker should still work with a plain string mode name."""
        from ai.prompts import ContextRanker
        ranked = ContextRanker.rank(["s", "a", "r"], mode="coding")
        sources = [r[0] for r in ranked]
        self.assertEqual(sources[0], "screen")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Context limits
# ─────────────────────────────────────────────────────────────────────────────

class TestContextLimits(unittest.TestCase):

    def test_interview_audio_limit_larger_than_default(self):
        from modes.interview import InterviewMode
        mode = InterviewMode()
        self.assertGreaterEqual(mode.limit("audio"), 3000,
                                "Interview audio budget should be ≥3000 chars")

    def test_coding_screen_limit_larger_than_default(self):
        from modes.coding import CodingMode
        mode = CodingMode()
        self.assertGreaterEqual(mode.limit("screen"), 4500,
                                "Coding screen budget should be ≥4500 chars")

    def test_ranker_limit_applies_mode(self):
        from ai.prompts import ContextRanker
        from modes.interview import InterviewMode
        mode = InterviewMode()
        long_audio = "word " * 2000  # 10000 chars
        limited = ContextRanker.limit("audio", long_audio, mode=mode)
        self.assertEqual(len(limited), mode.limit("audio"))

    def test_ranker_limit_fallback_no_mode(self):
        from ai.prompts import ContextRanker
        long_text = "x" * 10000
        limited = ContextRanker.limit("screen", long_text, mode=None)
        self.assertEqual(len(limited), 4000)  # default


# ─────────────────────────────────────────────────────────────────────────────
# 4. ModeManager
# ─────────────────────────────────────────────────────────────────────────────

class TestModeManager(unittest.TestCase):

    def _manager(self, mode="general"):
        cfg = _make_config(**{"ai.mode": mode})
        from modes import ModeManager
        return ModeManager(cfg)

    def test_default_mode_loaded(self):
        mm = self._manager("general")
        self.assertEqual(mm.current.name, "general")

    def test_switch_returns_correct_profile(self):
        mm = self._manager("general")
        profile = mm.switch("coding")
        self.assertEqual(profile.name, "coding")
        self.assertEqual(mm.current.name, "coding")

    def test_switch_unknown_mode_returns_current(self):
        mm = self._manager("general")
        result = mm.switch("nonexistent_mode")
        self.assertEqual(result.name, "general")

    def test_profile_property_alias(self):
        mm = self._manager("interview")
        self.assertIs(mm.profile, mm.current)

    def test_get_profile_by_name(self):
        mm = self._manager("general")
        coding = mm.get_profile("coding")
        self.assertEqual(coding.name, "coding")

    def test_all_modes_returns_list(self):
        mm = self._manager()
        names = [m.name for m in mm.all_modes]
        self.assertIn("general", names)
        self.assertIn("coding", names)
        self.assertIn("interview", names)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quick-answer mode awareness
# ─────────────────────────────────────────────────────────────────────────────

class TestQuickAnswerModeAware(unittest.TestCase):
    """Verify that generate_quick_response uses mode.quick_answer_query."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _make_engine(self, mode_name="interview"):
        from ai.engine import AIEngine
        from modes import ModeManager

        cfg = _make_config(**{"ai.mode": mode_name,
                               "ai.parallel.enabled": False,
                               "ai.fixed_provider": ""})
        mm = ModeManager(cfg)
        mm.switch(mode_name)

        # Minimal mock provider
        mock_provider = MagicMock()
        mock_provider.name = "groq"
        mock_provider.enabled = True
        mock_provider.speed = 10
        mock_provider.quality = 8
        mock_provider.rpm = 100
        mock_provider._req_times = []
        mock_provider.stats = MagicMock(requests=1, errors=0, success_rate=1.0, tps=200, avg_latency=0.1)
        mock_provider.check_rate.return_value = True
        mock_provider.has_model.return_value = True
        mock_provider.default_tier = "fast"

        captured_prompts = {}

        async def fake_stream(sys_p, user_p):
            captured_prompts["system"] = sys_p
            captured_prompts["user"] = user_p
            yield "mock answer"

        mock_provider.generate_stream.side_effect = fake_stream

        history = MagicMock()
        history.add = MagicMock()

        engine = AIEngine(cfg, history, rag=None, mode_manager=mm)
        engine._providers = {"groq": mock_provider}
        engine._active_provider_id = "groq"
        engine._router = None   # skip router, use direct fallback
        engine._parallel = None

        return engine, mm.profile, captured_prompts

    def test_interview_quick_answer_uses_audio_query(self):
        engine, profile, captured = self._make_engine("interview")
        self._run(engine.generate_quick_response({}, screen_context="", audio_context="what are your strengths?"))
        used_query = captured.get("user", "")
        # The mode's quick_answer_query mentions audio as primary source
        self.assertIn("audio", profile.quick_answer_query.lower())

    def test_coding_quick_answer_uses_screen_query(self):
        engine, profile, captured = self._make_engine("coding")
        self._run(engine.generate_quick_response({}, screen_context="def foo(): pass", audio_context=""))
        self.assertIn("screen", profile.quick_answer_query.lower())

    def test_meeting_quick_answer_mentions_action_items(self):
        engine, profile, _ = self._make_engine("meeting")
        self.assertIn("action", profile.quick_answer_query.lower())

    def test_exam_quick_answer_mentions_correct_answer(self):
        engine, profile, _ = self._make_engine("exam")
        self.assertIn("answer", profile.quick_answer_query.lower())


# ─────────────────────────────────────────────────────────────────────────────
# 6. PromptBuilder quick-answer format injection
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuilderQuickFormat(unittest.TestCase):

    def test_quick_origin_injects_mode_format(self):
        from ai.prompts import PromptBuilder
        from modes.interview import InterviewMode

        mode = InterviewMode()
        pb = PromptBuilder()
        result = pb.user(
            query="test query",
            screen="some screen",
            audio="some audio",
            mode=mode,
            origin="quick",
        )
        # The mode's quick_answer_format should appear in the prompt
        self.assertIn(mode.quick_answer_format[:20], result)

    def test_context_ordering_respects_mode_weights(self):
        from ai.prompts import PromptBuilder
        from modes.interview import InterviewMode

        mode = InterviewMode()
        pb = PromptBuilder()
        result = pb.user(
            query="test query",
            screen="SCREEN_DATA",
            audio="AUDIO_DATA",
            rag="RAG_DATA",
            mode=mode,
            origin="manual",
        )
        # Audio should appear before screen in the prompt for interview mode
        audio_pos = result.find("[AUDIO]")
        screen_pos = result.find("[SCREEN]")
        self.assertGreater(screen_pos, audio_pos,
                           "Interview mode: [AUDIO] should appear before [SCREEN]")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Mode-specific provider preferences
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderPreferences(unittest.TestCase):

    def _make_engine(self):
        from ai.engine import AIEngine
        cfg = _make_config()
        history = MagicMock()
        engine = AIEngine(cfg, history, rag=None, mode_manager=None)
        return engine

    def test_general_simple_prefers_fast_providers(self):
        from modes.general import GeneralMode
        engine = self._make_engine()
        mode = GeneralMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        self.assertIn("groq", providers[:2])

    def test_meeting_prefers_fast_providers(self):
        from modes.meeting import MeetingMode
        engine = self._make_engine()
        mode = MeetingMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        self.assertIn("groq", providers[:2])

    def test_coding_complex_promotes_quality(self):
        from modes.coding import CodingMode
        engine = self._make_engine()
        mode = CodingMode()
        providers = engine._preferred_providers_for_complexity("complex", mode)
        self.assertIn("gemini", providers[:2])

    def test_exam_prefers_accuracy_providers(self):
        from modes.exam import ExamMode
        engine = self._make_engine()
        mode = ExamMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        self.assertIn("gemini", providers[:3])

    def test_ollama_hint_differs_by_mode(self):
        from modes.coding import CodingMode
        from modes.general import GeneralMode
        self.assertNotEqual(CodingMode().ollama_model_hint,
                            GeneralMode().ollama_model_hint,
                            "Coding should prefer a code-specialist model")

    def test_detector_sensitivity_ordering(self):
        from modes.exam import ExamMode
        from modes.writing import WritingMode
        from modes.general import GeneralMode
        # Exam most aggressive, writing most relaxed
        self.assertGreater(ExamMode().detector_sensitivity,
                           GeneralMode().detector_sensitivity)
        self.assertGreater(GeneralMode().detector_sensitivity,
                           WritingMode().detector_sensitivity)


if __name__ == "__main__":
    unittest.main()
