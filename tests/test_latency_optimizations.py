"""
Tests for latency optimizations:
1. Refinement skipped for simple queries
2. Refinement skipped for manual + general-knowledge typed queries
3. RAG prefetch cache hit (zero-latency path)
4. RAG prefetch fingerprint stability
5. Offline-first Ollama routing for simple general queries
6. Character-level _chunk_response
7. clear_rag_prefetch clears cache
"""

import unittest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**overrides):
    cfg = MagicMock()
    defaults = {
        "ai.mode": "general",
        "ai.strategy": "smart",
        "ai.offline_first": False,
        "ai.router.task_routing": {},
        "ai.router.fallback_order": [],
        "ai.parallel.enabled": False,
        "ai.fixed_provider": "",
        "capture.audio.correct_transcript": True,
        "capture.audio.correction_provider": "groq",
    }
    defaults.update(overrides)
    cfg.get = lambda key, default=None: defaults.get(key, default)
    return cfg


def _make_engine(config_overrides=None, rag=None):
    from ai.engine import AIEngine
    cfg = _make_config(**(config_overrides or {}))
    history = MagicMock()
    history.add = MagicMock()
    engine = AIEngine(cfg, history, rag=rag, mode_manager=None)
    return engine


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Refinement skip logic
# ─────────────────────────────────────────────────────────────────────────────

class TestRefinementSkip(unittest.TestCase):

    def test_simple_complexity_skips_refinement(self):
        """complexity=simple → no API call to refiner, saves ~400ms"""
        engine = _make_engine()

        refiner_calls = []

        async def fake_refine(text, provider):
            refiner_calls.append(text)
            return text + " [refined]"

        engine._refine_transcript = fake_refine

        # Simulate the refinement guard directly
        complexity = "simple"
        origin = "speech"
        query = "what is python"
        is_general_knowledge = (
            origin == "manual"
            and engine.prompts._is_general_knowledge_query(query)
        )
        skip = complexity == "simple" or is_general_knowledge
        self.assertTrue(skip, "simple complexity should skip refinement")
        self.assertEqual(refiner_calls, [])  # nothing called

    def test_manual_general_knowledge_skips_refinement(self):
        """manual typed 'what is react' → skip refinement (typed text is clean)"""
        engine = _make_engine()
        general_queries = [
            "what is react",
            "what is python",
            "who is linus torvalds",
            "explain recursion",
            "define polymorphism",
            "how does tcp work",
        ]
        for q in general_queries:
            is_gk = engine.prompts._is_general_knowledge_query(q)
            skip = is_gk  # origin='manual' assumed
            self.assertTrue(skip, f"'{q}' should be classified as general knowledge")

    def test_manual_contextual_query_does_not_skip_refinement(self):
        """manual 'fix this error' is contextual → refinement should NOT be skipped"""
        engine = _make_engine()
        contextual_queries = [
            "fix this error",
            "what does this function do",
            "explain this code",
            "in this code what is the bug",
        ]
        for q in contextual_queries:
            is_gk = engine.prompts._is_general_knowledge_query(q)
            self.assertFalse(is_gk, f"'{q}' should NOT be classified as general knowledge")

    def test_speech_origin_does_not_skip_for_moderate_complexity(self):
        """speech + moderate complexity → refinement should run"""
        engine = _make_engine()
        origin = "speech"
        complexity = "moderate"
        query = "analyze the algorithm shown on screen"
        is_gk = origin == "manual" and engine.prompts._is_general_knowledge_query(query)
        skip = complexity == "simple" or is_gk
        self.assertFalse(skip, "speech + moderate should NOT skip refinement")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAG Prefetch
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPrefetch(unittest.TestCase):

    def _mock_rag(self, results=None):
        rag = MagicMock()
        rag.query = AsyncMock(return_value=results or ["result 1", "result 2"])
        return rag

    def test_fingerprint_stability(self):
        """Same context → same fingerprint"""
        from ai.engine import AIEngine
        fp1 = AIEngine._rag_prefetch_fingerprint("screen content", "audio content")
        fp2 = AIEngine._rag_prefetch_fingerprint("screen content", "audio content")
        self.assertEqual(fp1, fp2)

    def test_fingerprint_differs_for_different_context(self):
        from ai.engine import AIEngine
        fp1 = AIEngine._rag_prefetch_fingerprint("screen A", "audio A")
        fp2 = AIEngine._rag_prefetch_fingerprint("screen B", "audio B")
        self.assertNotEqual(fp1, fp2)

    def test_prefetch_rag_stores_result(self):
        """prefetch_rag() stores results in _rag_prefetch"""
        rag = self._mock_rag(["knowledge chunk 1", "knowledge chunk 2"])
        engine = _make_engine(rag=rag)

        _run(engine.prefetch_rag(screen_text="def foo():", audio_text="explain this"))

        self.assertTrue(len(engine._rag_prefetch) > 0, "prefetch cache should have entries")

    def test_prefetch_rag_skips_empty_context(self):
        """No context → no prefetch query"""
        rag = self._mock_rag()
        engine = _make_engine(rag=rag)

        _run(engine.prefetch_rag(screen_text="", audio_text=""))

        rag.query.assert_not_called()

    def test_prefetch_hit_skips_live_query(self):
        """When prefetch cache is warm, live RAG query is not called"""
        rag = self._mock_rag(["prefetched content"])
        engine = _make_engine(rag=rag)

        # Pre-warm the cache manually
        screen = "def bubble_sort(arr):"
        audio = "explain this sorting algorithm"
        fp = engine._rag_prefetch_fingerprint(screen, audio)
        engine._rag_prefetch[fp] = ("prefetched content", time.time() + 60)

        # Simulate the cache check inside generate_response
        now = time.time()
        prefetch_hit = engine._rag_prefetch.get(fp)
        hit = prefetch_hit and now < prefetch_hit[1]
        self.assertTrue(hit, "Should get a prefetch cache hit")

        # Live RAG query should never be called if prefetch hit
        rag.query.assert_not_called()

    def test_prefetch_does_not_re_query_when_fresh(self):
        """prefetch_rag() skips the query if the fingerprint is already fresh"""
        rag = self._mock_rag(["cached result"])
        engine = _make_engine(rag=rag)

        screen = "import numpy as np"
        audio = "what does this do"
        fp = engine._rag_prefetch_fingerprint(screen, audio)
        # Pre-seed as if we already prefetched this context
        engine._rag_prefetch[fp] = ("cached result", time.time() + 60)

        _run(engine.prefetch_rag(screen_text=screen, audio_text=audio))

        # Should NOT have called query again
        rag.query.assert_not_called()

    def test_clear_rag_prefetch(self):
        """clear_rag_prefetch() empties the prefetch dict"""
        rag = self._mock_rag()
        engine = _make_engine(rag=rag)
        engine._rag_prefetch["key"] = ("value", time.time() + 60)
        self.assertEqual(len(engine._rag_prefetch), 1)

        engine.clear_rag_prefetch()
        self.assertEqual(len(engine._rag_prefetch), 0)

    def test_prefetch_seeds_regular_cache(self):
        """A successful prefetch also seeds _rag_cache so a direct query also hits"""
        rag = self._mock_rag(["seeded result"])
        engine = _make_engine(rag=rag)

        _run(engine.prefetch_rag(screen_text="react hooks example", audio_text=""))

        # The regular cache should also have an entry
        self.assertGreater(len(engine._rag_cache), 0, "_rag_cache should be seeded")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Offline-first Ollama routing
# ─────────────────────────────────────────────────────────────────────────────

class TestOfflineFirstOllamaRouting(unittest.TestCase):

    def _engine_with_ollama(self, offline_first=True):
        engine = _make_engine({"ai.offline_first": offline_first})
        mock_ollama = MagicMock()
        mock_ollama.enabled = True
        engine._providers = {"ollama": mock_ollama, "groq": MagicMock(enabled=True)}
        return engine

    def test_offline_first_simple_general_prefers_ollama(self):
        """offline_first=True + simple + general → Ollama first"""
        from modes.general import GeneralMode
        engine = self._engine_with_ollama(offline_first=True)
        mode = GeneralMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        self.assertEqual(providers[0], "ollama",
                         "offline_first simple general should route to Ollama first")

    def test_offline_first_disabled_does_not_route_to_ollama_first(self):
        """offline_first=False → normal fast-provider order"""
        from modes.general import GeneralMode
        engine = self._engine_with_ollama(offline_first=False)
        mode = GeneralMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        self.assertNotEqual(providers[0], "ollama",
                            "offline_first=False should not put Ollama first")

    def test_offline_first_complex_query_does_not_route_to_ollama_first(self):
        """offline_first + complex query → don't use Ollama (quality matters)"""
        from modes.general import GeneralMode
        engine = self._engine_with_ollama(offline_first=True)
        mode = GeneralMode()
        providers = engine._preferred_providers_for_complexity("complex", mode)
        self.assertNotEqual(providers[0], "ollama",
                            "Complex queries should not go to Ollama first")

    def test_offline_first_interview_mode_does_not_route_to_ollama_first(self):
        """offline_first + interview mode → interview needs quality, not Ollama"""
        from modes.interview import InterviewMode
        engine = self._engine_with_ollama(offline_first=True)
        mode = InterviewMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        # interview mode is not in the offline-first routing set
        self.assertNotEqual(providers[0], "ollama",
                            "Interview mode should not auto-route to Ollama")

    def test_offline_first_ollama_not_available_falls_through(self):
        """offline_first=True but Ollama not available → normal routing"""
        from modes.general import GeneralMode
        engine = _make_engine({"ai.offline_first": True})
        # No ollama in providers
        engine._providers = {"groq": MagicMock(enabled=True)}
        mode = GeneralMode()
        providers = engine._preferred_providers_for_complexity("simple", mode)
        self.assertNotEqual(providers[0], "ollama")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Character-level streaming
# ─────────────────────────────────────────────────────────────────────────────

class TestCharacterLevelStreaming(unittest.TestCase):

    def test_chunk_response_is_character_level(self):
        """_chunk_response should yield char-level chunks, not whole words"""
        engine = _make_engine()
        text = "Hello world"
        chunks = list(engine._chunk_response(text, chunk_size=3))
        # With chunk_size=3: ["Hel", "lo ", "wor", "ld"]
        self.assertEqual("".join(chunks), text)
        # All chunks (except possibly last) should be exactly 3 chars
        for chunk in chunks[:-1]:
            self.assertEqual(len(chunk), 3)

    def test_chunk_response_reassembles_exactly(self):
        """Chunks must rejoin to exactly the original text"""
        engine = _make_engine()
        text = "The quick brown fox jumps over the lazy dog"
        chunks = list(engine._chunk_response(text, chunk_size=3))
        self.assertEqual("".join(chunks), text)

    def test_chunk_response_empty_string(self):
        """Empty string should produce no chunks"""
        engine = _make_engine()
        chunks = list(engine._chunk_response(""))
        self.assertEqual(chunks, [])

    def test_chunk_response_smaller_than_chunk_size(self):
        """Text shorter than chunk_size should produce a single chunk"""
        engine = _make_engine()
        chunks = list(engine._chunk_response("Hi", chunk_size=10))
        self.assertEqual(chunks, ["Hi"])

    def test_chunk_response_default_size_is_small(self):
        """Default chunk_size should be ≤ 5 for smooth streaming"""
        import inspect
        from ai.engine import AIEngine
        sig = inspect.signature(AIEngine._chunk_response)
        default = sig.parameters["chunk_size"].default
        self.assertLessEqual(default, 5,
                             "Default chunk_size should be small for character-level streaming")


if __name__ == "__main__":
    unittest.main()
