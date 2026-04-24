import unittest
import time
import numpy as np
from unittest.mock import patch, MagicMock
from ai.cache import ShortQueryCache, EmbeddingTier, _EmbedRecord


class TestShortQueryCacheSemantic(unittest.TestCase):
    def setUp(self):
        # Disable embedding tier for signature/fuzzy tests (avoids model download)
        self.cache = ShortQueryCache(ttl_s=10.0, enable_fuzzy=True, enable_embedding=False)
        self.ctx = "visual_state_1"
        self.hist = "history_1"

    def test_semantic_signature_normalization(self):
        # "What is React?" and "Could you explain React library?" should have the same signature
        sig1 = self.cache._get_semantic_signature("What is React?")
        sig2 = self.cache._get_semantic_signature("Could you explain React library?")
        
        self.assertEqual(sig1, "v1:EXPLAIN:react")
        self.assertEqual(sig2, "v1:EXPLAIN:react")

    def test_semantic_cache_hit_paraphrased(self):
        # Set cache with one phrasing
        self.cache.set(
            mode="general",
            query="What is React?",
            context_fp=self.ctx,
            history_fp=self.hist,
            response="React is a UI library.",
            provider="test"
        )
        
        # Get cache with totally different phrasing (but same semantic meaning)
        entry = self.cache.get(
            mode="general",
            query="Can you explain the React library to me?",
            context_fp=self.ctx,
            history_fp=self.hist
        )
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.response, "React is a UI library.")

    def test_semantic_cache_entity_sorting(self):
        # "React hooks" vs "hooks in React" -> order shouldn't matter
        sig1 = self.cache._get_semantic_signature("Tell me about React hooks")
        sig2 = self.cache._get_semantic_signature("How do hooks work in React?")
        
        self.assertEqual(sig1, "v1:EXPLAIN:hooks:react")
        self.assertEqual(sig2, "v1:HOW_TO:hooks:react") # Intent differs, but entities are sorted
        
        # Test exact match on entities if intent is same
        sig3 = self.cache._get_semantic_signature("Explain React hooks")
        self.assertEqual(sig1, sig3)

    def test_context_stability(self):
        # Context fingerprint should ignore audio
        fp1 = self.cache.context_fingerprint(active_window="VSCode", screen="code_snippet", audio="person talking")
        fp2 = self.cache.context_fingerprint(active_window="VSCode", screen="code_snippet", audio="silence")
        
        self.assertEqual(fp1, fp2)

    def test_long_query_support(self):
        # Should support queries longer than 80 chars now
        long_query = "I am looking for a way to implement a custom hook in React that handles window resizing events efficiently"
        self.cache.set(
            mode="general",
            query=long_query,
            context_fp=self.ctx,
            history_fp=self.hist,
            response="Use useWindowSize hook.",
            provider="test"
        )
        
        q2 = "How can I implement custom hooks in React?"
        
        entry = self.cache.get(
            mode="general",
            query=q2,
            context_fp=self.ctx,
            history_fp=self.hist
        )
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.response, "Use useWindowSize hook.")

    def test_multi_mode_isolation(self):
        """Semantic cache must not bleed between different modes."""
        self.cache.set(
            mode="general",
            query="What is React?",
            context_fp=self.ctx, history_fp=self.hist,
            response="React is a UI library.", provider="test"
        )
        self.cache.set(
            mode="coding",
            query="What is React?",
            context_fp=self.ctx, history_fp=self.hist,
            response="React: component-based architecture.", provider="test"
        )

        entry_general = self.cache.get(
            mode="general", query="Can you explain the React framework?",
            context_fp=self.ctx, history_fp=self.hist
        )
        entry_coding = self.cache.get(
            mode="coding", query="Can you explain the React framework?",
            context_fp=self.ctx, history_fp=self.hist
        )

        self.assertIsNotNone(entry_general)
        self.assertIsNotNone(entry_coding)
        self.assertNotEqual(entry_general.response, entry_coding.response)

    def test_signature_no_trailing_colon_when_no_entities(self):
        """Vague queries without known entities must not produce a trailing colon."""
        sig = self.cache._get_semantic_signature("Fix this bug")
        self.assertFalse(sig.endswith(":"), f"Trailing colon found in: {sig!r}")
        self.assertEqual(sig, "v1:TROUBLESHOOT")

    def test_intent_word_boundary_no_false_positive(self):
        """'fixed' should NOT match the TROUBLESHOOT 'fix' intent pattern."""
        sig = self.cache._get_semantic_signature("I already fixed this")
        # Should NOT classify as TROUBLESHOOT
        self.assertNotIn("TROUBLESHOOT", sig)

    def test_multi_context_isolation(self):
        """Semantic cache must not bleed between different context fingerprints."""
        ctx1 = "ctx_window1"
        ctx2 = "ctx_window2"
        
        self.cache.set(
            mode="general",
            query="What is React?",
            context_fp=ctx1, history_fp=self.hist,
            response="Response for window 1.", provider="test"
        )
        self.cache.set(
            mode="general",
            query="What is React?",
            context_fp=ctx2, history_fp=self.hist,
            response="Response for window 2.", provider="test"
        )

        entry1 = self.cache.get(
            mode="general", query="Can you explain React?",
            context_fp=ctx1, history_fp=self.hist
        )
        entry2 = self.cache.get(
            mode="general", query="Can you explain React?",
            context_fp=ctx2, history_fp=self.hist
        )

        self.assertIsNotNone(entry1)
        self.assertIsNotNone(entry2)
        self.assertNotEqual(entry1.response, entry2.response)
        self.assertEqual(entry1.response, "Response for window 1.")
        self.assertEqual(entry2.response, "Response for window 2.")


class TestEmbeddingTier(unittest.TestCase):
    """Tests for the ONNX embedding tier using a mocked model."""

    def _make_tier(self, threshold=0.88):
        tier = EmbeddingTier(
            threshold=threshold,
            vectors_path=Path("data/cache/_test_embed_vectors.npy"),
            meta_path=Path("data/cache/_test_embed_meta.json"),
            ttl_s=30.0,
        )
        return tier

    def _inject_mock_embed(self, tier, vectors: dict):
        """Patch _embed to return deterministic vectors from a dict."""
        def fake_embed(text):
            v = np.array(vectors.get(text, [0.0] * 384), dtype=np.float32)
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-8 else v
        tier._embed = fake_embed

    def test_exact_hit_above_threshold(self):
        tier = self._make_tier(threshold=0.85)
        # Two nearly identical vectors → cosine ~ 1.0 → hit
        v1 = np.random.rand(384).astype(np.float32)
        v2 = v1 + np.random.rand(384).astype(np.float32) * 0.02  # tiny noise

        def fake_embed(text):
            raw = v1 if text == "stored" else v2
            norm = np.linalg.norm(raw)
            return raw / norm
        tier._embed = fake_embed

        rec = _EmbedRecord(mode="general", context_fp="ctx1", cache_query="stored", history_fp="h1")
        tier.add("stored", rec)

        result = tier.find("query", mode="general", context_fp="ctx1")
        self.assertIsNotNone(result)
        self.assertEqual(result.cache_query, "stored")

    def test_no_hit_below_threshold(self):
        tier = self._make_tier(threshold=0.99)  # impossibly high → no hit
        v1 = np.random.rand(384).astype(np.float32)
        v2 = np.random.rand(384).astype(np.float32)  # completely different

        def fake_embed(text):
            raw = v1 if text == "stored" else v2
            norm = np.linalg.norm(raw)
            return raw / norm
        tier._embed = fake_embed

        rec = _EmbedRecord(mode="general", context_fp="ctx1", cache_query="stored", history_fp="h1")
        tier.add("stored", rec)
        result = tier.find("query", mode="general", context_fp="ctx1")
        self.assertIsNone(result)

    def test_mode_isolation(self):
        tier = self._make_tier(threshold=0.85)
        v = np.random.rand(384).astype(np.float32)

        def fake_embed(text):
            norm = np.linalg.norm(v)
            return v / norm
        tier._embed = fake_embed

        rec = _EmbedRecord(mode="coding", context_fp="ctx1", cache_query="stored", history_fp="h1")
        tier.add("stored", rec)

        # Same vector but looking up with mode="general" — must NOT return
        result = tier.find("query", mode="general", context_fp="ctx1")
        self.assertIsNone(result)

    def test_embedding_tier_disabled_by_default_in_base_tests(self):
        """ShortQueryCache with enable_embedding=False must have _embed=None."""
        cache = ShortQueryCache(ttl_s=10.0, enable_embedding=False)
        self.assertIsNone(cache._embed)

    def test_embedding_tier_created_when_enabled(self):
        """ShortQueryCache with enable_embedding=True must create EmbeddingTier."""
        cache = ShortQueryCache(ttl_s=10.0, enable_embedding=True)
        self.assertIsInstance(cache._embed, EmbeddingTier)

    def test_ttl_expiry_skips_record(self):
        """Records older than TTL must not be returned."""
        tier = EmbeddingTier(threshold=0.50, ttl_s=0.01)  # 10ms TTL
        v = np.random.rand(384).astype(np.float32)

        def fake_embed(text):
            norm = np.linalg.norm(v)
            return v / norm
        tier._embed = fake_embed

        rec = _EmbedRecord(mode="general", context_fp="ctx1", cache_query="stored", history_fp="h1")
        tier.add("stored", rec)
        time.sleep(0.05)  # let TTL expire

        result = tier.find("query", mode="general", context_fp="ctx1")
        self.assertIsNone(result)


# Resolve Path for tests
from pathlib import Path

if __name__ == "__main__":
    unittest.main()
