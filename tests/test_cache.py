import unittest
import time
from ai.cache import ShortQueryCache

class TestShortQueryCacheSemantic(unittest.TestCase):
    def setUp(self):
        self.cache = ShortQueryCache(ttl_s=10.0, enable_fuzzy=True)
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


if __name__ == "__main__":
    unittest.main()
