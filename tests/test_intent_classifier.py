"""Tests for the embedding-based IntentClassifier with learning safeguards."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntentClassifierTests(unittest.TestCase):
    """Test that the IntentClassifier correctly categorizes speech intents."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import ai.intent_classifier as intent_classifier

        cls._tmpdir = tempfile.TemporaryDirectory()
        intent_classifier._LEARNED_INTENTS_FILE = (
            Path(cls._tmpdir.name) / "learned_intents.jsonl"
        )
        intent_classifier.IntentClassifier._instance = None
        IntentClassifier = intent_classifier.IntentClassifier
        cls.classifier = IntentClassifier()
        result = cls.classifier.classify("test warmup")
        cls.available = result is not None

    @classmethod
    def tearDownClass(cls):
        tmpdir = getattr(cls, "_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()

    # Question classification

    def test_classify_direct_question(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("What is a closure in JavaScript?")
        self.assertEqual(scores.best_intent, "question")

    def test_classify_how_question(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("How do I merge two dictionaries in Python?")
        self.assertEqual(scores.best_intent, "question")

    def test_classify_imperative_command(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Explain how you would implement a caching layer.")
        self.assertEqual(scores.best_intent, "question")

    def test_classify_modal_question(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Can you explain the difference between TCP and UDP?")
        self.assertEqual(scores.best_intent, "question")

    def test_classify_compound_question(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "What is the expected time complexity in big O notation "
            "and what is the worst case if the tree is unbalanced?"
        )
        self.assertEqual(scores.best_intent, "question")

    def test_classify_novel_question_not_in_exemplars(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "What are the tradeoffs between event sourcing and CQRS?"
        )
        self.assertEqual(scores.best_intent, "question")

    def test_classify_short_technical_question(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("What is React Router and how does it work?")
        self.assertEqual(scores.best_intent, "question")

    # Setup classification

    def test_classify_transition_phrase(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Let's pivot to some CSS basics.")
        self.assertEqual(scores.best_intent, "setup")

    def test_classify_imagination_prompt(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "Imagine we are designing a new public-facing API for our mobile app."
        )
        self.assertEqual(scores.best_intent, "setup")

    def test_classify_context_observation(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "A lot of developers get confused between CSS Grid and Flexbox."
        )
        self.assertEqual(scores.best_intent, "setup")

    def test_classify_resume_observation(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "I was looking at your resume and I see you've used React quite a bit."
        )
        self.assertEqual(scores.best_intent, "setup")

    def test_classify_topic_shift(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Alright, let's talk about scaling.")
        self.assertEqual(scores.best_intent, "setup")

    def test_classify_novel_setup_not_in_exemplars(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "Before we dive in, let me give you some background on the architecture."
        )
        self.assertEqual(scores.best_intent, "setup")

    # Greeting classification

    def test_classify_greeting(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Hello there, can you help me with some coding?")
        self.assertIn(scores.best_intent, ("greeting", "question"))

    def test_classify_acknowledgement(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Okay, got it. That makes sense.")
        self.assertEqual(scores.best_intent, "greeting")

    # Followup classification

    def test_classify_followup_prompt(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify(
            "How did you approach the situation and what was the outcome?"
        )
        self.assertIn(scores.best_intent, ("followup", "question"))

    def test_classify_elaboration_request(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("Can you elaborate on that?")
        self.assertIn(scores.best_intent, ("followup", "question"))

    # Combined is_likely_setup / is_likely_question

    def test_is_likely_setup_agrees_with_regex(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertTrue(self.classifier.is_likely_setup(
            "Let's pivot to some CSS basics.", regex_says_setup=True
        ))

    def test_is_likely_setup_overrides_regex_for_question(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.is_likely_setup(
            "What is the difference between CSS Grid and Flexbox?",
            regex_says_setup=True,
        ))

    def test_is_likely_setup_overrides_regex_for_acknowledgement(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.is_likely_setup(
            "Okay, got it. That makes sense.",
            regex_says_setup=True,
        ))

    def test_is_likely_question_agrees_with_regex(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertTrue(self.classifier.is_likely_question(
            "How do I merge two dictionaries in Python?",
            regex_says_question=True,
        ))

    def test_is_likely_question_does_not_promote_setup(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.is_likely_question(
            "Let's pivot to some CSS basics.", regex_says_question=False,
        ))

    # Edge cases

    def test_classify_empty_returns_none(self):
        self.assertIsNone(self.classifier.classify(""))

    def test_classify_none_returns_none(self):
        self.assertIsNone(self.classifier.classify(None))

    def test_intent_scores_repr(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        scores = self.classifier.classify("What is a closure?")
        self.assertIn("question=", repr(scores))

    def test_classifier_is_singleton(self):
        from ai.intent_classifier import IntentClassifier
        self.assertIs(IntentClassifier(), IntentClassifier())

    # Learning: quality gate tests
    # Reset learned data before learning tests to ensure clean slate.
    # The singleton persists across test modules in the same pytest process,
    # and prior test runs persist to disk; both can cause false failures.

    def test_learn_00_reset_before_tests(self):
        """Reset learned state so learning tests are idempotent."""
        if not self.available: self.skipTest("Embeddings unavailable")
        self.classifier.reset_learned()
        stats = self.classifier.get_stats()
        for cat in ("question", "setup", "greeting", "followup"):
            self.assertEqual(stats["categories"][cat]["learned"], 0)

    def test_learn_rejects_too_short(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn("hello there", "greeting", confidence=0.9))

    def test_learn_rejects_invalid_intent(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "What is a closure in JavaScript?", "invalid_cat", confidence=0.9
        ))

    def test_learn_rejects_exact_duplicate(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "What is a closure in JavaScript?", "question", confidence=0.9
        ))

    def test_learn_rejects_near_duplicate(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "What is a closure in JS?", "question", confidence=0.9
        ))

    def test_learn_rejects_garbled_audio(self):
        """Garbled audio with too many filler words should be rejected."""
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "um yeah so like basically um just like you know basically um",
            "question", confidence=0.8,
        ))

    def test_learn_rejects_system_ui_noise(self):
        """UI/system messages that leak through should be rejected."""
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "Click-through enabled. Press Ctrl+M to restore interaction.",
            "setup", confidence=0.7,
        ))

    def test_learn_rejects_low_alpha_ratio(self):
        """Text with too many non-alphabetic characters should be rejected."""
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "--- ... ??? !!! ### $$$ *** &&& @@@ ~~~",
            "question", confidence=0.7,
        ))

    def test_learn_rejects_repetitive_stutter(self):
        """STT stuttering pattern should be rejected."""
        if not self.available: self.skipTest("Embeddings unavailable")
        self.assertFalse(self.classifier.learn(
            "what are some what are some strategies what are some strategies for scaling",
            "question", confidence=0.8,
        ))

    def test_learn_accepts_novel_exemplar(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        unique = f"How would you design a real-time collaborative document editor using operational transforms for session {int(time.time())}?"
        self.assertTrue(self.classifier.learn(unique, "question", confidence=0.85))

    # Learning: rate limiting

    def test_learn_cooldown_rejects_rapid_same_category(self):
        """Second learn in same category within cooldown should be rejected."""
        if not self.available: self.skipTest("Embeddings unavailable")
        # The test_learn_accepts_novel_exemplar already learned a question
        # within this session, so this should be on cooldown
        result = self.classifier.learn(
            f"What is the CAP theorem and how does it apply to distributed databases designed in session {int(time.time())}?",
            "question", confidence=0.8,
        )
        self.assertFalse(result)

    # Management

    def test_get_stats_structure(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        stats = self.classifier.get_stats()
        self.assertTrue(stats["available"])
        self.assertIn("question", stats["categories"])
        self.assertIn("session_learns", stats)
        self.assertIn("session_limit", stats)
        self.assertIn("max_age_days", stats)
        self.assertGreater(stats["total_exemplars"], 0)
        for cat in ("question", "setup", "greeting", "followup"):
            self.assertIn("builtin", stats["categories"][cat])
            self.assertIn("learned", stats["categories"][cat])
            self.assertIn("capacity_remaining", stats["categories"][cat])

    def test_text_quality_rejects_empty(self):
        from ai.intent_classifier import _text_quality_ok
        ok, reason = _text_quality_ok("")
        self.assertFalse(ok)
        self.assertEqual(reason, "empty")

    def test_text_quality_accepts_clean_question(self):
        from ai.intent_classifier import _text_quality_ok
        ok, reason = _text_quality_ok(
            "What are the tradeoffs between horizontal and vertical database scaling?"
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_text_quality_rejects_fillers(self):
        from ai.intent_classifier import _text_quality_ok
        # Single-word fillers
        ok, reason = _text_quality_ok(
            "um yeah so like basically um just well you know basically like"
        )
        self.assertFalse(ok)
        self.assertIn("filler", reason)

    def test_text_quality_rejects_multiword_fillers(self):
        from ai.intent_classifier import _text_quality_ok
        # Multi-word fillers: "you know" and "i mean" should be counted
        ok, reason = _text_quality_ok(
            "you know i mean sort of kind of you know i mean"
        )
        self.assertFalse(ok)

    def test_learn_persists_to_disk(self):
        if not self.available: self.skipTest("Embeddings unavailable")
        from ai.intent_classifier import _LEARNED_INTENTS_FILE
        if _LEARNED_INTENTS_FILE.exists():
            with open(_LEARNED_INTENTS_FILE, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertGreater(len(lines), 0)
            last = json.loads(lines[-1])
            self.assertIn("text", last)
            self.assertIn("intent", last)
            self.assertIn("confidence", last)
            self.assertIn("learned_at", last)
            self.assertIn("source", last)


if __name__ == "__main__":
    unittest.main()
