import unittest
from unittest.mock import MagicMock, patch


def _make_config(**overrides):
    cfg = MagicMock()
    defaults = {
        "ai.mode": "general",
        "detection.auto_detect_questions": True,
        "detection.min_words": 3,
        "detection.sensitivity": 0.5,
        "detection.auto_response_threshold": 0.7,
    }
    defaults.update(overrides)
    cfg.get = lambda key, default=None: defaults.get(key, default)
    return cfg


class TestQuestionDetectorClauseExtraction(unittest.TestCase):
    def setUp(self):
        from ai.detectors.question_detector import QuestionDetector

        self.detector = QuestionDetector(_make_config())

    def test_detect_extracts_last_question_sentence(self):
        text = "I am working on my UI. What is React?"
        res = self.detector.detect(text)
        self.assertEqual(res, "What is React?")

    def test_detect_extracts_mid_sentence_question_clause(self):
        text = "Looking at this file, how can I fix it"
        res = self.detector.detect(text)
        self.assertEqual(res, "how can I fix it")

    def test_extract_question_clause_keeps_real_question_not_trailing_fragment(self):
        text = "what is this? Hello copy."
        res = self.detector.detect(text)
        self.assertEqual(res, "what is this?")

    def test_detect_accepts_natural_compare_question_without_wh_prefix(self):
        text = "difference between setTimeout and setInterval"
        res = self.detector.detect(text)
        self.assertEqual(res, "difference between setTimeout and setInterval")

    def test_detect_accepts_interview_style_prompt(self):
        text = "walk me through a time you handled conflict in your team"
        res = self.detector.detect(text)
        self.assertEqual(res, text)

    def test_detect_prefers_richer_question_clause_over_vague_followup(self):
        text = "what is the difference between double equals and triple equals? what does it do in JavaScript?"
        res = self.detector.detect(text)
        self.assertEqual(
            res,
            "what is the difference between double equals and triple equals?",
        )


class TestQuestionDetectorLearning(unittest.TestCase):
    def setUp(self):
        from ai.detectors.question_detector import QuestionDetector

        self.detector = QuestionDetector(_make_config())

    def test_learn_from_query_uses_extracted_question_clause(self):
        self.detector.learn_from_query("I am working on my UI. What is React?")
        self.assertIn("what is ", self.detector.question_prefixes)
        self.assertIn("what is react ", self.detector.question_prefixes)

    def test_learn_from_query_normalizes_merged_speech_tokens(self):
        self.detector.learn_from_query("Howcan yousee the screen even when the OCR is open?")
        self.assertIn("how can ", self.detector.question_prefixes)
        self.assertIn("how can you ", self.detector.question_prefixes)
        self.assertNotIn("howcan yousee ", self.detector.question_prefixes)


class TestQuestionDetectorHints(unittest.TestCase):
    def setUp(self):
        from ai.detectors.question_detector import QuestionDetector

        self.detector = QuestionDetector(_make_config())

    def test_detect_with_confidence_attaches_language_hint(self):
        res = self.detector.detect_with_confidence(
            "I am working on my UI. What is React?", source="audio"
        )
        self.assertTrue(res.triggered)
        self.assertTrue(res.is_code)
        self.assertEqual(res.language, "react")


class TestQuestionDetectorInterimGuardrails(unittest.TestCase):
    def setUp(self):
        from ai.detectors.question_detector import QuestionDetector

        self.detector = QuestionDetector(
            _make_config(
                **{
                    "detection.interim.enabled": True,
                    "detection.interim.min_words": 3,
                    "detection.interim.stability_ms": 800,
                    "detection.interim.min_confidence": 0.6,
                    "detection.interim.cooldown_s": 6.0,
                    "detection.interim.require_question_signal": True,
                }
            )
        )

    def test_interim_requires_stability_window(self):
        text = "I am working on my UI. What is React"
        with patch("ai.detectors.question_detector.time.time", side_effect=[0.0, 0.9]):
            self.assertIsNone(self.detector.detect_interim_with_guardrails(text))
            self.assertEqual(self.detector.detect_interim_with_guardrails(text), "What is React")

    def test_interim_enforces_cooldown(self):
        text = "What is React"
        with patch("ai.detectors.question_detector.time.time", side_effect=[0.0, 1.0, 2.0]):
            self.assertIsNone(self.detector.detect_interim_with_guardrails(text))
            self.assertEqual(self.detector.detect_interim_with_guardrails(text), "What is React")
            self.assertIsNone(self.detector.detect_interim_with_guardrails(text))


class TestQuestionDetectorFragmentReset(unittest.TestCase):
    def setUp(self):
        from ai.detectors.question_detector import QuestionDetector

        self.detector = QuestionDetector(_make_config(**{"detection.fragment_ttl_s": 4.0}))

    def test_detect_resets_stale_fragment_buffer(self):
        with patch("ai.detectors.question_detector.time.time", side_effect=[0.0, 6.0, 6.1, 6.2]):
            self.assertIsNone(self.detector.detect("for cloud helps"))
            self.assertIsNotNone(self.detector.detect("What is React?"))
        self.assertEqual(list(self.detector.fragment_buffer), [])

    def test_reset_turn_state_clears_debounce_and_interim_state(self):
        self.detector._last_text = "what is react"
        self.detector._last_trigger_time = 12.0
        self.detector.fragment_buffer.append("stale fragment")
        self.detector._fragment_last_seen_at = 9.0
        self.detector._interim_last_key = "what is react"
        self.detector._interim_first_seen_at = 8.0
        self.detector._interim_last_trigger_at = 11.0

        self.detector.reset_turn_state("unit-test")

        self.assertEqual(self.detector._last_text, "")
        self.assertEqual(self.detector._last_trigger_time, 0.0)
        self.assertEqual(list(self.detector.fragment_buffer), [])
        self.assertEqual(self.detector._fragment_last_seen_at, 0.0)
        self.assertEqual(self.detector._interim_last_key, "")
        self.assertEqual(self.detector._interim_first_seen_at, 0.0)
        self.assertEqual(self.detector._interim_last_trigger_at, 0.0)


if __name__ == "__main__":
    unittest.main()
