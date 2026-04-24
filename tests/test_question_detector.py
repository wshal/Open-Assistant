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


class TestQuestionDetectorLearning(unittest.TestCase):
    def setUp(self):
        from ai.detectors.question_detector import QuestionDetector

        self.detector = QuestionDetector(_make_config())

    def test_learn_from_query_uses_extracted_question_clause(self):
        self.detector.learn_from_query("I am working on my UI. What is React?")
        self.assertIn("what is ", self.detector.question_prefixes)
        self.assertIn("what is react ", self.detector.question_prefixes)


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


if __name__ == "__main__":
    unittest.main()
