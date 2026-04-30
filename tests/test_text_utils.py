import unittest

from utils.text_utils import is_likely_fragment, normalize_transcript


class TextUtilsTests(unittest.TestCase):
    def test_normalize_transcript_repairs_merged_question_words(self):
        self.assertEqual(
            normalize_transcript("Whocan tellme what is React?"),
            "Who can tell me what is React?",
        )

    def test_normalize_transcript_repairs_merged_followup_words(self):
        self.assertEqual(
            normalize_transcript("and why do weuse hooksin React?"),
            "and why do we use hooks in React?",
        )

    def test_is_likely_fragment_rejects_low_information_scrap(self):
        self.assertTrue(is_likely_fragment("for Cloud Helps."))

    def test_is_likely_fragment_keeps_real_question(self):
        self.assertFalse(is_likely_fragment("Why do we use hooks in React?"))

    def test_normalize_transcript_keeps_useeffect_phrase_intact(self):
        self.assertEqual(
            normalize_transcript("Why do we use useEffect hook?"),
            "Why do we use useEffect hook?",
        )


if __name__ == "__main__":
    unittest.main()
