"""Regression tests for the Third Wave audio-pipeline fixes.

Covers:
  * H-A1 — `_is_hall` whole-utterance matching (must NOT eat legitimate speech
    that contains a hallucination phrase as a substring).
  * H-A4 — `_required_silence_blocks` honours `set_question_complete_hint`
    by returning a tighter tail.
  * M-F1 — `OpenAssistApp.toggle_audio` updates `state.is_muted` before the
    hardware toggle and reconciles when the hardware response differs.

Cleanly skipped when PyQt6 is not installed (matches existing test policy).
"""
import sys
import unittest
from pathlib import Path

import pytest

# Subjects pull in PyQt6 transitively.  Skip the file early on minimal envs
# (the existing pytest subset already reports ~36 such skips).
pytest.importorskip("PyQt6")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.audio import AudioCapture  # noqa: E402


class _ConfigStub:
    def __init__(self, settings=None):
        self._settings = settings or {}

    def get(self, path, default=None):
        return self._settings.get(path, default)

    def set(self, path, value):
        self._settings[path] = value

    def save(self):
        pass

    def get_api_key(self, _provider):
        return ""


def _make_audio(settings=None):
    return AudioCapture(_ConfigStub(settings or {"capture.audio.mode": "system"}))


class IsHallTests(unittest.TestCase):
    """H-A1: whole-utterance matching, hard substrings match anywhere."""

    def setUp(self):
        self.audio = _make_audio()

    def test_short_hallucinations_are_filtered(self):
        for t in (
            "thank you",
            "Thank you.",
            "Thanks for watching!",
            "please subscribe",
            "Subscribe.",
            "  thank   you  ",
        ):
            with self.subTest(text=t):
                self.assertTrue(self.audio._is_hall(t))

    def test_legitimate_speech_containing_phrase_passes(self):
        for t in (
            "thank you for that explanation",
            "thank you for that very thorough explanation of the algorithm",
            "I would like to subscribe to this newsletter",
            "how does the cache work",
        ):
            with self.subTest(text=t):
                self.assertFalse(self.audio._is_hall(t))

    def test_hard_substrings_match_anywhere(self):
        for t in ("Visit www.example.com", "[music]", "subtitles by amara.org"):
            with self.subTest(text=t):
                self.assertTrue(self.audio._is_hall(t))

    def test_empty_input_is_filtered(self):
        self.assertTrue(self.audio._is_hall(""))
        self.assertTrue(self.audio._is_hall(None))


class RequiredSilenceBlocksTests(unittest.TestCase):
    """H-A4: question-complete hint shortens the silence tail."""

    def setUp(self):
        self.audio = _make_audio()

    def test_hint_off_returns_full_window_for_long_speech(self):
        # Simulate an utterance that has been speaking for 5s (above all
        # short-utterance thresholds).  Without the hint we expect the
        # full configured silence_blocks.
        import time as _t
        speech_started = _t.time() - 5.0
        self.audio._question_complete_hint = False
        baseline = self.audio._required_silence_blocks(speech_started)
        self.assertEqual(baseline, self.audio.silence_blocks)

    def test_hint_on_returns_tighter_window(self):
        import time as _t
        speech_started = _t.time() - 5.0
        self.audio._question_complete_hint = False
        baseline = self.audio._required_silence_blocks(speech_started)
        self.audio._question_complete_hint = True
        hinted = self.audio._required_silence_blocks(speech_started)
        self.assertLessEqual(hinted, baseline)
        # And the hinted tail must be floored at _post_chunk_silence_blocks.
        self.assertGreaterEqual(hinted, self.audio._post_chunk_silence_blocks)

    def test_setter_clears_and_sets_hint(self):
        self.audio.set_question_complete_hint(True)
        self.assertTrue(self.audio._question_complete_hint)
        self.audio.set_question_complete_hint(False)
        self.assertFalse(self.audio._question_complete_hint)


class ToggleAudioReconciliationTests(unittest.TestCase):
    """M-F1: state.is_muted reconciled to audio.toggle() return value.

    In the real app AudioCapture._muted is synced via the muted_changed signal,
    so audio.toggle() is the authoritative source.  state.is_muted is set to
    whatever audio.toggle() returns.  Pre-setting state before the toggle would
    cause a double-flip via the signal and must NOT be done.
    """

    def test_state_reflects_audio_toggle_return_value(self):
        from types import SimpleNamespace

        toggle_calls = []

        class _Audio:
            def __init__(self, response):
                self._response = response

            def toggle(self):
                toggle_calls.append(True)
                return self._response

        class _State:
            is_muted = False

        # Happy path: audio.toggle() returns True (muted).
        app = SimpleNamespace(state=_State(), audio=_Audio(response=True))
        from core.app import OpenAssistApp
        OpenAssistApp.toggle_audio(app)
        self.assertTrue(app.state.is_muted)
        self.assertEqual(len(toggle_calls), 1)

        # Second toggle: audio.toggle() returns False (unmuted).
        toggle_calls.clear()
        app2 = SimpleNamespace(state=_State(), audio=_Audio(response=False))
        app2.state.is_muted = True
        OpenAssistApp.toggle_audio(app2)
        self.assertFalse(app2.state.is_muted)
        self.assertEqual(len(toggle_calls), 1)




if __name__ == "__main__":
    unittest.main()
