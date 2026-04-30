# Audio Fixture Format

This directory holds real recorded speech fixtures for the local ASR benchmark.

Fastest way to add one:

```powershell
python scripts/capture_audio_fixture.py --name react_what_is_react_01 --transcript "what is react" --mode interview --tags react,frontend
```

The helper records one utterance from your microphone, saves the `.wav`, and creates the matching `.wav.json` automatically.

Each fixture is a `.wav` file with a matching `.wav.json` metadata file:

`react_question.wav`
`react_question.wav.json`

## WAV Requirements

- 16-bit PCM WAV
- mono preferred
- 16 kHz preferred
- include natural leading/trailing silence

The benchmark will resample to `16 kHz` if needed, but keeping fixtures at the target rate makes comparisons cleaner.

## Metadata Schema

```json
{
  "expected_transcript": "what is react",
  "expected_utterance_end_ms": 1850,
  "expected_segments": 1,
  "mode": "interview",
  "tags": ["react", "frontend", "short_question"],
  "notes": "Mild background fan noise."
}
```

## Field Notes

- `expected_transcript`: required ground-truth text for WER/CER scoring
- `expected_utterance_end_ms`: optional expected end-of-speech point in milliseconds from the start of the file
- `expected_segments`: optional expected number of emitted chunks for the utterance
- `mode`: optional scenario label such as `general`, `interview`, or `coding`
- `tags`: optional keywords for filtering and later analysis
- `notes`: optional free-form capture notes

## Recording Guidance

- Capture short spoken questions first.
- Keep one utterance per file.
- Include realistic frontend/coding vocabulary.
- Add ambiguous cases separately instead of mixing them into the same fixture.
- Prefer a mix of clean audio and mildly noisy real desktop conditions.

## Suggested Starter Corpus

- "What is React?"
- "Can you explain hooks in React?"
- "How can I boost performance?"
- "What do you see on the screen now?"
- "Why do we use `useEffect`?"
- "What is act?" with interview/frontend context noted in metadata
