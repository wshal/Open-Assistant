import json

data = json.loads(open(r'C:\Users\Vishal\Desktop\Open Assist\benchmarks\audio_asr_matrix_sweep.json').read())

# Focus: fixtures that had missing-intro-word pattern
target_fixtures = [
    "js_closure_different_values_01.wav",
    "react_unexpected_rerenders_props_01.wav",
    "react_useeffect_vs_usememo_01.wav",
    "screen_code_obvious_api_bug_01.wav",
    "screen_error_message_check_first_01.wav",
    "react_context_api_vs_prop_drilling_01.wav",
    "react_performance_rerenders_01.wav",
    "js_var_let_const_difference_01.wav",
    "react_why_is_usereducer_used_01.wav",
]

# Find best transcript for each across all runs
fixture_best = {}
fixture_ground_truth = {}
for run in data['runs']:
    model = run['whisper_model']
    dec = run['decoder_settings']
    for fx in run['fixtures']:
        fn = fx['filename']
        if fn not in target_fixtures:
            continue
        w = fx.get('wer_normalized', 1.0)
        if fn not in fixture_best or w < fixture_best[fn]['wer']:
            fixture_best[fn] = {
                'wer': w,
                'raw': fx.get('transcript_raw', ''),
                'norm': fx.get('transcript_normalized', ''),
                'model': model,
                'beam': dec['beam_size'],
                'prompt': dec['use_initial_prompt'],
            }

# Load expected transcripts
import os, json as json2
fixture_dir = r'C:\Users\Vishal\Desktop\Open Assist\tests\fixtures\audio_ground_truth'
for f in os.listdir(fixture_dir):
    if f.endswith('.wav.json'):
        meta = json2.loads(open(os.path.join(fixture_dir, f)).read())
        fn = f.replace('.json', '')
        fixture_ground_truth[fn] = meta.get('expected_transcript', meta.get('transcript', ''))

print("=== BEFORE/AFTER: Missing-Intro-Word Fixtures ===\n")
print("NOTE: Previous run (sweep 2) transcripts shown for comparison where known\n")

PREV_KNOWN = {
    "js_closure_different_values_01.wav": "happening in the JavaScript Floujo example. And why does each function remember a different value?",
    "react_unexpected_rerenders_props_01.wav": "Why this React component keeps re-rendering even when I",
    "react_useeffect_vs_usememo_01.wav": "When I should use useEffect, instead of useMemo, in Rea",
    "screen_code_obvious_api_bug_01.wav": "in the code on my screen, and is there an obvious bugin",
    "screen_error_message_check_first_01.wav": "and what this error message on the screen means and wha",
    "react_context_api_vs_prop_drilling_01.wav": "me, what the context API does, and when it is better th",
}

for fn in target_fixtures:
    best = fixture_best.get(fn, {})
    gt = fixture_ground_truth.get(fn, 'N/A')
    prev = PREV_KNOWN.get(fn, '(no prev data)')
    print(f"{'='*70}")
    print(f"FIXTURE : {fn}")
    print(f"EXPECTED: {gt}")
    print(f"PREV    : {prev}")
    print(f"NOW BEST: {best.get('raw', 'N/A')}  [WER={best.get('wer',1):.3f}, {best.get('model')} beam={best.get('beam')} prompt={best.get('prompt')}]")
    wer_change = "SAME"
    print()
