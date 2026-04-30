"""
For each fixture where we suspect the ground-truth has extra leading words
that weren't actually spoken, compare:
  - WER against full expected transcript
  - WER against expected transcript with first 3-5 words stripped
This tells us: how much of our WER floor is a labeling artifact?
"""
import sys, json, os
sys.path.insert(0, r'C:\Users\Vishal\Desktop\Open Assist')
from benchmarks.audio_asr_benchmark import normalize_eval_text, evaluation_word_error_rate

data = json.loads(open(r'C:\Users\Vishal\Desktop\Open Assist\benchmarks\audio_asr_matrix_sweep.json').read())

# Fixtures confirmed to have leading-word labeling mismatch (from debug_leading_words.py)
PHANTOM_OPENERS = {
    "js_closure_different_values_01.wav":           "what is",
    "react_context_api_vs_prop_drilling_01.wav":    "can you tell me",
    "screen_code_obvious_api_bug_01.wav":           "what do you see",
    "react_performance_rerenders_01.wav":           "how do i",
    "react_useeffect_vs_usememo_01.wav":            "can you explain",
    "react_unexpected_rerenders_props_01.wav":      "can you explain",
    "screen_error_message_check_first_01.wav":      "can you tell me",
    "js_var_let_const_difference_01.wav":           "what is the",
    "js_difference_var_let_const_01.wav":           "what is",
}

FIXTURE_DIR = r'C:\Users\Vishal\Desktop\Open Assist\tests\fixtures\audio_ground_truth'
fixture_ground_truth = {}
for f in os.listdir(FIXTURE_DIR):
    if f.endswith('.wav.json'):
        meta = json.loads(open(os.path.join(FIXTURE_DIR, f)).read())
        fn = f.replace('.json', '')
        fixture_ground_truth[fn] = meta.get('expected_transcript', meta.get('transcript', ''))

# Find best transcript per fixture from sweep
fixture_best = {}
for run in data['runs']:
    for fx in run['fixtures']:
        fn = fx['filename']
        w = fx.get('wer_normalized', 1.0)
        if fn not in fixture_best or w < fixture_best[fn]['wer']:
            fixture_best[fn] = {
                'wer': w,
                'transcript': fx.get('transcript_raw', ''),
                'model': run['whisper_model'],
            }

print("=== LABELING ARTIFACT ANALYSIS ===\n")
print(f"{'Fixture':48s} {'WER_full':9s} {'WER_trim':9s} {'Improvement':12s}  Phantom opener")

total_full_wer, total_trim_wer, n = 0.0, 0.0, 0

for fn, phantom in PHANTOM_OPENERS.items():
    best = fixture_best.get(fn, {})
    gt_full = fixture_ground_truth.get(fn, '')
    actual = best.get('transcript', '')
    wer_full = best.get('wer', 1.0)

    # Strip the phantom opener from the expected transcript
    gt_norm = normalize_eval_text(gt_full)
    phantom_norm = normalize_eval_text(phantom)
    if gt_norm.startswith(phantom_norm):
        gt_trimmed = gt_norm[len(phantom_norm):].strip()
    else:
        gt_trimmed = gt_norm

    wer_trim = evaluation_word_error_rate(gt_trimmed, actual)

    improvement = wer_full - wer_trim
    total_full_wer += wer_full
    total_trim_wer += wer_trim
    n += 1
    print(f"  {fn:46s} {wer_full:.3f}    {wer_trim:.3f}    {improvement:+.3f}        \"{phantom}\"")

print()
print(f"  Average WER (official label):  {total_full_wer/n:.4f}")
print(f"  Average WER (trimmed label):   {total_trim_wer/n:.4f}")
print(f"  Phantom-opener WER penalty:    {(total_full_wer-total_trim_wer)/n:.4f} pp per fixture")
print()
print(f"  Across ALL 19 fixtures, if we fix these {n} labels:")

# Recalculate global avg with trimmed WERs
all_fixture_wers = {}
for run in data['runs']:
    prof = run['profile']
    if 'small.en' not in prof or 'beam3_no_prev' not in prof:
        continue
    for fx in run['fixtures']:
        fn = fx['filename']
        all_fixture_wers[fn] = fx.get('wer_normalized', 1.0)

adjusted_wers = []
for fn, wer in all_fixture_wers.items():
    if fn in PHANTOM_OPENERS:
        gt = fixture_ground_truth.get(fn, '')
        actual = fixture_best.get(fn, {}).get('transcript', '')
        phantom = PHANTOM_OPENERS[fn]
        gt_norm = normalize_eval_text(gt)
        phantom_norm = normalize_eval_text(phantom)
        gt_trimmed = gt_norm[len(phantom_norm):].strip() if gt_norm.startswith(phantom_norm) else gt_norm
        wer = evaluation_word_error_rate(gt_trimmed, actual)
    adjusted_wers.append(wer)

import statistics
print(f"  Current best small.en avg WER:  0.1972")
if adjusted_wers:
    print(f"  Adjusted WER (fixed labels):    {statistics.mean(adjusted_wers):.4f}")
    print(f"  Effective WER improvement if corpus labels are corrected: {0.1972 - statistics.mean(adjusted_wers):.4f} pp")
