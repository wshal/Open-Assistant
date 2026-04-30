import json, statistics
from collections import defaultdict, Counter

data = json.loads(open(r'C:\Users\Vishal\Desktop\Open Assist\benchmarks\audio_asr_matrix_sweep.json').read())
models_in_sweep = data.get('models', [])

profiles = {}
for run in data['runs']:
    prof = run['profile']
    model = run['whisper_model']
    dec = run['decoder_settings']
    wers, deltas, trans_ms, false_late, false_early = [], [], [], 0, 0
    for fx in run['fixtures']:
        w = fx.get('wer_normalized')
        if w is not None: wers.append(w)
        d = fx.get('endpoint_delta_ms')
        if d is not None: deltas.append(d)
        t = fx.get('transcribe_only_ms')
        if t is not None: trans_ms.append(t)
        if fx.get('false_late_cut'): false_late += 1
        if fx.get('false_early_cut'): false_early += 1
    profiles[prof] = {
        'model': model, 'beam': dec['beam_size'],
        'cond_prev': dec['condition_on_previous_text'],
        'vad_filter': dec['vad_filter'], 'prompt': dec['use_initial_prompt'],
        'avg_wer': statistics.mean(wers) if wers else None,
        'median_delta': statistics.median(deltas) if deltas else None,
        'p95_delta': sorted(deltas)[int(len(deltas)*0.95)] if len(deltas) > 1 else None,
        'median_transcribe_ms': statistics.median(trans_ms) if trans_ms else None,
        'false_late': false_late, 'false_early': false_early, 'n': len(wers),
    }

by_model = defaultdict(list)
for k, v in profiles.items():
    by_model[v['model']].append(v)

print('=== PER-MODEL BEST PROFILE (lowest avg WER) ===')
for model in models_in_sweep:
    rows = by_model.get(model, [])
    if not rows: continue
    best = min(rows, key=lambda x: x['avg_wer'] or 99)
    print(f"{model:12s}  best_avg_wer={best['avg_wer']:.4f}  median_delta={best['median_delta']}ms  transcribe_p50={best['median_transcribe_ms']:.1f}ms  false_late={best['false_late']}/{best['n']}  beam={best['beam']} prompt={best['prompt']}")

print()
print('=== ALL PROFILES SORTED BY WER (top 10 best) ===')
all_p = sorted(profiles.values(), key=lambda x: x['avg_wer'] or 99)
for p in all_p[:10]:
    print(f"  wer={p['avg_wer']:.4f}  delta={p['median_delta']}ms  trans={p['median_transcribe_ms']:.1f}ms  beam={p['beam']} cond={p['cond_prev']} vad={p['vad_filter']} prompt={p['prompt']}  [{p['model']}]")

print()
total_early = sum(v['false_early'] for v in profiles.values())
print(f'=== FALSE EARLY CUTS: {total_early} total across {len(profiles)} profiles x 19 fixtures ===')

print()
print('=== ENDPOINT DELTA DISTRIBUTION ===')
for model in models_in_sweep:
    rows = by_model.get(model, [])
    if not rows: continue
    best = min(rows, key=lambda x: x['avg_wer'] or 99)
    print(f"  {model}: median={best['median_delta']}ms  p95={best['p95_delta']}ms")

print()
print('=== PER-FIXTURE BEST WER ACROSS ALL PROFILES ===')
fixture_best = {}
for run in data['runs']:
    for fx in run['fixtures']:
        fn = fx['filename']
        w = fx.get('wer_normalized', 1.0)
        if fn not in fixture_best or w < fixture_best[fn]['wer']:
            fixture_best[fn] = {'wer': w, 'transcript': fx.get('transcript_raw', ''), 'model': run['whisper_model']}

sorted_fx = sorted(fixture_best.items(), key=lambda x: x[1]['wer'], reverse=True)
print('  {:45s} {:7s}  {:10s}  {}'.format('Fixture','BestWER','Best model','Transcript snippet'))
for fn, info in sorted_fx:
    wer_str = f"{info['wer']:.3f}"
    print('  {:45s} {:7s}  {:10s}  "{}"'.format(fn, wer_str, info['model'], info['transcript'][:60]))

print()
print('=== DECODER CONFIG THAT WINS MOST FIXTURES ===')
winner_configs = Counter()
summary = data.get('summary', {})
for item in summary.get('best_fixtures', []):
    ds = item.get('winning_decoder_settings', {})
    key = f"beam={ds.get('beam_size')} cond={ds.get('condition_on_previous_text')} vad={ds.get('vad_filter')} prompt={ds.get('use_initial_prompt')}"
    winner_configs[key] += 1
for cfg, cnt in winner_configs.most_common():
    print(f"  {cnt}x  {cfg}")
if not winner_configs:
    print("  (No summary/best_fixtures section in this sweep)")
