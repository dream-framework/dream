#!/usr/bin/env python3
"""Run T7 test on a curated subset of 20 solid datasets (faster, no timeouts)."""
import os, sys, json
import numpy as np

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from t7_hypothesis_test import (
    refetch_values, test_P1_scale_dependent_D,
    test_P2_multicomponent_at_finer_resolution,
    test_P3_no_universal_D, test_P4_meta_s2_exploratory
)
from dream_auto_scanner import retention_curve

with open(os.path.join(REPO, 'en/tests.json')) as f:
    data = json.load(f)

# Curated: 20 solid datasets known to have downloadable URLs
candidates = []
seen_urls = set()
for t in data['tests']:
    if t.get('D') is None or (t.get('r2') or 0) < 0.3: continue
    url = t.get('url', '')
    if url in seen_urls: continue
    if any(x in url for x in ['fredgraph.csv', 'CSSEGISandData', 'wikimedia.org',
                                'open-meteo', 'ndbc.noaa.gov', 'binance.com']):
        seen_urls.add(url)
        candidates.append(t)
candidates = candidates[:20]

print(f'Testing {len(candidates)} datasets...')
print()

p1_results = []
p2_results = []
for i, entry in enumerate(candidates):
    vals = refetch_values(entry.get('url', ''))
    if not vals or len(vals) < 100:
        print(f'  [{i+1}/20] SKIP (fetch failed): {entry["name"][:45]}')
        continue
    taus, acf = retention_curve(vals)
    if taus is None or len(taus) < 20:
        print(f'  [{i+1}/20] SKIP (ACF failed): {entry["name"][:45]}')
        continue
    t_arr = np.array(taus, dtype=float)
    R_arr = np.array(acf, dtype=float)
    name = entry['name'][:45]
    p1 = test_P1_scale_dependent_D(t_arr, R_arr, name)
    p2 = test_P2_multicomponent_at_finer_resolution(t_arr, R_arr, name)
    if 'D_fine' in p1:
        p1_results.append(p1)
    if 'delta_fine' in p2:
        p2_results.append(p2)
    p1_mark = '✓' if p1.get('pass') else '✗'
    p2_mark = '✓' if p2.get('pass') else '✗'
    print(f'  [{i+1}/20] {name:<45} P1={p1_mark} P2={p2_mark}')

# P3 + P4
all_D = [t['D'] for t in data['tests']
         if t.get('D') is not None and 0 < t['D'] < 4.99 and (t.get('r2') or 0) >= 0.3]
p3 = test_P3_no_universal_D(all_D)
p4 = test_P4_meta_s2_exploratory()

# Aggregate
n_p1 = len(p1_results)
n_p1_pass = sum(1 for r in p1_results if r['pass'])
pct_p1 = 100 * n_p1_pass / n_p1 if n_p1 > 0 else 0
p1_pass = pct_p1 >= 50

n_p2 = len(p2_results)
n_p2_pass = sum(1 for r in p2_results if r['pass'])
pct_p2 = 100 * n_p2_pass / n_p2 if n_p2 > 0 else 0
p2_pass = pct_p2 >= 40

print()
print('=' * 70)
print('T7 HYPOTHESIS TEST RESULTS')
print('=' * 70)
print()
print(f'P1: Scale-dependent D_eff')
print(f'  {n_p1_pass}/{n_p1} datasets ({pct_p1:.0f}%) show D variation > 0.3 across scales')
if p1_results:
    print(f'  Mean D range: {np.mean([r["d_range"] for r in p1_results]):.3f}')
print(f'  VERDICT: {"PASS" if p1_pass else "FAIL"}')
print()
print(f'P2: Multi-component fits improve at finer resolution')
print(f'  {n_p2_pass}/{n_p2} datasets ({pct_p2:.0f}%) show S2+dust more favored at fine scale')
if p2_results:
    print(f'  Mean improvement: {np.mean([r["improvement"] for r in p2_results]):.2f} ΔAICc')
print(f'  VERDICT: {"PASS" if p2_pass else "FAIL"}')
print()
print(f'P3: No universal D')
print(f'  n={p3["n"]}, mean={p3["mean"]:.3f}, std={p3["std"]:.3f}, CV={p3["cv"]:.3f}')
print(f'  VERDICT: {"PASS" if p3["pass"] else "FAIL"}')
print()
print(f'P4: Meta-S2 exploratory')
print(f'  Lilliefors p={p4["p_lilliefors"]}')
print(f'  VERDICT: {"PASS" if p4["pass"] else "FAIL"}')
print()

passes = sum([p1_pass, p2_pass, p3['pass'], p4['pass']])
if passes >= 4: verdict = 'SUPPORTED'
elif passes >= 2: verdict = 'PARTIAL'
else: verdict = 'REFUTED'

print('=' * 70)
print(f'T7 HYPOTHESIS: {verdict} ({passes}/4 predictions confirmed)')
print('=' * 70)

# Save
results = {
    'verdict': verdict,
    'predictions_confirmed': int(passes),
    'predictions_total': 4,
    'P1': {'pass': bool(p1_pass), 'n_tested': int(n_p1), 'n_pass': int(n_p1_pass), 'pct_pass': float(pct_p1),
           'mean_d_range': float(np.mean([r['d_range'] for r in p1_results])) if p1_results else None},
    'P2': {'pass': bool(p2_pass), 'n_tested': int(n_p2), 'n_pass': int(n_p2_pass), 'pct_pass': float(pct_p2),
           'mean_improvement': float(np.mean([r['improvement'] for r in p2_results])) if p2_results else None},
    'P3': {'pass': bool(p3['pass']), 'n': int(p3['n']), 'mean': float(p3['mean']),
           'std': float(p3['std']), 'cv': float(p3['cv']), 'threshold': float(p3['threshold'])},
    'P4': {'pass': bool(p4['pass']), 'p_lilliefors': float(p4['p_lilliefors']) if p4['p_lilliefors'] else None,
           'threshold': float(p4['threshold'])},
}
out = os.path.join(REPO, 't7_test_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out}')
