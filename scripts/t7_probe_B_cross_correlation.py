#!/usr/bin/env python3
"""
T7 PROBE B — Domain-pair cross-correlation of S2 residuals
================================================================

For each pair of fetched real datasets from DIFFERENT domains:
  1. Fit single-S2 to each curve
  2. Compute residuals ε_i(t) = R_i(t) - m_s2(t; θ̂_i)
  3. Resample both residuals to a common time grid (linear interpolation)
  4. Compute cross-correlation C_ij(τ) at multiple lags
  5. Take max |C_ij(τ)| over lags as the interference statistic

Null model: independently permute residuals in time within each curve
and recompute max |C_ij(τ)|. Repeat 200 times to get the null
distribution.

If multi-MM interference is real: observed max |C_ij| exceeds 95th
percentile of null distribution at the SAME characteristic scale across
multiple domain pairs.

If single kernel is correct: no systematic cross-correlation structure.
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_model_compare import m_s2
from t7_probe_A_drift_mixture import get_raw_curve, fetch

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(REPO, 'en/tests.json')) as f:
    tests = json.load(f)['tests']

# Use the same fetchable entries as Probe A — fetch real raw curves
# Pick entries we know are fetchable based on Probe A results
fetchable = []
for t in tests:
    if not t.get('url'):
        continue
    from urllib.parse import urlparse
    host = urlparse(t['url']).netloc
    if host in ('api.worldbank.org', 'api.coingecko.com', 'api.binance.com',
                'archive-api.open-meteo.com', 'raw.githubusercontent.com',
                'earthquake.usgs.gov'):
        fetchable.append(t)

print(f'Fetchable: {len(fetchable)} entries')

# Fetch them, store (name, domain, t_array, R_array) tuples
fetched_data = []
seen = set()
for i, t in enumerate(fetchable):
    if t['url'] in seen:
        continue
    seen.add(t['url'])
    try:
        result = get_raw_curve(t['url'])
    except Exception:
        continue
    if result is None:
        continue
    t_arr, R_arr = result
    if len(t_arr) < 30:
        continue
    fetched_data.append({
        'name': t.get('name', '')[:40],
        'domain': t.get('domain', 'unknown'),
        'url': t['url'],
        't': t_arr,
        'R': R_arr,
    })

print(f'Fetched successfully: {len(fetched_data)} curves')

# ─────────────────────────────────────────────────────────────────────
# For each curve: fit S2, compute residuals
# ─────────────────────────────────────────────────────────────────────

def fit_s2(t, R):
    if R[0] != 0:
        R_n = R / R[0]
    else:
        R_n = R / max(abs(R))
    tm = float(t[len(t)//2])
    best = None
    for p0 in [[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]]:
        try:
            popt, _ = curve_fit(m_s2, t, R_n, p0=p0,
                bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]),
                maxfev=20000)
            rss = float(np.sum((R_n - m_s2(t, *popt))**2))
            if best is None or rss < best[1]:
                best = (popt, rss, R_n)
        except Exception:
            continue
    return best

print('\nFitting S2 to each curve:')
for d in fetched_data:
    f = fit_s2(d['t'], d['R'])
    if f:
        popt, rss, R_n = f
        d['s2_params'] = list(popt)
        d['residuals'] = R_n - m_s2(d['t'], *popt)
        print(f"  [{d['domain']:>14s}] {d['name']:<40s}  D={popt[2]:.3f}  RSS={rss:.4g}")
    else:
        d['s2_params'] = None
        d['residuals'] = None

# Filter to successful fits
fitted = [d for d in fetched_data if d.get('residuals') is not None]
print(f'\nSuccessfully fitted: {len(fitted)}')

# ─────────────────────────────────────────────────────────────────────
# Pairwise cross-correlation across DIFFERENT domains
# ─────────────────────────────────────────────────────────────────────

def cross_correlation(eps1, t1, eps2, t2, max_lag_frac=0.2):
    """Resample both residuals to a common time grid, compute cross-correlation
    at multiple lags. Return max |C(τ)| and the lag where it occurs."""
    n1, n2 = len(t1), len(t2)
    # Common time grid: use the shorter range
    t_min = max(t1[0], t2[0])
    t_max = min(t1[-1], t2[-1])
    if t_max <= t_min:
        return None
    # Use n = min(n1, n2) points
    n = min(n1, n2)
    t_common = np.linspace(t_min, t_max, n)
    e1 = np.interp(t_common, t1, eps1)
    e2 = np.interp(t_common, t2, eps2)
    # Detrend
    e1 = e1 - np.mean(e1)
    e2 = e2 - np.mean(e2)
    if np.std(e1) < 1e-10 or np.std(e2) < 1e-10:
        return None
    e1 = e1 / np.std(e1)
    e2 = e2 / np.std(e2)
    # Cross-correlation at multiple lags
    max_lag = max(int(max_lag_frac * n), 1)
    ccs = []
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            c = np.sum(e1 * e2) / n
        elif lag > 0:
            c = np.sum(e1[:-lag] * e2[lag:]) / (n - lag)
        else:
            c = np.sum(e1[-lag:] * e2[:lag]) / (n + lag)
        ccs.append(c)
    ccs = np.array(ccs)
    max_abs = float(np.max(np.abs(ccs)))
    lag_at_max = int(np.argmax(np.abs(ccs)) - max_lag)
    return {'max_abs_cc': max_abs, 'lag_at_max': lag_at_max, 'n': n}


def null_cc_distribution(eps1, t1, eps2, t2, n_boot=200, seed=0):
    """Permute residuals in time within each series, recompute max |C(τ)|."""
    rng = np.random.RandomState(seed)
    nulls = []
    for _ in range(n_boot):
        e1_p = rng.permutation(eps1)
        e2_p = rng.permutation(eps2)
        r = cross_correlation(e1_p, t1, e2_p, t2)
        if r is not None:
            nulls.append(r['max_abs_cc'])
    return np.array(nulls) if nulls else None


print('\n' + '='*72)
print('PROBE B — Domain-pair cross-correlations of S2 residuals')
print('='*72)

# Test all cross-domain pairs (cap to ~30 pairs to keep runtime reasonable)
pairs = []
domains_present = sorted(set(d['domain'] for d in fitted))
print(f'Domains with fitted curves: {domains_present}')
for i in range(len(fitted)):
    for j in range(i+1, len(fitted)):
        if fitted[i]['domain'] == fitted[j]['domain']:
            continue  # only cross-domain
        pairs.append((i, j))

print(f'Cross-domain pairs: {len(pairs)}')

# Cap pairs
if len(pairs) > 30:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(pairs), 30, replace=False)
    pairs = [pairs[i] for i in sorted(idx)]
    print(f'Subsampled to: {len(pairs)} pairs')

results = []
for k, (i, j) in enumerate(pairs):
    d1, d2 = fitted[i], fitted[j]
    print(f'\n[{k+1}/{len(pairs)}] ({d1["domain"]} x {d2["domain"]})')
    print(f'  {d1["name"]}')
    print(f'  {d2["name"]}')
    obs = cross_correlation(d1['residuals'], d1['t'], d2['residuals'], d2['t'])
    if obs is None:
        print('  No overlap')
        continue
    null = null_cc_distribution(d1['residuals'], d1['t'],
                                 d2['residuals'], d2['t'],
                                 n_boot=200, seed=k)
    if null is None or len(null) < 10:
        print('  Null failed')
        continue
    p = float((null >= obs['max_abs_cc']).mean())
    null_95 = float(np.percentile(null, 95))
    sig = p < 0.05
    print(f"  Observed max|C|={obs['max_abs_cc']:.3f} at lag={obs['lag_at_max']}")
    print(f'  Null 95th pct: {null_95:.3f}  p-value: {p:.4g}  {"SIGNIFICANT" if sig else "ns"}')
    results.append({
        'name1': d1['name'], 'domain1': d1['domain'],
        'name2': d2['name'], 'domain2': d2['domain'],
        'observed_max_cc': obs['max_abs_cc'],
        'lag_at_max': obs['lag_at_max'],
        'null_95th': null_95,
        'p_value': p,
        'significant': sig,
        'n_overlap': obs['n'],
    })

# ─────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('PROBE B — AGGREGATE VERDICT')
print('='*72)
n_pairs = len(results)
n_sig = sum(1 for r in results if r['significant'])
print(f'Pairs tested: {n_pairs}')
print(f'Significant cross-correlations (p<0.05): {n_sig}/{n_pairs} ({100*n_sig/max(n_pairs,1):.0f}%)')
print(f'Expected by chance at p<0.05: {n_pairs*0.05:.1f}')
print(f'\nIf interference is real, we expect MORE significant pairs than chance.')
print(f'Observed: {n_sig}, expected: {n_pairs*0.05:.1f}')
if n_sig > n_pairs * 0.05 * 1.5:
    print(f'  -> EXCESS SIGNIFICANT PAIRS — interference signal possible')
elif n_sig > n_pairs * 0.1:
    print(f'  -> SOME EXCESS — weak interference signal')
else:
    print(f'  -> NO EXCESS — single kernel consistent with no interference')

# Show significant pairs
if n_sig > 0:
    print(f'\nSignificant pairs:')
    for r in results:
        if r['significant']:
            print(f"  {r['domain1']:>14s} x {r['domain2']:<14s}  cc={r['observed_max_cc']:.3f}  p={r['p_value']:.4g}  lag={r['lag_at_max']}")
            print(f"    {r['name1']}")
            print(f"    {r['name2']}")

# Check if significant lags cluster at a characteristic scale
if n_sig >= 3:
    sig_lags = [r['lag_at_max'] for r in results if r['significant']]
    print(f'\nLag distribution for significant pairs:')
    print(f'  values: {sig_lags}')
    print(f'  median: {np.median(sig_lags):.1f}')
    # If lags cluster around a single value, that's the characteristic interference scale

out_path = os.path.join(OUT_DIR, 't7_probe_B_cross_correlation.json')
with open(out_path, 'w') as f:
    json.dump({
        'probe': 'B — domain-pair cross-correlation of S2 residuals',
        'n_fetched_curves': len(fitted),
        'n_pairs_tested': n_pairs,
        'n_significant_pairs': n_sig,
        'expected_by_chance': n_pairs * 0.05,
        'results': results,
    }, f, indent=2)
print(f'\nSaved: {out_path}')
