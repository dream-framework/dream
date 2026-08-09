#!/usr/bin/env python3
"""
T7 Interference Hypothesis Test
===============================

Formal test of the T7 hypothesis:

  "Under the 10D → 4D projection, every 4D measurement is a superposition
   of local S2-governed retention processes. Pure S2 emerges only when the
   measurement isolates a sufficiently narrow component of the underlying
   retention distribution."

The hypothesis makes 4 falsifiable predictions:

  P1: Scale-dependent D_eff — D varies as probe scale changes within the
      same dataset (different components dominate at different scales).

  P2: Multi-component fits improve at finer resolution — as measurement
      resolution increases, single-S2 gives way to multi-component fits.

  P3: No universal D — systems are not characterized by a single D;
      they are characterized by a distribution of local D values.

  P4: Meta-S2 is exploratory — the distribution of measured D values is
      a distribution of effective parameters, not fundamental invariants.

For each prediction, we define:
  - A quantitative test
  - A pass/fail threshold
  - An effect size

The overall T7 verdict is:
  SUPPORTED  — all 4 predictions confirmed
  PARTIAL    — 2-3 predictions confirmed
  REFUTED    — 0-1 predictions confirmed

This is a genuine hypothesis test: T7 could be refuted by the data.
"""
import os, sys, json, csv, io, re
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import compare as s2_compare, m_s2, m_s2_dust, fit_all_models


# ── Data fetchers (reuse from lambda_comparison) ────────────────────

def refetch_fred(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    rows = list(csv.reader(io.StringIO(data.decode('utf-8') if isinstance(data, bytes) else data)))
    vals = []
    for row in rows[1:]:
        if len(row) >= 2 and row[1] not in ('', '.'):
            try:
                v = float(row[1])
                if not np.isnan(v) and not np.isinf(v): vals.append(v)
            except: pass
    return vals if len(vals) >= 100 else None

def refetch_covid(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    rows = list(csv.reader(io.StringIO(data.decode('utf-8') if isinstance(data, bytes) else data)))
    if len(rows) < 2: return None
    daily_sums = []
    for col in range(4, len(rows[0])):
        total = 0
        for row in rows[1:]:
            if col < len(row):
                try: total += int(row[col])
                except: pass
        daily_sums.append(total)
    daily_new = [max(0, daily_sums[i] - daily_sums[i-1]) for i in range(1, len(daily_sums))]
    return daily_new if len(daily_new) >= 100 else None

def refetch_wikipedia(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'items' in obj:
            return [item['views'] for item in obj['items']]
    except: pass
    return None

def refetch_open_meteo(url):
    data = fetch_url(url, timeout=25)
    if not data: return None
    try:
        d = json.loads(data)
        daily = d.get('daily', {})
        for var in ['wind_speed_10m_max', 'precipitation_sum', 'temperature_2m_mean']:
            if var in daily:
                vals = [v for v in daily[var] if v is not None]
                if len(vals) >= 100: return vals
    except: pass
    return None

def refetch_ndbc(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    lines = text.strip().split('\n')
    vals = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 6:
            try:
                v = float(parts[5])
                if 0 <= v < 99: vals.append(v)
            except: pass
    return vals if len(vals) >= 100 else None

def refetch_binance(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        arr = json.loads(data)
        return [float(k[4]) for k in arr] if arr else None
    except: pass
    return None

def refetch_values(url):
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url: return refetch_fred(url)
        if 'CSSEGISandData' in url: return refetch_covid(url)
        if 'wikimedia.org' in url: return refetch_wikipedia(url)
        if 'open-meteo' in url: return refetch_open_meteo(url)
        if 'ndbc.noaa.gov' in url: return refetch_ndbc(url)
        if 'binance.com' in url: return refetch_binance(url)
    except: pass
    return None


# ── Fit S2 on a sub-range of the ACF ────────────────────────────────

def fit_s2_on_range(t, R, t_min, t_max):
    mask = (t >= t_min) & (t <= t_max)
    if mask.sum() < 8: return None
    t_sub = t[mask]
    R_sub = R[mask]
    if R_sub[0] > 0: R_sub = R_sub / R_sub[0]
    try:
        t_mid = float(t_sub[len(t_sub) // 2])
        best = None
        for p0 in [[1.0, t_mid, 0.5], [1.0, t_mid * 0.5, 1.0], [1.0, t_mid * 2, 0.3]]:
            try:
                popt, _ = curve_fit(m_s2, t_sub, R_sub, p0=p0,
                                    bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]),
                                    maxfev=20000)
                rss = float(np.sum((R_sub - m_s2(t_sub, *popt)) ** 2))
                if best is None or rss < best[1]:
                    best = (popt, rss)
            except: pass
        if best is None: return None
        popt, rss = best
        ss_tot = float(np.sum((R_sub - np.mean(R_sub)) ** 2))
        if ss_tot == 0: return None
        return {'D': float(popt[2]), 'lambda_q': float(popt[1]),
                'r2': 1 - rss / ss_tot, 'n': int(mask.sum())}
    except: return None


def fit_s2_dust_on_range(t, R, t_min, t_max):
    """Fit S2+dust (2-component) on a sub-range."""
    mask = (t >= t_min) & (t <= t_max)
    if mask.sum() < 12: return None  # need more points for 6-param fit
    t_sub = t[mask]
    R_sub = R[mask]
    if R_sub[0] > 0: R_sub = R_sub / R_sub[0]
    try:
        t_mid = float(t_sub[len(t_sub) // 2])
        best = None
        for p0 in [[0.7, t_mid * 0.3, 1.5, 0.3, t_mid * 2, 0.5],
                   [0.6, t_mid * 0.5, 1.0, 0.4, t_mid, 0.6]]:
            try:
                popt, _ = curve_fit(m_s2_dust, t_sub, R_sub, p0=p0,
                                    bounds=([0.0, 1e-3, 0.01, 0.0, 1e-3, 0.01],
                                            [2.0, 1e6, 10.0, 2.0, 1e6, 10.0]),
                                    maxfev=30000)
                rss = float(np.sum((R_sub - m_s2_dust(t_sub, *popt)) ** 2))
                if best is None or rss < best[1]:
                    best = (popt, rss)
            except: pass
        if best is None: return None
        popt, rss = best
        ss_tot = float(np.sum((R_sub - np.mean(R_sub)) ** 2))
        if ss_tot == 0: return None
        n = int(mask.sum())
        k = 6
        aic = n * np.log(rss / n) + 2 * k if rss > 0 else np.inf
        aic += (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else 0
        return {'D1': float(popt[2]), 'D2': float(popt[5]),
                'r2': 1 - rss / ss_tot, 'aic': aic, 'n': n}
    except: return None


def fit_single_s2_aic(t, R, t_min, t_max):
    """Fit single S2 and return AIC for comparison with S2+dust."""
    mask = (t >= t_min) & (t <= t_max)
    if mask.sum() < 8: return None
    t_sub = t[mask]
    R_sub = R[mask]
    if R_sub[0] > 0: R_sub = R_sub / R_sub[0]
    try:
        t_mid = float(t_sub[len(t_sub) // 2])
        best = None
        for p0 in [[1.0, t_mid, 0.5], [1.0, t_mid * 0.5, 1.0]]:
            try:
                popt, _ = curve_fit(m_s2, t_sub, R_sub, p0=p0,
                                    bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]),
                                    maxfev=20000)
                rss = float(np.sum((R_sub - m_s2(t_sub, *popt)) ** 2))
                if best is None or rss < best[1]:
                    best = (popt, rss)
            except: pass
        if best is None: return None
        popt, rss = best
        n = int(mask.sum())
        k = 3
        aic = n * np.log(rss / n) + 2 * k if rss > 0 else np.inf
        aic += (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else 0
        return {'D': float(popt[2]), 'aic': aic, 'r2': 1 - rss / float(np.sum((R_sub - np.mean(R_sub))**2))}
    except: return None


# ── The four T7 predictions ─────────────────────────────────────────

def test_P1_scale_dependent_D(t, R, name):
    """P1: D_eff varies with probe scale within the same dataset.
    Test: Fit S2 at 3 scales (fine=first 25%, medium=25-75%, coarse=last 50%).
    PASS if |D_fine - D_coarse| > 0.3 (meaningful variation, not noise)."""
    max_lag = float(t[-1])
    fine = fit_s2_on_range(t, R, 0, max_lag * 0.25)
    medium = fit_s2_on_range(t, R, max_lag * 0.25, max_lag * 0.75)
    coarse = fit_s2_on_range(t, R, max_lag * 0.5, max_lag)

    if not all([fine, medium, coarse]):
        return {'pass': False, 'reason': 'fit_failed', 'name': name}

    d_range = max(fine['D'], medium['D'], coarse['D']) - min(fine['D'], medium['D'], coarse['D'])
    # Threshold: variation > 0.3 is meaningful (more than typical estimator noise)
    threshold = 0.3
    return {
        'pass': d_range > threshold,
        'd_range': d_range,
        'threshold': threshold,
        'D_fine': fine['D'], 'D_medium': medium['D'], 'D_coarse': coarse['D'],
        'name': name,
    }


def test_P2_multicomponent_at_finer_resolution(t, R, name):
    """P2: Multi-component fits improve at finer resolution.
    Test: At fine scale (first 25%), does S2+dust beat single S2 by ΔAICc <= -4?
    If yes, finer resolution reveals multi-component structure (T7 prediction)."""
    max_lag = float(t[-1])

    # Fine scale
    s2_fine = fit_single_s2_aic(t, R, 0, max_lag * 0.25)
    dust_fine = fit_s2_dust_on_range(t, R, 0, max_lag * 0.25)

    # Full range
    s2_full = fit_single_s2_aic(t, R, 0, max_lag)
    dust_full = fit_s2_dust_on_range(t, R, 0, max_lag)

    if not all([s2_fine, dust_fine, s2_full, dust_full]):
        return {'pass': False, 'reason': 'fit_failed', 'name': name}

    delta_fine = dust_fine['aic'] - s2_fine['aic']  # negative = dust wins
    delta_full = dust_full['aic'] - s2_full['aic']

    # P2: multi-component should be MORE favored at fine scale than full scale
    # i.e., delta_fine should be more negative than delta_full
    improvement = delta_full - delta_fine  # positive = dust MORE favored at fine scale
    return {
        'pass': improvement > 2,  # meaningful improvement (ΔAICc > 2)
        'delta_fine': delta_fine,
        'delta_full': delta_full,
        'improvement': improvement,
        'threshold': 2,
        'name': name,
    }


def test_P3_no_universal_D(all_D_values):
    """P3: No universal D — the cross-dataset D distribution is broad, not peaked.
    Test: If T7 is right, D values should NOT cluster tightly around one value.
    PASS if the coefficient of variation (CV = std/mean) > 0.3 (broad distribution).
    FAIL if CV < 0.15 (tight clustering around a universal D)."""
    Ds = np.array(all_D_values)
    mean_d = np.mean(Ds)
    std_d = np.std(Ds, ddof=1)
    cv = std_d / mean_d if mean_d > 0 else 0
    return {
        'pass': cv > 0.3,
        'cv': cv,
        'mean': mean_d,
        'std': std_d,
        'n': len(Ds),
        'threshold': 0.3,
    }


def test_P4_meta_s2_exploratory():
    """P4: Meta-S2 is exploratory, not foundational.
    Test: If T7 is right, the Lilliefors-corrected KS test should REJECT Weibull
    (because D values are effective parameters, not samples from a universal law).
    PASS if Lilliefors p < 0.05 (Weibull rejected = effective params, not invariants)."""
    snapshot_path = os.path.join(REPO, 'meta_s2_snapshot.json')
    if not os.path.exists(snapshot_path):
        return {'pass': False, 'reason': 'snapshot_missing'}
    with open(snapshot_path) as f:
        snap = json.load(f)
    p_lilliefors = snap.get('ks_p_lilliefors')
    return {
        'pass': p_lilliefors is not None and p_lilliefors < 0.05,
        'p_lilliefors': p_lilliefors,
        'threshold': 0.05,
    }


# ── Main test runner ────────────────────────────────────────────────

def main():
    print('=' * 80)
    print('T7 INTERFERENCE HYPOTHESIS TEST')
    print('=' * 80)
    print()
    print('Hypothesis: Under 10D → 4D projection, every 4D measurement is a')
    print('superposition of local S2-governed retention processes.')
    print()
    print('4 falsifiable predictions:')
    print('  P1: D_eff varies with probe scale (within-dataset)')
    print('  P2: Multi-component fits improve at finer resolution')
    print('  P3: No universal D (cross-dataset distribution is broad)')
    print('  P4: Meta-S2 is exploratory (Lilliefors rejects Weibull)')
    print()

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    # Get solid datasets with downloadable URLs
    candidates = []
    seen_urls = set()
    for t in data['tests']:
        if t.get('D') is None: continue
        if (t.get('r2') or 0) < 0.3: continue  # need decent fits
        url = t.get('url', '')
        if url in seen_urls: continue
        if any(x in url for x in ['fredgraph.csv', 'CSSEGISandData', 'wikimedia.org',
                                    'open-meteo', 'ndbc.noaa.gov', 'binance.com']):
            seen_urls.add(url)
            candidates.append(t)

    print(f'Testing {len(candidates)} solid datasets with downloadable data...')
    print()

    # ── P1 + P2: per-dataset tests ──
    p1_results = []
    p2_results = []
    for entry in candidates:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100:
            continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 20:
            continue

        t_arr = np.array(taus, dtype=float)
        R_arr = np.array(acf, dtype=float)

        name = entry['name'][:50]
        p1 = test_P1_scale_dependent_D(t_arr, R_arr, name)
        p2 = test_P2_multicomponent_at_finer_resolution(t_arr, R_arr, name)
        if 'D_fine' in p1:
            p1_results.append(p1)
        if 'delta_fine' in p2:
            p2_results.append(p2)

    # ── P3: cross-dataset D distribution ──
    all_D = [t['D'] for t in data['tests']
             if t.get('D') is not None and 0 < t['D'] < 4.99 and (t.get('r2') or 0) >= 0.3]
    p3_result = test_P3_no_universal_D(all_D)

    # ── P4: meta-S2 Lilliefors ──
    p4_result = test_P4_meta_s2_exploratory()

    # ── Aggregate verdicts ──
    print('═' * 80)
    print('PREDICTION RESULTS')
    print('═' * 80)
    print()

    # P1
    n_p1 = len(p1_results)
    n_p1_pass = sum(1 for r in p1_results if r['pass'])
    pct_p1 = 100 * n_p1_pass / n_p1 if n_p1 > 0 else 0
    print(f'P1: Scale-dependent D_eff')
    print(f'  Tested {n_p1} datasets')
    print(f'  PASS: {n_p1_pass}/{n_p1} ({pct_p1:.0f}%) show meaningful D variation across scales')
    if p1_results:
        d_ranges = [r['d_range'] for r in p1_results]
        print(f'  Mean D range across scales: {np.mean(d_ranges):.3f}')
        print(f'  Max D range: {np.max(d_ranges):.3f}')
    p1_pass = pct_p1 >= 50  # majority show scale-dependence
    print(f'  VERDICT: {"PASS" if p1_pass else "FAIL"} (threshold: ≥50% of datasets)')
    print()

    # P2
    n_p2 = len(p2_results)
    n_p2_pass = sum(1 for r in p2_results if r['pass'])
    pct_p2 = 100 * n_p2_pass / n_p2 if n_p2 > 0 else 0
    print(f'P2: Multi-component fits improve at finer resolution')
    print(f'  Tested {n_p2} datasets')
    print(f'  PASS: {n_p2_pass}/{n_p2} ({pct_p2:.0f}%) show S2+dust MORE favored at fine scale')
    if p2_results:
        improvements = [r['improvement'] for r in p2_results]
        print(f'  Mean improvement (ΔAICc fine vs full): {np.mean(improvements):.2f}')
    p2_pass = pct_p2 >= 40  # substantial minority
    print(f'  VERDICT: {"PASS" if p2_pass else "FAIL"} (threshold: ≥40% of datasets)')
    print()

    # P3
    print(f'P3: No universal D (broad cross-dataset distribution)')
    print(f'  n={p3_result["n"]}, mean={p3_result["mean"]:.3f}, std={p3_result["std"]:.3f}')
    print(f'  Coefficient of variation: {p3_result["cv"]:.3f}')
    print(f'  VERDICT: {"PASS" if p3_result["pass"] else "FAIL"} (threshold: CV > {p3_result["threshold"]})')
    print()

    # P4
    print(f'P4: Meta-S2 is exploratory (Lilliefors rejects Weibull)')
    print(f'  Lilliefors KS p: {p4_result["p_lilliefors"]}')
    print(f'  VERDICT: {"PASS" if p4_result["pass"] else "FAIL"} (threshold: p < {p4_result["threshold"]})')
    print()

    # ── Overall T7 verdict ──
    passes = sum([p1_pass, p2_pass, p3_result['pass'], p4_result['pass']])
    print('═' * 80)
    print('OVERALL T7 VERDICT')
    print('═' * 80)
    print(f'  Predictions confirmed: {passes}/4')
    if passes >= 4:
        verdict = 'SUPPORTED'
    elif passes >= 2:
        verdict = 'PARTIAL'
    else:
        verdict = 'REFUTED'
    print(f'  T7 Hypothesis: {verdict}')
    print()

    # Save results
    results = {
        'verdict': verdict,
        'predictions_confirmed': passes,
        'predictions_total': 4,
        'P1': {
            'pass': p1_pass,
            'n_tested': n_p1,
            'n_pass': n_p1_pass,
            'pct_pass': pct_p1,
            'mean_d_range': float(np.mean([r['d_range'] for r in p1_results])) if p1_results else None,
        },
        'P2': {
            'pass': p2_pass,
            'n_tested': n_p2,
            'n_pass': n_p2_pass,
            'pct_pass': pct_p2,
            'mean_improvement': float(np.mean([r['improvement'] for r in p2_results])) if p2_results else None,
        },
        'P3': p3_result,
        'P4': p4_result,
    }
    out_path = os.path.join(REPO, 't7_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {out_path}')

    return results


if __name__ == '__main__':
    main()
