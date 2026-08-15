#!/usr/bin/env python3
"""
T7 Deep Analysis: S2 Family Persistence Under Aggregation
==========================================================

Tests the user's key hypothesis: S2 mixtures occupy a constrained
function family, and this constraint is why S2-like structure
persists across heterogeneous datasets.

Three experiments:

1. S2-MIXTURE vs GENERIC FLEXIBLE MODELS
   - Fit S2_DUST (2-component S2 mixture) vs equally flexible non-S2 models
     (2-component power law, 2-component lognormal, 2-component Gaussian)
   - Does S2-mixture systematically win across the registry?

2. PIPELINE BIAS CONTROL
   - Generate synthetic non-S2 processes (power-law mixtures, lognormal
     mixtures, sums of exponentials, damped oscillations)
   - Run the IDENTICAL automated detector
   - Does it falsely detect S2 in non-S2 data?

3. S2 ANCESTRY TEST
   - For each INTERFERENCE/MULTI-SCALE entry, check whether S2_DUST
     (which is an S2 mixture) beats BIEXP (which is NOT an S2 mixture
     but has the same number of parameters: 4)
   - If S2_DUST systematically beats BIEXP on real data but not on
     synthetic non-S2 data, that's evidence for S2 ancestry.
"""
import os, sys, json, csv, io, re
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import fit_all_models, m_s2, m_biexp, m_power, m_lognormal, m_gaussian, m_exp


# ── Non-S2 mixture models (same parameter count as S2_DUST for fair comparison) ──

def m_biexp_2comp(t, A1, l1, A2, l2):
    """2-component exponential (NOT S2) — 4 params, same as BIEXP."""
    return A1 * np.exp(-t / max(l1, 1e-6)) + A2 * np.exp(-t / max(l2, 1e-6))

def m_power_2comp(t, A1, a1, A2, a2):
    """2-component power law (NOT S2) — 4 params."""
    return A1 * np.power(np.maximum(t, 1e-6), -max(a1, 0.01)) + \
           A2 * np.power(np.maximum(t, 1e-6), -max(a2, 0.01))

def m_lognormal_2comp(t, A1, m1, s1, A2, m2, s2):
    """2-component lognormal (NOT S2) — 6 params, same as S2_DUST."""
    s1, s2 = max(s1, 1e-3), max(s2, 1e-3)
    return A1 / (np.maximum(t, 1e-6) * s1 * np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(np.maximum(t, 1e-6))-m1)/s1)**2) + \
           A2 / (np.maximum(t, 1e-6) * s2 * np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(np.maximum(t, 1e-6))-m2)/s2)**2)

def m_gaussian_2comp(t, A1, s1, A2, s2):
    """2-component Gaussian (NOT S2) — 4 params."""
    return A1 * np.exp(-0.5*(t/max(s1, 1e-6))**2) + A2 * np.exp(-0.5*(t/max(s2, 1e-6))**2)


def safe_fit(func, t, R, p0_list, bounds=None, maxfev=20000):
    best = None
    for p0 in p0_list:
        try:
            if bounds:
                popt, _ = curve_fit(func, t, R, p0=p0, bounds=bounds, maxfev=maxfev)
            else:
                popt, _ = curve_fit(func, t, R, p0=p0, maxfev=maxfev)
            rss = float(np.sum((R - func(t, *popt)) ** 2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except:
            pass
    return best


def compute_aicc(rss, n, k):
    if n - k - 1 <= 0 or rss <= 0:
        return float('inf')
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


# ── Experiment 1: S2-mixture vs non-S2 flexible models on real data ──

def refetch_values(url):
    """Refetch values from URL (reused from other scripts)."""
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url:
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
        if 'CSSEGISandData' in url:
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
        if 'wikimedia.org' in url:
            data = fetch_url(url, timeout=20)
            if not data: return None
            try:
                obj = json.loads(data)
                if 'items' in obj:
                    return [item['views'] for item in obj['items']]
            except: pass
            return None
        if 'open-meteo' in url:
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
        if 'ndbc.noaa.gov' in url:
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
        if 'binance.com' in url:
            data = fetch_url(url, timeout=20)
            if not data: return None
            try:
                arr = json.loads(data)
                return [float(k[4]) for k in arr] if arr else None
            except: pass
            return None
    except: pass
    return None


def experiment1_s2mix_vs_generic():
    """Compare S2_DUST (S2 mixture, 6 params) vs lognormal_2comp (non-S2, 6 params)
    and S2 (3 params) vs BIEXP (4 params, non-S2) across real datasets."""
    print('=' * 80)
    print('EXPERIMENT 1: S2-mixture vs non-S2 flexible models on real data')
    print('=' * 80)

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

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

    results = []
    for entry in candidates[:25]:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 20: continue

        t = np.array(taus, dtype=float)
        R = np.array(acf, dtype=float)
        t = t - t[0]
        if R[0] > 0: R = R / R[0]
        n = len(t)
        ss_tot = float(np.sum((R - np.mean(R))**2))
        if ss_tot == 0: continue

        # Fit all models
        fits = fit_all_models(t, R)

        # Also fit 2-component non-S2 models
        # 2-comp lognormal (6 params, same as S2_DUST)
        t_mid = float(t[len(t)//2])
        best_ln2 = safe_fit(m_lognormal_2comp, t, R,
            p0_list=[[0.5, np.log(t_mid*0.3), 0.5, 0.5, np.log(t_mid*2), 0.8],
                     [0.6, np.log(t_mid*0.5), 0.8, 0.4, np.log(t_mid), 0.6]])
        if best_ln2:
            popt, rss = best_ln2
            k = 6
            aic_ln2 = compute_aicc(rss, n, k)
            r2_ln2 = 1 - rss / ss_tot
        else:
            aic_ln2 = float('inf'); r2_ln2 = 0

        # 2-comp Gaussian (4 params, same as BIEXP)
        best_g2 = safe_fit(m_gaussian_2comp, t, R,
            p0_list=[[0.7, t_mid*0.3, 0.3, t_mid*2],
                     [0.5, t_mid*0.5, 0.5, t_mid]])
        if best_g2:
            popt, rss = best_g2
            k = 4
            aic_g2 = compute_aicc(rss, n, k)
            r2_g2 = 1 - rss / ss_tot
        else:
            aic_g2 = float('inf'); r2_g2 = 0

        s2_aicc = fits.get('S2', {}).get('aicc', float('inf'))
        s2dust_aicc = fits.get('S2_DUST', {}).get('aicc', float('inf'))
        biexp_aicc = fits.get('BIEXP', {}).get('aicc', float('inf'))

        results.append({
            'name': entry['name'][:45],
            's2_aicc': s2_aicc,
            's2dust_aicc': s2dust_aicc,
            'biexp_aicc': biexp_aicc,
            'ln2_aicc': aic_ln2,
            'g2_aicc': aic_g2,
            'delta_s2dust_vs_ln2': s2dust_aicc - aic_ln2,  # negative = S2 mixture wins
            'delta_s2_vs_biexp': s2_aicc - biexp_aicc,
            'delta_s2_vs_g2': s2_aicc - aic_g2,
        })

    print(f'\nTested {len(results)} datasets')
    print()

    # Analysis
    s2dust_wins = sum(1 for r in results if r['delta_s2dust_vs_ln2'] < -2)
    s2dust_ties = sum(1 for r in results if abs(r['delta_s2dust_vs_ln2']) <= 2)
    ln2_wins = sum(1 for r in results if r['delta_s2dust_vs_ln2'] > 2)

    print('S2_DUST (6-param S2 mixture) vs 2-comp Lognormal (6-param non-S2):')
    print(f'  S2_DUST wins (ΔAICc < -2): {s2dust_wins}/{len(results)} ({100*s2dust_wins/len(results):.0f}%)')
    print(f'  Tie (|ΔAICc| ≤ 2):          {s2dust_ties}/{len(results)} ({100*s2dust_ties/len(results):.0f}%)')
    print(f'  Lognormal wins (ΔAICc > 2): {ln2_wins}/{len(results)} ({100*ln2_wins/len(results):.0f}%)')
    if results:
        mean_delta = np.mean([r['delta_s2dust_vs_ln2'] for r in results])
        print(f'  Mean ΔAICc: {mean_delta:.1f} (negative = S2 mixture favored)')
    print()

    s2_vs_biexp_wins = sum(1 for r in results if r['delta_s2_vs_biexp'] < -2)
    s2_vs_g2_wins = sum(1 for r in results if r['delta_s2_vs_g2'] < -2)
    print('S2 (3-param) vs BIEXP (4-param non-S2):')
    print(f'  S2 wins: {s2_vs_biexp_wins}/{len(results)} ({100*s2_vs_biexp_wins/len(results):.0f}%)')
    print('S2 (3-param) vs 2-comp Gaussian (4-param non-S2):')
    print(f'  S2 wins: {s2_vs_g2_wins}/{len(results)} ({100*s2_vs_g2_wins/len(results):.0f}%)')

    return results


# ── Experiment 2: Pipeline bias control ──

def experiment2_pipeline_bias():
    """Generate synthetic non-S2 data, run the detector, check for false S2 detection."""
    print()
    print('=' * 80)
    print('EXPERIMENT 2: Pipeline bias control — does the detector falsely detect S2?')
    print('=' * 80)

    np.random.seed(42)
    n_points = 200
    t = np.arange(n_points, dtype=float)

    results = []

    # Type 1: Pure power-law decay (NOT S2)
    print('\n  Type 1: Pure power-law decay R = t^-1.5')
    for i in range(20):
        R = np.power(t + 1, -1.5) + np.random.normal(0, 0.01, n_points)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        fits = fit_all_models(t, R)
        s2_aicc = fits.get('S2', {}).get('aicc', float('inf'))
        s2dust_aicc = fits.get('S2_DUST', {}).get('aicc', float('inf'))
        power_aicc = fits.get('POWER', {}).get('aicc', float('inf'))
        best = min(fits.items(), key=lambda x: x[1]['aicc'])
        results.append({'type': 'power_law', 'best_model': best[0],
                        's2_wins': best[0] == 'S2', 's2dust_wins': best[0] == 'S2_DUST',
                        'power_wins': best[0] == 'POWER',
                        'delta_s2_vs_power': s2_aicc - power_aicc})

    # Type 2: Sum of exponentials (NOT S2 — this is what BIEXP fits)
    print('  Type 2: Sum of 2 exponentials R = 0.6*exp(-t/5) + 0.4*exp(-t/30)')
    for i in range(20):
        R = 0.6 * np.exp(-t/5) + 0.4 * np.exp(-t/30) + np.random.normal(0, 0.01, n_points)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        fits = fit_all_models(t, R)
        best = min(fits.items(), key=lambda x: x[1]['aicc'])
        results.append({'type': 'biexp_true', 'best_model': best[0],
                        's2_wins': best[0] == 'S2', 's2dust_wins': best[0] == 'S2_DUST',
                        'biexp_wins': best[0] == 'BIEXP'})

    # Type 3: Lognormal decay (NOT S2)
    print('  Type 3: Lognormal decay')
    for i in range(20):
        mu, sigma = np.log(10), 0.8
        R = (1 / (t * sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(t)-mu)/sigma)**2)
        R[0] = R[1]  # avoid inf at t=0
        R = R + np.random.normal(0, 0.01, n_points)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        fits = fit_all_models(t, R)
        best = min(fits.items(), key=lambda x: x[1]['aicc'])
        results.append({'type': 'lognormal_true', 'best_model': best[0],
                        's2_wins': best[0] == 'S2', 's2dust_wins': best[0] == 'S2_DUST'})

    # Type 4: Damped oscillation (NOT S2, NOT monotonic)
    print('  Type 4: Damped oscillation R = exp(-t/15)*cos(t/3)')
    for i in range(20):
        R = np.exp(-t/15) * np.cos(t/3) + np.random.normal(0, 0.01, n_points)
        R = np.clip(R, -0.5, 2.0)
        R = np.abs(R)  # retention must be non-negative
        if R[0] > 0: R = R / R[0]
        fits = fit_all_models(t, R)
        best = min(fits.items(), key=lambda x: x[1]['aicc'])
        results.append({'type': 'damped_osc', 'best_model': best[0],
                        's2_wins': best[0] == 'S2', 's2dust_wins': best[0] == 'S2_DUST'})

    # Type 5: Pure S2 (positive control — should be detected)
    print('  Type 5: Pure S2 R = exp[-(t/10)^0.8] (positive control)')
    for i in range(20):
        R = np.exp(-np.power(t/10, 0.8)) + np.random.normal(0, 0.01, n_points)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        fits = fit_all_models(t, R)
        best = min(fits.items(), key=lambda x: x[1]['aicc'])
        results.append({'type': 'pure_s2', 'best_model': best[0],
                        's2_wins': best[0] == 'S2', 's2dust_wins': best[0] == 'S2_DUST'})

    # Summary
    print(f'\n  Results ({len(results)} synthetic datasets):')
    print(f'  {"Type":<20} {"N":>3} {"S2 wins":>8} {"S2_DUST wins":>13} {"Correct":>8} {"False S2":>9}')
    print('  ' + '-' * 65)
    for typ in ['power_law', 'biexp_true', 'lognormal_true', 'damped_osc', 'pure_s2']:
        subset = [r for r in results if r['type'] == typ]
        n = len(subset)
        s2_w = sum(1 for r in subset if r['s2_wins'])
        s2d_w = sum(1 for r in subset if r['s2dust_wins'])
        if typ == 'pure_s2':
            correct = s2_w + s2d_w  # S2 or S2_DUST
            false = n - correct
            label = 'S2 detected'
        else:
            correct = n - s2_w - s2d_w  # correctly NOT S2
            false = s2_w + s2d_w  # falsely detected as S2
            label = 'rejected'
        print(f'  {typ:<20} {n:>3} {s2_w:>8} {s2d_w:>13} {correct:>8} {false:>9}')

    false_positive_rate = sum(1 for r in results if r['type'] != 'pure_s2' and (r['s2_wins'] or r['s2dust_wins'])) / \
                          sum(1 for r in results if r['type'] != 'pure_s2')
    true_positive_rate = sum(1 for r in results if r['type'] == 'pure_s2' and (r['s2_wins'] or r['s2dust_wins'])) / \
                         sum(1 for r in results if r['type'] == 'pure_s2')
    print(f'\n  False positive rate (non-S2 called S2): {false_positive_rate:.1%}')
    print(f'  True positive rate (S2 called S2):     {true_positive_rate:.1%}')

    return results


# ── Experiment 3: S2 ancestry — does S2_DUST beat BIEXP on real but not synthetic? ──

def experiment3_s2_ancestry(exp1_results, exp2_results):
    """Key test: S2_DUST (S2 mixture) vs BIEXP (non-S2 mixture, same param count area).

    If S2_DUST systematically beats BIEXP on real data but NOT on synthetic
    non-S2 data, that's evidence for S2 ancestry.
    """
    print()
    print('=' * 80)
    print('EXPERIMENT 3: S2 ancestry — S2_DUST vs BIEXP on real vs synthetic')
    print('=' * 80)

    # Real data: from experiment 1
    real_deltas = [r['delta_s2dust_vs_ln2'] for r in exp1_results]
    real_s2dust_wins = sum(1 for d in real_deltas if d < -2)
    real_total = len(real_deltas)

    # Synthetic non-S2: from experiment 2, biexp_true type
    synth_results = [r for r in exp2_results if r['type'] == 'biexp_true']
    synth_s2dust_false = sum(1 for r in synth_results if r['s2dust_wins'])

    print(f'\nReal data (n={real_total}):')
    print(f'  S2_DUST beats non-S2 lognormal: {real_s2dust_wins}/{real_total} ({100*real_s2dust_wins/real_total:.0f}%)')
    if real_deltas:
        print(f'  Mean ΔAICc (S2_DUST - Lognormal_2comp): {np.mean(real_deltas):.1f}')

    print(f'\nSynthetic biexp data (n={len(synth_results)}):')
    print(f'  S2_DUST falsely wins on pure BIEXP data: {synth_s2dust_false}/{len(synth_results)} ({100*synth_s2dust_false/len(synth_results):.0f}%)')

    print(f'\nInterpretation:')
    if real_s2dust_wins / real_total > 0.5 and synth_s2dust_false / len(synth_results) < 0.3:
        print('  S2_DUST systematically beats non-S2 models on real data')
        print('  but does NOT falsely win on synthetic non-S2 data.')
        print('  → Evidence for S2 ancestry in real datasets.')
    elif synth_s2dust_false / len(synth_results) > 0.5:
        print('  S2_DUST frequently wins even on synthetic non-S2 data.')
        print('  → Cannot distinguish S2 ancestry from fitting flexibility.')
    else:
        print('  Mixed results — inconclusive.')


# ── Main ──

def main():
    print('=' * 80)
    print('T7 DEEP ANALYSIS: S2 Family Persistence Under Aggregation')
    print('=' * 80)
    print()
    print('Hypothesis: S2 mixtures occupy a constrained function family.')
    print('The persistence of S2-like structure across heterogeneous datasets')
    print('is evidence for a common S2-generating mechanism, not just curve fitting.')
    print()

    exp1 = experiment1_s2mix_vs_generic()
    exp2 = experiment2_pipeline_bias()
    experiment3_s2_ancestry(exp1, exp2)

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)

    # Experiment 1 summary
    s2dust_wins = sum(1 for r in exp1 if r['delta_s2dust_vs_ln2'] < -2)
    print(f'\n1. S2-mixture vs non-S2 flexible models:')
    print(f'   S2_DUST beats 2-comp Lognormal on {s2dust_wins}/{len(exp1)} real datasets')
    if exp1:
        mean_d = np.mean([r['delta_s2dust_vs_ln2'] for r in exp1])
        print(f'   Mean ΔAICc: {mean_d:.1f} ({"S2 mixture favored" if mean_d < 0 else "non-S2 favored"})')

    # Experiment 2 summary
    fp_rate = sum(1 for r in exp2 if r['type'] != 'pure_s2' and (r['s2_wins'] or r['s2dust_wins'])) / \
              sum(1 for r in exp2 if r['type'] != 'pure_s2')
    tp_rate = sum(1 for r in exp2 if r['type'] == 'pure_s2' and (r['s2_wins'] or r['s2dust_wins'])) / \
              sum(1 for r in exp2 if r['type'] == 'pure_s2')
    print(f'\n2. Pipeline bias control:')
    print(f'   False positive rate: {fp_rate:.1%} (non-S2 data falsely called S2)')
    print(f'   True positive rate:  {tp_rate:.1%} (S2 data correctly detected)')

    print(f'\n3. S2 ancestry verdict:')
    if s2dust_wins / len(exp1) > 0.5 and fp_rate < 0.3:
        print('   SUPPORTED: S2-mixture systematically wins on real data,')
        print('   detector does NOT falsely detect S2 in non-S2 controls.')
        print('   → The persistence of S2 is not a fitting artifact.')
    elif fp_rate > 0.4:
        print('   INCONCLUSIVE: Detector has high false positive rate.')
        print('   → Cannot distinguish S2 ancestry from fitting flexibility.')
    else:
        print('   PARTIAL: Some evidence for S2 ancestry but not decisive.')


if __name__ == '__main__':
    main()
