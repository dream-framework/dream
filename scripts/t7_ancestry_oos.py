#!/usr/bin/env python3
"""
T7 S2 Ancestry Test: Out-of-Sample Prediction
================================================

The decisive test. S2_DUST has 6 parameters and wins in-sample on
96% of real datasets — but it also wins on 85% of synthetic non-S2
data. So in-sample AICc cannot distinguish "S2 ancestry" from
"fitting flexibility."

Out-of-sample prediction CAN.

If S2_DUST wins because of fitting flexibility:
  → It overfits training data
  → It predicts held-out data WORSE than equally flexible non-S2 models

If S2_DUST wins because of S2 ancestry:
  → The S2 functional form genuinely matches the data's generating structure
  → It predicts held-out data BETTER than non-S2 models with same k

Protocol:
  1. Split each ACF curve: first 70% = training, last 30% = test
  2. Fit S2_DUST (6-param) and 2-comp Lognormal (6-param) on training only
  3. Compute prediction RSS on held-out test portion
  4. Compare: does S2_DUST predict better out-of-sample?

  5. Repeat on synthetic non-S2 data (BIEXP, power-law, lognormal)
  6. Check: does S2_DUST predict WORSE out-of-sample when the true
     process is non-S2?

If S2_DUST predicts better on real data AND worse on synthetic non-S2
data, that's evidence for S2 ancestry.
"""
import os, sys, json, csv, io
import numpy as np
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import m_s2_dust, m_lognormal


# ── 6-param non-S2 models (same k as S2_DUST) ──

def m_lognormal_2comp(t, A1, m1, s1, A2, m2, s2):
    """2-component lognormal — 6 params, NOT an S2 mixture."""
    s1, s2 = max(s1, 1e-3), max(s2, 1e-3)
    t = np.maximum(t, 1e-6)
    return A1 / (t * s1 * np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(t)-m1)/s1)**2) + \
           A2 / (t * s2 * np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(t)-m2)/s2)**2)

def m_biexp_2comp(t, A1, l1, A2, l2):
    """2-component exponential — 4 params, NOT S2. For comparison with single S2."""
    return A1 * np.exp(-t / max(l1, 1e-6)) + A2 * np.exp(-t / max(l2, 1e-6))


def safe_fit(func, t, R, p0_list, bounds=None, maxfev=30000):
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


def refetch_values(url):
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
                if 'items' in obj: return [item['views'] for item in obj['items']]
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


def run_out_of_sample(t, R, split_frac=0.7):
    """Split data, fit on training, predict on test. Return prediction RSS for each model."""
    n = len(t)
    split = int(n * split_frac)
    if split < 10 or n - split < 5:
        return None

    t_train, R_train = t[:split], R[:split]
    t_test, R_test = t[split:], R[split:]

    if R_train[0] > 0:
        R_train_n = R_train / R_train[0]
    else:
        R_train_n = R_train.copy()

    t_mid = float(t_train[len(t_train)//2])

    # --- S2_DUST (6-param S2 mixture) ---
    s2dust_fit = safe_fit(m_s2_dust, t_train, R_train_n,
        p0_list=[
            [0.7, t_mid*0.3, 1.5, 0.3, t_mid*2, 0.5],
            [0.6, t_mid*0.5, 1.0, 0.4, t_mid, 0.6],
            [0.5, t_mid, 1.2, 0.5, t_mid*3, 0.4],
        ],
        bounds=([0.0, 1e-3, 0.01, 0.0, 1e-3, 0.01], [2.0, 1e6, 10.0, 2.0, 1e6, 10.0]),
        maxfev=30000)

    # --- 2-comp Lognormal (6-param non-S2) ---
    ln2_fit = safe_fit(m_lognormal_2comp, t_train, R_train_n,
        p0_list=[
            [0.5, np.log(max(t_mid*0.3, 1)), 0.5, 0.5, np.log(max(t_mid*2, 1)), 0.8],
            [0.6, np.log(max(t_mid*0.5, 1)), 0.8, 0.4, np.log(max(t_mid, 1)), 0.6],
        ])

    if not s2dust_fit or not ln2_fit:
        return None

    # Predict on test set
    # Need to normalize test R the same way
    if R_train[0] > 0:
        R_test_n = R_test / R_train[0]
    else:
        R_test_n = R_test.copy()

    s2dust_pred = m_s2_dust(t_test, *s2dust_fit[0])
    ln2_pred = m_lognormal_2comp(t_test, *ln2_fit[0])

    s2dust_rss_test = float(np.sum((R_test_n - s2dust_pred)**2))
    ln2_rss_test = float(np.sum((R_test_n - ln2_pred)**2))

    # Also training RSS for reference
    s2dust_rss_train = s2dust_fit[1]
    ln2_rss_train = ln2_fit[1]

    return {
        's2dust_rss_train': s2dust_rss_train,
        'ln2_rss_train': ln2_rss_train,
        's2dust_rss_test': s2dust_rss_test,
        'ln2_rss_test': ln2_rss_test,
        's2dust_wins_train': s2dust_rss_train < ln2_rss_train,
        's2dust_wins_test': s2dust_rss_test < ln2_rss_test,
        'ratio_test': s2dust_rss_test / max(ln2_rss_test, 1e-12),  # <1 = S2_DUST predicts better
        'ratio_train': s2dust_rss_train / max(ln2_rss_train, 1e-12),
        'overfit_s2dust': s2dust_rss_test / max(s2dust_rss_train, 1e-12),  # how much worse test is
        'overfit_ln2': ln2_rss_test / max(ln2_rss_train, 1e-12),
    }


def test_real_data():
    """Test on real registry datasets."""
    print('=' * 80)
    print('REAL DATA: Out-of-sample S2_DUST vs 2-comp Lognormal')
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
        if taus is None or len(taus) < 30: continue

        t = np.array(taus, dtype=float)
        R = np.array(acf, dtype=float)
        t = t - t[0]
        if R[0] > 0: R = R / R[0]

        res = run_out_of_sample(t, R)
        if res:
            res['name'] = entry['name'][:45]
            results.append(res)

    print(f'\nTested {len(results)} real datasets (70% train / 30% test)')
    print()

    s2dust_train_wins = sum(1 for r in results if r['s2dust_wins_train'])
    s2dust_test_wins = sum(1 for r in results if r['s2dust_wins_test'])

    print(f'In-sample (training): S2_DUST wins {s2dust_train_wins}/{len(results)} ({100*s2dust_train_wins/len(results):.0f}%)')
    print(f'Out-of-sample (test): S2_DUST wins {s2dust_test_wins}/{len(results)} ({100*s2dust_test_wins/len(results):.0f}%)')
    print()

    mean_ratio_test = np.mean([r['ratio_test'] for r in results])
    median_ratio_test = np.median([r['ratio_test'] for r in results])
    print(f'Test RSS ratio (S2_DUST / Lognormal):')
    print(f'  Mean: {mean_ratio_test:.3f} ({"S2_DUST predicts better" if mean_ratio_test < 1 else "Lognormal predicts better"})')
    print(f'  Median: {median_ratio_test:.3f}')
    print()

    mean_overfit_s2dust = np.mean([r['overfit_s2dust'] for r in results])
    mean_overfit_ln2 = np.mean([r['overfit_ln2'] for r in results])
    print(f'Overfitting ratio (test RSS / train RSS):')
    print(f'  S2_DUST:    {mean_overfit_s2dust:.2f}x')
    print(f'  Lognormal:  {mean_overfit_ln2:.2f}x')
    print(f'  (lower = less overfitting)')

    return results


def test_synthetic_non_s2():
    """Test on synthetic non-S2 data — S2_DUST should predict WORSE here."""
    print()
    print('=' * 80)
    print('SYNTHETIC NON-S2: Out-of-sample (S2_DUST should LOSE here)')
    print('=' * 80)

    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=float)
    results = []

    # Type 1: True BIEXP (sum of 2 exponentials)
    print('\n  Type 1: True BIEXP data')
    for i in range(25):
        R = 0.6 * np.exp(-t/5) + 0.4 * np.exp(-t/30) + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_out_of_sample(t, R)
        if res:
            res['type'] = 'biexp_true'
            results.append(res)

    # Type 2: True power-law
    print('  Type 2: True power-law data')
    for i in range(25):
        R = np.power(t + 1, -1.5) + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_out_of_sample(t, R)
        if res:
            res['type'] = 'power_law'
            results.append(res)

    # Type 3: True lognormal
    print('  Type 3: True lognormal data')
    for i in range(25):
        mu, sigma = np.log(10), 0.8
        R = (1 / (np.maximum(t,1) * sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(np.maximum(t,1))-mu)/sigma)**2)
        R[0] = R[1]
        R = R + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_out_of_sample(t, R)
        if res:
            res['type'] = 'lognormal_true'
            results.append(res)

    # Type 4: True S2 (positive control — S2_DUST SHOULD win here)
    print('  Type 4: True S2 data (positive control)')
    for i in range(25):
        R = np.exp(-np.power(t/10, 0.8)) + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_out_of_sample(t, R)
        if res:
            res['type'] = 'pure_s2'
            results.append(res)

    print(f'\n  Results ({len(results)} synthetic datasets):')
    print(f'  {"Type":<20} {"N":>3} {"Train S2 wins":>14} {"Test S2 wins":>13} {"Median ratio":>13}')
    print('  ' + '-' * 65)
    for typ in ['biexp_true', 'power_law', 'lognormal_true', 'pure_s2']:
        subset = [r for r in results if r['type'] == typ]
        if not subset: continue
        n = len(subset)
        train_w = sum(1 for r in subset if r['s2dust_wins_train'])
        test_w = sum(1 for r in subset if r['s2dust_wins_test'])
        med_ratio = np.median([r['ratio_test'] for r in subset])
        print(f'  {typ:<20} {n:>3} {train_w:>14} {test_w:>13} {med_ratio:>13.3f}')

    return results


def main():
    print('=' * 80)
    print('T7 S2 ANCESTRY: OUT-OF-SAMPLE PREDICTION TEST')
    print('=' * 80)
    print()
    print('The decisive test. In-sample AICc cannot distinguish S2 ancestry')
    print('from fitting flexibility (S2_DUST has 6 params, wins on everything).')
    print()
    print('Out-of-sample CAN: if S2_DUST predicts held-out data better than')
    print('an equally flexible non-S2 model, the S2 functional form genuinely')
    print('matches the data structure. If it just overfits, it will predict worse.')
    print()

    real_results = test_real_data()
    synth_results = test_synthetic_non_s2()

    # ── Verdict ──
    print()
    print('=' * 80)
    print('VERDICT')
    print('=' * 80)

    real_test_wins = sum(1 for r in real_results if r['s2dust_wins_test'])
    real_ratio = np.median([r['ratio_test'] for r in real_results])

    biexp_subset = [r for r in synth_results if r['type'] == 'biexp_true']
    biexp_test_wins = sum(1 for r in biexp_subset if r['s2dust_wins_test'])
    biexp_ratio = np.median([r['ratio_test'] for r in biexp_subset]) if biexp_subset else float('nan')

    power_subset = [r for r in synth_results if r['type'] == 'power_law']
    power_test_wins = sum(1 for r in power_subset if r['s2dust_wins_test'])
    power_ratio = np.median([r['ratio_test'] for r in power_subset]) if power_subset else float('nan')

    s2_subset = [r for r in synth_results if r['type'] == 'pure_s2']
    s2_test_wins = sum(1 for r in s2_subset if r['s2dust_wins_test'])
    s2_ratio = np.median([r['ratio_test'] for r in s2_subset]) if s2_subset else float('nan')

    print(f'\n  Real data:     S2_DUST predicts better on {real_test_wins}/{len(real_results)} datasets (median ratio = {real_ratio:.3f})')
    print(f'  True BIEXP:    S2_DUST predicts better on {biexp_test_wins}/{len(biexp_subset)} datasets (median ratio = {biexp_ratio:.3f})')
    print(f'  True power:    S2_DUST predicts better on {power_test_wins}/{len(power_subset)} datasets (median ratio = {power_ratio:.3f})')
    print(f'  True S2:       S2_DUST predicts better on {s2_test_wins}/{len(s2_subset)} datasets (median ratio = {s2_ratio:.3f})')
    print()
    print(f'  Ratio < 1 = S2_DUST predicts held-out data better')
    print(f'  Ratio > 1 = Lognormal predicts held-out data better')
    print()

    # Decision logic
    real_s2_better = real_ratio < 1
    biexp_s2_worse = biexp_ratio > 1
    power_s2_worse = power_ratio > 1
    s2_control_works = s2_ratio < 1

    if real_s2_better and (biexp_s2_worse or power_s2_worse) and s2_control_works:
        verdict = 'SUPPORTED'
        reason = ('S2_DUST predicts held-out real data better than non-S2 models, '
                  'but predicts WORSE on synthetic non-S2 data. The S2 functional '
                  'form genuinely matches real data structure — not just fitting flexibility.')
    elif real_s2_better and not (biexp_s2_worse or power_s2_worse):
        verdict = 'INCONCLUSIVE'
        reason = ('S2_DUST predicts real data better, but ALSO predicts synthetic '
                  'non-S2 data better. Cannot distinguish ancestry from flexibility.')
    elif not real_s2_better:
        verdict = 'NOT SUPPORTED'
        reason = ('S2_DUST does NOT predict held-out real data better than non-S2 '
                  'models. The in-sample dominance was fitting flexibility, not ancestry.')
    else:
        verdict = 'PARTIAL'
        reason = 'Mixed results across conditions.'

    print(f'  S2 ANCESTRY: {verdict}')
    print(f'  {reason}')


if __name__ == '__main__':
    main()
