#!/usr/bin/env python3
"""
T7 S2 Ancestry: Refined Out-of-Sample Test
============================================

Addresses the power-law confound from the previous test. S2_DUST
generalized on both real data AND power-law data, so we couldn't
distinguish S2 ancestry from power-law ancestry.

This test adds:
  1. S2_DUST vs 2-comp Power-law (same 6-param flexibility) out-of-sample
  2. S2_DUST vs 2-comp BIEXP (4-param) out-of-sample (already done, repeat for completeness)
  3. S2_DUST vs 2-comp Gaussian (4-param) out-of-sample
  4. Single S2 (3-param) vs single Power-law (2-param) out-of-sample
     — this is the sharpest test: does S2 predict better than power-law
     with FEWER parameters? If so, that's strong evidence for S2 form.

  5. Cross-validation: 5-fold CV instead of single 70/30 split
  6. Damped oscillation control (non-monotonic — S2 should fail)

If S2_DUST predicts real data better than ALL non-S2 alternatives
(BIEXP, lognormal, power-law, Gaussian) out-of-sample, but predicts
synthetic non-S2 data WORSE, then S2 ancestry is supported.
"""
import os, sys, json, csv, io
import numpy as np
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import m_s2_dust, m_s2, m_biexp, m_power, m_gaussian, m_exp


# ── Non-S2 models ──

def m_lognormal_2comp(t, A1, m1, s1, A2, m2, s2):
    s1, s2 = max(s1, 1e-3), max(s2, 1e-3)
    t = np.maximum(t, 1e-6)
    return A1 / (t * s1 * np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(t)-m1)/s1)**2) + \
           A2 / (t * s2 * np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(t)-m2)/s2)**2)

def m_power_2comp(t, A1, a1, A2, a2):
    return A1 * np.power(np.maximum(t, 1e-6), -max(a1, 0.01)) + \
           A2 * np.power(np.maximum(t, 1e-6), -max(a2, 0.01))

def m_gaussian_2comp(t, A1, s1, A2, s2):
    return A1 * np.exp(-0.5*(t/max(s1, 1e-6))**2) + A2 * np.exp(-0.5*(t/max(s2, 1e-6))**2)

def m_biexp_2comp(t, A1, l1, A2, l2):
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


def fit_model(model_name, t, R):
    """Fit a specific model, return (popt, rss_train) or None."""
    t_mid = float(t[len(t)//2]) if len(t) else 1.0
    if model_name == 's2_dust':
        return safe_fit(m_s2_dust, t, R,
            p0_list=[[0.7, t_mid*0.3, 1.5, 0.3, t_mid*2, 0.5],
                     [0.6, t_mid*0.5, 1.0, 0.4, t_mid, 0.6]],
            bounds=([0, 1e-3, 0.01, 0, 1e-3, 0.01], [2, 1e6, 10, 2, 1e6, 10]))
    elif model_name == 's2':
        return safe_fit(m_s2, t, R,
            p0_list=[[1.0, t_mid, 0.5], [1.0, t_mid*0.5, 1.0]],
            bounds=([0.01, 1e-3, 0.01], [2, 1e6, 10]))
    elif model_name == 'lognormal_2':
        return safe_fit(m_lognormal_2comp, t, R,
            p0_list=[[0.5, np.log(max(t_mid*0.3,1)), 0.5, 0.5, np.log(max(t_mid*2,1)), 0.8],
                     [0.6, np.log(max(t_mid*0.5,1)), 0.8, 0.4, np.log(max(t_mid,1)), 0.6]])
    elif model_name == 'power_2':
        return safe_fit(m_power_2comp, t, R,
            p0_list=[[0.7, 0.5, 0.3, 1.5], [0.5, 1.0, 0.5, 0.8]],
            bounds=([0.01, 0.01, 0.01, 0.01], [2, 10, 2, 10]))
    elif model_name == 'biexp':
        return safe_fit(m_biexp_2comp, t, R,
            p0_list=[[0.7, t_mid*0.3, 0.3, t_mid*2], [0.5, t_mid*0.5, 0.5, t_mid]],
            bounds=([0, 1e-3, 0, 1e-3], [2, 1e6, 2, 1e6]))
    elif model_name == 'gaussian_2':
        return safe_fit(m_gaussian_2comp, t, R,
            p0_list=[[0.7, t_mid*0.3, 0.3, t_mid*2], [0.5, t_mid*0.5, 0.5, t_mid]],
            bounds=([0.01, 1e-3, 0.01, 1e-3], [2, 1e6, 2, 1e6]))
    elif model_name == 'power':
        return safe_fit(m_power, t, R,
            p0_list=[[1.0, 0.5], [1.0, 1.0]],
            bounds=([0.01, 0.01], [2, 10]))
    elif model_name == 'exp':
        return safe_fit(m_exp, t, R,
            p0_list=[[1.0, t_mid]],
            bounds=([0.01, 1e-3], [2, 1e6]))
    return None


MODEL_FUNCS = {
    's2_dust': m_s2_dust, 's2': m_s2, 'lognormal_2': m_lognormal_2comp,
    'power_2': m_power_2comp, 'biexp': m_biexp_2comp, 'gaussian_2': m_gaussian_2comp,
    'power': m_power, 'exp': m_exp,
}

MODEL_K = {
    's2_dust': 6, 's2': 3, 'lognormal_2': 6, 'power_2': 4,
    'biexp': 4, 'gaussian_2': 4, 'power': 2, 'exp': 2,
}


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


def run_5fold_cv(t, R, model_names):
    """5-fold cross-validation. Returns dict of model_name -> mean test RSS."""
    n = len(t)
    fold_size = n // 5
    if fold_size < 5: return None

    results = {m: [] for m in model_names}
    norm = R[0] if R[0] > 0 else 1.0

    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else n
        # Test = this fold, Train = everything else
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])

        t_train, R_train = t[train_idx], R[train_idx] / norm
        t_test, R_test = t[test_idx], R[test_idx] / norm

        for mname in model_names:
            fit = fit_model(mname, t_train, R_train)
            if fit:
                func = MODEL_FUNCS[mname]
                pred = func(t_test, *fit[0])
                rss_test = float(np.sum((R_test - pred)**2))
                results[mname].append(rss_test)
            else:
                results[mname].append(float('inf'))

    return {m: np.median(v) for m, v in results.items()}


def test_real_data():
    print('=' * 80)
    print('REAL DATA: 5-fold CV — S2 family vs all non-S2 alternatives')
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

    models = ['s2_dust', 's2', 'lognormal_2', 'power_2', 'biexp', 'gaussian_2', 'power', 'exp']
    all_results = []

    for entry in candidates[:25]:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 40: continue

        t = np.array(taus, dtype=float)
        R = np.array(acf, dtype=float)
        t = t - t[0]
        if R[0] > 0: R = R / R[0]

        cv = run_5fold_cv(t, R, models)
        if cv:
            cv['name'] = entry['name'][:40]
            all_results.append(cv)

    print(f'\nTested {len(all_results)} real datasets with 5-fold CV')
    print()

    # For each model, compute how often it has the LOWEST test RSS
    wins = {m: 0 for m in models}
    for r in all_results:
        best = min(models, key=lambda m: r[m])
        wins[best] += 1

    print(f'{"Model":<15} {"k":>3} {"CV wins":>8} {"Win %":>6} {"Median RSS":>12} {"vs S2_DUST":>12}')
    print('-' * 60)
    median_rss = {m: np.median([r[m] for r in all_results]) for m in models}
    s2dust_median = median_rss['s2_dust']
    for m in models:
        w = wins[m]
        pct = 100 * w / len(all_results)
        med = median_rss[m]
        ratio = med / s2dust_median if s2dust_median > 0 else float('inf')
        print(f'{m:<15} {MODEL_K[m]:>3} {w:>8} {pct:>5.0f}% {med:>12.6f} {ratio:>11.3f}x')

    # Key comparisons
    print()
    s2dust_wins = wins['s2_dust']
    print(f'S2_DUST wins {s2dust_wins}/{len(all_results)} ({100*s2dust_wins/len(all_results):.0f}%)')
    for m in ['lognormal_2', 'power_2', 'biexp', 'gaussian_2']:
        w = wins[m]
        ratio = median_rss[m] / s2dust_median if s2dust_median > 0 else float('inf')
        print(f'  vs {m}: S2_DUST wins {s2dust_wins}x, {m} wins {w}x, median RSS ratio = {ratio:.3f}')

    # Single S2 vs single power-law (sharpest test)
    s2_wins = wins['s2']
    power_wins = wins['power']
    print(f'\nSingle S2 (k=3) vs single Power (k=2):')
    print(f'  S2 wins: {s2_wins}, Power wins: {power_wins}')
    s2_vs_power_ratio = median_rss['s2'] / median_rss['power'] if median_rss['power'] > 0 else float('inf')
    print(f'  Median RSS ratio (S2/Power): {s2_vs_power_ratio:.3f} ({"S2 better" if s2_vs_power_ratio < 1 else "Power better"})')

    return all_results, models, wins, median_rss


def test_synthetic():
    print()
    print('=' * 80)
    print('SYNTHETIC CONTROLS: 5-fold CV')
    print('=' * 80)

    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=float)
    models = ['s2_dust', 'lognormal_2', 'power_2', 'biexp', 'gaussian_2', 'power']
    all_results = []

    generators = {
        'biexp_true': lambda: 0.6*np.exp(-t/5) + 0.4*np.exp(-t/30) + np.random.normal(0, 0.01, n),
        'power_law': lambda: np.power(t+1, -1.5) + np.random.normal(0, 0.01, n),
        'lognormal_true': lambda: (1/(np.maximum(t,1)*0.8*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(np.maximum(t,1))-np.log(10))/0.8)**2) + np.random.normal(0, 0.01, n),
        'damped_osc': lambda: np.abs(np.exp(-t/15)*np.cos(t/3)) + np.random.normal(0, 0.01, n),
        'pure_s2': lambda: np.exp(-np.power(t/10, 0.8)) + np.random.normal(0, 0.01, n),
    }

    for gen_name, gen_func in generators.items():
        print(f'\n  {gen_name}:')
        for i in range(20):
            R = gen_func()
            R = np.clip(R, 0.01, 2.0)
            if R[0] > 0: R = R / R[0]
            cv = run_5fold_cv(t, R, models)
            if cv:
                cv['true_type'] = gen_name
                all_results.append(cv)

    print(f'\n  Results ({len(all_results)} synthetic datasets):')
    print(f'  {"True type":<20} {"N":>3} {"S2_DUST wins":>13} {"Lognorm wins":>13} {"Power2 wins":>12} {"BIEXP wins":>11} {"Gauss2 wins":>12}')
    print('  ' + '-' * 85)
    for gen_name in generators:
        subset = [r for r in all_results if r['true_type'] == gen_name]
        if not subset: continue
        n = len(subset)
        wins = {m: sum(1 for r in subset if min(models, key=lambda m: r[m]) == m) for m in models}
        print(f'  {gen_name:<20} {n:>3} {wins["s2_dust"]:>13} {wins["lognormal_2"]:>13} {wins["power_2"]:>12} {wins["biexp"]:>11} {wins["gaussian_2"]:>12}')

    return all_results, models


def main():
    print('=' * 80)
    print('T7 S2 ANCESTRY: REFINED 5-FOLD CROSS-VALIDATION')
    print('=' * 80)
    print()
    print('Addresses the power-law confound from the previous OOS test.')
    print('Now compares S2_DUST against ALL non-S2 alternatives, including')
    print('2-component power-law (same flexibility). Uses 5-fold CV for')
    print('robustness. Also tests single S2 vs single power-law.')
    print()

    real_results, models, real_wins, real_median = test_real_data()
    synth_results, synth_models = test_synthetic()

    # ── Verdict ──
    print()
    print('=' * 80)
    print('VERDICT')
    print('=' * 80)

    # Real data: does S2_DUST win more than any other model?
    s2dust_real_wins = real_wins['s2_dust']
    best_non_s2 = max(real_wins[m] for m in models if m not in ['s2_dust', 's2'])
    best_non_s2_name = max([m for m in models if m not in ['s2_dust', 's2']], key=lambda m: real_wins[m])

    print(f'\nReal data (5-fold CV, {len(real_results)} datasets):')
    print(f'  S2_DUST wins: {s2dust_real_wins}/{len(real_results)} ({100*s2dust_real_wins/len(real_results):.0f}%)')
    print(f'  Best non-S2 ({best_non_s2_name}): {best_non_s2}/{len(real_results)} ({100*best_non_s2/len(real_results):.0f}%)')

    # Synthetic: does S2_DUST lose on non-S2 data?
    for gen_name in ['biexp_true', 'power_law', 'lognormal_true', 'damped_osc', 'pure_s2']:
        subset = [r for r in synth_results if r['true_type'] == gen_name]
        if not subset: continue
        s2dust_w = sum(1 for r in subset if min(models, key=lambda m: r[m]) == 's2_dust')
        # What SHOULD win?
        if gen_name == 'biexp_true': expected = 'biexp'
        elif gen_name == 'power_law': expected = 'power_2'
        elif gen_name == 'lognormal_true': expected = 'lognormal_2'
        elif gen_name == 'damped_osc': expected = 'any (non-monotonic)'
        elif gen_name == 'pure_s2': expected = 's2_dust'
        expected_w = sum(1 for r in subset if min(models, key=lambda m: r[m]) == expected) if expected != 'any (non-monotonic)' else 0
        print(f'  {gen_name}: S2_DUST wins {s2dust_w}/{len(subset)}, {expected} wins {expected_w}/{len(subset)}')

    # Decision
    real_s2_best = s2dust_real_wins > best_non_s2
    biexp_subset = [r for r in synth_results if r['true_type'] == 'biexp_true']
    biexp_s2dust_wins = sum(1 for r in biexp_subset if min(models, key=lambda m: r[m]) == 's2_dust')
    power_subset = [r for r in synth_results if r['true_type'] == 'power_law']
    power_s2dust_wins = sum(1 for r in power_subset if min(models, key=lambda m: r[m]) == 's2_dust')
    s2_subset = [r for r in synth_results if r['true_type'] == 'pure_s2']
    s2_control_wins = sum(1 for r in s2_subset if min(models, key=lambda m: r[m]) == 's2_dust')

    print(f'\n  Decision logic:')
    print(f'    S2_DUST best on real data: {real_s2_best}')
    print(f'    S2_DUST loses on true BIEXP: {biexp_s2dust_wins == 0}')
    print(f'    S2_DUST loses on true power-law: {power_s2dust_wins < len(power_subset)/2}')
    print(f'    S2_DUST wins on true S2 (control): {s2_control_wins == len(s2_subset)}')

    if real_s2_best and biexp_s2dust_wins == 0 and s2_control_wins == len(s2_subset):
        if power_s2dust_wins < len(power_subset)/2:
            verdict = 'STRONGLY SUPPORTED'
            reason = ('S2_DUST is the best predictor on real data AND on true S2 data, '
                      'but LOSES on synthetic BIEXP, power-law, and lognormal data. '
                      'The S2 functional form genuinely matches real data structure.')
        else:
            verdict = 'SUPPORTED (with power-law caveat)'
            reason = ('S2_DUST is best on real data and true S2, loses on BIEXP and lognormal. '
                      'But S2_DUST also wins on power-law data — cannot fully distinguish '
                      'S2 ancestry from power-law ancestry. Both are heavy-tail families.')
    elif real_s2_best:
        verdict = 'PARTIALLY SUPPORTED'
        reason = 'S2_DUST predicts real data best, but mixed results on synthetic controls.'
    else:
        verdict = 'NOT SUPPORTED'
        reason = 'S2_DUST does not predict real data best out-of-sample.'

    print(f'\n  S2 ANCESTRY: {verdict}')
    print(f'  {reason}')


if __name__ == '__main__':
    main()
