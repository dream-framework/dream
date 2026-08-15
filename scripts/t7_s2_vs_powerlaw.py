#!/usr/bin/env python3
"""
S2 vs Power-Law Discriminator: Out-of-Sample
==============================================

The refined T7 test narrowed S2 ancestry vs all alternatives to one
remaining confound: S2_DUST predicts both real data AND power-law
data well out-of-sample. We need to distinguish these.

Key insight: S2 and power-law differ most in the SHORT-LAG regime.

  S2:    R(λ) = exp[-(λ/λ_q)^D]  ≈ 1 - (λ/λ_q)^D + ...  (polynomial departure from 1)
  Power:  R(λ) = A·λ^{-α}         ≈ A·λ^{-α}              (diverges at λ→0)

Near λ=0:
  S2 starts at R=1 and departs as a power of λ
  Power-law either diverges or starts at A≠1

At large λ:
  Both are heavy-tail, hard to distinguish

So the discriminator should:
  1. Fit S2_DUST and 2-comp Power on TRAINING data (first 70%)
  2. Predict TEST data (last 30%)
  3. BUT ALSO: fit on LAGS 1-10 only, predict lags 11-50
     (short-lag training, long-lag prediction — where they diverge)
  4. AND: fit on lags 30-50, predict lags 1-10
     (long-lag training, short-lag prediction — the reverse)

If S2 ancestry is correct:
  - S2_DUST should predict short→long AND long→short well
  - Power should predict short→long poorly (can't extrapolate to R→1)

If power-law ancestry is correct:
  - Power should predict well in both directions
  - S2_DUST should also do well (since it can mimic power-law)

The decisive test: long→short extrapolation.
  Power-law fit on large λ will predict R(0)→∞ or R(0)=A≠1
  S2 fit on large λ will predict R(0)→1 (by construction)

Real ACF curves have R(0)=1. If power-law mispredicts R(0) badly
while S2 gets it right, that's the discriminator.
"""
import os, sys, json, csv, io
import numpy as np
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import m_s2_dust, m_s2, m_power


def m_power_2comp(t, A1, a1, A2, a2):
    """2-component power law — 4 params."""
    return A1 * np.power(np.maximum(t, 1e-6), -max(a1, 0.01)) + \
           A2 * np.power(np.maximum(t, 1e-6), -max(a2, 0.01))


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


def refetch_values(url):
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url:
            data = fetch_url(url, timeout=15)
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
        if 'wikimedia.org' in url:
            data = fetch_url(url, timeout=15)
            if not data: return None
            try:
                obj = json.loads(data)
                if 'items' in obj: return [item['views'] for item in obj['items']]
            except: pass
            return None
        if 'open-meteo' in url:
            data = fetch_url(url, timeout=20)
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
        if 'binance.com' in url:
            data = fetch_url(url, timeout=15)
            if not data: return None
            try:
                arr = json.loads(data)
                return [float(k[4]) for k in arr] if arr else None
            except: pass
            return None
        if 'CSSEGISandData' in url:
            data = fetch_url(url, timeout=15)
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
    except: pass
    return None


def run_discriminator(t, R):
    """Run S2_DUST vs 2-comp Power-law on three split regimes.

    Returns dict with RSS for each model in each split direction.
    """
    n = len(t)
    if n < 30: return None

    t_mid = float(t[n//2])
    norm = R[0] if R[0] > 0 else 1.0
    R_n = R / norm

    results = {}

    # ── Test 1: Standard 70/30 (train early, predict late) ──
    split = int(n * 0.7)
    t_tr, R_tr = t[:split], R_n[:split]
    t_te, R_te = t[split:], R_n[split:]

    s2d = safe_fit(m_s2_dust, t_tr, R_tr,
        p0_list=[[0.7,t_mid*0.3,1.5,0.3,t_mid*2,0.5],[0.6,t_mid*0.5,1.0,0.4,t_mid,0.6]],
        bounds=([0,1e-3,0.01,0,1e-3,0.01],[2,1e6,10,2,1e6,10]))
    pw2 = safe_fit(m_power_2comp, t_tr, R_tr,
        p0_list=[[0.7,0.5,0.3,1.5],[0.5,1.0,0.5,0.8]],
        bounds=([0.01,0.01,0.01,0.01],[2,10,2,10]))

    results['standard_s2dust'] = float(np.sum((R_te - m_s2_dust(t_te, *s2d[0]))**2)) if s2d else float('inf')
    results['standard_power2'] = float(np.sum((R_te - m_power_2comp(t_te, *pw2[0]))**2)) if pw2 else float('inf')

    # ── Test 2: Short→Long (train on first 30%, predict last 70%) ──
    split2 = int(n * 0.3)
    t_tr2, R_tr2 = t[:split2], R_n[:split2]
    t_te2, R_te2 = t[split2:], R_n[split2:]

    t_mid2 = float(t_tr2[len(t_tr2)//2]) if len(t_tr2) > 0 else 1.0
    s2d2 = safe_fit(m_s2_dust, t_tr2, R_tr2,
        p0_list=[[0.7,t_mid2*0.3,1.5,0.3,t_mid2*2,0.5]],
        bounds=([0,1e-3,0.01,0,1e-3,0.01],[2,1e6,10,2,1e6,10]))
    pw2_2 = safe_fit(m_power_2comp, t_tr2, R_tr2,
        p0_list=[[0.7,0.5,0.3,1.5]],
        bounds=([0.01,0.01,0.01,0.01],[2,10,2,10]))

    results['short2long_s2dust'] = float(np.sum((R_te2 - m_s2_dust(t_te2, *s2d2[0]))**2)) if s2d2 else float('inf')
    results['short2long_power2'] = float(np.sum((R_te2 - m_power_2comp(t_te2, *pw2_2[0]))**2)) if pw2_2 else float('inf')

    # ── Test 3: Long→Short (train on last 50%, predict first 50%) ──
    # THIS IS THE DISCRIMINATOR. Power-law fit on large λ will mispredict R(0).
    split3 = int(n * 0.5)
    t_tr3, R_tr3 = t[split3:], R_n[split3:]
    t_te3, R_te3 = t[:split3], R_n[:split3]

    t_mid3 = float(t_tr3[len(t_tr3)//2]) if len(t_tr3) > 0 else 1.0
    s2d3 = safe_fit(m_s2_dust, t_tr3, R_tr3,
        p0_list=[[0.7,t_mid3*0.3,1.5,0.3,t_mid3*2,0.5]],
        bounds=([0,1e-3,0.01,0,1e-3,0.01],[2,1e6,10,2,1e6,10]))
    pw2_3 = safe_fit(m_power_2comp, t_tr3, R_tr3,
        p0_list=[[0.7,0.5,0.3,1.5]],
        bounds=([0.01,0.01,0.01,0.01],[2,10,2,10]))

    results['long2short_s2dust'] = float(np.sum((R_te3 - m_s2_dust(t_te3, *s2d3[0]))**2)) if s2d3 else float('inf')
    results['long2short_power2'] = float(np.sum((R_te3 - m_power_2comp(t_te3, *pw2_3[0]))**2)) if pw2_3 else float('inf')

    # ── Test 4: R(0) prediction error (the sharpest discriminator) ──
    # When fit on large λ, what does each model predict at λ=0?
    if s2d3 and pw2_3:
        # S2_DUST at t=0: A1*exp(0) + A2*exp(0) = A1+A2 (should be ~1)
        s2d_pred_0 = m_s2_dust(np.array([0.0]), *s2d3[0])[0]
        # Power_2 at t=0: diverges! Use t=0.01 instead
        pw_pred_0 = m_power_2comp(np.array([0.01]), *pw2_3[0])[0]
        # True R(0) = 1.0
        results['s2dust_R0_error'] = abs(s2d_pred_0 - 1.0)
        results['power_R0_error'] = abs(pw_pred_0 - 1.0)
    else:
        results['s2dust_R0_error'] = float('nan')
        results['power_R0_error'] = float('nan')

    return results


def main():
    print('=' * 80)
    print('S2 vs POWER-LAW DISCRIMINATOR: Out-of-Sample')
    print('=' * 80)
    print()
    print('The remaining confound: S2_DUST and power-law both predict real')
    print('data well out-of-sample. This test exploits their key difference:')
    print()
    print('  S2:  R(0) = 1 by construction (exp[-0] = 1)')
    print('  Pow: R(0) → ∞ or A≠1 (power law diverges at 0)')
    print()
    print('Three split regimes:')
    print('  1. Standard 70/30 (train early, predict late)')
    print('  2. Short→Long (train first 30%, predict last 70%)')
    print('  3. Long→Short (train last 50%, predict first 50%) ← DISCRIMINATOR')
    print()
    print('Plus: R(0) prediction error when fit on large λ only.')
    print()

    # ── REAL DATA ──
    print('=' * 80)
    print('REAL DATA')
    print('=' * 80)

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    candidates = []
    seen = set()
    for t in data['tests']:
        if t.get('D') is None or (t.get('r2') or 0) < 0.3: continue
        url = t.get('url', '')
        if url in seen: continue
        if any(x in url for x in ['fredgraph.csv', 'wikimedia.org', 'open-meteo',
                                    'binance.com', 'CSSEGISandData']):
            seen.add(url)
            candidates.append(t)

    real_results = []
    for entry in candidates[:20]:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 40: continue
        t = np.array(taus, dtype=float)
        R = np.array(acf, dtype=float)
        t = t - t[0]
        if R[0] > 0: R = R / R[0]
        res = run_discriminator(t, R)
        if res:
            res['name'] = entry['name'][:40]
            real_results.append(res)

    print(f'\nTested {len(real_results)} real datasets\n')

    # Standard split
    s2w = sum(1 for r in real_results if r['standard_s2dust'] < r['standard_power2'])
    print(f'Standard 70/30: S2_DUST wins {s2w}/{len(real_results)} ({100*s2w/len(real_results):.0f}%)')

    # Short→Long
    s2w2 = sum(1 for r in real_results if r['short2long_s2dust'] < r['short2long_power2'])
    print(f'Short→Long:      S2_DUST wins {s2w2}/{len(real_results)} ({100*s2w2/len(real_results):.0f}%)')

    # Long→Short (THE DISCRIMINATOR)
    s2w3 = sum(1 for r in real_results if r['long2short_s2dust'] < r['long2short_power2'])
    print(f'Long→Short:      S2_DUST wins {s2w3}/{len(real_results)} ({100*s2w3/len(real_results):.0f}%)')

    # R(0) prediction error
    s2_r0 = [r['s2dust_R0_error'] for r in real_results if not np.isnan(r['s2dust_R0_error'])]
    pw_r0 = [r['power_R0_error'] for r in real_results if not np.isnan(r['power_R0_error'])]
    print(f'\nR(0) prediction error (fit on large λ, predict at λ=0):')
    print(f'  S2_DUST:     mean={np.mean(s2_r0):.4f}, median={np.median(s2_r0):.4f}')
    print(f'  Power-law:   mean={np.mean(pw_r0):.4f}, median={np.median(pw_r0):.4f}')
    r0_wins = sum(1 for r in real_results if not np.isnan(r['s2dust_R0_error']) and not np.isnan(r['power_R0_error']) and r['s2dust_R0_error'] < r['power_R0_error'])
    print(f'  S2_DUST closer to R(0)=1: {r0_wins}/{len(real_results)}')

    # ── SYNTHETIC CONTROLS ──
    print()
    print('=' * 80)
    print('SYNTHETIC CONTROLS')
    print('=' * 80)

    np.random.seed(42)
    n = 200
    t_syn = np.arange(n, dtype=float)
    synth_results = []

    # True S2
    print('\n  True S2: exp[-(t/10)^0.8]')
    for i in range(15):
        R = np.exp(-np.power(t_syn/10, 0.8)) + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_discriminator(t_syn, R)
        if res:
            res['type'] = 'pure_s2'
            synth_results.append(res)

    # True power-law
    print('  True power-law: (t+1)^-1.5')
    for i in range(15):
        R = np.power(t_syn+1, -1.5) + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_discriminator(t_syn, R)
        if res:
            res['type'] = 'power_law'
            synth_results.append(res)

    # True BIEXP
    print('  True BIEXP: 0.6*exp(-t/5) + 0.4*exp(-t/30)')
    for i in range(15):
        R = 0.6*np.exp(-t_syn/5) + 0.4*np.exp(-t_syn/30) + np.random.normal(0, 0.01, n)
        R = np.clip(R, 0.01, 2.0)
        if R[0] > 0: R = R / R[0]
        res = run_discriminator(t_syn, R)
        if res:
            res['type'] = 'biexp_true'
            synth_results.append(res)

    print(f'\n  {"Type":<15} {"N":>3} {"Std S2 wins":>12} {"S→L S2 wins":>12} {"L→S S2 wins":>12} {"R(0) S2 wins":>13}')
    print('  ' + '-' * 70)
    for typ in ['pure_s2', 'power_law', 'biexp_true']:
        subset = [r for r in synth_results if r['type'] == typ]
        if not subset: continue
        n_s = len(subset)
        std_w = sum(1 for r in subset if r['standard_s2dust'] < r['standard_power2'])
        s2l_w = sum(1 for r in subset if r['short2long_s2dust'] < r['short2long_power2'])
        l2s_w = sum(1 for r in subset if r['long2short_s2dust'] < r['long2short_power2'])
        r0_w = sum(1 for r in subset if not np.isnan(r['s2dust_R0_error']) and not np.isnan(r['power_R0_error']) and r['s2dust_R0_error'] < r['power_R0_error'])
        print(f'  {typ:<15} {n_s:>3} {std_w:>12} {s2l_w:>12} {l2s_w:>12} {r0_w:>13}')

    # ── VERDICT ──
    print()
    print('=' * 80)
    print('VERDICT')
    print('=' * 80)

    real_l2s = sum(1 for r in real_results if r['long2short_s2dust'] < r['long2short_power2'])
    real_r0 = sum(1 for r in real_results if not np.isnan(r['s2dust_R0_error']) and not np.isnan(r['power_R0_error']) and r['s2dust_R0_error'] < r['power_R0_error'])

    s2_l2s = sum(1 for r in synth_results if r['type']=='pure_s2' and r['long2short_s2dust'] < r['long2short_power2'])
    pw_l2s = sum(1 for r in synth_results if r['type']=='power_law' and r['long2short_s2dust'] < r['long2short_power2'])

    print(f'\n  Long→Short extrapolation (the discriminator):')
    print(f'    Real data:    S2_DUST wins {real_l2s}/{len(real_results)} ({100*real_l2s/len(real_results):.0f}%)')
    print(f'    True S2:      S2_DUST wins {s2_l2s}/{sum(1 for r in synth_results if r["type"]=="pure_s2")} (control)')
    print(f'    True power:   S2_DUST wins {pw_l2s}/{sum(1 for r in synth_results if r["type"]=="power_law")} (should LOSE)')

    print(f'\n  R(0) prediction error (fit on large λ, predict R(0)=1):')
    print(f'    Real data:    S2_DUST closer to 1 on {real_r0}/{len(real_results)} datasets')

    if real_l2s > len(real_results) * 0.6 and pw_l2s < sum(1 for r in synth_results if r["type"]=="power_law") * 0.3:
        verdict = 'S2 ANCESTRY SUPPORTED OVER POWER-LAW'
        reason = 'S2_DUST predicts long→short extrapolation better on real data, while power-law fails on the same test.'
    elif real_l2s > len(real_results) * 0.5:
        verdict = 'PARTIALLY SUPPORTED'
        reason = 'S2_DUST has some advantage on long→short, but not decisive.'
    else:
        verdict = 'INCONCLUSIVE'
        reason = 'Cannot distinguish S2 from power-law on this test.'

    print(f'\n  {verdict}')
    print(f'  {reason}')


if __name__ == '__main__':
    main()
