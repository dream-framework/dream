#!/usr/bin/env python3
"""
S2 vs Power-Law: Scale Discriminator
=====================================

The key structural difference:
  S2:    R(λ) = exp[-(λ/λq)^D]  — has a CHARACTERISTIC SCALE λq
  Power:  R(λ) = A·λ^{-α}       — SCALE-FREE, no preferred scale

Test: Does real data show a characteristic scale?

Method 1: LOG-LOG LINEARITY TEST
  Power-law is linear in log-log space: ln R = ln A - α·ln λ
  S2 is NOT linear: ln R = -(λ/λq)^D, which curves in log-log
  
  Fit a line to ln(R) vs ln(λ). If power-law is correct, residuals
  should be flat. If S2 is correct, residuals should show a systematic
  curve (the "cliff" at λq).

  Metric: R² of the linear fit. High R² = power-law. Low R² = S2.

Method 2: SCALE BREAK DETECTION
  S2 has a transition at λq where the local slope changes.
  Power-law has constant local slope everywhere.
  
  Compute local slope d(ln R)/d(ln λ) in a sliding window.
  If slope is constant → power-law.
  If slope varies systematically → S2 (characteristic scale exists).

Method 3: OUT-OF-SAMPLE WITH SHIFTED SCALE
  Fit on λ ∈ [10, 30], predict λ ∈ [30, 50].
  Then fit on λ ∈ [30, 50], predict λ ∈ [50, 70].
  
  Power-law: same α everywhere → predicts equally well in both.
  S2: the cliff at λq means behavior changes → one direction predicts
  better than the other.

  If power-law ancestry: both directions predict equally well.
  If S2 ancestry: prediction quality differs (scale break exists).
"""
import os, sys, json, csv, io
import numpy as np
from scipy.optimize import curve_fit

REPO = '/home/z/my-project/dream_repo'
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve
from s2_model_compare import m_s2_dust, m_s2, m_power


def m_power_2comp(t, A1, a1, A2, a2):
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


def test_loglog_linearity(t, R):
    """Method 1: Is ln(R) vs ln(λ) linear (power-law) or curved (S2)?

    Power-law: ln R = ln A - α ln λ  (linear)
    S2:        ln R = -(λ/λq)^D       (curves downward)

    Metric: R² of linear fit to ln(R) vs ln(λ).
    High R² (>0.99) → power-law (scale-free).
    Low R² (<0.98) → S2 (characteristic scale exists).
    """
    # Skip λ=0 and any R≤0
    mask = (t > 0) & (R > 0.001)
    if mask.sum() < 5:
        return None

    ln_t = np.log(t[mask])
    ln_R = np.log(R[mask])

    # Linear fit
    coeffs = np.polyfit(ln_t, ln_R, 1)
    pred = np.polyval(coeffs, ln_t)
    ss_res = np.sum((ln_R - pred)**2)
    ss_tot = np.sum((ln_R - np.mean(ln_R))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Also fit quadratic — if quadratic is much better, there's curvature
    coeffs2 = np.polyfit(ln_t, ln_R, 2)
    pred2 = np.polyval(coeffs2, ln_t)
    ss_res2 = np.sum((ln_R - pred2)**2)
    r2_quad = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0

    # The quadratic coefficient: if significantly negative, S2-like curvature
    quad_coeff = coeffs2[0]

    return {
        'r2_linear': r2,
        'r2_quad': r2_quad,
        'r2_improvement': r2_quad - r2,  # how much curvature helps
        'quad_coeff': quad_coeff,  # negative = downward curve (S2-like)
        'linear_slope': coeffs[0],  # = -α for power-law
        'n_points': int(mask.sum()),
    }


def test_local_slope(t, R):
    """Method 2: Does local slope d(ln R)/d(ln λ) vary (S2) or stay constant (power-law)?

    Power-law: slope = -α everywhere (constant)
    S2: slope starts at 0 (near λ=0, R≈1), increases to -D at large λ

    Metric: coefficient of variation of local slope.
    Low CV → power-law (constant slope).
    High CV → S2 (slope varies = characteristic scale exists).
    """
    mask = (t > 0) & (R > 0.001)
    if mask.sum() < 10:
        return None

    ln_t = np.log(t[mask])
    ln_R = np.log(R[mask])

    # Sliding window local slope (window=5)
    window = 5
    slopes = []
    for i in range(len(ln_t) - window):
        t_win = ln_t[i:i+window]
        R_win = ln_R[i:i+window]
        if len(t_win) >= 3 and np.var(t_win) > 0:
            c = np.polyfit(t_win, R_win, 1)
            slopes.append(c[0])

    if len(slopes) < 3:
        return None

    slopes = np.array(slopes)
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    cv = std_slope / abs(mean_slope) if abs(mean_slope) > 0 else float('inf')

    # Range of slopes
    slope_range = np.max(slopes) - np.min(slopes)

    return {
        'mean_slope': float(mean_slope),
        'std_slope': float(std_slope),
        'cv': float(cv),
        'slope_range': float(slope_range),
        'n_slopes': len(slopes),
        'slopes': slopes.tolist(),
    }


def test_scale_shifted_oos(t, R):
    """Method 3: Fit on [a,b], predict [b,c]. Then fit [b,c], predict [c,d].

    Power-law: same α everywhere → both predictions equally good.
    S2: cliff at λq → one direction better than the other.
    """
    n = len(t)
    if n < 30:
        return None

    # Split into three equal segments
    seg = n // 3
    t1, R1 = t[:seg], R[:seg]
    t2, R2 = t[seg:2*seg], R[seg:2*seg]
    t3, R3 = t[2*seg:], R[2*seg:]

    norm = R[0] if R[0] > 0 else 1.0
    R1n, R2n, R3n = R1/norm, R2/norm, R3/norm

    t_mid = float(t[n//2])

    results = {}

    # S2_DUST: fit seg1, predict seg2; fit seg2, predict seg3
    for model_name, func, p0s, bounds in [
        ('s2_dust', m_s2_dust,
         [[0.7, t_mid*0.3, 1.5, 0.3, t_mid*2, 0.5]],
         ([0, 1e-3, 0.01, 0, 1e-3, 0.01], [2, 1e6, 10, 2, 1e6, 10])),
        ('power_2', m_power_2comp,
         [[0.7, 0.5, 0.3, 1.5]],
         ([0.01, 0.01, 0.01, 0.01], [2, 10, 2, 10])),
    ]:
        # Fit seg1, predict seg2
        f12 = safe_fit(func, t1, R1n, p0s, bounds)
        rss12 = float(np.sum((R2n - func(t2, *f12[0]))**2)) if f12 else float('inf')

        # Fit seg2, predict seg3
        f23 = safe_fit(func, t2, R2n, p0s, bounds)
        rss23 = float(np.sum((R3n - func(t3, *f23[0]))**2)) if f23 else float('inf')

        # Fit seg1, predict seg3 (skip seg2)
        f13 = safe_fit(func, t1, R1n, p0s, bounds)
        rss13 = float(np.sum((R3n - func(t3, *f13[0]))**2)) if f13 else float('inf')

        results[model_name] = {
            'rss_12': rss12,  # early→mid
            'rss_23': rss23,  # mid→late
            'rss_13': rss13,  # early→late (skip)
            'ratio_12_23': rss12 / max(rss23, 1e-12),  # if power-law, should be ~1
        }

    return results


def main():
    print('=' * 80)
    print('S2 vs POWER-LAW: SCALE DISCRIMINATOR')
    print('=' * 80)
    print()
    print('Key structural difference:')
    print('  S2:    R = exp[-(λ/λq)^D]  — has CHARACTERISTIC SCALE λq')
    print('  Power: R = A·λ^{-α}       — SCALE-FREE')
    print()
    print('Three tests:')
    print('  1. Log-log linearity (power-law is linear, S2 curves)')
    print('  2. Local slope variation (power-law constant, S2 varies)')
    print('  3. Scale-shifted OOS (power-law equal in all directions, S2 differs)')
    print()

    # Load real data
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

    real_loglog = []
    real_slopes = []
    real_shifted = []

    for entry in candidates[:20]:
        vals = refetch_values(entry.get('url', ''))
        if not vals or len(vals) < 100: continue
        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 30: continue
        t = np.array(taus, dtype=float)
        R = np.array(acf, dtype=float)
        t = t - t[0]
        if R[0] > 0: R = R / R[0]

        name = entry['name'][:40]

        ll = test_loglog_linearity(t, R)
        if ll:
            ll['name'] = name
            real_loglog.append(ll)

        ls = test_local_slope(t, R)
        if ls:
            ls['name'] = name
            real_slopes.append(ls)

        ss = test_scale_shifted_oos(t, R)
        if ss:
            ss['name'] = name
            real_shifted.append(ss)

    # ── Synthetic controls ──
    np.random.seed(42)
    n = 200
    t_syn = np.arange(n, dtype=float)

    synth_loglog = []
    synth_slopes = []
    synth_shifted = []

    generators = {
        'pure_s2': lambda: np.exp(-np.power(t_syn/10, 0.8)),
        'power_law': lambda: np.power(t_syn+1, -1.5) / np.power(1, -1.5),
        'biexp': lambda: 0.6*np.exp(-t_syn/5) + 0.4*np.exp(-t_syn/30),
    }

    for gen_name, gen_func in generators.items():
        for i in range(15):
            R = gen_func() + np.random.normal(0, 0.01, n)
            R = np.clip(R, 0.001, 2.0)
            if R[0] > 0: R = R / R[0]

            ll = test_loglog_linearity(t_syn, R)
            if ll:
                ll['type'] = gen_name
                synth_loglog.append(ll)

            ls = test_local_slope(t_syn, R)
            if ls:
                ls['type'] = gen_name
                synth_slopes.append(ls)

            ss = test_scale_shifted_oos(t_syn, R)
            if ss:
                ss['type'] = gen_name
                synth_shifted.append(ss)

    # ── Method 1: Log-log linearity ──
    print('=' * 80)
    print('METHOD 1: LOG-LOG LINEARITY')
    print('=' * 80)
    print()
    print('Power-law: ln R vs ln λ is LINEAR (R² should be high)')
    print('S2:        ln R vs ln λ CURVES (R² lower, quadratic improves)')
    print()

    print(f'{"Source":<15} {"R² linear":>10} {"R² quad":>10} {"Improvement":>12} {"Quad coeff":>11} {"Verdict":>10}')
    print('-' * 70)

    for label, results_list in [('REAL', real_loglog), ('S2 (synth)', [r for r in synth_loglog if r['type']=='pure_s2']),
                                 ('Power (synth)', [r for r in synth_loglog if r['type']=='power_law']),
                                 ('BIEXP (synth)', [r for r in synth_loglog if r['type']=='biexp'])]:
        if not results_list: continue
        r2l = np.mean([r['r2_linear'] for r in results_list])
        r2q = np.mean([r['r2_quad'] for r in results_list])
        imp = np.mean([r['r2_improvement'] for r in results_list])
        qc = np.mean([r['quad_coeff'] for r in results_list])
        verdict = 'S2-like' if imp > 0.005 else 'power-like'
        print(f'{label:<15} {r2l:>10.6f} {r2q:>10.6f} {imp:>12.6f} {qc:>11.4f} {verdict:>10}')

    print()
    real_improvements = [r['r2_improvement'] for r in real_loglog]
    s2_improvements = [r['r2_improvement'] for r in synth_loglog if r['type'] == 'pure_s2']
    pw_improvements = [r['r2_improvement'] for r in synth_loglog if r['type'] == 'power_law']

    print(f'Real data: mean R² improvement from quadratic = {np.mean(real_improvements):.6f}')
    print(f'True S2:   mean R² improvement from quadratic = {np.mean(s2_improvements):.6f}')
    print(f'True Power: mean R² improvement from quadratic = {np.mean(pw_improvements):.6f}')
    print()

    # ── Method 2: Local slope variation ──
    print('=' * 80)
    print('METHOD 2: LOCAL SLOPE VARIATION')
    print('=' * 80)
    print()
    print('Power-law: local slope = -α everywhere (CONSTANT)')
    print('S2:        local slope varies from 0 to -D (VARIES)')
    print()
    print('Metric: coefficient of variation (CV) of local slope')
    print('  Low CV → constant slope → power-law')
    print('  High CV → varying slope → S2 (characteristic scale)')
    print()

    print(f'{"Source":<15} {"Mean slope":>11} {"Std slope":>10} {"CV":>8} {"Range":>8} {"Verdict":>10}')
    print('-' * 65)

    for label, results_list in [('REAL', real_slopes), ('S2 (synth)', [r for r in synth_slopes if r['type']=='pure_s2']),
                                 ('Power (synth)', [r for r in synth_slopes if r['type']=='power_law']),
                                 ('BIEXP (synth)', [r for r in synth_slopes if r['type']=='biexp'])]:
        if not results_list: continue
        ms = np.mean([r['mean_slope'] for r in results_list])
        ss = np.mean([r['std_slope'] for r in results_list])
        cv = np.mean([r['cv'] for r in results_list])
        rng = np.mean([r['slope_range'] for r in results_list])
        verdict = 'S2-like' if cv > 0.3 else 'power-like'
        print(f'{label:<15} {ms:>11.4f} {ss:>10.4f} {cv:>8.3f} {rng:>8.4f} {verdict:>10}')

    print()
    real_cvs = [r['cv'] for r in real_slopes]
    s2_cvs = [r['cv'] for r in synth_slopes if r['type'] == 'pure_s2']
    pw_cvs = [r['cv'] for r in synth_slopes if r['type'] == 'power_law']

    print(f'Real data:   mean CV = {np.mean(real_cvs):.3f}, median = {np.median(real_cvs):.3f}')
    print(f'True S2:     mean CV = {np.mean(s2_cvs):.3f}, median = {np.median(s2_cvs):.3f}')
    print(f'True Power:  mean CV = {np.mean(pw_cvs):.3f}, median = {np.median(pw_cvs):.3f}')
    print()

    # ── Method 3: Scale-shifted OOS ──
    print('=' * 80)
    print('METHOD 3: SCALE-SHIFTED OOS')
    print('=' * 80)
    print()
    print('Power-law: same α everywhere → predicts equally in all directions')
    print('S2: cliff at λq → prediction quality DIFFERS across segments')
    print()

    print(f'{"Source":<15} {"S2→S2 1→2":>11} {"S2→S2 2→3":>11} {"Ratio":>8} {"PW→PW 1→2":>11} {"PW→PW 2→3":>11} {"Ratio":>8}')
    print('-' * 80)

    for label, results_list in [('REAL', real_shifted), ('S2 (synth)', [r for r in synth_shifted if r['type']=='pure_s2']),
                                 ('Power (synth)', [r for r in synth_shifted if r['type']=='power_law']),
                                 ('BIEXP (synth)', [r for r in synth_shifted if r['type']=='biexp'])]:
        if not results_list: continue
        s2_12 = np.mean([r['s2_dust']['rss_12'] for r in results_list])
        s2_23 = np.mean([r['s2_dust']['rss_23'] for r in results_list])
        s2_ratio = np.mean([r['s2_dust']['ratio_12_23'] for r in results_list])
        pw_12 = np.mean([r['power_2']['rss_12'] for r in results_list])
        pw_23 = np.mean([r['power_2']['rss_23'] for r in results_list])
        pw_ratio = np.mean([r['power_2']['ratio_12_23'] for r in results_list])
        print(f'{label:<15} {s2_12:>11.6f} {s2_23:>11.6f} {s2_ratio:>8.2f} {pw_12:>11.6f} {pw_23:>11.6f} {pw_ratio:>8.2f}')

    print()
    print('Ratio ≈ 1.0 → scale-free (power-law)')
    print('Ratio ≠ 1.0 → characteristic scale exists (S2)')
    print()

    # ── VERDICT ──
    print('=' * 80)
    print('VERDICT')
    print('=' * 80)
    print()

    # Method 1 verdict
    real_imp = np.mean(real_improvements)
    pw_imp = np.mean(pw_improvements)
    s2_imp = np.mean(s2_improvements)
    print(f'Method 1 (log-log linearity):')
    print(f'  Real R² improvement from quadratic: {real_imp:.6f}')
    print(f'  True S2 improvement:  {s2_imp:.6f}')
    print(f'  True Power improvement: {pw_imp:.6f}')
    m1_s2 = real_imp > (s2_imp + pw_imp) / 2
    print(f'  → {"Curvature detected — favors S2" if m1_s2 else "Minimal curvature — favors power-law"}')
    print()

    # Method 2 verdict
    real_cv = np.median(real_cvs)
    s2_cv = np.median(s2_cvs)
    pw_cv = np.median(pw_cvs)
    print(f'Method 2 (local slope variation):')
    print(f'  Real median CV: {real_cv:.3f}')
    print(f'  True S2 median CV: {s2_cv:.3f}')
    print(f'  True Power median CV: {pw_cv:.3f}')
    m2_s2 = real_cv > (s2_cv + pw_cv) / 2
    print(f'  → {"Slope varies — favors S2" if m2_s2 else "Slope constant — favors power-law"}')
    print()

    # Method 3 verdict
    real_s2_ratio = np.mean([r['s2_dust']['ratio_12_23'] for r in real_shifted])
    real_pw_ratio = np.mean([r['power_2']['ratio_12_23'] for r in real_shifted])
    s2_s2_ratio = np.mean([r['s2_dust']['ratio_12_23'] for r in synth_shifted if r['type']=='pure_s2'])
    pw_pw_ratio = np.mean([r['power_2']['ratio_12_23'] for r in synth_shifted if r['type']=='power_law'])
    print(f'Method 3 (scale-shifted OOS):')
    print(f'  Real S2_DUST ratio (1→2 / 2→3): {real_s2_ratio:.2f}')
    print(f'  Real Power_2 ratio:              {real_pw_ratio:.2f}')
    print(f'  True S2 S2_DUST ratio:           {s2_s2_ratio:.2f}')
    print(f'  True Power Power_2 ratio:        {pw_pw_ratio:.2f}')
    m3_s2 = abs(real_s2_ratio - 1) > abs(pw_pw_ratio - 1) * 1.5
    print(f'  → {"Scale break detected — favors S2" if m3_s2 else "No scale break — favors power-law"}')
    print()

    # Overall
    s2_votes = sum([m1_s2, m2_s2, m3_s2])
    print(f'Overall: {s2_votes}/3 methods favor S2 over power-law')
    if s2_votes >= 2:
        print('→ S2 ANCESTRY SUPPORTED OVER POWER-LAW')
    elif s2_votes == 1:
        print('→ INCONCLUSIVE — mixed evidence')
    else:
        print('→ POWER-LAW ANCESTRY FAVORED (S2 not uniquely supported)')


if __name__ == '__main__':
    main()
