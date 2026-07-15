#!/usr/bin/env python3
"""
S2 Model Comparison Layer
=========================
Fits the S2 retention law R(t) = A·exp[-(t/λq)^D] against 5 alternative
decay models on the SAME retention curve, then ranks them by AICc
(Akaike Information Criterion with finite-sample correction).

Models (k = parameter count):
  1. S2 stretched exponential   A, λq, D          k=3   (our hypothesis)
  2. Pure exponential           A, λ              k=2   (special case D=1)
  3. Biexponential              A1, λ1, A2, λ2    k=4   (two-pool decay)
  4. Power law                  A, α              k=2   (heavy tail)
  5. Lognormal                  A, μ, σ           k=3   (peak-then-decay)
  6. Gaussian                   A, σ              k=2   (special case D=2)

Decision rule (Burnham & Anderson 2002):
  ΔAICc = AICc_S2 − AICc_best_alternative
  ΔAICc ≤ −2  →  S2_WINS    (strong evidence, add to registry)
  −2 < Δ ≤ 2  →  S2_TIES    (weak evidence, add with caveat)
  Δ > 2       →  S2_LOSES   (do NOT add — flag inconclusive)

This is the local "S2 vs other models" check the auto-scanner runs
before promoting any candidate fit into tests.html.
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ── Models ───────────────────────────────────────────────────────────

def m_s2(t, A, lambda_q, D):
    return A * np.exp(-np.power(np.maximum(t, 1e-6) / max(lambda_q, 1e-6), D))

def m_exp(t, A, lam):
    return A * np.exp(-t / max(lam, 1e-6))

def m_biexp(t, A1, l1, A2, l2):
    return A1 * np.exp(-t / max(l1, 1e-6)) + A2 * np.exp(-t / max(l2, 1e-6))

def m_power(t, A, alpha):
    return A * np.power(np.maximum(t, 1e-6), -alpha)

def m_lognormal(t, A, mu, sigma):
    sigma = max(sigma, 1e-3)
    return A / (np.maximum(t, 1e-6) * sigma * np.sqrt(2 * np.pi)) * \
           np.exp(-0.5 * ((np.log(np.maximum(t, 1e-6)) - mu) / sigma) ** 2)

def m_gaussian(t, A, sigma):
    return A * np.exp(-0.5 * (t / max(sigma, 1e-6)) ** 2)

# ── AICc ─────────────────────────────────────────────────────────────

def aicc(rss, n, k):
    """Akaike Information Criterion with finite-sample correction."""
    if n - k - 1 <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)

def bic(rss, n, k):
    """Bayesian Information Criterion (bonus check)."""
    return n * np.log(rss / n) + k * np.log(max(n, 2))

# ── Fit one model with multiple restarts ────────────────────────────

def _safe_fit(func, t, R, p0_list, bounds=None, maxfev=20000):
    """Try several initial guesses, return best (popt, rss) or None."""
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
        except Exception:
            continue
    return best

def fit_all_models(t, R):
    """
    Fit every candidate model to (t, R).
    Returns dict:  {model_name: {popt, rss, k, aicc, bic, r2}}
    """
    t = np.asarray(t, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(t)
    t_mid = float(t[len(t) // 2]) if len(t) else 1.0
    t_mid = max(t_mid, 1e-3)
    ss_tot = float(np.sum((R - np.mean(R)) ** 2))
    if ss_tot == 0:
        return None

    results = {}

    # 1. S2
    fit = _safe_fit(m_s2, t, R,
                    p0_list=[[1.0, t_mid, 0.5], [1.0, t_mid * 0.5, 0.8], [1.0, t_mid * 2, 0.3]],
                    bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]))
    if fit:
        popt, rss = fit
        k = 3
        results['S2'] = {'popt': popt, 'rss': rss, 'k': k,
                         'aicc': aicc(rss, n, k), 'bic': bic(rss, n, k),
                         'r2': 1 - rss / ss_tot, 'D': float(popt[2]),
                         'lambda_q': float(popt[1])}

    # 2. Pure exponential
    fit = _safe_fit(m_exp, t, R,
                    p0_list=[[1.0, t_mid], [1.0, t_mid * 0.5], [1.0, t_mid * 2]],
                    bounds=([0.01, 1e-3], [2.0, 1e6]))
    if fit:
        popt, rss = fit
        k = 2
        results['EXP'] = {'popt': popt, 'rss': rss, 'k': k,
                          'aicc': aicc(rss, n, k), 'bic': bic(rss, n, k),
                          'r2': 1 - rss / ss_tot}

    # 3. Biexponential
    fit = _safe_fit(m_biexp, t, R,
                    p0_list=[[0.7, t_mid * 0.3, 0.3, t_mid * 2],
                             [0.5, t_mid * 0.5, 0.5, t_mid],
                             [0.6, t_mid, 0.4, t_mid * 3]],
                    bounds=([0.0, 1e-3, 0.0, 1e-3], [2.0, 1e6, 2.0, 1e6]))
    if fit:
        popt, rss = fit
        k = 4
        results['BIEXP'] = {'popt': popt, 'rss': rss, 'k': k,
                            'aicc': aicc(rss, n, k), 'bic': bic(rss, n, k),
                            'r2': 1 - rss / ss_tot}

    # 4. Power law
    fit = _safe_fit(m_power, t, R,
                    p0_list=[[1.0, 0.5], [1.0, 1.0], [1.0, 2.0]],
                    bounds=([0.01, 0.01], [2.0, 10.0]))
    if fit:
        popt, rss = fit
        k = 2
        results['POWER'] = {'popt': popt, 'rss': rss, 'k': k,
                            'aicc': aicc(rss, n, k), 'bic': bic(rss, n, k),
                            'r2': 1 - rss / ss_tot}

    # 5. Lognormal
    fit = _safe_fit(m_lognormal, t, R,
                    p0_list=[[1.0, np.log(t_mid), 0.5],
                             [1.0, np.log(t_mid * 0.5), 1.0],
                             [1.0, np.log(max(t_mid * 2, 1.0)), 0.8]],
                    bounds=([0.01, -10, 0.01], [10.0, 10, 10]))
    if fit:
        popt, rss = fit
        k = 3
        results['LOGNORM'] = {'popt': popt, 'rss': rss, 'k': k,
                              'aicc': aicc(rss, n, k), 'bic': bic(rss, n, k),
                              'r2': 1 - rss / ss_tot}

    # 6. Gaussian
    fit = _safe_fit(m_gaussian, t, R,
                    p0_list=[[1.0, t_mid], [1.0, t_mid * 0.5], [1.0, t_mid * 2]],
                    bounds=([0.01, 1e-3], [2.0, 1e6]))
    if fit:
        popt, rss = fit
        k = 2
        results['GAUSS'] = {'popt': popt, 'rss': rss, 'k': k,
                            'aicc': aicc(rss, n, k), 'bic': bic(rss, n, k),
                            'r2': 1 - rss / ss_tot}

    return results

# ── Verdict ──────────────────────────────────────────────────────────

def compare(t, R, label=''):
    """
    Run full model comparison.
    Returns:
      {
        'verdict': 'S2_WINS' | 'S2_TIES' | 'S2_LOSES',
        'best_model': str,           # name of lowest-AICc model
        's2_aicc': float,
        'best_alt_aicc': float,
        'delta_aicc': float,         # s2_aicc - best_alt_aicc  (≤−2 = win)
        'rank': [(model, aicc), ...] sorted ascending,
        's2': {D, lambda_q, r2} | None,
        'label': str,
      }
    """
    fits = fit_all_models(t, R)
    if not fits or 'S2' not in fits:
        return {'verdict': 'NO_FIT', 'best_model': None, 'label': label,
                'rank': [], 's2': None, 'delta_aicc': None}

    ranked = sorted(fits.items(), key=lambda kv: kv[1]['aicc'])
    s2_aicc = fits['S2']['aicc']

    best_alt_name, best_alt = None, None
    for name, f in ranked:
        if name != 'S2':
            best_alt_name, best_alt = name, f
            break

    if best_alt is None:
        delta = -np.inf
        verdict = 'S2_WINS'
    else:
        delta = s2_aicc - best_alt['aicc']
        if delta <= -2:
            verdict = 'S2_WINS'
        elif delta <= 2:
            verdict = 'S2_TIES'
        else:
            verdict = 'S2_LOSES'

    return {
        'verdict': verdict,
        'best_model': ranked[0][0],
        's2_aicc': round(float(s2_aicc), 2),
        'best_alt_aicc': round(float(best_alt['aicc']), 2) if best_alt else None,
        'best_alt_name': best_alt_name,
        'delta_aicc': round(float(delta), 2) if delta != -np.inf else -999,
        'rank': [(name, round(f['aicc'], 2), round(f['r2'], 4), f['k'])
                 for name, f in ranked],
        's2': {'D': round(fits['S2']['D'], 4),
               'lambda_q': round(fits['S2']['lambda_q'], 2),
               'r2': round(fits['S2']['r2'], 4)},
        'label': label,
        'n': len(t),
    }

# ── CLI / self-test ─────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 70)
    print('S2 MODEL COMPARISON — self-test on synthetic curves')
    print('=' * 70)

    np.random.seed(42)
    t = np.arange(1, 50, dtype=float)

    cases = [
        ('True S2 (D=1.5, extraction)',   m_s2(t, 1.0, 10.0, 1.5) + 0.005 * np.random.randn(len(t))),
        ('True S2 (D=0.6, natural)',      m_s2(t, 1.0, 10.0, 0.6) + 0.005 * np.random.randn(len(t))),
        ('Pure exponential',              m_exp(t, 1.0, 8.0) + 0.005 * np.random.randn(len(t))),
        ('Power law',                     m_power(t, 1.0, 1.2) + 0.005 * np.random.randn(len(t))),
        ('Gaussian decay',                m_gaussian(t, 1.0, 6.0) + 0.005 * np.random.randn(len(t))),
    ]

    for label, R in cases:
        R = np.clip(R, 1e-4, 2.0)
        res = compare(t, R, label)
        print(f'\n── {label} ──')
        print(f'  Verdict: {res["verdict"]}  (best overall: {res["best_model"]})')
        print(f'  S2: D={res["s2"]["D"]}, λq={res["s2"]["lambda_q"]}, R²={res["s2"]["r2"]}')
        print(f'  ΔAICc vs best alt: {res["delta_aicc"]}  '
              f'(S2={res["s2_aicc"]}, {res["best_alt_name"]}={res["best_alt_aicc"]})')
        print(f'  Ranking:')
        for name, aicc_val, r2, k in res['rank']:
            mark = ' ★' if name == 'S2' else ''
            print(f'    {name:10s}  AICc={aicc_val:8.2f}  R²={r2:.4f}  k={k}{mark}')
