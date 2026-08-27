#!/usr/bin/env python3
"""
T7.1 POSITIVE CONTROL — sequential with SIGALRM (no multiprocessing).
Smaller N (60) but should actually complete.
"""
import json, os, sys, signal
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_model_compare import m_s2


class TimeoutError_(Exception): pass
def _timeout_handler(signum, frame):
    raise TimeoutError_('fit timeout')


def piecewise_s2(t, A1, lam1, D1, A2, lam2, D2, t_break):
    mask = t < t_break
    out = np.zeros_like(t, dtype=float)
    out[mask] = A1 * np.exp(-np.power(np.maximum(t[mask], 1e-6) / max(lam1, 1e-6), D1))
    out[~mask] = A2 * np.exp(-np.power(np.maximum(t[~mask], 1e-6) / max(lam2, 1e-6), D2))
    return out


def aicc(rss, n, k):
    if n - k - 1 <= 0: return float('inf')
    return n * np.log(rss/n) + 2*k + (2*k*(k+1))/(n-k-1)


def fit_s2_one(t, R, p0_list, bounds=None, maxfev=5000):
    best = None
    for p0 in p0_list:
        try:
            if bounds:
                popt, _ = curve_fit(m_s2, t, R, p0=p0, bounds=bounds, maxfev=maxfev)
            else:
                popt, _ = curve_fit(m_s2, t, R, p0=p0, maxfev=maxfev)
            rss = float(np.sum((R - m_s2(t, *popt))**2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except: continue
    return best


def fit_single_s2(t, R):
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    tm = float(t[len(t)//2])
    f = fit_s2_one(t, R_n,
        p0_list=[[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]],
        bounds=([0.01, 1e-2, 0.01], [10.0, 1e6, 10.0]))
    if not f: return None
    return {'params': list(f[0]), 'rss': f[1], 'aicc': aicc(f[1], len(t), 3)}


def fit_piecewise_s2_sig(t, R, n_break_candidates=5, per_fit_timeout=2.0):
    """Piecewise S2 fit with SIGALRM timeout."""
    n = len(t)
    if n < 30: return None
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    R_n = np.clip(R_n, 1e-6, None)
    t_min, t_max = t[0], t[-1]
    t_breaks = np.linspace(t_min + 0.25*(t_max-t_min), t_min + 0.75*(t_max-t_min), n_break_candidates)
    best = None
    bounds_lower = [0.001, 1e-2, 0.01, 0.001, 1e-2, 0.01]
    bounds_upper = [10.0, 1e6, 10.0, 10.0, 1e6, 10.0]
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    for t_b in t_breaks:
        mask = t < t_b
        if mask.sum() < 5 or (~mask).sum() < 5: continue
        tm1 = float(t[mask][len(t[mask])//2])
        tm2 = float(t[~mask][len(t[~mask])//2])
        R1 = R_n[mask]; R2 = R_n[~mask]
        if R1[0] > 0: R1 = R1 / R1[0]
        if R2[0] > 0: R2 = R2 / R2[0]
        f1 = fit_s2_one(t[mask], R1,
            p0_list=[[1.0, tm1, 0.5], [1.0, tm1*0.5, 1.0]],
            bounds=([0.01, 1e-2, 0.01], [10.0, 1e6, 10.0]))
        f2 = fit_s2_one(t[~mask], R2,
            p0_list=[[1.0, tm2, 0.5], [1.0, tm2*0.5, 1.0]],
            bounds=([0.01, 1e-2, 0.01], [10.0, 1e6, 10.0]))
        if not f1 or not f2: continue
        A1_0, lam1_0, D1_0 = f1[0]
        A2_0, lam2_0, D2_0 = f2[0]
        val1 = A1_0 * np.exp(-np.power(t_b / max(lam1_0, 1e-6), D1_0))
        val2 = A2_0 * np.exp(-np.power(t_b / max(lam2_0, 1e-6), D2_0))
        A2_init = A2_0 * (val1 / val2) if val2 > 0 else A2_0
        p0 = [max(bounds_lower[i], min(bounds_upper[i]-1e-9, v))
              for i, v in enumerate([A1_0, lam1_0, D1_0, A2_init, lam2_0, D2_0])]
        signal.setitimer(signal.ITIMER_REAL, per_fit_timeout)
        try:
            popt, _ = curve_fit(
                lambda tt, A1, lam1, D1, A2, lam2, D2: piecewise_s2(tt, A1, lam1, D1, A2, lam2, D2, t_b),
                t, R_n, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=800
            )
            signal.setitimer(signal.ITIMER_REAL, 0)
            rss = float(np.sum((R_n - piecewise_s2(t, *popt, t_break=t_b))**2))
            a = aicc(rss, n, 6)
            if best is None or a < best[1]:
                best = (list(popt), a, float(t_b), rss)
        except TimeoutError_:
            continue
        except Exception:
            signal.setitimer(signal.ITIMER_REAL, 0)
            continue
    signal.signal(signal.SIGALRM, old_handler)
    return best


# ─────────────────────────────────────────────────────────────────────
# Synthetic generators
# ─────────────────────────────────────────────────────────────────────

def null_single_s2(n, t_min, t_max, seed):
    rng = np.random.RandomState(seed)
    t = np.linspace(t_min, t_max, n)
    D = rng.uniform(0.1, 5.0)
    lam_q = rng.uniform(t_max * 0.1, t_max * 0.9)
    R = np.exp(-np.power(t / lam_q, D))
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 1e-6, None)
    return t, R / R[0]


def null_two_regime_s2(n, t_min, t_max, seed, t_break_frac=0.3):
    rng = np.random.RandomState(seed)
    t = np.linspace(t_min, t_max, n)
    t_break = t_min + t_break_frac * (t_max - t_min)
    D1 = rng.uniform(0.3, 1.2)
    D2 = rng.uniform(1.5, 4.0)
    lam1 = rng.uniform(t_break * 0.5, t_break * 1.5)
    lam2 = rng.uniform(t_break * 0.5, t_break * 1.5)
    mask = t < t_break
    R = np.zeros(n)
    R[mask] = np.exp(-np.power(t[mask] / lam1, D1))
    R[~mask] = np.exp(-np.power(t[~mask] / lam2, D2))
    val1 = np.exp(-np.power(t_break / lam1, D1))
    val2 = np.exp(-np.power(t_break / lam2, D2))
    R[~mask] = R[~mask] * val1 / val2
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 1e-6, None)
    return t, R / R[0], t_break_frac, D1, D2


# ─────────────────────────────────────────────────────────────────────
# Main — sequential
# ─────────────────────────────────────────────────────────────────────

print('='*72)
print('T7.1 POSITIVE CONTROL + NULL 1 — SEQUENTIAL')
print('='*72)
print()

N_NULL = 60
N_PER = 80
T_MIN = 0.0
T_MAX = 79.0

# NULL 1
print(f'NULL 1: {N_NULL} pure single-S2 curves...')
null1_results = []
for seed in range(N_NULL):
    try:
        t, R = null_single_s2(N_PER, T_MIN, T_MAX, seed)
        single = fit_single_s2(t, R)
        pw = fit_piecewise_s2_sig(t, R, n_break_candidates=5, per_fit_timeout=2.0)
        if single and pw:
            null1_results.append(((pw[2] - T_MIN) / (T_MAX - T_MIN), single['aicc'] - pw[1]))
    except Exception:
        pass
    if (seed + 1) % 20 == 0:
        print(f'  {seed+1}/{N_NULL} done (success: {len(null1_results)})', flush=True)

null1_fracs = np.array([r[0] for r in null1_results])
null1_deltas = np.array([r[1] for r in null1_results])
print(f'\nNull 1: n={len(null1_fracs)}  mean={null1_fracs.mean():.3f}  std={null1_fracs.std():.3f}')
print(f'  Median ΔAICc: {np.median(null1_deltas):.2f}  % > 4: {100*np.mean(null1_deltas > 4):.1f}%')

# NULL 2 (positive control)
print()
print(f'NULL 2 (positive control, true t_break=0.3): {N_NULL} 2-regime S2 curves...')
null2_results = []
for seed in range(N_NULL):
    try:
        t, R, true_frac, true_D1, true_D2 = null_two_regime_s2(N_PER, T_MIN, T_MAX, seed, t_break_frac=0.3)
        single = fit_single_s2(t, R)
        pw = fit_piecewise_s2_sig(t, R, n_break_candidates=5, per_fit_timeout=2.0)
        if single and pw:
            null2_results.append(((pw[2] - T_MIN) / (T_MAX - T_MIN), single['aicc'] - pw[1],
                                  pw[0][2], pw[0][5], true_frac, true_D1, true_D2))
    except Exception:
        pass
    if (seed + 1) % 20 == 0:
        print(f'  {seed+1}/{N_NULL} done (success: {len(null2_results)})', flush=True)

null2_fracs = np.array([r[0] for r in null2_results])
null2_deltas = np.array([r[1] for r in null2_results])
null2_d1s = np.array([r[2] for r in null2_results])
null2_d2s = np.array([r[3] for r in null2_results])

print(f'\nNull 2: n={len(null2_fracs)}')
print(f'  t_break fractions (true=0.3): mean={null2_fracs.mean():.3f}  median={np.median(null2_fracs):.3f}  std={null2_fracs.std():.3f}')
print(f'  Recovery error: median={np.median(np.abs(null2_fracs - 0.3)):.3f}  % within 0.1: {100*np.mean(np.abs(null2_fracs - 0.3) < 0.1):.0f}%')
print(f'  ΔAICc: median={np.median(null2_deltas):.2f}  % > 4: {100*np.mean(null2_deltas > 4):.1f}%')
print(f'  D1 fit: mean={null2_d1s.mean():.3f}')
print(f'  D2 fit: mean={null2_d2s.mean():.3f}')

# Save
out = {
    'null_test': 'T7.1 positive control + null 1 (sequential)',
    'N_NULL': N_NULL,
    'null1_single_s2': {
        'n': int(len(null1_fracs)),
        't_break_frac_mean': float(null1_fracs.mean()) if len(null1_fracs) else None,
        't_break_frac_median': float(np.median(null1_fracs)) if len(null1_fracs) else None,
        't_break_frac_std': float(null1_fracs.std()) if len(null1_fracs) else None,
        'delta_aicc_median': float(np.median(null1_deltas)) if len(null1_deltas) else None,
        'pct_delta_aicc_gt_4': float(100 * np.mean(null1_deltas > 4)) if len(null1_deltas) else None,
    },
    'null2_positive_control': {
        'n': int(len(null2_fracs)),
        'true_t_break_frac': 0.3,
        't_break_frac_mean': float(null2_fracs.mean()) if len(null2_fracs) else None,
        't_break_frac_median': float(np.median(null2_fracs)) if len(null2_fracs) else None,
        't_break_frac_std': float(null2_fracs.std()) if len(null2_fracs) else None,
        'recovery_error_median': float(np.median(np.abs(null2_fracs - 0.3))) if len(null2_fracs) else None,
        'pct_within_0.1_of_true': float(100 * np.mean(np.abs(null2_fracs - 0.3) < 0.1)) if len(null2_fracs) else None,
        'delta_aicc_median': float(np.median(null2_deltas)) if len(null2_deltas) else None,
        'pct_delta_aicc_gt_4': float(100 * np.mean(null2_deltas > 4)) if len(null2_deltas) else None,
        'D1_fit_mean': float(null2_d1s.mean()) if len(null2_d1s) else None,
        'D2_fit_mean': float(null2_d2s.mean()) if len(null2_d2s) else None,
    },
}
out_path = '/home/z/my-project/download/t71_null_complete.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {out_path}')
