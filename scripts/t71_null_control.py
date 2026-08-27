#!/usr/bin/env python3
"""
T7.1 NULL-MODEL CONTROL
========================

The critical test the previous run was missing: are the breakpoint
fractional positions clustered because of genuine multi-segment S2
structure, OR because any piecewise fit on a curve of similar length
will naturally place the breakpoint in the middle of the t-range?

Test:
  1. Generate N=200 synthetic single-S2 curves with the SAME n, t-range,
     and noise level as the real datasets.
  2. Run the SAME piecewise S2 fit on each synthetic curve.
  3. Collect the fitted t_break fractions.
  4. Compare to the real-data t_break fractions:
     - If real clusters significantly tighter than null → real signal
     - If real matches null → geometric artifact

Also: a "two-regime mixture" positive control — generate synthetic
2-S2 piecewise data and confirm the pipeline recovers it.
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


def fit_piecewise_s2(t, R, n_break_candidates=5, per_fit_timeout=2.0):
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
                t, R_n, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=1000
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
# NULL 1: Single-S2 synthetic through same pipeline
# ─────────────────────────────────────────────────────────────────────

def null_single_s2(n, t_min, t_max, seed):
    """Generate a synthetic single-S2 curve. Random (D, lambda_q) within
    typical ranges from the real registry."""
    rng = np.random.RandomState(seed)
    t = np.linspace(t_min, t_max, n)
    # Pick D in [0.1, 5.0], lambda_q in [t_max*0.1, t_max*0.9]
    D = rng.uniform(0.1, 5.0)
    lam_q = rng.uniform(t_max * 0.1, t_max * 0.9)
    R = np.exp(-np.power(t / lam_q, D))
    # Add noise similar to real data (1% std)
    noise_level = 0.01
    R += rng.normal(0, noise_level, n)
    R = np.clip(R, 1e-6, None)
    return t, R / R[0], D, lam_q


def null_two_regime_s2(n, t_min, t_max, seed, t_break_frac=0.3):
    """Generate a synthetic 2-regime piecewise S2 curve."""
    rng = np.random.RandomState(seed)
    t = np.linspace(t_min, t_max, n)
    t_break = t_min + t_break_frac * (t_max - t_min)
    D1 = rng.uniform(0.2, 1.0)
    D2 = rng.uniform(1.5, 4.0)
    lam1 = rng.uniform(t_break * 0.5, t_break * 1.5)
    lam2 = rng.uniform(t_break * 0.5, t_break * 1.5)
    mask = t < t_break
    R = np.zeros(n)
    R[mask] = np.exp(-np.power(t[mask] / lam1, D1))
    R[~mask] = np.exp(-np.power(t[~mask] / lam2, D2))
    # Continuity at t_break
    val1 = np.exp(-np.power(t_break / lam1, D1))
    val2 = np.exp(-np.power(t_break / lam2, D2))
    R[~mask] = R[~mask] * val1 / val2
    R += rng.normal(0, 0.01, n)
    R = np.clip(R, 1e-6, None)
    return t, R / R[0], (D1, D2, t_break, lam1, lam2)


# ─────────────────────────────────────────────────────────────────────
# Run null tests
# ─────────────────────────────────────────────────────────────────────

print('='*72)
print('T7.1 NULL-MODEL CONTROL')
print('='*72)
print()
print('Q: Is t_break clustering a real signal, or a geometric artifact?')
print()
print('NULL HYPOTHESIS: piecewise fit on curves of similar length will')
print('naturally place breakpoints at the middle of the t-range, regardless')
print('of whether the underlying data is single-S2 or 2-regime S2.')
print()

# Match real data parameters
# Real curves had n~50-80, t_max 49-79
N_NULL = 100  # reduced from 200 to avoid hangs
N_PER = 80  # match Open-Meteo (most common real dataset)
T_MIN = 0.0
T_MAX = 79.0

print(f'Generating {N_NULL} null curves per condition (n={N_PER}, t=[{T_MIN}, {T_MAX}])')
print()

# NULL 1: pure single-S2 (no regime change)
print('='*72)
print('NULL 1: Pure single-S2 through same piecewise pipeline')
print('='*72)
import multiprocessing as mp

def run_one_null1(seed):
    """Run null 1 - returns (frac, delta) or None."""
    try:
        # Disable signal-based timeout inside subprocess (won't work)
        # Use try/except instead with maxfev cap
        t, R, true_D, true_lam = null_single_s2(N_PER, T_MIN, T_MAX, seed)
        single = fit_single_s2(t, R)
        # Run piecewise without SIGALRM (won't work in subprocess)
        pw = fit_piecewise_s2_nosig(t, R, n_break_candidates=5)
        if single and pw:
            delta = single['aicc'] - pw[1]
            frac = (pw[2] - T_MIN) / (T_MAX - T_MIN)
            return (frac, delta)
    except Exception:
        pass
    return None

def fit_piecewise_s2_nosig(t, R, n_break_candidates=5):
    """Piecewise S2 fit without SIGALRM (for use in subprocesses)."""
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
        try:
            popt, _ = curve_fit(
                lambda tt, A1, lam1, D1, A2, lam2, D2: piecewise_s2(tt, A1, lam1, D1, A2, lam2, D2, t_b),
                t, R_n, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=500
            )
            rss = float(np.sum((R_n - piecewise_s2(t, *popt, t_break=t_b))**2))
            a = aicc(rss, n, 6)
            if best is None or a < best[1]:
                best = (list(popt), a, float(t_b), rss)
        except Exception:
            continue
    return best

# Spawn pool with hard per-task timeout
ctx = mp.get_context('fork')
pool = ctx.Pool(processes=4)
null1_results = []
for seed in range(N_NULL):
    res = pool.apply_async(run_one_null1, (seed,))
    try:
        r = res.get(timeout=15)
        if r is not None:
            null1_results.append(r)
    except mp.TimeoutError:
        # Kill the slow worker and respawn
        pool.terminate()
        pool = ctx.Pool(processes=4)
    except Exception:
        pass
    if (seed + 1) % 25 == 0:
        print(f'  {seed+1}/{N_NULL} done (success: {len(null1_results)})', flush=True)
pool.close()
pool.join()

null1_t_break_fracs = [r[0] for r in null1_results]
null1_delta_aiccs = [r[1] for r in null1_results]

null1_fracs = np.array(null1_t_break_fracs)
null1_deltas = np.array(null1_delta_aiccs)
print(f'\nNull 1 (pure single-S2) results:')
print(f'  Successful piecewise fits: {len(null1_fracs)}/{N_NULL}')
if len(null1_fracs) >= 10:
    print(f'  t_break fractions:')
    print(f'    n = {len(null1_fracs)}')
    print(f'    mean = {null1_fracs.mean():.3f}  median = {np.median(null1_fracs):.3f}  std = {null1_fracs.std():.3f}')
    print(f'    IQR  = [{np.percentile(null1_fracs, 25):.3f}, {np.percentile(null1_fracs, 75):.3f}]')
    print(f'  ΔAICc (single - piecewise):')
    print(f'    mean = {null1_deltas.mean():.2f}  median = {np.median(null1_deltas):.2f}')
    print(f'    % with ΔAICc > 4: {100 * np.mean(null1_deltas > 4):.1f}%')
    print(f'    (Note: piecewise has 6 params, single has 3, so it has an')
    print(f'     inherent advantage of ~6 AICc per fit by overfitting.)')

# NULL 2: Positive control — true 2-regime S2
print()
print('='*72)
print('NULL 2: POSITIVE CONTROL — true 2-regime S2 through same pipeline')
print('='*72)

def run_one_null2(seed):
    try:
        t, R, true_params = null_two_regime_s2(N_PER, T_MIN, T_MAX, seed, t_break_frac=0.3)
        single = fit_single_s2(t, R)
        pw = fit_piecewise_s2_nosig(t, R, n_break_candidates=5)
        if single and pw:
            delta = single['aicc'] - pw[1]
            frac = (pw[2] - T_MIN) / (T_MAX - T_MIN)
            return (frac, delta, pw[0][2], pw[0][5])
    except Exception:
        pass
    return None

ctx2 = mp.get_context('fork')
pool2 = ctx2.Pool(processes=4)
null2_results = []
for seed in range(N_NULL):
    res = pool2.apply_async(run_one_null2, (seed,))
    try:
        r = res.get(timeout=15)
        if r is not None:
            null2_results.append(r)
    except mp.TimeoutError:
        pool2.terminate()
        pool2 = ctx2.Pool(processes=4)
    except Exception:
        pass
    if (seed + 1) % 25 == 0:
        print(f'  {seed+1}/{N_NULL} done (success: {len(null2_results)})', flush=True)
pool2.close()
pool2.join()

null2_t_break_fracs = [r[0] for r in null2_results]
null2_delta_aiccs = [r[1] for r in null2_results]
null2_d1s = [r[2] for r in null2_results]
null2_d2s = [r[3] for r in null2_results]

null2_fracs = np.array(null2_t_break_fracs)
null2_deltas = np.array(null2_delta_aiccs)
print(f'\nNull 2 (positive control) results:')
print(f'  Successful piecewise fits: {len(null2_fracs)}/{N_NULL}')
if len(null2_fracs) >= 10:
    print(f'  t_break fractions (true = 0.3):')
    print(f'    n = {len(null2_fracs)}')
    print(f'    mean = {null2_fracs.mean():.3f}  median = {np.median(null2_fracs):.3f}  std = {null2_fracs.std():.3f}')
    print(f'    IQR  = [{np.percentile(null2_fracs, 25):.3f}, {np.percentile(null2_fracs, 75):.3f}]')
    print(f'  ΔAICc (single - piecewise):')
    print(f'    mean = {null2_deltas.mean():.2f}  median = {np.median(null2_deltas):.2f}')
    print(f'    % with ΔAICc > 4: {100 * np.mean(null2_deltas > 4):.1f}%')

# Compare to real data
print()
print('='*72)
print('NULL-MODEL vs REAL DATA COMPARISON')
print('='*72)

# Load real data results
real_path = '/home/z/my-project/download/t71_final_results.json'
with open(real_path) as f:
    real_data = json.load(f)

real_b_strong = [r for r in real_data['results'] if r['t71b_delta_aicc'] is not None and r['t71b_delta_aicc'] > 4]
real_fracs = np.array([(r['t71b_t_break'] - r['t_range'][0]) / max(r['t_range'][1]-r['t_range'][0], 1e-10)
                       for r in real_b_strong])
real_deltas = np.array([r['t71b_delta_aicc'] for r in real_b_strong])

print(f'\nReal data (n={len(real_fracs)}):')
print(f'  t_break fractions: mean={real_fracs.mean():.3f}  median={np.median(real_fracs):.3f}  std={real_fracs.std():.3f}')
print(f'  ΔAICc: median={np.median(real_deltas):.1f}  (% with ΔAICc>4: {100*np.mean(real_deltas>4):.0f}%)')

if len(null1_fracs) >= 10:
    print(f'\nNull 1 — single-S2 control (n={len(null1_fracs)}):')
    print(f'  t_break fractions: mean={null1_fracs.mean():.3f}  median={np.median(null1_fracs):.3f}  std={null1_fracs.std():.3f}')
    print(f'  ΔAICc: median={np.median(null1_deltas):.2f}  (% with ΔAICc>4: {100*np.mean(null1_deltas>4):.1f}%)')

if len(null2_fracs) >= 10:
    print(f'\nNull 2 — positive control (true 2-regime, n={len(null2_fracs)}):')
    print(f'  t_break fractions: mean={null2_fracs.mean():.3f}  median={np.median(null2_fracs):.3f}  std={null2_fracs.std():.3f}')
    print(f'  ΔAICc: median={np.median(null2_deltas):.2f}  (% with ΔAICc>4: {100*np.mean(null2_deltas>4):.1f}%)')

# Statistical tests
print()
print('='*72)
print('STATISTICAL TESTS')
print('='*72)

if len(null1_fracs) >= 10 and len(real_fracs) >= 3:
    # KS test: does real distribution differ from null 1?
    ks, p_ks = stats.ks_2samp(real_fracs, null1_fracs)
    u, p_u = stats.mannwhitneyu(real_fracs, null1_fracs, alternative='two-sided')
    print(f'\nReal vs Null 1 (single-S2):')
    print(f'  KS test:    D={ks:.3f}  p={p_ks:.4g}')
    print(f'  Mann-Whitney: p={p_u:.4g}')
    if p_ks < 0.05:
        print(f'  -> Real t_break distribution DIFFERS from single-S2 null')
        print(f'     (T7.1 signal is NOT a geometric artifact)')
    else:
        print(f'  -> Real t_break distribution NOT distinguishable from single-S2 null')
        print(f'     (clustering could be geometric artifact)')

    # Effect size
    print(f'\n  Real std: {real_fracs.std():.3f}')
    print(f'  Null 1 std: {null1_fracs.std():.3f}')
    print(f'  Std ratio: {null1_fracs.std()/max(real_fracs.std(), 1e-6):.2f}')
    if real_fracs.std() < null1_fracs.std() * 0.7:
        print(f'  -> Real clustering is TIGHTER than null (real signal)')
    else:
        print(f'  -> Real clustering is similar to null (likely artifact)')

if len(null2_fracs) >= 10:
    # Recovery test for positive control
    true_frac = 0.3
    recovery_error = np.abs(null2_fracs - true_frac)
    print(f'\nNull 2 — recovery of true t_break=0.3:')
    print(f'  Mean recovery error: {recovery_error.mean():.3f}')
    print(f'  Median error: {np.median(recovery_error):.3f}')
    print(f'  % within 0.1 of true: {100 * np.mean(recovery_error < 0.1):.1f}%')

# Save
out = {
    'null_test': 'T7.1 — null model control for piecewise S2 clustering',
    'n_null_per_condition': N_NULL,
    'real_data': {
        'n': int(len(real_fracs)),
        't_break_frac_mean': float(real_fracs.mean()) if len(real_fracs) else None,
        't_break_frac_median': float(np.median(real_fracs)) if len(real_fracs) else None,
        't_break_frac_std': float(real_fracs.std()) if len(real_fracs) else None,
        'delta_aicc_median': float(np.median(real_deltas)) if len(real_deltas) else None,
        'pct_delta_aicc_gt_4': float(100 * np.mean(real_deltas > 4)) if len(real_deltas) else None,
    },
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
        't_break_frac_mean': float(null2_fracs.mean()) if len(null2_fracs) else None,
        't_break_frac_median': float(np.median(null2_fracs)) if len(null2_fracs) else None,
        't_break_frac_std': float(null2_fracs.std()) if len(null2_fracs) else None,
        'delta_aicc_median': float(np.median(null2_deltas)) if len(null2_deltas) else None,
        'pct_delta_aicc_gt_4': float(100 * np.mean(null2_deltas > 4)) if len(null2_deltas) else None,
        'true_t_break_frac': 0.3,
        'recovery_error_median': float(np.median(np.abs(null2_fracs - 0.3))) if len(null2_fracs) else None,
    },
    'statistical_tests': {
        'ks_real_vs_null1_p': float(p_ks) if len(null1_fracs) >= 10 and len(real_fracs) >= 3 else None,
        'mannwhitney_real_vs_null1_p': float(p_u) if len(null1_fracs) >= 10 and len(real_fracs) >= 3 else None,
        'real_std': float(real_fracs.std()) if len(real_fracs) else None,
        'null1_std': float(null1_fracs.std()) if len(null1_fracs) else None,
    },
}
out_path = '/home/z/my-project/download/t71_null_control.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {out_path}')

# Verdict
print()
print('='*72)
print('VERDICT')
print('='*72)
if len(null1_fracs) >= 10 and len(real_fracs) >= 3:
    if p_ks < 0.05 and real_fracs.std() < null1_fracs.std() * 0.7:
        print('T7.1 SURVIVES NULL CONTROL.')
        print('  Real t_break distribution is significantly tighter than the')
        print('  single-S2 null. The clustering is NOT a geometric artifact.')
        print('  Multi-segment S2 is empirically supported.')
    elif p_ks < 0.05:
        print('T7.1 PARTIALLY SURVIVES NULL CONTROL.')
        print('  Real distribution differs from null, but std comparison')
        print('  is ambiguous. Need more data.')
    else:
        print('T7.1 FAILS NULL CONTROL.')
        print('  Real t_break distribution is NOT distinguishable from')
        print('  what pure single-S2 + piecewise-fit would produce.')
        print('  The clustering is likely a geometric artifact.')
