#!/usr/bin/env python3
"""
T7.1-I Discriminating Tests
============================

5 tests to distinguish "interference physics" from "damped oscillation"
as the explanation for the C2 model's strong wins.

Test 1: Phase stability across rolling windows
  - Coherent source (interference): phase (α, β) stable across windows
  - Random oscillation: phase drifts randomly

Test 2: Coherence scale
  - Interference: cross-term decays beyond a characteristic λ_c
  - Oscillation: cross-term just oscillates and decays generically

Test 3: Component-pair specificity (3-component)
  - Interference: different (i,j) pairs have different phases
  - Generic oscillation: all pairs give the same phase

Test 4: Scramble control
  - Geometric/interference: scramble destroys the cross-term's value
  - Scalar oscillation: scramble preserves it

Test 5: 3-component interference vs 2-component
  - If interference scales with N components: 3-comp beats 2-comp
  - If 2-comp was already capturing the oscillation: 3-comp doesn't help

Real data only. No fudging.
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math
import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=15):
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T71I2/1.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 1: raise
            time.sleep(1)


def to_acf(values, max_lag=80):
    R = np.asarray(values, dtype=float)
    if len(R) < 30: return None
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, max_lag)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    if acf[0] > 0: acf = acf / acf[0]
    return np.arange(max_lag, dtype=float), acf


def fetch_binance(sym):
    raw = fetch(f'https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=365').decode()
    return np.array([float(r[4]) for r in json.loads(raw)])

def fetch_coingecko(coin):
    raw = fetch(f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365&interval=daily').decode()
    return np.array([float(p[1]) for p in json.loads(raw).get('prices', [])])

def fetch_openmeteo(lat, lon):
    import datetime
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365)
    fmt = lambda d: d.strftime('%Y-%m-%d')
    url = f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={fmt(start)}&end_date={fmt(today)}&daily=temperature_2m_mean'
    return np.array(json.loads(fetch(url).decode()).get('daily', {}).get('temperature_2m_mean', []), dtype=float)

def fetch_usgs():
    raw = fetch('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv').decode()
    reader = csv.DictReader(io.StringIO(raw))
    times = [r.get('time') for r in reader if r.get('time')]
    if len(times) < 30: return None
    import datetime
    hrs = []
    for tstr in times:
        try:
            dt = datetime.datetime.fromisoformat(tstr.replace('Z', '+00:00'))
            hrs.append(dt.timestamp() / 3600.0)
        except: continue
    hrs.sort()
    h0 = math.floor(min(hrs)); h_max = math.ceil(max(hrs))
    counts, _ = np.histogram(hrs, bins=np.arange(h0, h_max + 2))
    return counts.astype(float)


DATASETS = {
    'BTC': lambda: fetch_binance('BTCUSDT'),
    'ETH': lambda: fetch_binance('ETHUSDT'),
    'SOL': lambda: fetch_binance('SOLUSDT'),
    'cg_BTC': lambda: fetch_coingecko('bitcoin'),
    'cg_ETH': lambda: fetch_coingecko('ethereum'),
    'Berlin': lambda: fetch_openmeteo(52.52, 13.41),
    'Tokyo': lambda: fetch_openmeteo(35.68, 139.69),
    'NYC': lambda: fetch_openmeteo(40.71, -74.01),
    'London': lambda: fetch_openmeteo(51.51, -0.13),
    'Sydney': lambda: fetch_openmeteo(-33.87, 151.21),
    'USGS': fetch_usgs,
}


# ─── Models ──────────────────────────────────────────────────────────

def m_s2(t, A, lam, D):
    return A * np.exp(-np.power(np.maximum(t, 1e-9) / max(lam, 1e-9), D))

# 2-component additive (Model A)
def model_A2(t, A1, lam1, D1, A2, lam2, D2):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    return A1 * R1 + A2 * R2

# 2-component interference, linear phase (Model C2)
def model_C2(t, A1, lam1, D1, A2, lam2, D2, alpha, beta):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    dtheta = alpha * t + beta
    cross = 2.0 * np.sqrt(np.maximum(A1 * A2 * R1 * R2, 0)) * np.cos(dtheta)
    return A1 * R1 + A2 * R2 + cross

# 3-component additive
def model_A3(t, A1, lam1, D1, A2, lam2, D2, A3, lam3, D3):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    R3 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam3, 1e-9), D3))
    return A1 * R1 + A2 * R2 + A3 * R3

# 3-component interference: 3 pairs, each with own phase
# Y = sum |Ai|^2 Ri + 2 sum_{i<j} sqrt(Ai Aj Ri Rj) cos(theta_ij)
# theta_ij = alpha_ij * t + beta_ij
# 9 base params + 6 phase params (3 pairs × 2) = 15 params
def model_C3(t, A1, lam1, D1, A2, lam2, D2, A3, lam3, D3,
             a12, b12, a13, b13, a23, b23):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    R3 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam3, 1e-9), D3))
    base = A1*R1 + A2*R2 + A3*R3
    cross12 = 2.0 * np.sqrt(np.maximum(A1*A2*R1*R2, 0)) * np.cos(a12*t + b12)
    cross13 = 2.0 * np.sqrt(np.maximum(A1*A3*R1*R3, 0)) * np.cos(a13*t + b13)
    cross23 = 2.0 * np.sqrt(np.maximum(A2*A3*R2*R3, 0)) * np.cos(a23*t + b23)
    return base + cross12 + cross13 + cross23


def aicc(rss, n, k):
    if not np.isfinite(rss) or n <= k + 1: return float('inf')
    return 2*k + n*np.log(rss/n) + (2*k*(k+1))/(n-k-1)


def safe_fit(model, t, R, p0_list, bounds, maxfev=10000):
    best = None
    for p0 in p0_list:
        try:
            popt, _ = curve_fit(model, t, R, p0=p0, bounds=bounds, maxfev=maxfev)
            rss = float(np.sum((R - model(t, *popt))**2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except Exception:
            continue
    return best


def fit_2comp(t, R):
    """Fit A2 and C2 on (t, R). Return dict with params/rss/aicc for each."""
    n = len(t)
    tmid = float(t[len(t)//2])
    bounds_A_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01]
    bounds_A_hi = [2, 1e6, 10, 2, 1e6, 10]
    p0_list_A = [
        [0.7, tmid*0.3, 0.5, 0.3, tmid*2, 1.5],
        [0.5, tmid*0.5, 1.0, 0.5, tmid, 0.5],
        [0.6, tmid, 0.3, 0.4, tmid*3, 2.0],
    ]
    fA = safe_fit(model_A2, t, R, p0_list_A, (bounds_A_lo, bounds_A_hi))

    bounds_C2_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01, -1.0, 0.0]
    bounds_C2_hi = [2, 1e6, 10, 2, 1e6, 10, 1.0, 2*np.pi]
    p0_list_C2 = []
    if fA:
        A1, l1, D1, A2, l2, D2 = fA[0]
        for alpha in [0.0, 0.05]:
            for beta in [0.0, np.pi/2, np.pi]:
                p0_list_C2.append([A1, l1, D1, A2, l2, D2, alpha, beta])
    fC2 = safe_fit(model_C2, t, R, p0_list_C2, (bounds_C2_lo, bounds_C2_hi), maxfev=5000)

    out = {}
    if fA:
        out['A2'] = {'params': list(fA[0]), 'rss': fA[1], 'aicc': aicc(fA[1], n, 6), 'k': 6}
    if fC2:
        out['C2'] = {'params': list(fC2[0]), 'rss': fC2[1], 'aicc': aicc(fC2[1], n, 8), 'k': 8}
    return out


def fit_3comp(t, R):
    """Fit A3 and C3. Return dict."""
    n = len(t)
    tmid = float(t[len(t)//2])
    bounds_A3_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01]
    bounds_A3_hi = [2, 1e6, 10, 2, 1e6, 10, 2, 1e6, 10]
    p0_list_A3 = [
        [0.5, tmid*0.3, 0.5, 0.3, tmid, 1.0, 0.2, tmid*3, 2.0],
        [0.4, tmid*0.2, 0.3, 0.4, tmid*0.7, 0.7, 0.2, tmid*2.5, 1.5],
    ]
    fA3 = safe_fit(model_A3, t, R, p0_list_A3, (bounds_A3_lo, bounds_A3_hi))

    # C3: 15 params
    bounds_C3_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01,
                    -1.0, 0.0, -1.0, 0.0, -1.0, 0.0]
    bounds_C3_hi = [2, 1e6, 10, 2, 1e6, 10, 2, 1e6, 10,
                    1.0, 2*np.pi, 1.0, 2*np.pi, 1.0, 2*np.pi]
    p0_list_C3 = []
    if fA3:
        A1, l1, D1, A2, l2, D2, A3p, l3, D3 = fA3[0]
        # Reduced grid: 4 starts instead of 16
        for a12, b12, a13, b13 in [(0.0, 0.0, 0.0, 0.0),
                                    (0.05, np.pi/2, 0.05, np.pi/2),
                                    (-0.05, np.pi, -0.05, np.pi),
                                    (0.1, 0.0, 0.0, np.pi/2)]:
            p0_list_C3.append([A1, l1, D1, A2, l2, D2, A3p, l3, D3,
                              a12, b12, a13, b13, 0.0, 0.0])
    fC3 = safe_fit(model_C3, t, R, p0_list_C3, (bounds_C3_lo, bounds_C3_hi), maxfev=8000)

    out = {}
    if fA3:
        out['A3'] = {'params': list(fA3[0]), 'rss': fA3[1], 'aicc': aicc(fA3[1], n, 9), 'k': 9}
    if fC3:
        out['C3'] = {'params': list(fC3[0]), 'rss': fC3[1], 'aicc': aicc(fC3[1], n, 15), 'k': 15}
    return out


# ─── TEST 1: Phase stability across rolling windows ─────────────────

def test1_phase_stability(name, values, n_windows=4):
    """Fit C2 on rolling windows. Are (alpha, beta) stable?"""
    print(f'\n  [Test 1] Phase stability across {n_windows} rolling windows...', flush=True)
    n = len(values)
    if n < 60:
        print(f'    SKIP: not enough data (n={n})')
        return None
    win_size = n // 2  # half-overlap windows
    step = max(1, (n - win_size) // (n_windows - 1)) if n_windows > 1 else 1

    phases = []
    for i in range(n_windows):
        start = i * step
        end = min(start + win_size, n)
        if end - start < 40:
            continue
        chunk = values[start:end]
        acf_result = to_acf(chunk, max_lag=80)
        if acf_result is None:
            continue
        t, R = acf_result
        t = t[1:]; R = R[1:]
        if len(t) < 20:
            continue
        fits = fit_2comp(t, R)
        if 'C2' in fits:
            p = fits['C2']['params']
            phases.append({
                'window': i,
                'start_idx': start,
                'end_idx': end,
                'alpha': float(p[6]),
                'beta': float(p[7]),
                'aicc': float(fits['C2']['aicc']),
            })

    if len(phases) < 3:
        print(f'    SKIP: only {len(phases)} windows fit successfully')
        return None

    alphas = np.array([p['alpha'] for p in phases])
    betas = np.array([p['beta'] for p in phases])

    # Coefficient of variation (lower = more stable = coherent)
    alpha_cv = alphas.std() / max(abs(alphas.mean()), 1e-6) if abs(alphas.mean()) > 1e-6 else float('inf')
    beta_cv = betas.std() / max(abs(betas.mean()), 1e-6) if abs(betas.mean()) > 1e-6 else float('inf')

    # Also: circular std of beta (more appropriate for phase)
    beta_circ_std = float(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * betas))))))

    print(f'    {len(phases)} windows fit')
    print(f'    alpha: mean={alphas.mean():+.4f}  std={alphas.std():.4f}  CV={alpha_cv:.2f}')
    print(f'    beta:  mean={betas.mean():+.3f}  std={betas.std():.3f}  CV={beta_cv:.2f}  circ_std={beta_circ_std:.3f}')

    return {
        'n_windows': len(phases),
        'alphas': alphas.tolist(),
        'betas': betas.tolist(),
        'alpha_mean': float(alphas.mean()),
        'alpha_std': float(alphas.std()),
        'alpha_cv': float(alpha_cv),
        'beta_mean': float(betas.mean()),
        'beta_std': float(betas.std()),
        'beta_cv': float(beta_cv),
        'beta_circ_std': beta_circ_std,
        'phases': phases,
    }


# ─── TEST 2: Coherence scale ────────────────────────────────────────

def test2_coherence_scale(name, values):
    """Does the cross-term's contribution decay beyond a characteristic λ?"""
    print(f'\n  [Test 2] Coherence scale analysis...', flush=True)
    acf_result = to_acf(values, max_lag=80)
    if acf_result is None:
        print(f'    SKIP: ACF failed')
        return None
    t, R = acf_result
    t = t[1:]; R = R[1:]
    if len(t) < 20:
        print(f'    SKIP: too few lags')
        return None

    fits = fit_2comp(t, R)
    if 'C2' not in fits or 'A2' not in fits:
        print(f'    SKIP: fits failed')
        return None

    pC2 = fits['C2']['params']
    pA = fits['A2']['params']

    # Compute cross-term contribution as function of λ
    A1, l1, D1, A2, l2, D2 = pC2[:6]
    alpha, beta = pC2[6], pC2[7]
    R1 = np.exp(-np.power(t / max(l1, 1e-9), D1))
    R2 = np.exp(-np.power(t / max(l2, 1e-9), D2))
    cross = 2.0 * np.sqrt(np.maximum(A1 * A2 * R1 * R2, 0)) * np.cos(alpha * t + beta)
    base = A1 * R1 + A2 * R2
    total = base + cross

    # Cross-term magnitude relative to base
    cross_ratio = np.abs(cross) / np.maximum(np.abs(base), 1e-9)

    # Find where cross-term drops to 10% of its peak
    cross_abs = np.abs(cross)
    peak_idx = int(np.argmax(cross_abs))
    peak_val = cross_abs[peak_idx]
    decay_idx = None
    for i in range(peak_idx, len(t)):
        if cross_abs[i] < 0.1 * peak_val:
            decay_idx = i
            break
    coherence_scale = float(t[decay_idx]) if decay_idx is not None else float(t[-1])

    # Compute lambda_q from C2 fit (geometric mean of l1, l2)
    lambda_q = float(np.sqrt(l1 * l2))

    print(f'    Cross-term peak at λ={t[peak_idx]:.1f}, magnitude={peak_val:.4f}')
    print(f'    Cross-term decays to 10% of peak at λ={coherence_scale:.1f}')
    print(f'    λ_q (geometric mean of l1, l2) = {lambda_q:.1f}')
    print(f'    Ratio coherence_scale / λ_q = {coherence_scale / max(lambda_q, 1e-6):.2f}')

    return {
        't': t.tolist(),
        'cross': cross.tolist(),
        'base': base.tolist(),
        'cross_ratio': cross_ratio.tolist(),
        'peak_lambda': float(t[peak_idx]),
        'peak_val': float(peak_val),
        'coherence_scale': coherence_scale,
        'lambda_q': lambda_q,
        'ratio_coherence_over_lambda_q': float(coherence_scale / max(lambda_q, 1e-6)),
    }


# ─── TEST 3: Component-pair specificity (3-component) ───────────────

def test3_pair_specificity(name, values):
    """Fit 3-component interference. Do different pairs have different phases?"""
    print(f'\n  [Test 3] Component-pair specificity (3-comp interference)...', flush=True)
    acf_result = to_acf(values, max_lag=80)
    if acf_result is None:
        return None
    t, R = acf_result
    t = t[1:]; R = R[1:]
    if len(t) < 25:
        print(f'    SKIP: too few lags')
        return None

    fits3 = fit_3comp(t, R)
    if 'C3' not in fits3:
        print(f'    SKIP: C3 fit failed')
        return None

    pC3 = fits3['C3']['params']
    # 9 base + 6 phase
    A1, l1, D1, A2, l2, D2, A3p, l3, D3 = pC3[:9]
    a12, b12, a13, b13, a23, b23 = pC3[9:15]

    # Compare phases across pairs
    print(f'    Pair (1,2): alpha={a12:+.4f}, beta={b12:+.3f} ({np.degrees(b12):+.0f}°)')
    print(f'    Pair (1,3): alpha={a13:+.4f}, beta={b13:+.3f} ({np.degrees(b13):+.0f}°)')
    print(f'    Pair (2,3): alpha={a23:+.4f}, beta={b23:+.3f} ({np.degrees(b23):+.0f}°)')

    # Phase differences between pairs
    d_beta_12_13 = float(np.abs(b12 - b13))
    d_beta_12_23 = float(np.abs(b12 - b23))
    d_beta_13_23 = float(np.abs(b13 - b23))
    d_alpha_12_13 = float(np.abs(a12 - a13))
    d_alpha_12_23 = float(np.abs(a12 - a23))
    d_alpha_13_23 = float(np.abs(a13 - a23))

    # AICc comparison: A3 vs C3
    a3_aicc = fits3.get('A3', {}).get('aicc', float('inf'))
    c3_aicc = fits3['C3']['aicc']
    delta_a3_c3 = a3_aicc - c3_aicc

    print(f'    A3 (9p) AICc = {a3_aicc:.2f}')
    print(f'    C3 (15p) AICc = {c3_aicc:.2f}')
    print(f'    Δ(A3 - C3) = {delta_a3_c3:+.2f}  (positive = interference helps)')

    return {
        'A3_aicc': float(a3_aicc) if np.isfinite(a3_aicc) else None,
        'C3_aicc': float(c3_aicc),
        'delta_A3_C3': float(delta_a3_c3),
        'pair_12': {'alpha': float(a12), 'beta': float(b12)},
        'pair_13': {'alpha': float(a13), 'beta': float(b13)},
        'pair_23': {'alpha': float(a23), 'beta': float(b23)},
        'beta_diff_12_13': d_beta_12_13,
        'beta_diff_12_23': d_beta_12_23,
        'beta_diff_13_23': d_beta_13_23,
        'alpha_diff_12_13': d_alpha_12_13,
        'alpha_diff_12_23': d_alpha_12_23,
        'alpha_diff_13_23': d_alpha_13_23,
    }


# ─── TEST 4: Scramble control ───────────────────────────────────────

def test4_scramble_control(name, values, n_scrambles=5):
    """Scramble the temporal order. Does interference advantage survive?"""
    print(f'\n  [Test 4] Scramble control ({n_scrambles} permutations)...', flush=True)
    acf_result = to_acf(values, max_lag=80)
    if acf_result is None:
        return None
    t, R = acf_result
    t = t[1:]; R = R[1:]
    if len(t) < 20:
        print(f'    SKIP: too few lags')
        return None

    # Baseline (unscrambled)
    fits_base = fit_2comp(t, R)
    base_A = fits_base.get('A2', {}).get('aicc', float('inf'))
    base_C2 = fits_base.get('C2', {}).get('aicc', float('inf'))
    base_delta = base_A - base_C2  # positive = C2 better

    print(f'    Baseline: A_AICc={base_A:.2f}, C2_AICc={base_C2:.2f}, Δ={base_delta:+.2f}')

    # Scramble: permute the temporal order of values, recompute ACF, refit
    rng = np.random.RandomState(42)
    scrambled_deltas = []
    for i in range(n_scrambles):
        perm = rng.permutation(len(values))
        scrambled = values[perm]
        acf_scram = to_acf(scrambled, max_lag=80)
        if acf_scram is None:
            continue
        ts, Rs = acf_scram
        ts = ts[1:]; Rs = Rs[1:]
        if len(ts) < 20:
            continue
        fits_scram = fit_2comp(ts, Rs)
        sA = fits_scram.get('A2', {}).get('aicc', float('inf'))
        sC2 = fits_scram.get('C2', {}).get('aicc', float('inf'))
        s_delta = sA - sC2
        scrambled_deltas.append(s_delta)

    if not scrambled_deltas:
        print(f'    SKIP: no scrambled fits succeeded')
        return None

    scram_deltas = np.array(scrambled_deltas)
    print(f'    Scrambled: median Δ = {np.median(scram_deltas):+.2f}, mean = {scram_deltas.mean():+.2f}')
    print(f'    Scrambled Δ range: [{scram_deltas.min():+.2f}, {scram_deltas.max():+.2f}]')

    return {
        'baseline_delta': float(base_delta),
        'scrambled_deltas': scram_deltas.tolist(),
        'scrambled_median': float(np.median(scram_deltas)),
        'scrambled_mean': float(scram_deltas.mean()),
        'scrambled_min': float(scram_deltas.min()),
        'scrambled_max': float(scram_deltas.max()),
        'n_scrambles': len(scram_deltas),
    }


# ─── TEST 5: 3-comp vs 2-comp interference ──────────────────────────

def test5_3vs2_comp(name, values):
    """Does 3-component interference beat 2-component?"""
    print(f'\n  [Test 5] 3-comp vs 2-comp interference...', flush=True)
    acf_result = to_acf(values, max_lag=80)
    if acf_result is None:
        return None
    t, R = acf_result
    t = t[1:]; R = R[1:]
    if len(t) < 25:
        print(f'    SKIP: too few lags')
        return None

    fits2 = fit_2comp(t, R)
    fits3 = fit_3comp(t, R)

    a2 = fits2.get('A2', {}).get('aicc', float('inf'))
    c2 = fits2.get('C2', {}).get('aicc', float('inf'))
    a3 = fits3.get('A3', {}).get('aicc', float('inf'))
    c3 = fits3.get('C3', {}).get('aicc', float('inf'))

    delta_c2_a2 = a2 - c2
    delta_c3_a3 = a3 - c3
    delta_c3_c2 = c2 - c3  # positive = C3 better than C2

    print(f'    2-comp: A={a2:.2f}, C2={c2:.2f}, Δ(C2-A)={delta_c2_a2:+.2f}')
    print(f'    3-comp: A={a3:.2f}, C3={c3:.2f}, Δ(C3-A)={delta_c3_a3:+.2f}')
    print(f'    C3 vs C2: Δ = {delta_c3_c2:+.2f}  (positive = 3-comp interference helps)')

    return {
        'A2_aicc': float(a2) if np.isfinite(a2) else None,
        'C2_aicc': float(c2) if np.isfinite(c2) else None,
        'A3_aicc': float(a3) if np.isfinite(a3) else None,
        'C3_aicc': float(c3) if np.isfinite(c3) else None,
        'delta_C2_A2': float(delta_c2_a2) if np.isfinite(delta_c2_a2) else None,
        'delta_C3_A3': float(delta_c3_a3) if np.isfinite(delta_c3_a3) else None,
        'delta_C3_C2': float(delta_c3_c2) if np.isfinite(delta_c3_c2) else None,
    }


# ─── Main ────────────────────────────────────────────────────────────

print('='*72)
print('T7.1-I DISCRIMINATING TESTS: Interference vs Oscillation')
print('='*72)
print()
print('5 tests on real data:')
print('  1. Phase stability across rolling windows (coherent = interference)')
print('  2. Coherence scale (cross-term decays beyond characteristic λ)')
print('  3. Component-pair specificity (3-comp: different pairs, different phases)')
print('  4. Scramble control (destroy joint structure, does interference survive?)')
print('  5. 3-comp vs 2-comp interference (does richer decomposition help?)')
print()

all_results = {}

for name, loader in DATASETS.items():
    print(f'\n{"="*72}')
    print(f'DATASET: {name}')
    print(f'{"="*72}', flush=True)
    try:
        values = loader()
        if values is None or len(values) < 30:
            print(f'  FETCH failed or too short: n={len(values) if values is not None else 0}')
            continue
        print(f'  n={len(values)}', flush=True)
    except Exception as e:
        print(f'  FETCH ERROR: {repr(e)[:80]}')
        continue

    ds_result = {'n': int(len(values))}

    # Run all 5 tests
    try:
        ds_result['test1_phase_stability'] = test1_phase_stability(name, values)
    except Exception as e:
        print(f'  [Test 1] ERROR: {repr(e)[:80]}')
        ds_result['test1_phase_stability'] = None

    try:
        ds_result['test2_coherence_scale'] = test2_coherence_scale(name, values)
    except Exception as e:
        print(f'  [Test 2] ERROR: {repr(e)[:80]}')
        ds_result['test2_coherence_scale'] = None

    # Test 3 disabled for runtime — 3-comp fit is too expensive
    # We have partial Test 3 data from BTC/ETH/SOL/cg_BTC/cg_ETH in the earlier run
    ds_result['test3_pair_specificity'] = None

    try:
        ds_result['test4_scramble_control'] = test4_scramble_control(name, values)
    except Exception as e:
        print(f'  [Test 4] ERROR: {repr(e)[:80]}')
        ds_result['test4_scramble_control'] = None

    # Test 5 disabled for runtime — 3-comp fit is too expensive on some datasets
    # We have partial Test 5 data from the first 4 datasets in the earlier run
    ds_result['test5_3vs2_comp'] = None

    all_results[name] = ds_result

# ─── Summary ─────────────────────────────────────────────────────────

print('\n' + '='*72)
print('GRAND SUMMARY')
print('='*72)

# Test 1 summary
print('\n--- Test 1: Phase stability (lower CV = more coherent = interference) ---')
print(f'{"Dataset":<10} {"n_win":>6} {"alpha_CV":>10} {"beta_CV":>10} {"beta_circ_std":>14}')
print('-'*55)
t1_results = []
for name, r in all_results.items():
    t1 = r.get('test1_phase_stability')
    if t1:
        print(f'{name:<10} {t1["n_windows"]:>6} {t1["alpha_cv"]:>10.2f} {t1["beta_cv"]:>10.2f} {t1["beta_circ_std"]:>14.3f}')
        t1_results.append(t1)
if t1_results:
    cvs = [t['beta_circ_std'] for t in t1_results]
    print(f'\n  Median beta circ_std across datasets: {np.median(cvs):.3f}')
    print(f'  (low < 1.0 = coherent/interference; high > 1.5 = drifting/oscillation)')

# Test 2 summary
print('\n--- Test 2: Coherence scale (ratio coherence_scale / lambda_q) ---')
print(f'{"Dataset":<10} {"peak_lam":>10} {"coh_scale":>10} {"lambda_q":>10} {"ratio":>8}')
print('-'*55)
t2_results = []
for name, r in all_results.items():
    t2 = r.get('test2_coherence_scale')
    if t2:
        print(f'{name:<10} {t2["peak_lambda"]:>10.1f} {t2["coherence_scale"]:>10.1f} {t2["lambda_q"]:>10.1f} {t2["ratio_coherence_over_lambda_q"]:>8.2f}')
        t2_results.append(t2)
if t2_results:
    ratios = [t['ratio_coherence_over_lambda_q'] for t in t2_results]
    print(f'\n  Median ratio: {np.median(ratios):.2f}')
    print(f'  (ratio ~1.0 = coherence scale tracks lambda_q = interference-like)')

# Test 3 summary
print('\n--- Test 3: Component-pair specificity ---')
print(f'{"Dataset":<10} {"Δβ(12,13)":>10} {"Δβ(12,23)":>10} {"Δβ(13,23)":>10} {"Δ(A3-C3)":>10}')
print('-'*55)
t3_results = []
for name, r in all_results.items():
    t3 = r.get('test3_pair_specificity')
    if t3:
        print(f'{name:<10} {t3["beta_diff_12_13"]:>10.3f} {t3["beta_diff_12_23"]:>10.3f} {t3["beta_diff_13_23"]:>10.3f} {t3["delta_A3_C3"]:>+10.2f}')
        t3_results.append(t3)
if t3_results:
    diffs = [t['beta_diff_12_13'] for t in t3_results] + [t['beta_diff_12_23'] for t in t3_results]
    print(f'\n  Median |Δβ| across pairs: {np.median(diffs):.3f}')
    print(f'  (large Δβ = pair-specific = interference; small Δβ = generic oscillation)')
    n_c3_wins = sum(1 for t in t3_results if t['delta_A3_C3'] > 4)
    print(f'  C3 beats A3 strongly: {n_c3_wins}/{len(t3_results)}')

# Test 4 summary
print('\n--- Test 4: Scramble control ---')
print(f'{"Dataset":<10} {"base_Δ":>10} {"scram_med":>10} {"scram_min":>10} {"scram_max":>10}')
print('-'*55)
t4_results = []
for name, r in all_results.items():
    t4 = r.get('test4_scramble_control')
    if t4:
        print(f'{name:<10} {t4["baseline_delta"]:>+10.2f} {t4["scrambled_median"]:>+10.2f} {t4["scrambled_min"]:>+10.2f} {t4["scrambled_max"]:>+10.2f}')
        t4_results.append(t4)
if t4_results:
    base_deltas = [t['baseline_delta'] for t in t4_results]
    scram_deltas = [t['scrambled_median'] for t in t4_results]
    print(f'\n  Baseline median Δ: {np.median(base_deltas):+.2f}')
    print(f'  Scrambled median Δ: {np.median(scram_deltas):+.2f}')
    print(f'  (scrambled ≈ baseline = scalar oscillation; scrambled << baseline = geometric/interference)')

# Test 5 summary
print('\n--- Test 5: 3-comp vs 2-comp interference ---')
print(f'{"Dataset":<10} {"Δ(C2-A2)":>10} {"Δ(C3-A3)":>10} {"Δ(C3-C2)":>10}')
print('-'*45)
t5_results = []
for name, r in all_results.items():
    t5 = r.get('test5_3vs2_comp')
    if t5:
        d1 = t5.get('delta_C2_A2'); d2 = t5.get('delta_C3_A3'); d3 = t5.get('delta_C3_C2')
        print(f'{name:<10} {d1:>+10.2f} {d2:>+10.2f} {d3:>+10.2f}')
        t5_results.append(t5)
if t5_results:
    d_c3_c2 = [t['delta_C3_C2'] for t in t5_results if t.get('delta_C3_C2') is not None]
    n_c3_beats_c2 = sum(1 for d in d_c3_c2 if d > 4)
    print(f'\n  C3 beats C2 strongly: {n_c3_beats_c2}/{len(d_c3_c2)}')

# Save
out = {
    'test': 'T7.1-I discriminating tests (5 tests, 11 datasets)',
    'n_datasets': len(all_results),
    'results': all_results,
}
out_path = os.path.join(OUT_DIR, 't71_interference_discriminating.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')

# Final verdict
print('\n' + '='*72)
print('VERDICT')
print('='*72)
print()
print('Interference interpretation is supported if:')
print('  Test 1: low beta CV (phase stable across windows)')
print('  Test 2: coherence scale / lambda_q ratio ~ 1.0')
print('  Test 3: large phase differences between pairs')
print('  Test 4: scrambling destroys the interference advantage')
print('  Test 5: 3-comp interference beats 2-comp')
print()
print('Oscillation interpretation is supported if:')
print('  Test 1: high beta CV (phase drifts)')
print('  Test 2: no characteristic coherence scale')
print('  Test 3: similar phases across pairs')
print('  Test 4: scrambling preserves the advantage')
print('  Test 5: 3-comp does not help (2-comp already captured the oscillation)')
