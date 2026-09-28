#!/usr/bin/env python3
"""
T7.1-I: Interference Extension Test
====================================

Tests whether an interference model (additive + cross-term with phase)
systematically beats the standard additive S2_DUST model on real data.

Three models:
  A:  Y = A1*R1 + A2*R2                                    (6 params)
  C1: Y = A1*R1 + A2*R2 + 2*sqrt(A1*A2*R1*R2)*cos(dtheta)  (7 params)
  C2: Y = A1*R1 + A2*R2 + 2*sqrt(A1*A2*R1*R2)*cos(alpha*lam + beta)  (8 params)

Where R_i = exp[-(lam/lam_qi)^Di]

Real data only. No fudging. AICc comparison.
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math, signal
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
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T71I/1.0'})
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


# Fetchers (same as t71 toy)
def fetch_binance(sym):
    raw = fetch(f'https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=365').decode()
    j = json.loads(raw)
    return np.array([float(r[4]) for r in j])

def fetch_coingecko(coin):
    raw = fetch(f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365&interval=daily').decode()
    j = json.loads(raw)
    return np.array([float(p[1]) for p in j.get('prices', [])])

def fetch_openmeteo(lat, lon):
    import datetime
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365)
    fmt = lambda d: d.strftime('%Y-%m-%d')
    url = f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={fmt(start)}&end_date={fmt(today)}&daily=temperature_2m_mean'
    raw = fetch(url).decode()
    j = json.loads(raw)
    return np.array(j.get('daily', {}).get('temperature_2m_mean', []), dtype=float)

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
    bins = np.arange(h0, h_max + 2)
    counts, _ = np.histogram(hrs, bins=bins)
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

# Model A: additive S2_DUST
def model_A(t, A1, lam1, D1, A2, lam2, D2):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    return A1 * R1 + A2 * R2

# Model C1: interference with constant phase
def model_C1(t, A1, lam1, D1, A2, lam2, D2, dtheta):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    cross = 2.0 * np.sqrt(np.maximum(A1 * A2 * R1 * R2, 0)) * np.cos(dtheta)
    return A1 * R1 + A2 * R2 + cross

# Model C2: interference with linear-in-lambda phase
def model_C2(t, A1, lam1, D1, A2, lam2, D2, alpha, beta):
    R1 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam1, 1e-9), D1))
    R2 = np.exp(-np.power(np.maximum(t, 1e-9) / max(lam2, 1e-9), D2))
    dtheta = alpha * t + beta
    cross = 2.0 * np.sqrt(np.maximum(A1 * A2 * R1 * R2, 0)) * np.cos(dtheta)
    return A1 * R1 + A2 * R2 + cross


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


def fit_all_three(t, R):
    """Fit models A, C1, C2. Return dict with params, rss, aicc."""
    n = len(t)
    tmid = float(t[len(t)//2])
    results = {}

    # Model A: additive, 6 params
    bounds_A_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01]
    bounds_A_hi = [2, 1e6, 10, 2, 1e6, 10]
    p0_list_A = [
        [0.7, tmid*0.3, 0.5, 0.3, tmid*2, 1.5],
        [0.5, tmid*0.5, 1.0, 0.5, tmid, 0.5],
        [0.6, tmid, 0.3, 0.4, tmid*3, 2.0],
    ]
    fA = safe_fit(model_A, t, R, p0_list_A, (bounds_A_lo, bounds_A_hi))
    if fA:
        results['A'] = {'params': list(fA[0]), 'rss': fA[1], 'aicc': aicc(fA[1], n, 6), 'k': 6}
    else:
        results['A'] = None

    # Model C1: interference, constant phase, 7 params
    bounds_C1_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01, 0.0]
    bounds_C1_hi = [2, 1e6, 10, 2, 1e6, 10, 2*np.pi]
    p0_list_C1 = []
    if fA:
        A1, l1, D1, A2, l2, D2 = fA[0]
        for dt in [0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            p0_list_C1.append([A1, l1, D1, A2, l2, D2, dt])
    else:
        p0_list_C1 = [
            [0.7, tmid*0.3, 0.5, 0.3, tmid*2, 1.5, 0.0],
            [0.7, tmid*0.3, 0.5, 0.3, tmid*2, 1.5, np.pi/2],
            [0.7, tmid*0.3, 0.5, 0.3, tmid*2, 1.5, np.pi],
        ]
    fC1 = safe_fit(model_C1, t, R, p0_list_C1, (bounds_C1_lo, bounds_C1_hi))
    if fC1:
        results['C1'] = {'params': list(fC1[0]), 'rss': fC1[1], 'aicc': aicc(fC1[1], n, 7), 'k': 7}
    else:
        results['C1'] = None

    # Model C2: interference, linear phase, 8 params
    bounds_C2_lo = [0.01, 1e-2, 0.01, 0.01, 1e-2, 0.01, -1.0, 0.0]
    bounds_C2_hi = [2, 1e6, 10, 2, 1e6, 10, 1.0, 2*np.pi]
    p0_list_C2 = []
    if fA:
        A1, l1, D1, A2, l2, D2 = fA[0]
        for alpha in [0.0, 0.01, 0.1]:
            for beta in [0.0, np.pi/2, np.pi]:
                p0_list_C2.append([A1, l1, D1, A2, l2, D2, alpha, beta])
    fC2 = safe_fit(model_C2, t, R, p0_list_C2, (bounds_C2_lo, bounds_C2_hi))
    if fC2:
        results['C2'] = {'params': list(fC2[0]), 'rss': fC2[1], 'aicc': aicc(fC2[1], n, 8), 'k': 8}
    else:
        results['C2'] = None

    return results


# ─── Main ────────────────────────────────────────────────────────────

print('='*72)
print('T7.1-I: INTERFERENCE EXTENSION TEST')
print('='*72)
print()
print('Three models on real ACF data:')
print('  A:  Y = A1*R1 + A2*R2                                    (6 params, additive)')
print('  C1: Y = A1*R1 + A2*R2 + 2*sqrt(A1*A2*R1*R2)*cos(dtheta)  (7 params, const phase)')
print('  C2: Y = A1*R1 + A2*R2 + 2*sqrt(A1*A2*R1*R2)*cos(a*l+b)   (8 params, linear phase)')
print()
print('If interference is real: C1 or C2 systematically beats A on AICc.')
print('If extra params just overfit: AICc penalizes them, A wins.')
print()

all_results = []

for name, loader in DATASETS.items():
    print(f'\n--- {name} ---', flush=True)
    try:
        values = loader()
        if values is None or len(values) < 30:
            print(f'  FETCH failed or too short: n={len(values) if values is not None else 0}')
            continue
        print(f'  n={len(values)}', flush=True)
    except Exception as e:
        print(f'  FETCH ERROR: {repr(e)[:80]}')
        continue

    result = to_acf(values)
    if result is None:
        print(f'  ACF failed')
        continue
    t, R = result
    # Drop lag 0
    t = t[1:]; R = R[1:]
    if len(t) < 20:
        print(f'  Too few lags after drop: n={len(t)}')
        continue
    print(f'  ACF lags: {len(t)}', flush=True)

    fits = fit_all_three(t, R)

    a_aicc = fits['A']['aicc'] if fits.get('A') else float('inf')
    c1_aicc = fits['C1']['aicc'] if fits.get('C1') else float('inf')
    c2_aicc = fits['C2']['aicc'] if fits.get('C2') else float('inf')

    min_aicc = min(a_aicc, c1_aicc, c2_aicc)

    print(f'  Model A  (6p): AICc = {a_aicc:>10.2f}  Δ = {a_aicc - min_aicc:>+7.2f}')
    print(f'  Model C1 (7p): AICc = {c1_aicc:>10.2f}  Δ = {c1_aicc - min_aicc:>+7.2f}')
    print(f'  Model C2 (8p): AICc = {c2_aicc:>10.2f}  Δ = {c2_aicc - min_aicc:>+7.2f}')

    winner = min([('A', a_aicc), ('C1', c1_aicc), ('C2', c2_aicc)], key=lambda x: x[1])[0]
    print(f'  Winner: {winner}')

    # Report interference params if C1 or C2 won
    if winner == 'C1' and fits.get('C1'):
        dtheta = fits['C1']['params'][6]
        print(f'    Δθ = {dtheta:.3f} rad ({np.degrees(dtheta):.1f}°)')
    if winner == 'C2' and fits.get('C2'):
        alpha = fits['C2']['params'][6]
        beta = fits['C2']['params'][7]
        print(f'    α = {alpha:.4f}, β = {beta:.3f} rad')

    all_results.append({
        'name': name,
        'n': int(len(t)),
        'A_aicc': float(a_aicc) if np.isfinite(a_aicc) else None,
        'C1_aicc': float(c1_aicc) if np.isfinite(c1_aicc) else None,
        'C2_aicc': float(c2_aicc) if np.isfinite(c2_aicc) else None,
        'A_C1_delta': float(a_aicc - c1_aicc) if np.isfinite(a_aicc) and np.isfinite(c1_aicc) else None,
        'A_C2_delta': float(a_aicc - c2_aicc) if np.isfinite(a_aicc) and np.isfinite(c2_aicc) else None,
        'winner': winner,
        'A_params': fits.get('A', {}).get('params') if fits.get('A') else None,
        'C1_params': fits.get('C1', {}).get('params') if fits.get('C1') else None,
        'C2_params': fits.get('C2', {}).get('params') if fits.get('C2') else None,
    })

# ─── Summary ─────────────────────────────────────────────────────────

print('\n' + '='*72)
print('SUMMARY')
print('='*72)
print()
print(f'{"Dataset":<10} {"A (6p)":>10} {"C1 (7p)":>10} {"C2 (8p)":>10} {"Δ(A-C1)":>8} {"Δ(A-C2)":>8} {"Winner":>8}')
print('-'*72)
for r in all_results:
    a = r['A_aicc']; c1 = r['C1_aicc']; c2 = r['C2_aicc']
    d1 = r['A_C1_delta']; d2 = r['A_C2_delta']
    print(f'{r["name"]:<10} {a:>10.2f} {c1:>10.2f} {c2:>10.2f} {d1:>+8.2f} {d2:>+8.2f} {r["winner"]:>8}')

# Aggregate
n_A = sum(1 for r in all_results if r['winner'] == 'A')
n_C1 = sum(1 for r in all_results if r['winner'] == 'C1')
n_C2 = sum(1 for r in all_results if r['winner'] == 'C2')
print(f'\nWins: A={n_A}  C1={n_C1}  C2={n_C2}  (total={len(all_results)})')

deltas_C1 = [r['A_C1_delta'] for r in all_results if r['A_C1_delta'] is not None]
deltas_C2 = [r['A_C2_delta'] for r in all_results if r['A_C2_delta'] is not None]
if deltas_C1:
    print(f'C1 vs A: median ΔAICc = {np.median(deltas_C1):+.2f}  (positive = C1 better)')
    print(f'          mean   ΔAICc = {np.mean(deltas_C1):+.2f}')
    n_c1_strong = sum(1 for d in deltas_C1 if d > 4)
    print(f'          strong C1 wins (Δ>4): {n_c1_strong}/{len(deltas_C1)}')
if deltas_C2:
    print(f'C2 vs A: median ΔAICc = {np.median(deltas_C2):+.2f}  (positive = C2 better)')
    print(f'          mean   ΔAICc = {np.mean(deltas_C2):+.2f}')
    n_c2_strong = sum(1 for d in deltas_C2 if d > 4)
    print(f'          strong C2 wins (Δ>4): {n_c2_strong}/{len(deltas_C2)}')

# Save
out = {
    'test': 'T7.1-I interference extension',
    'n_datasets': len(all_results),
    'wins': {'A': n_A, 'C1': n_C1, 'C2': n_C2},
    'C1_vs_A_median_delta': float(np.median(deltas_C1)) if deltas_C1 else None,
    'C2_vs_A_median_delta': float(np.median(deltas_C2)) if deltas_C2 else None,
    'C1_strong_wins': n_c1_strong if deltas_C1 else 0,
    'C2_strong_wins': n_c2_strong if deltas_C2 else 0,
    'results': all_results,
}
out_path = os.path.join(OUT_DIR, 't71_interference_test.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')

# Verdict
print('\n' + '='*72)
print('VERDICT')
print('='*72)
total_interference_wins = n_C1 + n_C2
total_strong = (n_c1_strong if deltas_C1 else 0) + (n_c2_strong if deltas_C2 else 0)
if total_strong > len(all_results) * 0.4:
    print(f'INTERFERENCE SUPPORTED: {total_strong}/{len(all_results)} datasets show strong interference wins (ΔAICc>4).')
    print('The cross-term with phase carries real predictive structure beyond additive S2_DUST.')
elif total_interference_wins > len(all_results) * 0.5:
    print(f'INTERFERENCE PARTIALLY SUPPORTED: {total_interference_wins}/{len(all_results)} wins but only {total_strong} strong.')
    print('Phase term helps marginally but AICc penalty mostly cancels the gain.')
else:
    print(f'INTERFERENCE NOT SUPPORTED: {n_A}/{len(all_results)} datasets won by additive model.')
    print('The cross-term does not carry structure beyond what additive S2_DUST captures.')
    print('The interference analogy is mathematically interesting but not empirically grounded on this data.')
