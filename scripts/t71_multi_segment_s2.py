#!/usr/bin/env python3
"""
T7.1 / T8 — Multi-Segment S2 Hypothesis Test
==============================================

Tests the user's hypothesis: the global retention curve is composed of
piecewise S2 regimes, with the kernel entering different "information
states" at reproducible scale thresholds.

  R(λ) ≈ exp[-(λ/λ_q,1)^D_1]   for  λ in [0, λ*_{1→2})
       ≈ exp[-(λ/λ_q,2)^D_2]   for  λ in [λ*_{1→2}, λ*_{2→3})
       ≈ exp[-(λ/λ_q,3)^D_3]   for  λ in [λ*_{2→3}, ∞)

The critical testable prediction:

  >>> Do the transition scales λ*_{k→k+1} cluster at reproducible values
  >>> across many independent datasets?

If yes → multi-segment S2 is a real refinement (T7.1 supported).
If no  → the apparent multi-segment structure is fitting noise.

Three concrete tests on real data:

  (T7.1-A) Sliding-window S2 fit + automatic breakpoint detection:
           fit S2 on rolling windows, find where (D, λ_q) jumps.
           Report the jump locations per dataset.

  (T7.1-B) Two-segment piecewise S2 fit with breakpoint search:
           R(λ) = exp[-(λ/λ_q,1)^D_1] for λ < λ*
                = c · exp[-(λ/λ_q,2)^D_2] for λ ≥ λ*
           Fit λ* by AICc minimization. Collect λ* across datasets.
           Test whether {λ*} clusters at characteristic scales.

  (T7.1-C) Information-rate test:
           Compute dR/dλ locally. Test whether regime transitions
           coincide with sharp changes in |dR/dλ| (the "information
           state changes" the user described).

Real data sources: fetch live curves from the registry, focusing on
the 50 entries with strongest S2_DUST wins (the closure-violation
candidates).
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_model_compare import m_s2, m_s2_dust

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=20):
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T71/1.0 (research)'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == 1: raise
            time.sleep(1)


# ─────────────────────────────────────────────────────────────────────
# Fetchers per source (reuse from probe A)
# ─────────────────────────────────────────────────────────────────────

def fetch_worldbank(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    rows = j[1] if len(j) > 1 and isinstance(j[1], list) else []
    pairs = []
    for r in rows:
        try:
            v = float(r['value']); y = int(r['date'])
            if v == v and v != 0: pairs.append((y, v))
        except: continue
    pairs.sort()
    if len(pairs) < 20: return None
    t = np.array([p[0] for p in pairs], dtype=float)
    R = np.array([p[1] for p in pairs], dtype=float)
    if R[0] != 0: R_norm = R / R[0]
    else: R_norm = R / max(abs(R))
    return t - t[0], R_norm

def fetch_coingecko(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    prices = j.get('prices', [])
    if len(prices) < 20: return None
    t = np.array([p[0] for p in prices], dtype=float) / 1000
    R = np.array([p[1] for p in prices], dtype=float)
    # Use ACF of returns for retention-like curve
    R = np.diff(np.log(R + 1e-10))
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 80)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0: acf = acf / acf[0]
    return t, acf

def fetch_binance(url):
    raw = fetch(url).decode('utf-8')
    rows = json.loads(raw)
    if len(rows) < 20: return None
    R = np.array([float(r[4]) for r in rows])
    # Use log-returns ACF
    R = np.diff(np.log(R + 1e-10))
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 80)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0: acf = acf / acf[0]
    return t, acf

def fetch_openmeteo(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    daily = j.get('daily', {})
    times = daily.get('time', [])
    temps = daily.get('temperature_2m_mean', [])
    if len(times) < 20: return None
    import datetime
    t0 = datetime.date.fromisoformat(times[0])
    t = np.array([(datetime.date.fromisoformat(s) - t0).days for s in times], dtype=float)
    R = np.array(temps, dtype=float)
    # ACF of temperature anomalies
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 80)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0: acf = acf / acf[0]
    return t, acf

def fetch_usgs(url):
    raw = fetch(url).decode('utf-8')
    reader = csv.DictReader(io.StringIO(raw))
    times = []
    for r in reader:
        t = r.get('time')
        if t: times.append(t)
    if len(times) < 20: return None
    import datetime
    hrs = []
    for tstr in times:
        try:
            dt = datetime.datetime.fromisoformat(tstr.replace('Z', '+00:00'))
            hrs.append(dt.timestamp() / 3600.0)
        except: continue
    if len(hrs) < 20: return None
    hrs.sort()
    h0 = math.floor(min(hrs))
    h_max = math.ceil(max(hrs))
    bins = np.arange(h0, h_max + 2)
    counts, _ = np.histogram(hrs, bins=bins)
    R = counts.astype(float)
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 80)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0: acf = acf / acf[0]
    return t, acf

def fetch_zenodo(url):
    """Try to fetch Zenodo CSV files. URL format: 10.5281/zenodo.XXXXX
    Try the Zenodo API to get the latest record files."""
    if not url.startswith('10.5281/zenodo.'):
        # Some have full URL
        if 'zenodo.org' in url:
            pass
        else:
            return None
    # Extract record ID
    import re
    m = re.search(r'zenodo\.(\d+)', url)
    if not m: return None
    rid = m.group(1)
    api_url = f'https://zenodo.org/api/records/{rid}'
    try:
        raw = fetch(api_url, timeout=15).decode('utf-8')
        j = json.loads(raw)
        files = j.get('files', [])
        # Pick a CSV file if available
        for f in files:
            key = f.get('key', '')
            if key.endswith('.csv'):
                csv_url = f.get('links', {}).get('self')
                if not csv_url:
                    continue
                csv_raw = fetch(csv_url, timeout=30).decode('utf-8', errors='ignore')
                # Parse CSV, take first numeric column
                reader = csv.reader(io.StringIO(csv_raw))
                rows = list(reader)
                # Skip header
                vals = []
                for row in rows[1:]:
                    for cell in row:
                        try:
                            v = float(cell)
                            vals.append(v)
                            break
                        except: continue
                if len(vals) < 30:
                    continue
                R = np.array(vals, dtype=float)
                # ACF
                R = R - R.mean()
                if np.std(R) > 0: R = R / np.std(R)
                n = len(R)
                max_lag = min(n - 1, 100)
                acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
                t = np.arange(max_lag, dtype=float)
                if acf[0] > 0: acf = acf / acf[0]
                return t, acf
    except Exception as e:
        return None
    return None

FETCHERS = {
    'api.worldbank.org': fetch_worldbank,
    'api.coingecko.com': fetch_coingecko,
    'api.binance.com': fetch_binance,
    'archive-api.open-meteo.com': fetch_openmeteo,
    'earthquake.usgs.gov': fetch_usgs,
    '10.5281/zenodo': fetch_zenodo,
    'doi.org': fetch_zenodo,  # Some URLs are DOI-style
}

def get_raw_curve(url):
    from urllib.parse import urlparse
    if not url: return None
    # Zenodo URLs often start with 10.5281 (no protocol)
    if url.startswith('10.5281/'):
        return fetch_zenodo(url)
    host = urlparse(url).netloc
    fetcher = FETCHERS.get(host)
    if not fetcher:
        # Try Zenodo DOI resolution
        if '10.5281' in url:
            return fetch_zenodo(url)
        return None
    try:
        return fetcher(url)
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────
# T7.1-A: Sliding-window S2 fit + breakpoint detection
# ─────────────────────────────────────────────────────────────────────

def fit_s2_one(t, R, p0_list, bounds=None, maxfev=20000):
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


def t71a_sliding_window_breakpoints(t, R, n_windows=10, overlap=0.6):
    """Sliding-window S2 fits. Return list of (window_center, D, lam_q)."""
    n = len(t)
    if n < n_windows * 5:
        return []
    step = max(int((1 - overlap) * n / n_windows), 1)
    win_size = int(n / n_windows) + step
    fits = []
    for i in range(n_windows):
        start = i * step
        end = min(start + win_size, n)
        if end - start < 8: continue
        tw = t[start:end]
        Rw = R[start:end]
        if Rw[0] > 0: Rw = Rw / Rw[0]
        tm = float(tw[len(tw)//2])
        f = fit_s2_one(tw, Rw,
            p0_list=[[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]],
            bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]))
        if f:
            fits.append({
                'center_t': float(np.mean(tw)),
                't_start': float(tw[0]), 't_end': float(tw[-1]),
                'A': float(f[0][0]), 'lambda_q': float(f[0][1]), 'D': float(f[0][2]),
                'rss': float(f[1]), 'n': end - start,
            })
    return fits


def detect_breakpoints(fits, jump_threshold=2.5):
    """Find t-values where (D, log lambda_q) jumps by more than threshold std."""
    if len(fits) < 4: return []
    Ds = np.array([f['D'] for f in fits])
    lams = np.array([np.log10(max(f['lambda_q'], 1e-6)) for f in fits])
    # Consecutive differences
    dD = np.abs(np.diff(Ds))
    dlam = np.abs(np.diff(lams))
    # Standardized jumps
    if dD.std() > 0: dD_z = dD / dD.std()
    else: dD_z = dD
    if dlam.std() > 0: dlam_z = dlam / dlam.std()
    else: dlam_z = dlam
    jump_score = np.sqrt(dD_z**2 + dlam_z**2)
    # Mark as breakpoint if jump > threshold
    bps = []
    for i, score in enumerate(jump_score):
        if score > jump_threshold:
            bps.append({
                't_break': (fits[i]['center_t'] + fits[i+1]['center_t']) / 2,
                'jump_score': float(score),
                'D_before': fits[i]['D'], 'D_after': fits[i+1]['D'],
                'lam_q_before': fits[i]['lambda_q'], 'lam_q_after': fits[i+1]['lambda_q'],
            })
    return bps


# ─────────────────────────────────────────────────────────────────────
# T7.1-B: Two-segment piecewise S2 with breakpoint search
# ─────────────────────────────────────────────────────────────────────

def piecewise_s2(t, A1, lam1, D1, A2, lam2, D2, t_break):
    """Piecewise S2: regime 1 for t<t_break, regime 2 for t>=t_break.
    Continuity not enforced — let A2 absorb the jump."""
    mask = t < t_break
    out = np.zeros_like(t, dtype=float)
    out[mask] = A1 * np.exp(-np.power(np.maximum(t[mask], 1e-6) / max(lam1, 1e-6), D1))
    out[~mask] = A2 * np.exp(-np.power(np.maximum(t[~mask], 1e-6) / max(lam2, 1e-6), D2))
    return out


def fit_piecewise_s2(t, R, n_break_candidates=11):
    """Search over candidate breakpoints, fit piecewise S2 at each.
    Return best (params, aicc, t_break). Reduced candidates for speed."""
    n = len(t)
    if n < 30: return None
    # Normalize R to start at 1
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    R_n = np.clip(R_n, 1e-6, None)
    # Candidate breakpoints: 20% to 80% of t-range
    t_min, t_max = t[0], t[-1]
    t_breaks = np.linspace(t_min + 0.2*(t_max-t_min), t_min + 0.8*(t_max-t_min), n_break_candidates)
    best = None
    for t_b in t_breaks:
        mask = t < t_b
        if mask.sum() < 5 or (~mask).sum() < 5: continue
        # Initial guesses for each regime
        tm1 = float(t[mask][len(t[mask])//2])
        tm2 = float(t[~mask][len(t[~mask])//2])
        # Use bounds wide enough to accommodate continuity
        bounds_lower = [0.001, 1e-2, 0.01, 0.001, 1e-2, 0.01]
        bounds_upper = [10.0, 1e6, 10.0, 10.0, 1e6, 10.0]
        # Per-regime S2 fits for initial guess (fit each regime separately)
        R1 = R_n[mask]; R2 = R_n[~mask]
        if R1[0] > 0: R1 = R1 / R1[0]
        if R2[0] > 0: R2 = R2 / R2[0]
        f1 = fit_s2_one(t[mask], R1,
            p0_list=[[1.0, tm1, 0.5], [1.0, tm1*0.5, 1.0], [1.0, tm1*2, 0.3]],
            bounds=([0.01, 1e-2, 0.01], [10.0, 1e6, 10.0]))
        f2 = fit_s2_one(t[~mask], R2,
            p0_list=[[1.0, tm2, 0.5], [1.0, tm2*0.5, 1.0], [1.0, tm2*2, 0.3]],
            bounds=([0.01, 1e-2, 0.01], [10.0, 1e6, 10.0]))
        if not f1 or not f2: continue
        A1_0, lam1_0, D1_0 = f1[0]
        A2_0, lam2_0, D2_0 = f2[0]
        # Adjust A2 to maintain continuity at t_break
        val_at_break_regime1 = A1_0 * np.exp(-np.power(t_b / max(lam1_0, 1e-6), D1_0))
        val_at_break_regime2 = A2_0 * np.exp(-np.power(t_b / max(lam2_0, 1e-6), D2_0))
        if val_at_break_regime2 > 0:
            A2_init = A2_0 * (val_at_break_regime1 / val_at_break_regime2)
        else:
            A2_init = A2_0
        # Clip p0 to bounds
        p0 = [max(bounds_lower[i], min(bounds_upper[i]-1e-9, v))
              for i, v in enumerate([A1_0, lam1_0, D1_0, A2_init, lam2_0, D2_0])]
        # Now do joint fit
        try:
            popt, _ = curve_fit(
                lambda tt, A1, lam1, D1, A2, lam2, D2: piecewise_s2(tt, A1, lam1, D1, A2, lam2, D2, t_b),
                t, R_n,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=5000
            )
            rss = float(np.sum((R_n - piecewise_s2(t, *popt, t_break=t_b))**2))
            n_p = 6
            if n - n_p - 1 > 0:
                aicc = n * np.log(rss/n) + 2*n_p + (2*n_p*(n_p+1))/(n-n_p-1)
            else:
                aicc = float('inf')
            if best is None or aicc < best[1]:
                best = (list(popt), aicc, float(t_b), rss)
        except Exception:
            continue
    return best


def fit_single_s2_aicc(t, R):
    """Single S2 fit, return AICc."""
    n = len(t)
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    tm = float(t[len(t)//2])
    f = fit_s2_one(t, R_n,
        p0_list=[[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]],
        bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]))
    if not f: return None
    rss = float(f[1])
    n_p = 3
    if n - n_p - 1 > 0:
        aicc = n * np.log(rss/n) + 2*n_p + (2*n_p*(n_p+1))/(n-n_p-1)
    else:
        aicc = float('inf')
    return {'params': list(f[0]), 'rss': rss, 'aicc': aicc, 'n_p': n_p}


# ─────────────────────────────────────────────────────────────────────
# T7.1-C: Information rate test — dR/dλ
# ─────────────────────────────────────────────────────────────────────

def info_rate_breakpoints(t, R, smooth_window=5):
    """Compute |dR/dt| locally (smoothed), find max-jump locations."""
    n = len(t)
    if n < 20: return []
    # Smooth R with rolling mean
    R_smooth = np.convolve(R, np.ones(smooth_window)/smooth_window, mode='valid')
    t_smooth = t[smooth_window-1:]
    # Compute |dR/dt|
    dR = np.abs(np.diff(R_smooth))
    dt = np.diff(t_smooth)
    # Avoid div by zero
    dt[dt == 0] = 1e-10
    rate = dR / dt
    # Find peaks in rate (information state changes)
    # A peak is where rate[i] > rate[i-1] and rate[i] > rate[i+1]
    peaks = []
    for i in range(1, len(rate) - 1):
        if rate[i] > rate[i-1] and rate[i] > rate[i+1] and rate[i] > 0.3 * rate.max():
            peaks.append({
                't_peak': float(t_smooth[i]),
                'rate': float(rate[i]),
                'rate_normalized': float(rate[i] / max(rate.max(), 1e-10)),
            })
    return peaks


# ─────────────────────────────────────────────────────────────────────
# Main: fetch real curves, run all three T7.1 tests
# ─────────────────────────────────────────────────────────────────────

with open(os.path.join(REPO, 'en/tests.json')) as f:
    tests = json.load(f)['tests']

# Select candidates: strongest S2_DUST wins + diversify by domain.
# ONLY use known-fetchable sources (WorldBank, Coingecko, Binance, Open-Meteo, USGS).
# Zenodo was tried but timed out — skip those for now.
FETCHABLE_HOSTS = {
    'api.worldbank.org', 'api.coingecko.com', 'api.binance.com',
    'archive-api.open-meteo.com', 'earthquake.usgs.gov',
}
def is_fetchable(url):
    if not url: return False
    if url.startswith('10.5281/'): return False  # skip Zenodo for this run
    from urllib.parse import urlparse
    return urlparse(url).netloc in FETCHABLE_HOSTS

candidates = []
for t in tests:
    ba = t.get('best_alt','')
    da = t.get('delta_aicc', None)
    mv = t.get('model_verdict','')
    if mv == 'S2_LOSES' and ba == 'S2_DUST' and da is not None and da >= 10 and is_fetchable(t.get('url','')):
        candidates.append((da, t))
candidates.sort(key=lambda x: -x[0])
print(f'Fetchable strong S2_DUST candidates: {len(candidates)}')

# Also pull in ALL fetchable S2_LOSES (regardless of best_alt) to maximize n
all_fetchable_loses = []
for t in tests:
    mv = t.get('model_verdict','')
    if mv == 'S2_LOSES' and is_fetchable(t.get('url','')):
        all_fetchable_loses.append(t)
print(f'All fetchable S2_LOSES entries: {len(all_fetchable_loses)}')

# Use ALL fetchable loses, plus all fetchable wins as control
all_fetchable_wins = []
for t in tests:
    mv = t.get('model_verdict','')
    if mv == 'S2_WINS' and is_fetchable(t.get('url','')):
        all_fetchable_wins.append(t)
print(f'All fetchable S2_WINS entries: {len(all_fetchable_wins)}')

# Combine all fetchable entries (any verdict) — gives the broadest real-data sample
all_fetchable = []
seen_urls = set()
for t in tests:
    if is_fetchable(t.get('url','')):
        if t['url'] in seen_urls: continue
        seen_urls.add(t['url'])
        all_fetchable.append(t)
print(f'Total unique fetchable entries: {len(all_fetchable)}')

# Build target list with kind label
all_targets = []
for t in all_fetchable:
    mv = t.get('model_verdict','')
    ba = t.get('best_alt','')
    if mv == 'S2_LOSES' and ba == 'S2_DUST':
        kind = 'S2_DUST_WIN'
    elif mv == 'S2_WINS':
        kind = 'S2_WINS_CTRL'
    elif mv == 'S2_LOSES':
        kind = 'S2_LOSES_OTHER'
    else:
        kind = 'OTHER'
    da = t.get('delta_aicc', 0) or 0
    all_targets.append((kind, da, t))
print(f'Total targets: {len(all_targets)}')

# ─────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('T7.1 MULTI-SEGMENT S2 HYPOTHESIS TEST (real fetched curves)')
print('='*72)
print()
print('Hypothesis: A single S2 law is the local retention law, but the')
print('kernel enters different regimes at reproducible scales. The global')
print('curve is a piecewise composition of S2 regimes:')
print('  R(λ) ≈ S2_1 → S2_2 → S2_3 ...')
print()
print('Tests:')
print('  T7.1-A: Sliding-window S2 + breakpoint detection')
print('  T7.1-B: Two-segment piecewise S2 fit, search for optimal t_break')
print('  T7.1-C: Information rate dR/dt peaks (state changes)')
print('  T7.1-D: Do transition scales cluster across datasets?')
print()

results = []
import sys as _sys
for i, (kind, da, t) in enumerate(all_targets):
    name = t.get('name', '')[:55]
    url = t.get('url', '')
    dom = t.get('domain', '')
    print(f'\n[{i+1}/{len(all_targets)}] ({kind}) ΔAICc={da:>7.1f} [{dom}]', flush=True)
    print(f'  {name}', flush=True)
    print(f'  URL: {url[:90]}', flush=True)

    curve = get_raw_curve(url)
    if curve is None:
        print(f'  FETCH failed — skip')
        continue
    t_arr, R_arr = curve
    if len(t_arr) < 30:
        print(f'  Too short: n={len(t_arr)}')
        continue
    print(f'  Fetched n={len(t_arr)}  t=[{t_arr[0]:.2f},{t_arr[-1]:.2f}]')

    # T7.1-A: Sliding window
    fits = t71a_sliding_window_breakpoints(t_arr, R_arr, n_windows=10, overlap=0.6)
    if len(fits) < 4:
        print(f'  Sliding-window: only {len(fits)} windows — skip')
        continue
    bps_a = detect_breakpoints(fits, jump_threshold=2.5)
    print(f'  T7.1-A: {len(bps_a)} breakpoints detected')
    for b in bps_a[:5]:
        print(f"    t_break={b['t_break']:.3f}  D:{b['D_before']:.3f}→{b['D_after']:.3f}  jump_score={b['jump_score']:.2f}")

    # T7.1-B: Two-segment piecewise fit
    single = fit_single_s2_aicc(t_arr, R_arr)
    pw = fit_piecewise_s2(t_arr, R_arr, n_break_candidates=15)
    if pw and single:
        delta_aicc_piecewise = single['aicc'] - pw[1]  # positive = piecewise better
        print(f"  T7.1-B: single AICc={single['aicc']:.1f}  piecewise AICc={pw[1]:.1f}  ΔAICc={delta_aicc_piecewise:.1f}")
        print(f"    piecewise params: A1={pw[0][0]:.2f}, lam1={pw[0][1]:.2f}, D1={pw[0][2]:.3f}")
        print(f"                    A2={pw[0][3]:.2f}, lam2={pw[0][4]:.2f}, D2={pw[0][5]:.3f}")
        print(f"    t_break = {pw[2]:.3f}")
    else:
        delta_aicc_piecewise = None
        pw = None
        print(f'  T7.1-B: piecewise fit failed')

    # T7.1-C: Information rate
    peaks = info_rate_breakpoints(t_arr, R_arr, smooth_window=5)
    print(f'  T7.1-C: {len(peaks)} info-rate peaks')
    for p in peaks[:3]:
        print(f"    t_peak={p['t_peak']:.3f}  rate_norm={p['rate_normalized']:.3f}")

    # Store
    results.append({
        'kind': kind,
        'delta_aicc_dust': float(da),
        'name': t.get('name', ''),
        'domain': dom,
        'url': url,
        'n': int(len(t_arr)),
        't_range': [float(t_arr[0]), float(t_arr[-1])],
        't71a_breakpoints': bps_a,
        't71a_n_windows': len(fits),
        't71a_window_fits': fits,
        't71b_piecewise_aicc': float(pw[1]) if pw else None,
        't71b_single_aicc': float(single['aicc']) if single else None,
        't71b_delta_aicc': float(delta_aicc_piecewise) if delta_aicc_piecewise is not None else None,
        't71b_t_break': float(pw[2]) if pw else None,
        't71b_params': list(pw[0]) if pw else None,
        't71c_info_peaks': peaks,
    })

# ─────────────────────────────────────────────────────────────────────
# T7.1-D: Do transition scales cluster across datasets?
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('T7.1-D: CROSS-DATASET TRANSITION SCALE CLUSTERING')
print('='*72)

# Collect all breakpoints from T7.1-A and T7.1-B
bp_t_values = []
for r in results:
    for b in r.get('t71a_breakpoints', []):
        bp_t_values.append({
            'source': 'T71A',
            't_break': b['t_break'],
            'kind': r['kind'],
            'domain': r['domain'],
            'name': r['name'][:40],
            't_range': r['t_range'],
        })
    if r.get('t71b_t_break') is not None:
        bp_t_values.append({
            'source': 'T71B',
            't_break': r['t71b_t_break'],
            'kind': r['kind'],
            'domain': r['domain'],
            'name': r['name'][:40],
            't_range': r['t_range'],
        })

print(f'\nTotal breakpoints collected: {len(bp_t_values)}')
print(f'  From T7.1-A: {sum(1 for b in bp_t_values if b["source"]=="T71A")}')
print(f'  From T7.1-B: {sum(1 for b in bp_t_values if b["source"]=="T71B")}')

# Normalize breakpoints to fractional position in their t-range
# (absolute scales differ by dataset units — days vs seconds vs years)
for b in bp_t_values:
    t_min, t_max = b['t_range']
    t_span = max(t_max - t_min, 1e-10)
    b['frac_pos'] = (b['t_break'] - t_min) / t_span

# Test 1: Is the distribution of fractional breakpoints multimodal?
frac_positions = np.array([b['frac_pos'] for b in bp_t_values])
print(f'\nFractional breakpoint positions (normalized 0-1 within each dataset):')
print(f'  n = {len(frac_positions)}')
if len(frac_positions) >= 5:
    print(f'  mean={frac_positions.mean():.3f}  median={np.median(frac_positions):.3f}  std={frac_positions.std():.3f}')
    # Bin into 5 bins, see if any bin dominates
    hist, edges = np.histogram(frac_positions, bins=10)
    print(f'  Histogram (10 bins):')
    for i, h in enumerate(hist):
        bar = '#' * int(40 * h / max(hist.max(), 1))
        print(f'    [{edges[i]:.2f}, {edges[i+1]:.2f}): {h:>3d}  {bar}')
    # If multi-segment is reproducible, we'd see clusters at specific fractions
    # Silverman test for multimodality
    from scipy.stats import gaussian_kde
    if len(frac_positions) >= 8 and frac_positions.std() > 0:
        kde = gaussian_kde(frac_positions, bw_method=0.15)
        xs = np.linspace(0, 1, 200)
        ys = kde(xs)
        n_modes = sum(1 for i in range(1, len(ys)-1) if ys[i] > ys[i-1] and ys[i] > ys[i+1])
        print(f'  KDE modes: {n_modes}')
        print(f'  Mode locations: {[float(xs[i]) for i in range(1, len(ys)-1) if ys[i] > ys[i-1] and ys[i] > ys[i+1]]}')

# Test 2: T71B breakpoints — does piecewise improve fit, and does t_break cluster?
t71b_results = [r for r in results if r.get('t71b_delta_aicc') is not None]
print(f'\nT7.1-B piecewise fits: {len(t71b_results)} datasets')
piecewise_improvements = [r['t71b_delta_aicc'] for r in t71b_results]
n_improve = sum(1 for d in piecewise_improvements if d > 4)
print(f'  Piecewise strongly improves single-S2 (ΔAICc > 4): {n_improve}/{len(t71b_results)} ({100*n_improve/max(len(t71b_results),1):.0f}%)')
if piecewise_improvements:
    print(f'  Median ΔAICc improvement: {np.median(piecewise_improvements):.2f}')

# T71B t_break fractions
t71b_fracs = np.array([(r['t71b_t_break'] - r['t_range'][0]) / max(r['t_range'][1]-r['t_range'][0], 1e-10)
                       for r in t71b_results if r.get('t71b_t_break')])
print(f'\nT7.1-B t_break fractional positions: n={len(t71b_fracs)}')
if len(t71b_fracs) >= 5:
    print(f'  mean={t71b_fracs.mean():.3f}  median={np.median(t71b_fracs):.3f}  std={t71b_fracs.std():.3f}')
    print(f'  IQR=[{np.percentile(t71b_fracs,25):.3f}, {np.percentile(t71b_fracs,75):.3f}]')
    # If breakpoints cluster at a specific fraction, that's the signature
    if t71b_fracs.std() < 0.15:
        print(f'  -> Std < 0.15: BREAKPOINTS CLUSTER (reproducible transition scale)')
    else:
        print(f'  -> Spread is broad — no single characteristic transition fraction')

# ─────────────────────────────────────────────────────────────────────
# DUMP & SYNTHESIZE
# ─────────────────────────────────────────────────────────────────────
out = {
    'hypothesis': 'T7.1 — multi-segment S2: piecewise S2 regimes at reproducible scales',
    'n_target_curves': len(all_targets),
    'n_fetched_and_analyzed': len(results),
    'n_S2_DUST_WINS_group': sum(1 for r in results if r['kind'] == 'S2_DUST_WIN'),
    'n_S2_WINS_CTRL_group': sum(1 for r in results if r['kind'] == 'S2_WINS_CTRL'),
    't71a_total_breakpoints': sum(len(r['t71a_breakpoints']) for r in results),
    't71b_n_piecewise_improved': n_improve,
    't71b_median_delta_aicc': float(np.median(piecewise_improvements)) if piecewise_improvements else None,
    't71b_t_break_fractions': {
        'n': int(len(t71b_fracs)),
        'mean': float(t71b_fracs.mean()) if len(t71b_fracs) else None,
        'median': float(np.median(t71b_fracs)) if len(t71b_fracs) else None,
        'std': float(t71b_fracs.std()) if len(t71b_fracs) else None,
    },
    'results': results,
}
out_path = os.path.join(OUT_DIR, 't71_multi_segment_s2.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')

# ─────────────────────────────────────────────────────────────────────
# Honest verdict
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('T7.1 HONEST VERDICT')
print('='*72)
print()
print('Two pieces of evidence are needed to support multi-segment S2:')
print('  (1) Piecewise S2 fits must strongly beat single-S2 on real curves.')
print('  (2) Transition scales must CLUSTER at reproducible values.')
print()
print(f'(1) Piecewise improvement: {n_improve}/{len(t71b_results)} datasets show ΔAICc>4 improvement')
print(f'    Median ΔAICc improvement: {np.median(piecewise_improvements):.2f}' if piecewise_improvements else '    (no data)')
print(f'(2) t_break fractions: n={len(t71b_fracs)}, std={t71b_fracs.std():.3f}' if len(t71b_fracs) else '    (no data)')

# Final verdict
if piecewise_improvements and len(t71b_fracs) >= 5:
    median_improvement = np.median(piecewise_improvements)
    frac_std = t71b_fracs.std()
    if median_improvement > 4 and frac_std < 0.15:
        print('\n>>> T7.1 SUPPORTED: piecewise S2 improves AND transitions cluster.')
        print('>>> Multi-segment S2 is a real refinement of the S2 law.')
    elif median_improvement > 4 and frac_std >= 0.15:
        print('\n>>> T7.1 PARTIALLY SUPPORTED: piecewise improves, but transitions do NOT cluster.')
        print('>>> Multi-segment S2 is fitting noise, not a real regime structure.')
    elif median_improvement <= 4:
        print('\n>>> T7.1 NOT SUPPORTED: piecewise S2 does not consistently beat single-S2.')
        print('>>> Single-S2 remains the best simple model on real data.')
