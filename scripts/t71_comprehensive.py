#!/usr/bin/env python3
"""
T7.1 COMPREHENSIVE EXPANDED FETCH
==================================

Hits as many unique sources as possible:
  - All 10 CoinGecko symbols
  - All 8 Binance symbols
  - All 8 energy-charts endpoints
  - All 6 NASA power endpoints
  - All 5 World Bank indicators
  - All 5 GitHub CSVs
  - All 3 NOAA tide stations
  - All 3 USGS water services
  - All 3 NDBC buoys
  - All 3 SWPC sunspot endpoints
  - Wikipedia pageviews (try with different user agent)
  - Up to 30 unique Zenodo records (smaller CSVs first)

Real data only.
"""
import json, os, ssl, urllib.request, csv, io, time, sys, signal, re
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_model_compare import m_s2

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=15):
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T71D/1.0 (research)'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 1: raise
            time.sleep(1)


def to_acf(R, max_lag=80):
    """Convert a 1D series to its autocorrelation function."""
    R = np.asarray(R, dtype=float)
    if len(R) < 30: return None
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, max_lag)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0: acf = acf / acf[0]
    return t, acf


# ─────────────────────────────────────────────────────────────────────
# Per-source fetchers
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
    if len(pairs) < 30: return None
    t = np.array([p[0] for p in pairs], dtype=float)
    R = np.array([p[1] for p in pairs], dtype=float)
    # ACF of detrended log
    R = np.log(R + 1e-10)
    return to_acf(R)


def fetch_coingecko(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    prices = j.get('prices', [])
    if len(prices) < 30: return None
    R = np.array([p[1] for p in prices], dtype=float)
    R = np.diff(np.log(R + 1e-10))
    return to_acf(R)


def fetch_binance(url):
    raw = fetch(url).decode('utf-8')
    rows = json.loads(raw)
    if len(rows) < 30: return None
    R = np.array([float(r[4]) for r in rows])
    R = np.diff(np.log(R + 1e-10))
    return to_acf(R)


def fetch_openmeteo(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    daily = j.get('daily', {})
    times = daily.get('time', [])
    if not times: return None
    # Try multiple variables
    for var in ['temperature_2m_mean', 'wind_speed_10m_max', 'precipitation_sum']:
        vals = daily.get(var, [])
        if len(vals) < 30: continue
        try:
            R = np.array(vals, dtype=float)
            return to_acf(R)
        except: continue
    return None


def fetch_usgs_quakes(url):
    raw = fetch(url).decode('utf-8')
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
    if len(hrs) < 30: return None
    hrs.sort()
    h0 = math.floor(min(hrs))
    h_max = math.ceil(max(hrs))
    bins = np.arange(h0, h_max + 2)
    counts, _ = np.histogram(hrs, bins=bins)
    return to_acf(counts.astype(float))


def fetch_usgs_water(url):
    """USGS water services — returns CSV."""
    try:
        raw = fetch(url, timeout=20).decode('utf-8')
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        # Find a numeric column
        if len(rows) < 30: return None
        vals = []
        for row in rows[1:]:
            for cell in row:
                try:
                    v = float(cell)
                    if v == v:
                        vals.append(v)
                        break
                except: continue
        if len(vals) < 30: return None
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_energy_charts(url):
    """energy-charts.info — returns CSV or JSON."""
    try:
        raw = fetch(url).decode('utf-8')
        # Try JSON first
        try:
            j = json.loads(raw)
            # Typically has 'array' or list of arrays
            if isinstance(j, list):
                # Find longest numeric series
                best = None
                for arr in j:
                    if isinstance(arr, list):
                        vals = []
                        for el in arr:
                            if isinstance(el, dict):
                                for v in el.values():
                                    try:
                                        vals.append(float(v))
                                    except: continue
                            else:
                                try: vals.append(float(el))
                                except: continue
                        if len(vals) > (best[1] if best else 0):
                            best = (vals, len(vals))
                if best and best[1] >= 30:
                    return to_acf(np.array(best[0]))
        except Exception:
            pass
        # Try CSV
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if len(rows) < 30: return None
        vals = []
        for row in rows:
            for cell in row:
                try:
                    v = float(cell)
                    if v == v: vals.append(v); break
                except: continue
        if len(vals) < 30: return None
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_nasa_power(url):
    """NASA POWER — returns JSON or CSV."""
    try:
        raw = fetch(url, timeout=20).decode('utf-8')
        # Try JSON
        try:
            j = json.loads(raw)
            props = j.get('properties', {}).get('parameter', {})
            for key, series in props.items():
                vals = []
                for date, v in series.items():
                    try:
                        v = float(v)
                        if v != -999:  # NASA's missing value code
                            vals.append(v)
                    except: continue
                if len(vals) >= 30:
                    return to_acf(np.array(vals))
        except Exception:
            pass
        # Try CSV
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if len(rows) < 30: return None
        vals = []
        for row in rows[5:]:  # skip header
            for cell in row:
                try:
                    v = float(cell)
                    if v == v and v > -900:
                        vals.append(v)
                        break
                except: continue
        if len(vals) < 30: return None
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_noaa_tides(url):
    """NOAA tides & currents — returns CSV."""
    try:
        raw = fetch(url, timeout=20).decode('utf-8')
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if len(rows) < 30: return None
        vals = []
        for row in rows:
            for cell in row:
                try:
                    v = float(cell)
                    if v == v: vals.append(v); break
                except: continue
        if len(vals) < 30: return None
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_ndbc(url):
    """NDBC buoy — text format."""
    try:
        raw = fetch(url, timeout=20).decode('utf-8', errors='ignore')
        lines = raw.splitlines()
        # NDBC typically has header lines then column data
        vals = []
        for line in lines:
            parts = line.split()
            # Try columns until we find numeric
            for cell in parts:
                try:
                    v = float(cell)
                    if v == v and abs(v) < 1000:
                        vals.append(v); break
                except: continue
        if len(vals) < 30: return None
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_swpc(url):
    """SWPC solar data — JSON."""
    try:
        raw = fetch(url, timeout=15).decode('utf-8')
        j = json.loads(raw)
        # Could be list of dicts
        if isinstance(j, list):
            vals = []
            for el in j:
                if isinstance(el, dict):
                    for k, v in el.items():
                        try:
                            vals.append(float(v)); break
                        except: continue
            if len(vals) >= 30: return to_acf(np.array(vals))
        elif isinstance(j, dict):
            for k, v in j.items():
                if isinstance(v, list):
                    vals = []
                    for el in v:
                        try: vals.append(float(el))
                        except: continue
                    if len(vals) >= 30: return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_github_csv(url):
    """GitHub raw CSV."""
    try:
        raw = fetch(url, timeout=20).decode('utf-8', errors='ignore')
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if len(rows) < 30: return None
        # Find a numeric column (most rows parse)
        best_col = -1
        best_count = 0
        for c in range(len(rows[0])):
            count = 0
            for row in rows[1:]:
                if c < len(row):
                    try:
                        v = float(row[c])
                        if v == v: count += 1
                    except: continue
            if count > best_count:
                best_count = count
                best_col = c
        if best_count < 30: return None
        vals = []
        for row in rows[1:]:
            if best_col < len(row):
                try:
                    v = float(row[best_col])
                    if v == v: vals.append(v)
                except: continue
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_wikimedia(url):
    """Wikipedia pageviews — JSON."""
    try:
        raw = fetch(url, timeout=15).decode('utf-8')
        j = json.loads(raw)
        items = j.get('items', [])
        vals = []
        for item in items:
            views = item.get('views', 0)
            try: vals.append(float(views))
            except: continue
        if len(vals) < 30: return None
        return to_acf(np.array(vals))
    except Exception:
        return None


def fetch_zenodo_size_sorted(url):
    """Zenodo: API call to find CSV files, take smallest <5MB."""
    m = re.search(r'zenodo\.(\d+)', url)
    if not m: return None
    rid = m.group(1)
    api_url = f'https://zenodo.org/api/records/{rid}'
    try:
        raw = fetch(api_url, timeout=10).decode('utf-8')
        j = json.loads(raw)
        files = j.get('files', [])
        candidates = []
        for f in files:
            key = f.get('key', '')
            size = f.get('size', 0)
            if key.endswith('.csv') and 1000 < size < 5_000_000:
                candidates.append((size, f))
        candidates.sort()
        for _, f in candidates[:3]:
            csv_url = f.get('links', {}).get('self')
            if not csv_url: continue
            try:
                csv_raw = fetch(csv_url, timeout=15).decode('utf-8', errors='ignore')
                reader = csv.reader(io.StringIO(csv_raw))
                rows = list(reader)
                if len(rows) < 30: continue
                vals = []
                for row in rows[1:]:
                    for cell in row:
                        try:
                            v = float(cell)
                            if v == v: vals.append(v); break
                        except: continue
                if len(vals) < 30: continue
                return to_acf(np.array(vals))
            except Exception: continue
    except Exception:
        return None


FETCHERS = {
    'api.worldbank.org': fetch_worldbank,
    'api.coingecko.com': fetch_coingecko,
    'api.binance.com': fetch_binance,
    'archive-api.open-meteo.com': fetch_openmeteo,
    'earthquake.usgs.gov': fetch_usgs_quakes,
    'waterservices.usgs.gov': fetch_usgs_water,
    'api.energy-charts.info': fetch_energy_charts,
    'power.larc.nasa.gov': fetch_nasa_power,
    'api.tidesandcurrents.noaa.gov': fetch_noaa_tides,
    'www.ndbc.noaa.gov': fetch_ndbc,
    'services.swpc.noaa.gov': fetch_swpc,
    'raw.githubusercontent.com': fetch_github_csv,
    'wikimedia.org': fetch_wikimedia,
}


def get_raw_curve(url):
    if not url: return None
    if url.startswith('10.5281/zenodo'):
        return fetch_zenodo_size_sorted(url)
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    fetcher = FETCHERS.get(host)
    if not fetcher: return None
    try: return fetcher(url)
    except Exception: return None


# ─────────────────────────────────────────────────────────────────────
# S2 fits (same as before)
# ─────────────────────────────────────────────────────────────────────

class TimeoutError_(Exception): pass
def _timeout_handler(signum, frame):
    raise TimeoutError_('fit timeout')

class FetchTimeout(Exception): pass
def _fetch_timeout_handler(signum, frame):
    raise FetchTimeout('fetch timeout')


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


def piecewise_s2(t, A1, lam1, D1, A2, lam2, D2, t_break):
    mask = t < t_break
    out = np.zeros_like(t, dtype=float)
    out[mask] = A1 * np.exp(-np.power(np.maximum(t[mask], 1e-6) / max(lam1, 1e-6), D1))
    out[~mask] = A2 * np.exp(-np.power(np.maximum(t[~mask], 1e-6) / max(lam2, 1e-6), D2))
    return out


def aicc(rss, n, k):
    if n - k - 1 <= 0: return float('inf')
    return n * np.log(rss/n) + 2*k + (2*k*(k+1))/(n-k-1)


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
    """Piecewise S2 fit. NOTE: per_fit_timeout is now ignored; relies on outer SIGALRM."""
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
                t, R_n, p0=p0, bounds=(bounds_lower, bounds_upper), maxfev=800
            )
            rss = float(np.sum((R_n - piecewise_s2(t, *popt, t_break=t_b))**2))
            a = aicc(rss, n, 6)
            if best is None or a < best[1]:
                best = (list(popt), a, float(t_b), rss)
        except Exception:
            continue
    return best


def sliding_window_fits(t, R, n_windows=10, overlap=0.6):
    n = len(t)
    if n < n_windows * 5: return []
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
            bounds=([0.01, 1e-2, 0.01], [2.0, 1e6, 10.0]))
        if f:
            fits.append({
                'center_t': float(np.mean(tw)),
                't_start': float(tw[0]), 't_end': float(tw[-1]),
                'A': float(f[0][0]), 'lambda_q': float(f[0][1]), 'D': float(f[0][2]),
                'rss': float(f[1]), 'n': end - start,
            })
    return fits


def detect_breakpoints(fits, jump_threshold=2.5):
    if len(fits) < 4: return []
    Ds = np.array([f['D'] for f in fits])
    lams = np.array([np.log10(max(f['lambda_q'], 1e-6)) for f in fits])
    dD = np.abs(np.diff(Ds))
    dlam = np.abs(np.diff(lams))
    if dD.std() > 0: dD_z = dD / dD.std()
    else: dD_z = dD * 0
    if dlam.std() > 0: dlam_z = dlam / dlam.std()
    else: dlam_z = dlam * 0
    jump_score = np.sqrt(dD_z**2 + dlam_z**2)
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
# Main
# ─────────────────────────────────────────────────────────────────────

import math

with open(os.path.join(REPO, 'en/tests.json')) as f:
    tests = json.load(f)['tests']

# Filter to entries with fetchable URLs
fetchable_tests = []
seen_urls = set()
for t in tests:
    url = t.get('url', '')
    if not url: continue
    if url in seen_urls: continue
    if url.startswith('10.5281/zenodo') or 'zenodo' in url:
        # Zenodo: cap at 30 to keep runtime bounded
        if sum(1 for f in fetchable_tests if f['url'].startswith('10.5281')) >= 30:
            continue
    seen_urls.add(url)
    fetchable_tests.append(t)

print(f'Total unique fetchable entries: {len(fetchable_tests)}')

# Take top 50 S2_DUST_WIN + 30 strongest S2_WINS for control
def get_score(t):
    if t.get('model_verdict') == 'S2_LOSES' and t.get('best_alt') == 'S2_DUST':
        return -t.get('delta_aicc', 0) if t.get('delta_aicc') else 0  # higher delta_aicc = stronger S2_DUST win
    elif t.get('model_verdict') == 'S2_WINS':
        return t.get('delta_aicc', 0) if t.get('delta_aicc') else 0  # negative = stronger S2 win
    return 0

s2_dust_wins = [t for t in fetchable_tests if t.get('model_verdict') == 'S2_LOSES' and t.get('best_alt') == 'S2_DUST']
s2_dust_wins.sort(key=lambda t: -(t.get('delta_aicc') or 0))
top_dust = s2_dust_wins[:50]

s2_wins = [t for t in fetchable_tests if t.get('model_verdict') == 'S2_WINS']
s2_wins.sort(key=lambda t: (t.get('delta_aicc') or 0))  # most negative = strongest S2 win
top_wins = s2_wins[:30]

# Add other categories to diversify
others = [t for t in fetchable_tests if t not in top_dust and t not in top_wins][:30]

all_targets = top_dust + top_wins + others
print(f'Targets: {len(top_dust)} S2_DUST_WIN + {len(top_wins)} S2_WINS + {len(others)} other = {len(all_targets)}')

results = []
for i, t in enumerate(all_targets):
    name = t.get('name', '')[:55]
    url = t.get('url', '')
    dom = t.get('domain', '')
    mv = t.get('model_verdict', '')
    da = t.get('delta_aicc', 0) or 0
    print(f'\n[{i+1}/{len(all_targets)}] [{dom}] mv={mv} ΔAICc={da:>7.1f}', flush=True)
    print(f'  {name}', flush=True)
    print(f'  URL: {url[:80]}', flush=True)

    try:
        # Wrap entire fetch + analysis in 25s timeout
        old_h = signal.signal(signal.SIGALRM, _fetch_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, 25.0)
        curve = get_raw_curve(url)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_h)
    except FetchTimeout:
        print(f'  FETCH TIMEOUT (25s)', flush=True)
        continue
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(f'  FETCH ERROR: {repr(e)[:80]}', flush=True)
        continue
    if curve is None:
        print(f'  FETCH returned None', flush=True)
        continue
    t_arr, R_arr = curve
    if len(t_arr) < 30:
        print(f'  Too short: n={len(t_arr)}', flush=True)
        continue
    print(f'  n={len(t_arr)}  t=[{t_arr[0]:.2f},{t_arr[-1]:.2f}]', flush=True)

    # T7.1-A and T7.1-B wrapped with overall 60s timeout
    try:
        old_h2 = signal.signal(signal.SIGALRM, _fetch_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, 60.0)
        # T7.1-A
        wins = sliding_window_fits(t_arr, R_arr, n_windows=10, overlap=0.6)
        bps_a = detect_breakpoints(wins, jump_threshold=2.5) if len(wins) >= 4 else []
        # T7.1-B
        single = fit_single_s2(t_arr, R_arr)
        pw = fit_piecewise_s2(t_arr, R_arr, n_break_candidates=5, per_fit_timeout=2.0)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_h2)
    except FetchTimeout:
        print(f'  ANALYSIS TIMEOUT (60s)', flush=True)
        continue
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(f'  ANALYSIS ERROR: {repr(e)[:80]}', flush=True)
        continue
    if pw and single:
        delta = single['aicc'] - pw[1]
        print(f"  T7.1-B: Δ={delta:.1f}  D1={pw[0][2]:.2f}→D2={pw[0][5]:.2f}  t_break={pw[2]:.1f}", flush=True)
    else:
        delta = None
        print(f'  T7.1-B: failed', flush=True)

    results.append({
        'name': t.get('name',''),
        'domain': dom,
        'url': url,
        'n': int(len(t_arr)),
        't_range': [float(t_arr[0]), float(t_arr[-1])],
        'model_verdict': mv,
        'delta_aicc_dust': da,
        't71a_breakpoints_count': len(bps_a),
        't71a_n_windows': len(wins),
        't71b_single_aicc': float(single['aicc']) if single else None,
        't71b_piecewise_aicc': float(pw[1]) if pw else None,
        't71b_delta_aicc': float(delta) if delta is not None else None,
        't71b_t_break': float(pw[2]) if pw else None,
        't71b_D1': float(pw[0][2]) if pw else None,
        't71b_D2': float(pw[0][5]) if pw else None,
        't71b_lam1': float(pw[0][1]) if pw else None,
        't71b_lam2': float(pw[0][4]) if pw else None,
    })

print(f'\nTotal fetched & analyzed: {len(results)}')

# Save
out = {
    'probe': 'T7.1 comprehensive expanded fetch',
    'n_targets': len(all_targets),
    'n_fetched': len(results),
    'results': results,
}
out_path = os.path.join(OUT_DIR, 't71_comprehensive.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
