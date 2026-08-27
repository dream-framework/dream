#!/usr/bin/env python3
"""
T7 PROBE A — Drift vs Mixture Discriminator (real raw curves)
================================================================

Fetches REAL raw (t, R(t)) curves from public URLs in the registry,
then for each:

  1. Fit single-S2 and S2_DUST (2-component S2)
  2. Run sliding-window S2 fits along the t-axis
  3. Examine the trajectory {(D_w, lam_q_w)} in parameter space
  4. DISCRIMINATOR:
       - DRIFT (single complicated kernel): trajectory is smooth, 1D-like
         → PCA: 1st PC explains >80% of variance; trajectory length is
         small relative to bounding-box span
       - MIXTURE (two static kernels): trajectory visits two clusters
         → PCA: 1st PC explains <60% OR distance distribution between
         consecutive windows is bimodal

  5. K-means (k=2) on sliding-window params:
       - If both clusters are well-populated and well-separated → MIXTURE
       - If one cluster dominates → DRIFT
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_model_compare import m_s2, m_biexp, m_power, m_exp, m_s2_dust

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch(url, timeout=20):
    """Fetch a URL with retries."""
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T7A/1.0 (research)'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == 1:
                raise
            time.sleep(1)


# ─────────────────────────────────────────────────────────────────────
# Real raw data fetchers per source
# ─────────────────────────────────────────────────────────────────────

def fetch_worldbank(url):
    """World Bank API → time series."""
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    rows = j[1] if len(j) > 1 and isinstance(j[1], list) else []
    pairs = []
    for r in rows:
        try:
            v = float(r['value'])
            y = int(r['date'])
            if v != 0 and v == v:  # not nan, not zero
                pairs.append((y, v))
        except Exception:
            continue
    pairs.sort()
    if len(pairs) < 20:
        return None
    t = np.array([p[0] for p in pairs], dtype=float)
    R = np.array([p[1] for p in pairs], dtype=float)
    # Convert to "retention-like" — use cumulative or autocorrelation?
    # For GDP/CPI: use absolute value divided by max as "retention from peak"
    # Actually: better to use ACF. But for the test, we can use the
    # value normalized to start = 1.
    if R[0] != 0:
        R_norm = R / R[0]
    else:
        R_norm = R / max(abs(R))
    return t - t[0], R_norm

def fetch_coingecko(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    prices = j.get('prices', [])
    if len(prices) < 20:
        return None
    t = np.array([p[0] for p in prices], dtype=float) / 1000  # ms→s
    R = np.array([p[1] for p in prices], dtype=float)
    # Use absolute returns normalized — |R(t)/R(0)|
    if R[0] != 0:
        return t - t[0], R / R[0]
    return t - t[0], R / max(abs(R))

def fetch_binance(url):
    raw = fetch(url).decode('utf-8')
    rows = json.loads(raw)
    if len(rows) < 20:
        return None
    t = np.array([r[0] for r in rows], dtype=float) / 1000
    R = np.array([float(r[4]) for r in rows])  # close price
    if R[0] != 0:
        return t - t[0], R / R[0]
    return t - t[0], R / max(abs(R))

def fetch_openmeteo(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    daily = j.get('daily', {})
    times = daily.get('time', [])
    temps = daily.get('temperature_2m_mean', [])
    if len(times) < 20:
        return None
    # Convert dates to day indices
    import datetime
    t0 = datetime.date.fromisoformat(times[0])
    t = np.array([(datetime.date.fromisoformat(s) - t0).days for s in times], dtype=float)
    R = np.array(temps, dtype=float)
    # Use |R - mean| / max as "retention-like"
    R = np.abs(R - R.mean())
    if R.max() > 0:
        R = R / R.max()
    return t, R

def fetch_github_csv(url, col_idx=1):
    raw = fetch(url).decode('utf-8')
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    pairs = []
    for row in rows:
        if len(row) <= col_idx:
            continue
        try:
            v = float(row[col_idx])
            pairs.append(v)
        except Exception:
            continue
    if len(pairs) < 20:
        return None
    R = np.array(pairs, dtype=float)
    t = np.arange(len(R), dtype=float)
    # Use ACF as the retention curve (autocorrelation lag → R)
    # Compute ACF
    R = R - R.mean()
    if np.std(R) > 0:
        R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 100)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    # Only keep positive part
    if acf[0] > 0:
        acf = acf / acf[0]
    return t, acf

def fetch_giss(url):
    raw = fetch(url).decode('utf-8')
    lines = raw.splitlines()
    # NASA GISS format: header lines, then month rows
    # Skip first ~8 header lines
    pairs = []
    for line in lines[8:]:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
            for m, val_str in enumerate(parts[1:13]):
                if val_str == '***':
                    continue
                v = float(val_str)
                pairs.append((year + m/12.0, v))
        except Exception:
            continue
    if len(pairs) < 20:
        return None
    pairs.sort()
    t = np.array([p[0] for p in pairs])
    R = np.array([p[1] for p in pairs])
    # Use ACF
    R = R - R.mean()
    if np.std(R) > 0:
        R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 100)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0:
        acf = acf / acf[0]
    return t, acf

def fetch_usgs(url):
    raw = fetch(url).decode('utf-8')
    reader = csv.DictReader(io.StringIO(raw))
    times = []
    for r in reader:
        try:
            t = r.get('time')
            if t:
                times.append(t)
        except Exception:
            continue
    if len(times) < 20:
        return None
    # Use histogram-based counts per hour
    import datetime
    hrs = []
    for tstr in times:
        try:
            dt = datetime.datetime.fromisoformat(tstr.replace('Z', '+00:00'))
            hrs.append(dt.timestamp() / 3600.0)
        except Exception:
            continue
    if len(hrs) < 20:
        return None
    hrs.sort()
    # Bin into 1-hour bins
    h0 = math.floor(min(hrs))
    h_max = math.ceil(max(hrs))
    bins = np.arange(h0, h_max + 2)
    counts, _ = np.histogram(hrs, bins=bins)
    R = counts.astype(float)
    t = np.arange(len(R), dtype=float)
    # ACF
    R = R - R.mean()
    if np.std(R) > 0:
        R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 100)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0:
        acf = acf / acf[0]
    return t, acf


# Dispatcher
FETCHERS = {
    'api.worldbank.org': fetch_worldbank,
    'api.coingecko.com': fetch_coingecko,
    'api.binance.com': fetch_binance,
    'archive-api.open-meteo.com': fetch_openmeteo,
    'raw.githubusercontent.com': fetch_github_csv,
    'data.giss.nasa.gov': fetch_giss,
    'earthquake.usgs.gov': fetch_usgs,
}


def get_raw_curve(url):
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    fetcher = FETCHERS.get(host)
    if not fetcher:
        return None
    try:
        return fetcher(url)
    except Exception as e:
        print(f'  FETCH FAIL ({host}): {repr(e)[:80]}')
        return None


# ─────────────────────────────────────────────────────────────────────
# Sliding-window S2 fits
# ─────────────────────────────────────────────────────────────────────

def fit_s2_one(t, R, p0_list, bounds=None, maxfev=20000):
    best = None
    for p0 in p0_list:
        try:
            if bounds:
                popt, _ = curve_fit(m_s2, t, R, p0=p0, bounds=bounds, maxfev=maxfev)
            else:
                popt, _ = curve_fit(m_s2, t, R, p0=p0, maxfev=maxfev)
            rss = float(np.sum((R - m_s2(t, *popt)) ** 2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except Exception:
            continue
    return best


def sliding_window_fit(t, R, n_windows=5, overlap=0.5):
    """Fit S2 on overlapping windows along t-axis."""
    n = len(t)
    if n < n_windows * 5:
        return []
    step = max(int((1 - overlap) * n / n_windows), 1)
    win_size = int(n / n_windows) + step
    wins = []
    for i in range(n_windows):
        start = i * step
        end = min(start + win_size, n)
        if end - start < 8:
            continue
        tw = t[start:end]
        Rw = R[start:end]
        if Rw[0] > 0:
            Rw = Rw / Rw[0]
        tm = float(tw[len(tw)//2])
        f = fit_s2_one(tw, Rw,
            p0_list=[[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]],
            bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]))
        if f:
            wins.append({
                'start_idx': start, 'end_idx': end,
                't_start': float(tw[0]), 't_end': float(tw[-1]),
                'A': float(f[0][0]), 'lambda_q': float(f[0][1]), 'D': float(f[0][2]),
                'rss': float(f[1]),
                'n': end - start,
            })
    return wins


def drift_vs_mixture(wins):
    """Classify the trajectory as drift or mixture.

    DRIFT: smooth, 1D-like trajectory in (D, lam_q) space.
    MIXTURE: trajectory visits two distinct clusters.
    """
    if len(wins) < 4:
        return None

    params = np.array([[w['D'], np.log10(max(w['lambda_q'], 1e-6))] for w in wins])
    # Standardize
    sc = StandardScaler().fit(params)
    P = sc.transform(params)

    # PCA: how much variance in 1st PC?
    if len(P) < 2:
        return None
    pca = PCA(n_components=min(2, len(P)-1))
    pca.fit(P)
    pc1_var = float(pca.explained_variance_ratio_[0])

    # K-means with k=2: cluster sizes
    if len(P) < 4:
        return None
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(P)
    labels = km.labels_
    sizes = np.bincount(labels)
    cluster_balance = float(min(sizes) / max(sizes)) if max(sizes) > 0 else 0

    # Inter-window distances
    dists = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if len(dists) < 2:
        return None
    median_dist = float(np.median(dists))
    max_dist = float(dists.max())
    jump_ratio = max_dist / max(median_dist, 1e-6)

    # Verdict
    # DRIFT: high PC1 variance (1D trajectory), no clear cluster separation
    # MIXTURE: lower PC1 variance AND balanced clusters AND high jump_ratio
    is_mixture = (pc1_var < 0.85) and (cluster_balance > 0.25) and (jump_ratio > 3.0)

    return {
        'pc1_variance': pc1_var,
        'pc2_variance': float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0,
        'cluster_sizes': sizes.tolist(),
        'cluster_balance': cluster_balance,
        'jump_ratio': jump_ratio,
        'median_dist': median_dist,
        'max_dist': max_dist,
        'verdict': 'MIXTURE' if is_mixture else 'DRIFT',
    }


# ─────────────────────────────────────────────────────────────────────
# Main: fetch real curves, run drift-vs-mixture
# ─────────────────────────────────────────────────────────────────────

with open(os.path.join(REPO, 'en/tests.json')) as f:
    tests = json.load(f)['tests']

print('='*72)
print('PROBE A — DRIFT vs MIXTURE DISCRIMINATOR (real fetched curves)')
print('='*72)
print(f'\nRegistry: {len(tests)} entries')
print(f'Will fetch from: {list(FETCHERS.keys())}')

results = []
fetched = 0
analyzed = 0

# Group entries by domain for cross-correlation (probe B) later
domain_to_entries = {}
for t in tests:
    if not t.get('url'):
        continue
    from urllib.parse import urlparse
    host = urlparse(t['url']).netloc
    if host not in FETCHERS:
        continue
    d = t.get('domain', 'unknown')
    domain_to_entries.setdefault(d, []).append(t)

print(f'Fetchable domains: {len(domain_to_entries)}')
for d, lst in sorted(domain_to_entries.items(), key=lambda kv: -len(kv[1])):
    print(f'  {d:>20s}: {len(lst)} fetchable entries')

# Fetch one per domain first to maximize diversity, then add more
all_fetchable = []
for d, lst in domain_to_entries.items():
    for t in lst:
        all_fetchable.append((d, t))

# Cap at 30 to keep runtime reasonable; distribute by domain
by_dom_count = {}
selected = []
for d, t in all_fetchable:
    if by_dom_count.get(d, 0) >= 6:  # max 6 per domain
        continue
    by_dom_count[d] = by_dom_count.get(d, 0) + 1
    selected.append((d, t))
    if len(selected) >= 30:
        break

print(f'\nSelected {len(selected)} entries to fetch (capped 6/domain, max 30 total)')

for i, (dom, t) in enumerate(selected):
    name = t.get('name', '')[:50]
    url = t['url']
    print(f'\n[{i+1}/{len(selected)}] ({dom}) {name}')
    print(f'  URL: {url[:80]}')
    try:
        result = get_raw_curve(url)
    except Exception as e:
        print(f'  FETCH ERROR: {repr(e)[:120]}')
        continue
    if result is None:
        print(f'  FETCH returned no data')
        continue
    t_arr, R_arr = result
    if len(t_arr) < 20:
        print(f'  Too short: n={len(t_arr)}')
        continue
    fetched += 1
    print(f'  Fetched n={len(t_arr)}  t range [{t_arr[0]:.2f}, {t_arr[-1]:.2f}]  R range [{R_arr.min():.3f}, {R_arr.max():.3f}]')

    # Fit single S2 on full curve
    if R_arr[0] != 0:
        R_n = R_arr / R_arr[0]
    else:
        R_n = R_arr / max(abs(R_arr))
    tm = float(t_arr[len(t_arr)//2])
    f_single = fit_s2_one(t_arr, R_n,
        p0_list=[[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]],
        bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]))

    # Fit S2_DUST (2-component) on full curve
    f_dust = None
    try:
        f_dust, _ = curve_fit(m_s2_dust, t_arr, R_n,
            p0=[1.0, tm*0.3, 0.5, 0.3, tm*2, 1.5],
            bounds=([0, 1e-3, 0.01, 0, 1e-3, 0.01], [2, 1e6, 10, 2, 1e6, 10]),
            maxfev=30000)
        rss_dust = float(np.sum((R_n - m_s2_dust(t_arr, *f_dust))**2))
    except Exception:
        f_dust = None
        rss_dust = float('inf')

    # Sliding window
    wins = sliding_window_fit(t_arr, R_n, n_windows=6, overlap=0.4)
    if len(wins) < 4:
        print(f'  Sliding-window: only {len(wins)} windows fit — skipping')
        continue
    print(f'  Sliding-window fits: {len(wins)}')
    for w in wins:
        print(f'    t=[{w["t_start"]:.1f},{w["t_end"]:.1f}]  D={w["D"]:.3f}  lam_q={w["lambda_q"]:.3f}  rss={w["rss"]:.4g}')

    disc = drift_vs_mixture(wins)
    if disc is None:
        print(f'  Discriminator: insufficient windows')
        continue
    analyzed += 1
    print(f'  DISCRIMINATOR verdict: {disc["verdict"]}')
    print(f'    PC1 var={disc["pc1_variance"]:.3f}  PC2 var={disc["pc2_variance"]:.3f}')
    print(f'    cluster_balance={disc["cluster_balance"]:.3f}  jump_ratio={disc["jump_ratio"]:.2f}')

    # Single-S2 params
    if f_single:
        A_s, lam_s, D_s = f_single[0]
        rss_s = f_single[1]
        n = len(t_arr)
        k_s, k_d = 3, 6
        # AICc
        def aicc(rss, n, k):
            return n*np.log(rss/n) + 2*k + (2*k*(k+1))/(n-k-1)
        aicc_s = aicc(rss_s, n, k_s)
        aicc_d = aicc(rss_dust, n, k_d) if rss_dust != float('inf') else float('inf')
        delta = aicc_s - aicc_d
        print(f'  Single-S2: D={D_s:.3f}  AICc={aicc_s:.2f}')
        print(f'  S2_DUST:   AICc={aicc_d:.2f}  Δ={delta:.2f}')
    else:
        D_s = lam_s = A_s = None
        rss_s = None
        delta = None

    results.append({
        'name': t.get('name', ''),
        'domain': dom,
        'url': url,
        'n': int(len(t_arr)),
        'single_S2': {'A': float(A_s) if A_s else None,
                      'lambda_q': float(lam_s) if lam_s else None,
                      'D': float(D_s) if D_s else None,
                      'rss': float(rss_s) if rss_s else None},
        's2_dust_delta_aicc': float(delta) if delta is not None else None,
        'sliding_windows': wins,
        'discriminator': disc,
    })

# ─────────────────────────────────────────────────────────────────────
# Aggregate verdict
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('PROBE A — AGGREGATE VERDICT')
print('='*72)
print(f'Fetched: {fetched}/{len(selected)}')
print(f'Analyzed (with discriminator): {analyzed}/{fetched}')

mixture_count = sum(1 for r in results if r['discriminator']['verdict'] == 'MIXTURE')
drift_count = sum(1 for r in results if r['discriminator']['verdict'] == 'DRIFT')
print(f'\nMIXTURE verdict: {mixture_count}/{analyzed} ({100*mixture_count/max(analyzed,1):.0f}%)')
print(f'DRIFT verdict:   {drift_count}/{analyzed} ({100*drift_count/max(analyzed,1):.0f}%)')

print(f'\nBy domain:')
by_dom = {}
for r in results:
    by_dom.setdefault(r['domain'], {'mixture': 0, 'drift': 0})
    if r['discriminator']['verdict'] == 'MIXTURE':
        by_dom[r['domain']]['mixture'] += 1
    else:
        by_dom[r['domain']]['drift'] += 1
for d, m in sorted(by_dom.items(), key=lambda kv: -kv[1]['mixture']-kv[1]['drift']):
    print(f'  {d:>20s}: mixture={m["mixture"]}  drift={m["drift"]}')

# Save
out_path = os.path.join(OUT_DIR, 't7_probe_A_drift_mixture.json')
with open(out_path, 'w') as f:
    json.dump({
        'probe': 'A — drift vs mixture discriminator',
        'n_attempted': len(selected),
        'n_fetched': fetched,
        'n_analyzed': analyzed,
        'n_mixture': mixture_count,
        'n_drift': drift_count,
        'pct_mixture': 100*mixture_count/max(analyzed,1),
        'results': results,
    }, f, indent=2)
print(f'\nSaved: {out_path}')
