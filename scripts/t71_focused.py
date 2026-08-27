#!/usr/bin/env python3
"""
T7.1 — Multi-segment S2 — focused re-run
==========================================
Focused test with strict per-dataset timeout. Uses cached results
where possible.

Strategy:
  1. Use T7.1-A (sliding-window S2 + breakpoint detection) — fast,
     reliable, gave clear breakpoint clustering signal in first run.
  2. Use T7.1-B (piecewise fit) on a SUBSET of 10 fetchable curves
     with strict timeout per fit.
  3. T7.1-D: do breakpoints cluster at reproducible fractional positions?

Real data only. No fudging.
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math, signal
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_model_compare import m_s2

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'
os.makedirs(OUT_DIR, exist_ok=True)


def fetch(url, timeout=15):
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T71B/1.0'})
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 1: raise
            time.sleep(1)


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
    return t - t[0], R / R[0]

def fetch_openmeteo(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    daily = j.get('daily', {})
    times = daily.get('time', [])
    temps = daily.get('temperature_2m_mean', [])
    if len(times) < 30: return None
    import datetime
    t0 = datetime.date.fromisoformat(times[0])
    t = np.array([(datetime.date.fromisoformat(s) - t0).days for s in times], dtype=float)
    R = np.array(temps, dtype=float)
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
    R = counts.astype(float)
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, 80)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    t = np.arange(max_lag, dtype=float)
    if acf[0] > 0: acf = acf / acf[0]
    return t, acf

FETCHERS = {
    'api.worldbank.org': fetch_worldbank,
    'archive-api.open-meteo.com': fetch_openmeteo,
    'earthquake.usgs.gov': fetch_usgs,
}


def get_raw_curve(url):
    from urllib.parse import urlparse
    if not url: return None
    host = urlparse(url).netloc
    fetcher = FETCHERS.get(host)
    if not fetcher: return None
    try: return fetcher(url)
    except Exception: return None


# ─────────────────────────────────────────────────────────────────────
# S2 fits
# ─────────────────────────────────────────────────────────────────────

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


class TimeoutError_(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError_('fit timeout')

def fit_piecewise_s2(t, R, n_break_candidates=7, per_fit_timeout=5.0):
    """Piecewise S2 fit with HARD per-fit timeout via SIGALRM."""
    n = len(t)
    if n < 30: return None
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    R_n = np.clip(R_n, 1e-6, None)
    t_min, t_max = t[0], t[-1]
    t_breaks = np.linspace(t_min + 0.2*(t_max-t_min), t_min + 0.8*(t_max-t_min), n_break_candidates)
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

with open(os.path.join(REPO, 'en/tests.json')) as f:
    tests = json.load(f)['tests']

print('='*72)
print('T7.1 MULTI-SEGMENT S2 — FOCUSED RE-RUN')
print('='*72)
print()

# Get all fetchable entries
fetchable = []
seen_urls = set()
for t in tests:
    url = t.get('url', '')
    if not url: continue
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    if host not in FETCHERS: continue
    if url in seen_urls: continue
    seen_urls.add(url)
    fetchable.append(t)
print(f'Fetchable entries: {len(fetchable)}')

# Filter to ones with n>=50 to ensure enough data
results = []
for i, t in enumerate(fetchable):
    name = t.get('name', '')[:55]
    url = t.get('url', '')
    dom = t.get('domain', '')
    print(f'\n[{i+1}/{len(fetchable)}] [{dom}] {name}', flush=True)
    print(f'  URL: {url[:80]}', flush=True)
    try:
        curve = get_raw_curve(url)
    except Exception as e:
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

    # T7.1-A
    wins = sliding_window_fits(t_arr, R_arr, n_windows=10, overlap=0.6)
    if len(wins) < 4:
        print(f'  T7.1-A: only {len(wins)} windows — skip', flush=True)
        continue
    bps_a = detect_breakpoints(wins, jump_threshold=2.5)
    print(f'  T7.1-A: {len(bps_a)} breakpoints', flush=True)
    for b in bps_a[:3]:
        print(f"    t_break={b['t_break']:.2f}  D:{b['D_before']:.2f}→{b['D_after']:.2f}  score={b['jump_score']:.2f}", flush=True)

    # T7.1-B (piecewise)
    single = fit_single_s2(t_arr, R_arr)
    pw = fit_piecewise_s2(t_arr, R_arr, n_break_candidates=9)
    if pw and single:
        delta = single['aicc'] - pw[1]
        print(f"  T7.1-B: single AICc={single['aicc']:.1f}  piecewise AICc={pw[1]:.1f}  Δ={delta:.1f}", flush=True)
        print(f"    D1={pw[0][2]:.3f}  D2={pw[0][5]:.3f}  t_break={pw[2]:.2f}", flush=True)
    else:
        delta = None
        print(f'  T7.1-B: failed', flush=True)

    results.append({
        'name': t.get('name',''),
        'domain': dom,
        'url': url,
        'n': int(len(t_arr)),
        't_range': [float(t_arr[0]), float(t_arr[-1])],
        'model_verdict': t.get('model_verdict',''),
        'delta_aicc_dust': t.get('delta_aicc'),
        't71a_breakpoints': bps_a,
        't71a_n_windows': len(wins),
        't71b_single_aicc': float(single['aicc']) if single else None,
        't71b_piecewise_aicc': float(pw[1]) if pw else None,
        't71b_delta_aicc': float(delta) if delta is not None else None,
        't71b_t_break': float(pw[2]) if pw else None,
        't71b_params': list(pw[0]) if pw else None,
    })

# ─────────────────────────────────────────────────────────────────────
# T7.1-D: Clustering analysis
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('T7.1-D: CROSS-DATASET TRANSITION SCALE CLUSTERING')
print('='*72)

# All T7.1-A breakpoints
all_bps = []
for r in results:
    for b in r['t71a_breakpoints']:
        t_min, t_max = r['t_range']
        span = max(t_max - t_min, 1e-10)
        all_bps.append({
            'source': 'T71A',
            't_break': b['t_break'],
            'frac_pos': (b['t_break'] - t_min) / span,
            'kind': r['model_verdict'],
            'domain': r['domain'],
            'name': r['name'][:40],
        })

# All T7.1-B breakpoints
t71b_bps = []
for r in results:
    if r.get('t71b_t_break') is not None:
        t_min, t_max = r['t_range']
        span = max(t_max - t_min, 1e-10)
        t71b_bps.append({
            'source': 'T71B',
            't_break': r['t71b_t_break'],
            'frac_pos': (r['t71b_t_break'] - t_min) / span,
            'delta_aicc': r['t71b_delta_aicc'],
            'kind': r['model_verdict'],
            'domain': r['domain'],
            'name': r['name'][:40],
        })

all_bps_combined = all_bps + t71b_bps
print(f'\nTotal breakpoints: {len(all_bps_combined)}')
print(f'  From T7.1-A: {len(all_bps)}')
print(f'  From T7.1-B: {len(t71b_bps)}')

# T7.1-A clustering
frac_a = np.array([b['frac_pos'] for b in all_bps])
if len(frac_a) >= 5:
    print(f'\nT7.1-A breakpoint fractional positions (normalized 0-1 within each dataset):')
    print(f'  n = {len(frac_a)}')
    print(f'  mean = {frac_a.mean():.3f}  median = {np.median(frac_a):.3f}  std = {frac_a.std():.3f}')
    print(f'  IQR  = [{np.percentile(frac_a, 25):.3f}, {np.percentile(frac_a, 75):.3f}]')
    hist, edges = np.histogram(frac_a, bins=10, range=(0, 1))
    print(f'  Histogram (10 bins 0-1):')
    for i, h in enumerate(hist):
        bar = '#' * int(40 * h / max(hist.max(), 1))
        print(f'    [{edges[i]:.2f}, {edges[i+1]:.2f}): {h:>3d}  {bar}')

# T7.1-B clustering
frac_b = np.array([b['frac_pos'] for b in t71b_bps if b.get('delta_aicc', 0) > 4])
if len(frac_b) >= 3:
    print(f'\nT7.1-B breakpoint fractional positions (only datasets where piecewise strongly improves, ΔAICc>4):')
    print(f'  n = {len(frac_b)}')
    print(f'  mean = {frac_b.mean():.3f}  median = {np.median(frac_b):.3f}  std = {frac_b.std():.3f}')
    print(f'  IQR  = [{np.percentile(frac_b, 25):.3f}, {np.percentile(frac_b, 75):.3f}]')
else:
    print(f'\nT7.1-B: only {len(frac_b)} strong-improvement breakpoints — too few for clustering test')

# T7.1-B: how often does piecewise improve?
n_pw_improve = sum(1 for r in results if r.get('t71b_delta_aicc', 0) and r['t71b_delta_aicc'] > 4)
n_pw_total = sum(1 for r in results if r.get('t71b_delta_aicc') is not None)
print(f'\nT7.1-B: piecewise strongly improves single-S2 (ΔAICc>4): {n_pw_improve}/{n_pw_total}')
if n_pw_total:
    deltas = [r['t71b_delta_aicc'] for r in results if r.get('t71b_delta_aicc') is not None]
    print(f'  Median ΔAICc: {np.median(deltas):.2f}')
    print(f'  Mean ΔAICc: {np.mean(deltas):.2f}')
    print(f'  Max ΔAICc: {np.max(deltas):.2f}')

# ─────────────────────────────────────────────────────────────────────
# Per-domain breakdown
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('T7.1 RESULTS BY DOMAIN')
print('='*72)
from collections import defaultdict
by_dom = defaultdict(lambda: {'n': 0, 'bps_A': 0, 'pw_improve': 0, 'pw_total': 0,
                              'bp_fracs': [], 't_break_fracs': []})
for r in results:
    d = r['domain']
    by_dom[d]['n'] += 1
    by_dom[d]['bps_A'] += len(r['t71a_breakpoints'])
    if r.get('t71b_delta_aicc') is not None:
        by_dom[d]['pw_total'] += 1
        if r['t71b_delta_aicc'] > 4:
            by_dom[d]['pw_improve'] += 1
    for b in r['t71a_breakpoints']:
        t_min, t_max = r['t_range']
        by_dom[d]['bp_fracs'].append((b['t_break'] - t_min) / max(t_max - t_min, 1e-10))
    if r.get('t71b_t_break') is not None:
        t_min, t_max = r['t_range']
        by_dom[d]['t_break_fracs'].append((r['t71b_t_break'] - t_min) / max(t_max - t_min, 1e-10))

print(f'  {"Domain":<20s}  n  bps_A  pw_imp/pw_tot  bp_frac_median  t_break_frac_median')
print('  ' + '-'*80)
for d, v in sorted(by_dom.items(), key=lambda kv: -kv[1]['n']):
    bp_med = np.median(v['bp_fracs']) if v['bp_fracs'] else float('nan')
    tb_med = np.median(v['t_break_fracs']) if v['t_break_fracs'] else float('nan')
    print(f'  {d:<20s}  {v["n"]:>2d}  {v["bps_A"]:>5d}  {v["pw_improve"]:>3d}/{v["pw_total"]:<3d}       {bp_med:>14.3f}  {tb_med:>17.3f}')

# Save
out = {
    'hypothesis': 'T7.1 — multi-segment S2: piecewise S2 regimes',
    'n_target_curves': len(fetchable),
    'n_fetched_and_analyzed': len(results),
    't71a_total_breakpoints': sum(len(r['t71a_breakpoints']) for r in results),
    't71a_frac_pos_stats': {
        'n': int(len(frac_a)),
        'mean': float(frac_a.mean()) if len(frac_a) else None,
        'median': float(np.median(frac_a)) if len(frac_a) else None,
        'std': float(frac_a.std()) if len(frac_a) else None,
        'iqr': [float(np.percentile(frac_a, 25)), float(np.percentile(frac_a, 75))] if len(frac_a) else None,
    },
    't71b_n_piecewise_improved': n_pw_improve,
    't71b_n_total': n_pw_total,
    't71b_median_delta_aicc': float(np.median([r['t71b_delta_aicc'] for r in results if r.get('t71b_delta_aicc') is not None])) if n_pw_total else None,
    't71b_frac_pos_stats': {
        'n': int(len(frac_b)),
        'mean': float(frac_b.mean()) if len(frac_b) else None,
        'median': float(np.median(frac_b)) if len(frac_b) else None,
        'std': float(frac_b.std()) if len(frac_b) else None,
        'iqr': [float(np.percentile(frac_b, 25)), float(np.percentile(frac_b, 75))] if len(frac_b) else None,
    },
    'results': results,
}
out_path = os.path.join(OUT_DIR, 't71_multi_segment_focused.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')

# ─────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────
print('\n' + '='*72)
print('T7.1 HONEST VERDICT')
print('='*72)
print()
print('Hypothesis: R(λ) is composed of piecewise S2 regimes, with')
print('transitions at reproducible scale thresholds.')
print()
print('Two pieces of evidence required:')
print('  (1) Piecewise S2 strongly beats single S2 (ΔAICc > 4)')
print('  (2) Transition scales cluster across datasets (low std of frac_pos)')
print()
print(f'(1) Piecewise improvement: {n_pw_improve}/{n_pw_total} datasets show ΔAICc>4')
if n_pw_total > 0:
    deltas = [r['t71b_delta_aicc'] for r in results if r.get('t71b_delta_aicc') is not None]
    print(f'    Median ΔAICc: {np.median(deltas):.2f}')
print(f'(2) T7.1-A breakpoint fractional std: {frac_a.std():.3f}' if len(frac_a) else '    (no data)')
print(f'    T7.1-B breakpoint fractional std: {frac_b.std():.3f}' if len(frac_b) else '    (no data)')
print()

if len(frac_a) >= 10:
    a_std = frac_a.std()
    if a_std < 0.12:
        print(f'>>> T7.1-A BREAKPOINTS CLUSTER (std<0.12) — transition scales are reproducible')
    elif a_std < 0.20:
        print(f'>>> T7.1-A breakpoints moderately clustered (std<0.20) — weak evidence of reproducibility')
    else:
        print(f'>>> T7.1-A breakpoints spread (std>0.20) — no reproducible transition scale')
