#!/usr/bin/env python3
"""
T7.1 EXPANDED DATASET FETCH
============================

Try harder to fetch real datasets that previously timed out:
  - Zenodo records via direct file URLs
  - FRED via direct CSV (not via fred.stlouisfed.org which timed out)
  - Wikipedia pageviews (was 403)
  - NASA GISS (was unreachable)

Run T7.1 piecewise S2 on whatever we successfully fetch.
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
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-T71C/1.0 (research)'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 1: raise
            time.sleep(1)


# ─────────────────────────────────────────────────────────────────────
# Zenodo direct file fetch (use API to list files, then fetch each)
# ─────────────────────────────────────────────────────────────────────

def fetch_zenodo_with_size_check(url):
    """Fetch a Zenodo record's first reasonable CSV file.
    Returns (t, R) or None."""
    m = re.search(r'zenodo\.(\d+)', url)
    if not m: return None
    rid = m.group(1)
    api_url = f'https://zenodo.org/api/records/{rid}'
    try:
        raw = fetch(api_url, timeout=10).decode('utf-8')
        j = json.loads(raw)
        files = j.get('files', [])
        # Sort by size — smaller files first (avoid 100MB+ files)
        candidates = []
        for f in files:
            key = f.get('key', '')
            size = f.get('size', 0)
            if key.endswith('.csv') and size > 1000 and size < 5_000_000:
                candidates.append((size, f))
        candidates.sort()
        for _, f in candidates[:3]:  # try top 3 small CSVs
            csv_url = f.get('links', {}).get('self')
            if not csv_url: continue
            try:
                csv_raw = fetch(csv_url, timeout=20).decode('utf-8', errors='ignore')
                # Parse CSV, take first numeric column
                reader = csv.reader(io.StringIO(csv_raw))
                rows = list(reader)
                if len(rows) < 30: continue
                # Find a numeric column
                header = rows[0] if rows else []
                # Skip header row, look at row 1
                vals = []
                for row in rows[1:]:
                    for cell in row:
                        try:
                            v = float(cell)
                            vals.append(v)
                            break
                        except: continue
                if len(vals) < 30: continue
                R = np.array(vals, dtype=float)
                # Compute ACF
                R = R - R.mean()
                if np.std(R) > 0: R = R / np.std(R)
                n = len(R)
                max_lag = min(n - 1, 80)
                acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
                t = np.arange(max_lag, dtype=float)
                if acf[0] > 0: acf = acf / acf[0]
                return t, acf
            except Exception:
                continue
    except Exception:
        return None
    return None


# ─────────────────────────────────────────────────────────────────────
# FRED via direct CSV (alternative to fred.stlouisfed.org which timed out)
# ─────────────────────────────────────────────────────────────────────

def fetch_fred_alt(url):
    """Try FRED via the API endpoint instead."""
    # Extract series ID
    m = re.search(r'id=([A-Z0-9\.]+)', url)
    if not m: return None
    sid = m.group(1)
    # Try API key-less CSV endpoint
    api_url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2000-01-01'
    try:
        raw = fetch(api_url, timeout=15).decode('utf-8')
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        pairs = []
        for row in rows[1:]:  # skip header
            if len(row) < 2: continue
            try:
                v = float(row[1])
                if v == v:
                    pairs.append(v)
            except: continue
        if len(pairs) < 30: return None
        R = np.array(pairs, dtype=float)
        # Compute ACF
        R = R - R.mean()
        if np.std(R) > 0: R = R / np.std(R)
        n = len(R)
        max_lag = min(n - 1, 80)
        acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
        t = np.arange(max_lag, dtype=float)
        if acf[0] > 0: acf = acf / acf[0]
        return t, acf
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# S2 fits (same as t71_focused.py)
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


class TimeoutError_(Exception): pass
def _timeout_handler(signum, frame):
    raise TimeoutError_('fit timeout')


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

# Pick top S2_DUST_WIN candidates from Zenodo + FRED
candidates = []
for t in tests:
    ba = t.get('best_alt','')
    da = t.get('delta_aicc', None)
    mv = t.get('model_verdict','')
    url = t.get('url', '')
    if not url: continue
    is_s2_dust_win = (mv == 'S2_LOSES' and ba == 'S2_DUST' and da is not None and da >= 10)
    if not is_s2_dust_win: continue
    # Filter to Zenodo or FRED
    if url.startswith('10.5281/zenodo') or 'zenodo' in url or 'fred.stlouisfed.org' in url:
        candidates.append((da, t))
candidates.sort(key=lambda x: -x[0])
print(f'Candidates: {len(candidates)} Zenodo/FRED S2_DUST_WIN entries')

# Take top 15
top = candidates[:15]
print(f'Trying top {len(top)} (highest ΔAICc)')

results = []
for i, (da, t) in enumerate(top):
    name = t.get('name', '')[:55]
    url = t.get('url', '')
    dom = t.get('domain', '')
    print(f'\n[{i+1}/{len(top)}] [{dom}] ΔAICc={da:.1f} {name}', flush=True)
    print(f'  URL: {url[:80]}', flush=True)

    # Try Zenodo first, then FRED alt
    curve = None
    if 'zenodo' in url or url.startswith('10.5281'):
        curve = fetch_zenodo_with_size_check(url)
    elif 'fred.stlouisfed' in url:
        curve = fetch_fred_alt(url)
    if curve is None:
        print(f'  FETCH failed', flush=True)
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

    # T7.1-B
    single = fit_single_s2(t_arr, R_arr)
    pw = fit_piecewise_s2(t_arr, R_arr, n_break_candidates=5, per_fit_timeout=2.0)
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
        't71b_D1': float(pw[0][2]) if pw else None,
        't71b_D2': float(pw[0][5]) if pw else None,
    })

print(f'\nFetched & analyzed: {len(results)} additional datasets')

# Save
out = {
    'probe': 'T7.1 expanded dataset fetch (Zenodo + FRED)',
    'n_candidates': len(candidates),
    'n_attempted': len(top),
    'n_fetched': len(results),
    'results': results,
}
out_path = os.path.join(OUT_DIR, 't71_expanded.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')

# Summary
if results:
    n_pw = sum(1 for r in results if r['t71b_delta_aicc'] is not None and r['t71b_delta_aicc'] > 4)
    n_total = sum(1 for r in results if r['t71b_delta_aicc'] is not None)
    print(f'\nResults:')
    print(f'  Successful piecewise fits: {n_total}')
    print(f'  Strong improvement (ΔAICc>4): {n_pw}/{n_total}')
    if n_total:
        deltas = [r['t71b_delta_aicc'] for r in results if r['t71b_delta_aicc'] is not None]
        print(f'  Median ΔAICc: {np.median(deltas):.1f}')
    fracs = [(r['t71b_t_break'] - r['t_range'][0]) / max(r['t_range'][1]-r['t_range'][0], 1e-10)
             for r in results if r['t71b_t_break'] is not None and r['t71b_delta_aicc'] > 4]
    if fracs:
        fracs = np.array(fracs)
        print(f'  t_break fractions: n={len(fracs)}  mean={fracs.mean():.3f}  std={fracs.std():.3f}')
        print(f'  IQR=[{np.percentile(fracs,25):.3f}, {np.percentile(fracs,75):.3f}]')
