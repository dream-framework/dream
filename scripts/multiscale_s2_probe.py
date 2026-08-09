#!/usr/bin/env python3
"""
Multi-scale S2 probe test.

Hypothesis (user): DREAM retention is like sifting through ever-coarser nets.
At each probe scale, a stretched exponential is the right local model —
but the D and λ_q CHANGE with scale. The "dust" we keep finding is
fine-scale structure that single-S2 ignores.

Test: Take a dust-dominated dataset. Fit S2 on:
  - Full ACF range [0, max_lag]
  - Fine scale: [0, max_lag/4]    (first quarter — fine probe)
  - Medium scale: [max_lag/4, max_lag/2]
  - Coarse scale: [max_lag/2, max_lag]   (last half — coarse probe)

If the hypothesis holds:
  - Each scale should give a BETTER S2 fit (higher R²) than the full-range fit
  - D should be different at each scale (regime-specific)
  - The fine scale should reveal structure that was "dust" at full range

This would be a genuine mathematical refinement: R(λ) is locally S2 at
each scale, but globally a superposition.
"""
import os, sys, json, csv, io, re
import numpy as np
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve, load_existing_tests
from s2_model_compare import compare as s2_compare, m_s2


def fit_s2_on_range(t, R, t_min, t_max):
    """Fit S2 only on the portion of the curve where t_min <= t <= t_max."""
    mask = (t >= t_min) & (t <= t_max)
    if mask.sum() < 5:
        return None
    t_sub = t[mask]
    R_sub = R[mask]
    # Normalize to R[0]=1 within this range
    if R_sub[0] > 0:
        R_sub = R_sub / R_sub[0]
    try:
        t_mid = float(t_sub[len(t_sub) // 2])
        p0_list = [
            [1.0, t_mid, 0.5],
            [1.0, t_mid * 0.5, 1.0],
            [1.0, t_mid * 2, 0.3],
        ]
        best = None
        for p0 in p0_list:
            try:
                popt, _ = curve_fit(m_s2, t_sub, R_sub, p0=p0,
                                    bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]),
                                    maxfev=20000)
                rss = float(np.sum((R_sub - m_s2(t_sub, *popt)) ** 2))
                if best is None or rss < best[1]:
                    best = (popt, rss)
            except: pass
        if best is None: return None
        popt, rss = best
        ss_tot = float(np.sum((R_sub - np.mean(R_sub)) ** 2))
        if ss_tot == 0: return None
        r2 = 1 - rss / ss_tot
        return {
            'D': float(popt[2]),
            'lambda_q': float(popt[1]),
            'r2': r2,
            'n_points': int(mask.sum()),
            't_range': (float(t_min), float(t_max)),
        }
    except Exception as e:
        return None


def refetch_fred(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    rows = list(csv.reader(io.StringIO(data.decode('utf-8') if isinstance(data, bytes) else data)))
    vals = []
    for row in rows[1:]:
        if len(row) >= 2 and row[1] not in ('', '.'):
            try:
                v = float(row[1])
                if not np.isnan(v) and not np.isinf(v): vals.append(v)
            except: pass
    return vals if len(vals) >= 50 else None

def refetch_covid(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    rows = list(csv.reader(io.StringIO(data.decode('utf-8') if isinstance(data, bytes) else data)))
    if len(rows) < 2: return None
    daily_sums = []
    for col in range(4, len(rows[0])):
        total = 0
        for row in rows[1:]:
            if col < len(row):
                try: total += int(row[col])
                except: pass
        daily_sums.append(total)
    daily_new = [max(0, daily_sums[i] - daily_sums[i-1]) for i in range(1, len(daily_sums))]
    return daily_new if len(daily_new) >= 50 else None

def refetch_wikipedia(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'items' in obj:
            return [item['views'] for item in obj['items']]
    except: pass
    return None

def refetch_open_meteo(url):
    data = fetch_url(url, timeout=25)
    if not data: return None
    try:
        d = json.loads(data)
        daily = d.get('daily', {})
        for var in ['wind_speed_10m_max', 'precipitation_sum']:
            if var in daily:
                return [v for v in daily[var] if v is not None]
    except: pass
    return None

def refetch_ndbc(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    lines = text.strip().split('\n')
    vals = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 6:
            try:
                v = float(parts[5])
                if 0 <= v < 99: vals.append(v)
            except: pass
    return vals if len(vals) >= 50 else None

def refetch_values(url):
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url: return refetch_fred(url)
        if 'CSSEGISandData' in url: return refetch_covid(url)
        if 'wikimedia.org' in url: return refetch_wikipedia(url)
        if 'open-meteo' in url: return refetch_open_meteo(url)
        if 'ndbc.noaa.gov' in url: return refetch_ndbc(url)
    except: pass
    return None


def main():
    print('=' * 90)
    print('MULTI-SCALE S2 PROBE TEST')
    print('=' * 90)
    print()
    print('Hypothesis: At each probe scale, S2 is the right local model.')
    print('The "dust" is fine-scale structure invisible to full-range S2.')
    print()
    print('Test: Fit S2 separately on fine / medium / coarse scale ranges.')
    print('If R² improves at finer scales, structure is being revealed.')
    print()

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    # Test on a sample of dust-dominated + dust-resolved entries
    test_entries = []
    for t in data['tests']:
        if t.get('model_verdict') in ('S2_LOSES', 'S2_DUST_WINS'):
            url = t.get('url', '')
            if any(x in url for x in ['fredgraph.csv', 'CSSEGISandData', 'wikimedia.org', 'open-meteo', 'ndbc.noaa.gov']):
                test_entries.append(t)

    # Dedupe by URL
    seen_urls = set()
    unique = []
    for t in test_entries:
        u = t.get('url', '')
        if u not in seen_urls:
            seen_urls.add(u)
            unique.append(t)
    test_entries = unique[:10]  # limit to 10 for speed

    print(f'Testing {len(test_entries)} datasets (deduplicated)')
    print()

    for entry in test_entries:
        name = entry['name'][:55]
        url = entry.get('url', '')
        vals = refetch_values(url)
        if not vals or len(vals) < 100:
            print(f'  ✗ {name}: fetch failed or too short')
            continue

        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 20:
            print(f'  ✗ {name}: ACF too short')
            continue

        t_arr = np.array(taus, dtype=float)
        R_arr = np.array(acf, dtype=float)
        max_lag = float(t_arr[-1])

        # Fit on full range
        full = fit_s2_on_range(t_arr, R_arr, 0, max_lag)
        # Fit on fine scale (first quarter)
        fine = fit_s2_on_range(t_arr, R_arr, 0, max_lag * 0.25)
        # Fit on medium scale (25%-75%)
        medium = fit_s2_on_range(t_arr, R_arr, max_lag * 0.25, max_lag * 0.75)
        # Fit on coarse scale (last half)
        coarse = fit_s2_on_range(t_arr, R_arr, max_lag * 0.5, max_lag)

        print(f'  {name}')
        if full:
            print(f'    Full range  [0, {max_lag:.0f}]:  D={full["D"]:.3f}, λ={full["lambda_q"]:.2f}, R²={full["r2"]:.4f}')
        else:
            print(f'    Full range  [0, {max_lag:.0f}]:  fit failed')
        if fine:
            better = '✓ better' if full and fine["r2"] > full["r2"] else '✗ worse'
            print(f'    Fine  [0, {max_lag*0.25:.0f}]:       D={fine["D"]:.3f}, λ={fine["lambda_q"]:.2f}, R²={fine["r2"]:.4f}  ({better})')
        if medium:
            better = '✓ better' if full and medium["r2"] > full["r2"] else '✗ worse'
            print(f'    Medium [{max_lag*0.25:.0f}, {max_lag*0.75:.0f}]:  D={medium["D"]:.3f}, λ={medium["lambda_q"]:.2f}, R²={medium["r2"]:.4f}  ({better})')
        if coarse:
            better = '✓ better' if full and coarse["r2"] > full["r2"] else '✓ better' if not full else '✗ worse'
            print(f'    Coarse [{max_lag*0.5:.0f}, {max_lag:.0f}]:   D={coarse["D"]:.3f}, λ={coarse["lambda_q"]:.2f}, R²={coarse["r2"]:.4f}  ({better})')

        # Check if D changes with scale
        Ds = [f["D"] for f in [full, fine, medium, coarse] if f]
        if len(Ds) >= 2:
            d_range = max(Ds) - min(Ds)
            print(f'    D range across scales: {min(Ds):.3f} → {max(Ds):.3f} (Δ={d_range:.3f})')
        print()


if __name__ == '__main__':
    main()
