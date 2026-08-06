#!/usr/bin/env python3
"""
Find REAL S2 failures: solid datasets with large n where S2 cannot fit.

User policy (strict):
  Record S2_NO_FIT ONLY when ALL of the following are true:
    1. Data was successfully fetched (n >= 100 valid observations)
    2. ACF was computed successfully (series has variance, not constant)
    3. S2 curve_fit optimizer failed to converge (returned None or NO_FIT)

DO NOT record:
  - Fetch failures (URL unreachable) — retry on next scout, internal only
  - Low-n datasets (< 100 obs) — not enough data to judge
  - Parse failures (no numeric column, CSV errors) — data quality issue, not S2
  - Constant series (var=0) — degenerate, not a real-world signal

The threshold n >= 100 ensures we only flag genuine S2 failures on
well-formed real-world datasets where we would EXPECT S2 to fit if
the framework holds.
"""
import os, sys, json, csv, io, urllib.request, re
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve, fit_s2, load_existing_tests, update_tests_html

# Minimum n for a "solid" dataset
MIN_N = 100

# ── Sources to check — large real-world datasets where S2 should fit ──
# Focus on well-formed public datasets with hundreds+ of observations
SOLID_DATASETS = [
    # FRED daily series (thousands of obs)
    ('FRED DFF (Federal Funds Effective Rate, daily)', 'financial',
     'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF'),
    ('FRED T10Y3MM (10Y-3M Treasury spread, daily)', 'financial',
     'https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3MM'),
    ('FRED M2SL (Money Stock M2, monthly)', 'financial',
     'https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL'),
    ('FRED CPIAUCSL (CPI All Urban, monthly)', 'financial',
     'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL'),
    ('FRED UNRATE (Unemployment, monthly)', 'financial',
     'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE'),
    ('FRED GDP (Quarterly)', 'financial',
     'https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP'),
    # Global temp (monthly, 1850-present)
    ('HadCRUT Global Temperature (monthly)', 'environmental',
     'https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv'),
    # NOAA daily weather (multiple stations, daily for years)
    ('NOAA Daily Weather: JFK airport', 'environmental',
     'https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094789&startDate=2020-01-01&endDate=2025-12-31&dataTypes=PRCP&format=csv'),
    # NDBC buoy (hourly, years of data)
    ('NDBC Buoy 46035 (Bering Sea, hourly)', 'oceanography',
     'https://www.ndbc.noaa.gov/data/realtime2/46035.txt'),
    # USGS daily discharge (decades of daily data)
    ('USGS: Colorado River at Lees Ferry (daily, 30y)', 'hydrology',
     'https://waterservices.usgs.gov/nwis/dv/?sites=09380000&parameterCd=00060&startDT=1995-01-01&endDT=2025-01-01&format=rdb'),
    # Open-Meteo daily (years)
    ('Open-Meteo: Reykjavik temperature_2m_mean (5y daily)', 'environmental',
     'https://archive-api.open-meteo.com/v1/archive?latitude=64.13&longitude=-21.94&start_date=2020-01-01&end_date=2025-12-31&daily=temperature_2m_mean'),
    # SWPC sunspot (monthly, 1749-present — 3000+ obs)
    ('SWPC Sunspot Number (monthly, 1749-present)', 'solar_physics',
     'https://www.swpc.noaa.gov/products/solar-cycle-progression-file'),
    # Wikipedia pageviews (daily, 5 years)
    ('Wikipedia: Donald Trump pageviews (5y daily)', 'cultural',
     'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Donald_Trump/daily/2020010100/2025123100'),
    # Binance daily klines (years)
    ('Binance ETHUSDT daily (3y)', 'crypto',
     'https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1d&limit=1000'),
    # NOAA tides (6-min interval, 1 year)
    ('NOAA Tide: The Battery NY (6-min, 1y)', 'oceanography',
     'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20240101&end_date=20241231&station=8518750&product=water_level&datum=mllw&units=metric&time_zone=gmt&format=json'),
]

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
    return vals

def refetch_globaltemp(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    rows = list(csv.reader(io.StringIO(data.decode('utf-8') if isinstance(data, bytes) else data)))
    vals = []
    for row in rows[1:]:
        if len(row) >= 2:
            try: vals.append(float(row[1]))
            except: pass
    return vals

def refetch_noaa_weather(url):
    data = fetch_url(url, timeout=30)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 5: return None
    # Find PRCP column
    val_idx = -1
    for i, h in enumerate(rows[0]):
        if 'PRCP' in h.upper(): val_idx = i; break
    if val_idx < 0:
        # Try last column
        val_idx = len(rows[0]) - 1
    vals = []
    for row in rows[1:]:
        if val_idx < len(row):
            try:
                v = float(row[val_idx])
                if v >= 0 and not np.isnan(v): vals.append(v)
            except: pass
    return vals

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
    return vals

def refetch_usgs_river(url):
    data = fetch_url(url, timeout=30)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    lines = text.strip().split('\n')
    vals = []
    for line in lines:
        if line.startswith('#') or not line.strip(): continue
        parts = line.split('\t')
        if len(parts) >= 5:
            try:
                v = float(parts[4])
                if v >= 0 and not np.isnan(v): vals.append(v)
            except: pass
    return vals

def refetch_open_meteo(url):
    data = fetch_url(url, timeout=30)
    if not data: return None
    try:
        d = json.loads(data)
        daily = d.get('daily', {})
        for var in ['temperature_2m_mean', 'wind_speed_10m_max', 'precipitation_sum']:
            if var in daily:
                return [v for v in daily[var] if v is not None]
    except: pass
    return None

def refetch_swpc_sunspot(url):
    # SWPC solar cycle progression — typically returns text file
    data = fetch_url(url, timeout=20)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    vals = []
    for line in text.split('\n'):
        parts = line.split()
        if len(parts) >= 2:
            try:
                v = float(parts[-1])
                if v >= 0 and not np.isnan(v): vals.append(v)
            except: pass
    return vals

def refetch_wikipedia(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'items' in obj:
            return [item['views'] for item in obj['items']]
    except: pass
    return None

def refetch_binance(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        arr = json.loads(data)
        return [float(k[4]) for k in arr] if arr else None
    except: pass
    return None

def refetch_noaa_tide(url):
    data = fetch_url(url, timeout=30)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'data' in obj:
            return [float(d['v']) for d in obj['data'] if d.get('v') not in ('', None)]
    except: pass
    return None

def refetch_values(url, name=''):
    """Refetch values from a URL. Returns list of floats or None on fetch failure."""
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url:           return refetch_fred(url)
        if 'global-temp' in url:             return refetch_globaltemp(url)
        if 'ncei.noaa.gov' in url:           return refetch_noaa_weather(url)
        if 'ndbc.noaa.gov' in url:           return refetch_ndbc(url)
        if 'waterservices.usgs.gov' in url:  return refetch_usgs_river(url)
        if 'open-meteo' in url:              return refetch_open_meteo(url)
        if 'swpc.noaa.gov' in url:           return refetch_swpc_sunspot(url)
        if 'wikimedia.org' in url:           return refetch_wikipedia(url)
        if 'binance.com' in url:             return refetch_binance(url)
        if 'tidesandcurrents.noaa.gov' in url: return refetch_noaa_tide(url)
    except Exception as e:
        print(f'    Refetch error: {e}')
    return None


def main():
    print('=' * 60)
    print(f'FINDING REAL S2 FAILURES (n >= {MIN_N}, data fetched, S2 cannot fit)')
    print('=' * 60)
    print('Fetch failures and low-n datasets are NOT recorded.')
    print()

    existing = load_existing_tests(os.path.join(REPO, 'en/tests.html'))
    existing_urls = set(e.get('url', '').rstrip('/') for e in existing if e.get('url'))

    real_failures = []
    for name, domain, url in SOLID_DATASETS:
        if url.rstrip('/') in existing_urls:
            print(f'  ⊙ Already in registry: {name[:55]}')
            continue

        print(f'\n  → {name[:60]}')
        vals = refetch_values(url, name)
        if vals is None:
            print(f'    ✗ fetch failed — SKIP (internal retry only)')
            continue
        if len(vals) < MIN_N:
            print(f'    ✗ only {len(vals)} obs (need {MIN_N}) — SKIP')
            continue

        print(f'    Fetched {len(vals)} values')
        taus, acf = retention_curve(vals)
        if taus is None:
            print(f'    ✗ ACF failed — SKIP (degenerate series)')
            continue
        if len(taus) < 5:
            print(f'    ✗ ACF too short ({len(taus)} lags) — SKIP')
            continue

        # Now try fit_s2 — if it returns D=None with no_fit, that's a REAL failure
        fit = fit_s2(taus, acf, name[:60], source_url=url)
        if fit is None:
            print(f'    ✗ fit_s2 returned None — SKIP (unknown)')
            continue
        if fit.get('D') is not None:
            print(f'    ✓ S2 fit succeeded: D={fit["D"]:.3f} — not a failure')
            continue

        # Real S2 failure: data was solid, ACF computed, but S2 couldn't fit
        reason = fit.get('rejection_reason', 'no_fit')
        print(f'    ⚠ REAL S2 FAILURE: {reason}')
        print(f'      n={len(vals)}, lags={len(taus)}, D=None')

        real_failures.append({
            'id': f'nofit-{len(real_failures)}',
            'name': f'{name} (ACF retention)',
            'domain': domain,
            'D': None,
            'r2': None,
            'verdict': 'REJECTED',
            'model_verdict': 'S2_NO_FIT',
            'model_note': fit.get('model_note', f'S2 fit rejected: {reason}.'),
            'rejection_reason': reason,
            'delta_aicc': None,
            'best_alt': None,
            'narrative': f'S2 fit rejected on solid dataset (n={len(vals)}): {reason}. Data was successfully fetched and ACF computed, but S2 curve_fit could not converge. This is a genuine S2 failure on well-formed real-world data.',
            'url': url,
        })

    print(f'\n{"=" * 60}')
    print(f'REAL S2 failures found: {len(real_failures)}')

    if real_failures:
        update_tests_html(real_failures, os.path.join(REPO, 'en/tests.html'), is_ru=False)
        update_tests_html(real_failures, os.path.join(REPO, 'ru/tests.html'), is_ru=True)

    # Re-export + reconcile + meta-s2
    import subprocess
    r = subprocess.run(['python3', 'scripts/export_tests_json.py', '.'],
                       cwd=REPO, capture_output=True, text=True, timeout=30)
    print(r.stdout.strip())
    r = subprocess.run(['python3', 'scripts/registry_integrity_reconcile.py'],
                       cwd=REPO, capture_output=True, text=True, timeout=60)
    for line in r.stdout.strip().split('\n')[-8:]:
        print(line)

    from dream_auto_scanner import update_meta_s2_article
    update_meta_s2_article(os.path.join(REPO, 'en/tests.html'), is_ru=False)
    update_meta_s2_article(os.path.join(REPO, 'ru/tests.html'), is_ru=True)

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        d = json.load(f)
    from collections import Counter
    mv = Counter(t.get('model_verdict', '?') for t in d['tests'])
    print(f'\n=== FINAL REGISTRY ===')
    print(f'Total: {d["total_tests"]}')
    for v, n in mv.most_common():
        print(f'  {v}: {n}')

if __name__ == '__main__':
    main()
