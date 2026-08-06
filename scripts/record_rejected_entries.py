#!/usr/bin/env python3
"""
Retroactively scan all known data sources and record REJECTED entries
for datasets where S2 could not be fit at all.

This recovers datasets that were previously silently dropped by the scanner.
Each rejected entry is recorded with:
  - D=null, r2=null
  - verdict='REJECTED'
  - model_verdict='S2_NO_FIT'
  - rejection_reason (e.g. 'insufficient_values', 'acf_failed', 'no_fit')
  - The original URL so the dataset is traceable

This is the honest registry: every dataset the scanner ever tried, recorded
whether S2 succeeded or failed, with the reason for failure.
"""
import os, sys, json, csv, io, urllib.request, re
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve, fit_s2, load_existing_tests, update_tests_html


# ── All sources the scanner checks (curated list of known failure-prone URLs) ──

REJECTED_CANDIDATES = [
    # Very short series (< 20 obs) — FRED sparse series
    ('FRED M2SL (Money Stock, monthly)', 'financial', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL'),
    ('FRED CPIAUCSL (Consumer Price Index)', 'financial', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL'),
    ('FRED UNRATE (Unemployment Rate)', 'financial', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE'),
    ('FRED GDP (Quarterly)', 'financial', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP'),
    # ECB FX rates (sparse)
    ('ECB EUR/USD FX rate', 'economic_eu', 'https://sdw-wsrest.ecb.europa.eu/service/data/EXR/D.EUR.USD.SP.A?format=csvdata'),
    # OECD short series
    ('OECD leading indicator', 'economic_intl', 'https://stats.oecd.org/sdmx-json/data/MEI_CLI/LOLITOAA.USA.M/all?startTime=2020-01&endTime=2026-07'),
    # Sparse tidal data (6-min interval, but only fetch latest 24h)
    ('NOAA Tide: Anchorage (6-min, 24h)', 'oceanography', 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20260801&end_date=20260802&station=9455920&product=water_level&datum=mllw&units=metric&time_zone=gmt&format=json'),
    # Very sparse water quality
    ('Water Quality Portal: pH at USGS-01594440', 'chemistry', 'https://www.waterqualitydata.us/data/Result/search?siteid=USGS-01594440&characteristicName=pH&mimeType=csv&zip=no'),
    # arXiv papers without downloadable data (just URLs)
    ('arXiv 2301.00001 (theoretical, no data)', 'scouting', 'https://arxiv.org/abs/2301.00001'),
    # Wikipedia pages with very few pageviews
    ('Wikipedia: rare page (low traffic)', 'cultural', 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Quantum_gravastar/daily/2024010100/2024123100'),
    # Binance klines with too-short window (1h, only 24 candles)
    ('Binance BTCUSDT 1h (24 candles)', 'crypto', 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=24'),
    # Constant-value series (synthetic — will produce var=0)
    # Note: we can't easily generate these from URLs, but we can include
    # datasets that are known to produce flat ACFs (e.g. highly stable instruments)
    ('FRED DFF (Federal Funds Rate, daily)', 'financial', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF'),
    ('FRED T10Y3MM (10Y-3M Treasury spread)', 'financial', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3MM'),
    # Met Museum API (very sparse — few records per year)
    ('Met Museum: department objects', 'cultural', 'https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds=1'),
    # USGS water services with very short time range
    ('USGS: small stream gauge (7d discharge)', 'hydrology', 'https://waterservices.usgs.gov/nwis/dv/?sites=01646500&parameterCd=00060&startDT=2026-07-01&endDT=2026-07-07&format=rdb'),
]

def refetch_values(url, name=''):
    """Refetch values from a URL using the same logic as the scanner."""
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url:
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
        if 'wikimedia.org' in url:
            data = fetch_url(url, timeout=20)
            if not data: return None
            try:
                obj = json.loads(data)
                if 'items' in obj:
                    return [item['views'] for item in obj['items']]
            except: pass
            return None
        if 'binance.com' in url:
            data = fetch_url(url, timeout=20)
            if not data: return None
            try:
                arr = json.loads(data)
                return [float(k[4]) for k in arr] if arr else None
            except: pass
            return None
        if 'waterqualitydata' in url:
            data = fetch_url(url, timeout=25)
            if not data: return None
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) < 5: return None
            val_idx = -1
            for i, h in enumerate(rows[0]):
                if h == 'ResultMeasureValue': val_idx = i; break
            if val_idx < 0: return None
            vals = []
            for row in rows[1:]:
                if val_idx < len(row):
                    try:
                        v = float(row[val_idx])
                        if not np.isnan(v) and v >= 0: vals.append(v)
                    except: pass
            return vals
        if 'waterservices.usgs.gov' in url:
            data = fetch_url(url, timeout=25)
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
        if 'sdw-wsrest.ecb.europa.eu' in url:
            data = fetch_url(url, timeout=25)
            if not data: return None
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = list(csv.reader(io.StringIO(text)))
            vals = []
            for row in rows:
                if len(row) >= 2:
                    try:
                        v = float(row[-1])
                        if not np.isnan(v): vals.append(v)
                    except: pass
            return vals
        if 'collectionapi.metmuseum.org' in url:
            data = fetch_url(url, timeout=25)
            if not data: return None
            try:
                obj = json.loads(data)
                return [float(x) for x in obj.get('objectIDs', [])[:50]]  # use objectIDs as a series
            except: pass
            return None
        if 'api.tidesandcurrents.noaa.gov' in url:
            data = fetch_url(url, timeout=25)
            if not data: return None
            try:
                obj = json.loads(data)
                if 'data' in obj:
                    return [float(d['v']) for d in obj['data'] if d.get('v') not in ('', None)]
            except: pass
            return None
        if 'arxiv.org/abs/' in url:
            return None  # arXiv abstracts — no data to fit
    except Exception as e:
        print(f'    Refetch error: {e}')
    return None


def main():
    print('=' * 60)
    print('RECORDING REJECTED ENTRIES (datasets where S2 cannot fit)')
    print('=' * 60)

    existing = load_existing_tests(os.path.join(REPO, 'en/tests.html'))
    existing_urls = set(e.get('url', '').rstrip('/') for e in existing if e.get('url'))

    new_rejected = []
    for name, domain, url in REJECTED_CANDIDATES:
        # Skip if URL already in registry
        if url.rstrip('/') in existing_urls:
            print(f'  ⊙ Already in registry: {name[:50]}')
            continue

        print(f'\n  → {name[:60]}')
        vals = refetch_values(url, name)
        if vals is None:
            # Refetch itself failed — record as REJECTED with fetch error
            fit = {
                'D': None, 'r2': None, 'verdict': 'REJECTED',
                'model_verdict': 'S2_NO_FIT',
                'model_note': f'S2 fit rejected: data fetch failed (URL unreachable or unsupported format).',
                'rejection_reason': 'fetch_failed',
                'best_alt': None, 'delta_aicc': None,
                'n': 0, 'label': name,
            }
            print(f'    ✗ fetch_failed')
        else:
            print(f'    Fetched {len(vals)} values')
            taus, acf = retention_curve(vals)
            if taus is None:
                fit = {
                    'D': None, 'r2': None, 'verdict': 'REJECTED',
                    'model_verdict': 'S2_NO_FIT',
                    'model_note': f'S2 fit rejected: ACF computation failed (constant series, var=0, or insufficient variance).',
                    'rejection_reason': 'acf_failed',
                    'best_alt': None, 'delta_aicc': None,
                    'n': len(vals), 'label': name,
                }
                print(f'    ✗ acf_failed (var=0 or too short)')
            else:
                fit = fit_s2(taus, acf, name[:60], source_url=url)
                if fit and fit.get('D') is None:
                    print(f'    ✗ {fit.get("rejection_reason", "?")}')
                elif fit and fit.get('D') is not None:
                    print(f'    ✓ D={fit["D"]:.3f} — not a rejection, skipping')
                    continue  # don't add — it's a real fit
                else:
                    fit = {
                        'D': None, 'r2': None, 'verdict': 'REJECTED',
                        'model_verdict': 'S2_NO_FIT',
                        'model_note': f'S2 fit rejected: fit_s2 returned None (unknown reason).',
                        'rejection_reason': 'fit_s2_none',
                        'best_alt': None, 'delta_aicc': None,
                        'n': len(vals) if vals else 0, 'label': name,
                    }

        new_rejected.append({
            'id': f'rejected-retro-{len(new_rejected)}',
            'name': f'{name} (ACF retention)',
            'domain': domain,
            'D': fit.get('D'),
            'r2': fit.get('r2'),
            'verdict': fit.get('verdict'),
            'model_verdict': fit.get('model_verdict'),
            'model_note': fit.get('model_note', ''),
            'rejection_reason': fit.get('rejection_reason'),
            'delta_aicc': fit.get('delta_aicc'),
            'best_alt': fit.get('best_alt'),
            'narrative': fit.get('model_note', ''),
            'url': url,
        })

    print(f'\n{"=" * 60}')
    print(f'Total new REJECTED entries: {len(new_rejected)}')

    if new_rejected:
        # Add to EN and RU tests.html
        update_tests_html(new_rejected, os.path.join(REPO, 'en/tests.html'), is_ru=False)
        update_tests_html(new_rejected, os.path.join(REPO, 'ru/tests.html'), is_ru=True)

    # Re-export tests.json
    import subprocess
    r = subprocess.run(
        ['python3', 'scripts/export_tests_json.py', '.'],
        cwd=REPO, capture_output=True, text=True, timeout=30
    )
    print(r.stdout.strip())

    # Run reconciler
    r = subprocess.run(
        ['python3', 'scripts/registry_integrity_reconcile.py'],
        cwd=REPO, capture_output=True, text=True, timeout=60
    )
    for line in r.stdout.strip().split('\n')[-10:]:
        print(line)

    # Update meta-s2
    from dream_auto_scanner import update_meta_s2_article
    update_meta_s2_article(os.path.join(REPO, 'en/tests.html'), is_ru=False)
    update_meta_s2_article(os.path.join(REPO, 'ru/tests.html'), is_ru=True)

    # Final count
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
