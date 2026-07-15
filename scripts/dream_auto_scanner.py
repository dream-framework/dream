#!/usr/bin/env python3
"""
DREAM Auto-Scanner: search scientific repositories, download datasets,
fit S2 retention law, update test registry automatically.

Sources:
  1. Zenodo — research datasets (REST API, no key)
  2. arXiv — papers with data (OAI-PMH API)
  3. FRED — economic time series (CSV download)
  4. USGS — earthquakes (REST API)
  5. NOAA — solar/weather (HTTPS)
  6. World Bank — GDP/CPI (REST API)

Pipeline: search → download → parse → fit S2 → narrate → update tests.html
"""

import os, sys, json, time, urllib.request, urllib.parse, csv, io
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime

OUT_DIR = os.environ.get('SCAN_OUT', '/tmp/dream_scan')
os.makedirs(OUT_DIR, exist_ok=True)

# ── S2 model ──
def s2_func(t, A, lambda_q, D):
    return A * np.exp(-np.power(np.maximum(t, 0.001) / lambda_q, D))

def fit_s2(t, R, label=''):
    t = np.array(t, dtype=float)
    R = np.array(R, dtype=float)
    if len(t) < 5: return None
    t = t - t[0]
    if R[0] > 0: R_norm = R / R[0]
    else: R_norm = R
    try:
        p0 = [1.0, t[len(t)//2], 0.5]
        bounds = ([0.01, 0.001, 0.01], [2.0, 1e6, 10.0])
        popt, _ = curve_fit(s2_func, t, R_norm, p0=p0, maxfev=20000, bounds=bounds)
        A, lambda_q, D = popt
        R_pred = s2_func(t, *popt)
        ss_res = np.sum((R_norm - R_pred) ** 2)
        ss_tot = np.sum((R_norm - np.mean(R_norm)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        verdict = 'EXTRACTION' if D > 1 else ('NATURAL' if D < 0.8 else 'THRESHOLD')
        return {'D': round(float(D), 4), 'lambda_q': round(float(lambda_q), 2),
                'r2': round(float(r2), 4), 'verdict': verdict, 'n': len(t),
                'label': label}
    except Exception as e:
        return {'error': str(e), 'label': label}

def retention_curve(values, max_lag=None):
    """ACF of |demeaned values| — the retention curve."""
    v = np.array(values, dtype=float)
    v = v - np.mean(v)
    n = len(v)
    if max_lag is None: max_lag = min(n // 4, 200)
    max_lag = min(max_lag, n // 4)
    if max_lag < 5: return None, None
    var = np.dot(v, v) / n
    if var == 0: return None, None
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.dot(v[:n-lag], v[lag:]) / (n * var)
    return np.arange(max_lag), acf

def fetch_url(url, timeout=30):
    """Fetch URL with User-Agent header."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'DREAM-Scanner/1.0 (https://dream-framework.github.io/dream/)'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f'  ✗ Fetch error: {e}')
        return None

def parse_csv(text):
    """Parse CSV text, return list of rows."""
    reader = csv.reader(io.StringIO(text.decode('utf-8') if isinstance(text, bytes) else text))
    return list(reader)

# ═══════════════════════════════════════════════════════════════════════
# DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════

def scan_zenodo(query='stretched exponential decay', size=10):
    """Search Zenodo for datasets."""
    print(f'\n📡 Zenodo: "{query}"')
    url = f'https://zenodo.org/api/records?q={urllib.parse.quote(query)}&size={size}&sort=mostrecent'
    data = fetch_url(url)
    if not data: return []
    try:
        results = json.loads(data)
    except: return []
    
    found = []
    for hit in results.get('hits', {}).get('hits', []):
        title = hit.get('metadata', {}).get('title', '')
        doi = hit.get('doi', '')
        files = hit.get('files', [])
        for f in files:
            if f.get('key', '').endswith(('.csv', '.json', '.tsv', '.txt')):
                found.append({
                    'source': 'zenodo',
                    'title': title,
                    'url': f.get('links', {}).get('self', ''),
                    'filename': f.get('key', ''),
                    'doi': doi,
                    'format': f.get('key', '').split('.')[-1],
                })
    print(f'  Found {len(found)} downloadable files')
    return found

def scan_arxiv(query='stretched exponential retention decay', max_results=10):
    """Search arXiv for papers."""
    print(f'\n📡 arXiv: "{query}"')
    url = f'http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&max_results={max_results}'
    data = fetch_url(url)
    if not data: return []
    text = data.decode('utf-8')
    
    # Simple XML parsing for entries
    import re
    entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
    found = []
    for entry in entries:
        title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
        id_m = re.search(r'<id>(.*?)</id>', entry)
        if title_m and id_m:
            found.append({
                'source': 'arxiv',
                'title': title_m.group(1).strip().replace('\n', ' '),
                'url': id_m.group(1).strip(),
                'format': 'paper',
            })
    print(f'  Found {len(found)} papers')
    return found

def scan_fred(series_ids=None):
    """Download FRED economic time series (no key needed for CSV)."""
    if series_ids is None:
        series_ids = ['GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'M2SL', 'DEXUSEU',
                      'SP500', 'VIXCLS', 'T10YIE', 'DGS10']
    print(f'\n📡 FRED: {len(series_ids)} series')
    found = []
    for sid in series_ids:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
        data = fetch_url(url, timeout=15)
        if not data: continue
        filepath = os.path.join(OUT_DIR, f'fred_{sid}.csv')
        with open(filepath, 'wb') as f:
            f.write(data)
        found.append({
            'source': 'fred',
            'title': f'FRED: {sid}',
            'url': url,
            'filename': filepath,
            'format': 'csv',
            'series_id': sid,
        })
    print(f'  Downloaded {len(found)} series')
    return found

def scan_usgs():
    """Download USGS earthquake data."""
    print('\n📡 USGS Earthquakes')
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
    data = fetch_url(url, timeout=30)
    if not data: return []
    filepath = os.path.join(OUT_DIR, 'usgs_earthquakes.csv')
    with open(filepath, 'wb') as f:
        f.write(data)
    return [{
        'source': 'usgs',
        'title': 'USGS Earthquakes (30 days)',
        'url': url,
        'filename': filepath,
        'format': 'csv',
    }]

def scan_worldbank():
    """Download World Bank indicators."""
    print('\n📡 World Bank')
    indicators = [('NY.GDP.MKTP.CD', 'GDP'), ('FP.CPI.TOTL', 'CPI'),
                  ('SL.UEM.TOTL.ZS', 'Unemployment')]
    found = []
    for code, name in indicators:
        url = f'https://api.worldbank.org/v2/country/US/indicator/{code}?format=json&per_page=100'
        data = fetch_url(url, timeout=15)
        if not data: continue
        try:
            wb = json.loads(data)
            if len(wb) > 1 and wb[1]:
                values = [d['value'] for d in wb[1] if d.get('value') is not None]
                filepath = os.path.join(OUT_DIR, f'wb_{name}.json')
                with open(filepath, 'w') as f:
                    json.dump(values, f)
                found.append({
                    'source': 'worldbank',
                    'title': f'World Bank: {name}',
                    'url': url,
                    'filename': filepath,
                    'format': 'json',
                    'values': values,
                })
        except: pass
    print(f'  Downloaded {len(found)} indicators')
    return found

# ═══════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════

def analyze_csv_timeseries(filepath, title):
    """Load CSV, find numeric column, compute retention curve, fit S2."""
    try:
        with open(filepath) as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 20: return None
        
        # Find first numeric column (skip date columns)
        header = rows[0] if rows else []
        for col_idx in range(len(header)):
            values = []
            for row in rows[1:]:
                if col_idx < len(row):
                    try:
                        v = float(row[col_idx])
                        if not np.isnan(v) and not np.isinf(v):
                            values.append(v)
                    except: pass
            if len(values) >= 20:
                taus, acf = retention_curve(values)
                if taus is not None and acf is not None:
                    fit = fit_s2(taus, acf, title)
                    if fit and 'D' in fit:
                        return fit
        return None
    except Exception as e:
        return {'error': str(e), 'label': title}

def analyze_json_values(values, title):
    """Fit S2 to ACF of a numeric array."""
    if not values or len(values) < 20: return None
    taus, acf = retention_curve(values)
    if taus is None: return None
    return fit_s2(taus, acf, title)

# ═══════════════════════════════════════════════════════════════════════
# GROQ NARRATION (optional — works without Groq)
# ═══════════════════════════════════════════════════════════════════════

def groq_narrate(fit, backend_url=None):
    """Ask Groq to write a 1-2 sentence narrative for the fit."""
    if not backend_url:
        # Fallback: template narrative
        D = fit.get('D', 0)
        r2 = fit.get('r2', 0)
        verdict = fit.get('verdict', 'UNKNOWN')
        if verdict == 'EXTRACTION':
            return f'D={D:.3f}, R²={r2:.4f}. D>1 indicates extraction regime — retention collapses super-exponentially.'
        elif verdict == 'NATURAL':
            return f'D={D:.3f}, R²={r2:.4f}. D<1 confirms natural retention — heavy-tailed, slow decay.'
        else:
            return f'D={D:.3f}, R²={r2:.4f}. D near threshold — regime transition zone.'
    
    try:
        import urllib.request
        msg = f"S2 fit result: D={fit.get('D')}, R²={fit.get('r2')}, verdict={fit.get('verdict')}, dataset={fit.get('label')}. Write a 1-2 sentence plain-English narrative."
        payload = json.dumps({'message': msg, 'lang': 'en'}).encode()
        req = urllib.request.Request(f'{backend_url}/groq-chat',
            data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get('reply', '')[:200]
    except:
        return groq_narrate(fit)  # fallback to template

# ═══════════════════════════════════════════════════════════════════════
# UPDATE TESTS.HTML
# ═══════════════════════════════════════════════════════════════════════

def update_tests_html(new_entries, html_path):
    """Append new entries to the TESTS array in tests.html."""
    with open(html_path) as f:
        html = f.read()
    
    # Find the closing ]; of TESTS array
    import re
    match = re.search(r'\n\];', html)
    if not match:
        print('  ✗ Could not find TESTS array closing')
        return
    
    insert_pos = match.start()
    
    # Build new entry strings
    today = datetime.now().strftime('%Y-%m-%d')
    new_js = ''
    for entry in new_entries:
        D_val = f'{entry["D"]:.4f}' if entry.get('D') else 'null'
        r2_val = f'{entry["r2"]:.4f}' if entry.get('r2') else 'null'
        new_js += f'\n  ,{{id:"auto-{today}-{entry["id"]}",name:"{entry["name"]}",domain:"{entry["domain"]}",D:{D_val},r2:{r2_val},verdict:"{entry["verdict"]}",narrative:"{entry["narrative"]}",source:"auto-scan {today}",date:"{today}",url:{("\"" + entry["url"] + "\"") if entry.get("url") else "null"},image:null}}'
    
    html = html[:insert_pos] + new_js + html[insert_pos:]
    
    with open(html_path, 'w') as f:
        f.write(html)
    print(f'  ✓ Added {len(new_entries)} entries to {html_path}')

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('DREAM AUTO-SCANNER')
    print(f'Date: {datetime.now().isoformat()}')
    print('=' * 60)
    
    groq_url = os.environ.get('GROQ_BACKEND_URL', '')
    tests_html = os.environ.get('TESTS_HTML', 'en/tests.html')
    
    all_results = []
    
    # 1. Search for datasets
    zenodo_results = scan_zenodo('stretched exponential decay')
    zenodo_results += scan_zenodo('retention forgetting curve data')
    arxiv_results = scan_arxiv('stretched exponential retention decay')
    
    # 2. Download known time series
    fred_results = scan_fred()
    usgs_results = scan_usgs()
    wb_results = scan_worldbank()
    
    # 3. Download and analyze Zenodo CSVs
    print('\n📊 Analyzing Zenodo datasets...')
    for item in zenodo_results[:5]:  # limit to 5
        if item['format'] in ('csv', 'tsv', 'txt'):
            data = fetch_url(item['url'])
            if data:
                filepath = os.path.join(OUT_DIR, f'zenodo_{item["filename"]}')
                with open(filepath, 'wb') as f:
                    f.write(data)
                fit = analyze_csv_timeseries(filepath, item['title'][:60])
                if fit and 'D' in fit:
                    narrative = groq_narrate(fit, groq_url if groq_url else None)
                    all_results.append({
                        'id': f'zenodo-{len(all_results)}',
                        'name': f'Zenodo: {item["title"][:50]}',
                        'domain': 'scouting',
                        'D': fit['D'], 'r2': fit['r2'],
                        'verdict': fit['verdict'],
                        'narrative': narrative,
                        'url': item.get('doi', item.get('url', '')),
                    })
                    print(f'  ✓ D={fit["D"]:.3f} R²={fit["r2"]:.4f} — {item["title"][:40]}')
    
    # 4. Analyze arXiv papers (just record as pending — can't fit without data)
    print('\n📄 Recording arXiv papers...')
    for item in arxiv_results[:5]:
        all_results.append({
            'id': f'arxiv-{len(all_results)}',
            'name': f'arXiv: {item["title"][:50]}',
            'domain': 'scouting',
            'D': None, 'r2': None,
            'verdict': 'PENDING',
            'narrative': f'Paper found via arXiv search. Data extraction and S2 fit pending. {item["title"][:80]}',
            'url': item.get('url', ''),
        })
        print(f'  ✓ {item["title"][:60]}')
    
    # 5. Analyze FRED economic data
    print('\n📊 Analyzing FRED economic data...')
    for item in fred_results:
        fit = analyze_csv_timeseries(item['filename'], f'FRED: {item["series_id"]}')
        if fit and 'D' in fit:
            narrative = groq_narrate(fit, groq_url if groq_url else None)
            all_results.append({
                'id': f'fred-{item["series_id"]}',
                'name': f'FRED {item["series_id"]} (ACF retention)',
                'domain': 'financial',
                'D': fit['D'], 'r2': fit['r2'],
                'verdict': fit['verdict'],
                'narrative': narrative,
                'url': item.get('url', ''),
            })
            print(f'  ✓ {item["series_id"]}: D={fit["D"]:.3f} R²={fit["r2"]:.4f} {fit["verdict"]}')
    
    # 6. Analyze USGS
    print('\n📊 Analyzing USGS earthquakes...')
    for item in usgs_results:
        fit = analyze_csv_timeseries(item['filename'], 'USGS Earthquakes')
        if fit and 'D' in fit:
            narrative = groq_narrate(fit, groq_url if groq_url else None)
            all_results.append({
                'id': 'usgs-quakes',
                'name': 'USGS Earthquakes (30d, ACF)',
                'domain': 'live',
                'D': fit['D'], 'r2': fit['r2'],
                'verdict': fit['verdict'],
                'narrative': narrative,
                'url': item.get('url', ''),
            })
            print(f'  ✓ D={fit["D"]:.3f} R²={fit["r2"]:.4f} {fit["verdict"]}')
    
    # 7. Analyze World Bank
    print('\n📊 Analyzing World Bank data...')
    for item in wb_results:
        fit = analyze_json_values(item.get('values', []), f'World Bank: {item["title"]}')
        if fit and 'D' in fit:
            narrative = groq_narrate(fit, groq_url if groq_url else None)
            all_results.append({
                'id': f'wb-{item["title"].split(": ")[1].lower()[:10]}',
                'name': f'{item["title"]} (ACF retention)',
                'domain': 'financial',
                'D': fit['D'], 'r2': fit['r2'],
                'verdict': fit['verdict'],
                'narrative': narrative,
                'url': item.get('url', ''),
            })
            print(f'  ✓ D={fit["D"]:.3f} R²={fit["r2"]:.4f} {fit["verdict"]}')
    
    # 8. Summary
    print(f'\n{"="*60}')
    print(f'SCAN COMPLETE')
    print(f'{"="*60}')
    print(f'Total new entries: {len(all_results)}')
    extraction = sum(1 for r in all_results if r.get('verdict') == 'EXTRACTION')
    natural = sum(1 for r in all_results if r.get('verdict') == 'NATURAL')
    pending = sum(1 for r in all_results if r.get('verdict') == 'PENDING')
    print(f'  EXTRACTION (D>1): {extraction}')
    print(f'  NATURAL (D<0.8): {natural}')
    print(f'  PENDING: {pending}')
    
    # 9. Save results JSON
    results_path = os.path.join(OUT_DIR, 'scan_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'total': len(all_results),
            'results': all_results,
        }, f, indent=2)
    print(f'\n✓ Results: {results_path}')
    
    # 10. Update tests.html if path provided
    if os.path.exists(tests_html):
        update_tests_html(all_results, tests_html)
    else:
        print(f'  tests.html not found at {tests_html} — skipping update')
    
    # 11. Also update RU tests.html
    ru_html = tests_html.replace('en/', 'ru/')
    if os.path.exists(ru_html):
        # Same entries but with source label in Russian
        for entry in all_results:
            entry['narrative'] = entry['narrative']  # keep English narrative (Groq handles RU if configured)
        update_tests_html(all_results, ru_html)
    
    return all_results

if __name__ == '__main__':
    results = main()
    sys.exit(0 if results else 1)
