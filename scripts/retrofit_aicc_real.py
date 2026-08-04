#!/usr/bin/env python3
"""
Retrofit AICc gate using ONLY real downloaded data.
No synthetic curves. If we can't download the actual data, we skip the entry.
"""
import os, sys, json, re, urllib.request, csv, io
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_model_compare import compare as s2_compare

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fetch_url(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-Scanner/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f'    Fetch error: {e}')
        return None

def acf_retention(values, max_lag=None):
    v = np.array(values, dtype=float) - np.mean(values)
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

def download_fred(series_id):
    """Download FRED CSV and extract numeric values."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    data = fetch_url(url, timeout=20)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    for col in range(len(rows[0])):
        vals = []
        for r in rows[1:]:
            if col < len(r):
                try:
                    v = float(r[col])
                    if not np.isnan(v) and not np.isinf(v): vals.append(v)
                except: pass
        if len(vals) >= 30: return np.array(vals)
    return None

def download_usgs():
    """Download USGS earthquake CSV."""
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
    data = fetch_url(url, timeout=30)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    # Find 'mag' column
    header = rows[0]
    mag_col = header.index('mag') if 'mag' in header else 4
    vals = []
    for r in rows[1:]:
        if mag_col < len(r):
            try:
                v = float(r[mag_col])
                if not np.isnan(v) and not np.isinf(v): vals.append(v)
            except: pass
    return np.array(vals) if len(vals) >= 30 else None

def download_worldbank(code):
    """Download World Bank indicator."""
    url = f'https://api.worldbank.org/v2/country/US/indicator/{code}?format=json&per_page=100'
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        wb = json.loads(data)
        if len(wb) > 1 and wb[1]:
            vals = [d['value'] for d in wb[1] if d.get('value') is not None]
            return np.array(vals) if len(vals) >= 20 else None
    except: pass
    return None

# Map entry IDs to real data sources
DATA_SOURCES = {
    # FRED series (these are the 'wb-usa-*' entries that were actually from FRED scout)
    'wb-usa-gdp': ('fred', 'GDP'),
    'wb-usa-cpi': ('fred', 'CPIAUCSL'),
    'wb-usa-unemp': ('fred', 'UNRATE'),
    # USGS
    'usgs-quakes': ('usgs', None),
    # World Bank
    'wb-gdp': ('worldbank', 'NY.GDP.MKTP.CD'),
    'wb-cpi': ('worldbank', 'FP.CPI.TOTL'),
}

def get_real_data(entry_id):
    """Download real data for an entry. Returns (t, R) or None."""
    if entry_id not in DATA_SOURCES:
        return None
    
    source_type, code = DATA_SOURCES[entry_id]
    values = None
    
    if source_type == 'fred':
        print(f'    Downloading FRED: {code}...')
        values = download_fred(code)
    elif source_type == 'usgs':
        print(f'    Downloading USGS earthquakes...')
        values = download_usgs()
    elif source_type == 'worldbank':
        print(f'    Downloading World Bank: {code}...')
        values = download_worldbank(code)
    
    if values is None or len(values) < 20:
        print(f'    ✗ Could not download data')
        return None
    
    print(f'    ✓ Got {len(values)} data points')
    t, R = acf_retention(values)
    if t is None or len(t) < 5:
        print(f'    ✗ ACF too short')
        return None
    
    print(f'    ✓ ACF: {len(t)} lags')
    return t, R

def main():
    print('=' * 60)
    print('AICc RETROFIT — REAL DATA ONLY')
    print('=' * 60)
    
    data = json.loads(open(os.path.join(REPO, 'en/tests.json')).read())
    tests = data['tests']
    
    # Find entries that: (a) have D+r2, (b) don't already have AICc in narrative,
    # (c) have a downloadable data source
    needs_retrofit = []
    for t in tests:
        D = t.get('D')
        r2 = t.get('r2')
        narr = t.get('narrative', '')
        eid = t.get('id', '')
        if D is not None and D > 0 and r2 is not None:
            if 'S2 beats' not in narr and 'S2 ties' not in narr and 'S2 loses' not in narr:
                if eid in DATA_SOURCES:
                    needs_retrofit.append(t)
    
    print(f'\nEntries with downloadable real data: {len(needs_retrofit)}')
    
    retrofits = {}
    for t in needs_retrofit:
        eid = t['id']
        print(f'\n  {eid} (D={t["D"]}, r2={t["r2"]})...')
        
        result = get_real_data(eid)
        if result is None:
            print(f'    → SKIPPED (no real data)')
            continue
        
        t_data, R_data = result
        cmp = s2_compare(t_data, R_data, eid)
        
        if not cmp or cmp.get('verdict') == 'NO_FIT' or not cmp.get('s2'):
            print(f'    → SKIPPED (fit failed)')
            continue
        
        model_verdict = cmp['verdict']
        delta = cmp.get('delta_aicc', 0)
        best_alt = cmp.get('best_alt_name', '')
        
        if model_verdict == 'S2_WINS':
            note = f'S2 beats {best_alt} (ΔAICc={delta}).'
        elif model_verdict == 'S2_TIES':
            note = f'S2 ties {best_alt} (ΔAICc={delta}, within ±2).'
        else:
            note = f'S2 loses to {best_alt} (ΔAICc={delta}).'
        
        print(f'    → {model_verdict}: {note}')
        retrofits[eid] = {
            'model_verdict': model_verdict,
            'delta_aicc': delta,
            'best_alt': best_alt,
            'narrative_note': note,
        }
    
    # Summary
    print(f'\n{"=" * 60}')
    print(f'RETROFIT COMPLETE (REAL DATA ONLY)')
    print(f'{"=" * 60}')
    wins = sum(1 for r in retrofits.values() if r['model_verdict'] == 'S2_WINS')
    ties = sum(1 for r in retrofits.values() if r['model_verdict'] == 'S2_TIES')
    loses = sum(1 for r in retrofits.values() if r['model_verdict'] == 'S2_LOSES')
    print(f'  S2 WINS: {wins}')
    print(f'  S2 TIES: {ties}')
    print(f'  S2 LOSES: {loses}')
    print(f'  Total retrofitted with REAL DATA: {len(retrofits)}')
    print(f'  Skipped (no data available): {len(needs_retrofit) - len(retrofits)}')
    
    # Update tests.html
    if retrofits:
        for lang in ['en', 'ru']:
            path = os.path.join(REPO, lang, 'tests.html')
            with open(path, encoding='utf-8') as f:
                html = f.read()
            
            updated = 0
            for eid, result in retrofits.items():
                note = result['narrative_note']
                # Find entry by id and append note to narrative
                pattern = re.compile(r'(id:"' + re.escape(eid) + r'"[^}]*?narrative:"((?:[^"\\]|\\.)*)")')
                m = pattern.search(html)
                if not m: continue
                old_narr = m.group(2)
                if 'S2 beats' in old_narr or 'S2 ties' in old_narr or 'S2 loses' in old_narr: continue
                new_narr = old_narr + ' ' + note
                new_narr_escaped = new_narr.replace('\\', '\\\\').replace('"', '\\"')
                old_full = m.group(1)
                new_full = old_full.replace(m.group(2), new_narr_escaped)
                html = html.replace(old_full, new_full, 1)
                updated += 1
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f'  {lang}/tests.html: updated {updated} entries')
    
    return retrofits

if __name__ == '__main__':
    main()
