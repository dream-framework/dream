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

import os, sys, json, time, re, urllib.request, urllib.parse, csv, io
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime

# Local model comparison — gates every fit before it can enter tests.html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_model_compare import compare as s2_compare, m_s2 as s2_func

OUT_DIR = os.environ.get('SCAN_OUT', '/tmp/dream_scan')
os.makedirs(OUT_DIR, exist_ok=True)

# ── S2 fit WITH model comparison ──
# Returns None if no model fits; otherwise returns a dict that ALWAYS
# includes the verdict ('S2_WINS' | 'S2_TIES' | 'S2_LOSES' | 'NO_FIT')
# plus the full AICc ranking. The caller decides whether to promote
# the entry to tests.html based on the verdict.
def fit_s2(t, R, label='', require_wins=True):
    t = np.array(t, dtype=float)
    R = np.array(R, dtype=float)
    if len(t) < 5: return None
    t = t - t[0]
    if R[0] > 0: R_norm = R / R[0]
    else: R_norm = R

    cmp = s2_compare(t, R_norm, label)
    if not cmp or cmp['verdict'] == 'NO_FIT' or cmp.get('s2') is None:
        return None

    # If S2 doesn't beat the alternatives, skip — don't pollute the registry
    if require_wins and cmp['verdict'] == 'S2_LOSES':
        print(f'    ✗ S2 loses to {cmp["best_alt_name"]} '
              f'(ΔAICc={cmp["delta_aicc"]}) — skipping {label[:40]}')
        return None

    D = cmp['s2']['D']
    s2_verdict = 'EXTRACTION' if D > 1 else ('NATURAL' if D < 0.8 else 'THRESHOLD')

    # Build the human narrative — include the model comparison result
    # so the registry card shows why S2 was promoted.
    if cmp['verdict'] == 'S2_WINS':
        model_note = f'S2 beats {cmp["best_alt_name"]} (ΔAICc={cmp["delta_aicc"]}).'
    else:  # S2_TIES
        model_note = f'S2 ties {cmp["best_alt_name"]} (ΔAICc={cmp["delta_aicc"]}, within ±2).'

    return {
        'D': D,
        'lambda_q': cmp['s2']['lambda_q'],
        'r2': cmp['s2']['r2'],
        'verdict': s2_verdict,
        'model_verdict': cmp['verdict'],      # S2_WINS | S2_TIES
        'model_note': model_note,
        'best_alt': cmp['best_alt_name'],
        'delta_aicc': cmp['delta_aicc'],
        'ranking': cmp['rank'],
        'n': len(t),
        'label': label,
    }

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
    model_note = fit.get('model_note', '')
    if not backend_url:
        # Fallback: template narrative
        D = fit.get('D', 0)
        r2 = fit.get('r2', 0)
        verdict = fit.get('verdict', 'UNKNOWN')
        if verdict == 'EXTRACTION':
            base = f'D={D:.3f}, R²={r2:.4f}. D>1 indicates extraction regime — retention collapses super-exponentially.'
        elif verdict == 'NATURAL':
            base = f'D={D:.3f}, R²={r2:.4f}. D<1 confirms natural retention — heavy-tailed, slow decay.'
        else:
            base = f'D={D:.3f}, R²={r2:.4f}. D near threshold — regime transition zone.'
        return f'{base} {model_note}'.strip()

    try:
        import urllib.request
        msg = (f"S2 fit result: D={fit.get('D')}, R²={fit.get('r2')}, "
               f"verdict={fit.get('verdict')}, dataset={fit.get('label')}. "
               f"{model_note} Write a 1-2 sentence plain-English narrative.")
        payload = json.dumps({'message': msg, 'lang': 'en'}).encode()
        req = urllib.request.Request(f'{backend_url}/groq-chat',
            data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get('reply', '')[:200]
    except:
        return groq_narrate(fit)  # fallback to template

# ═══════════════════════════════════════════════════════════════════════
# DEDUPLICATION & PENDING RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

def load_existing_tests(html_path):
    """Parse the TESTS array from tests.html. Returns list of dicts."""
    if not os.path.exists(html_path):
        return []
    with open(html_path) as f:
        html = f.read()
    # Extract the array body between "const TESTS = [" and "];"
    m = re.search(r'const\s+TESTS\s*=\s*\[(.*?)\n\];', html, re.DOTALL)
    if not m:
        print('  ! Could not parse TESTS array — treating as empty')
        return []
    body = m.group(1)
    # Parse each {...} entry. We use a simple brace-matching parser
    # because the entries are JS object literals (not strict JSON).
    entries = []
    i = 0
    while i < len(body):
        # Find next '{'
        brace = body.find('{', i)
        if brace < 0:
            break
        # Match braces to find the closing '}'
        depth = 0
        j = brace
        in_str = False
        esc = False
        while j < len(body):
            c = body[j]
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth != 0:
            break
        entry_str = body[brace:j+1]
        # Extract key fields by regex (good enough for dedup)
        entry = {}
        for field in ('id', 'name', 'url', 'verdict', 'date', 'D', 'r2'):
            fm = re.search(field + r'\s*:\s*"?(.*?)"?\s*[,}]', entry_str)
            if fm:
                val = fm.group(1)
                if field in ('D', 'r2'):
                    try:
                        entry[field] = float(val) if val != 'null' else None
                    except:
                        entry[field] = None
                else:
                    entry[field] = val
        entries.append(entry)
        i = j + 1
    return entries

def is_duplicate(new_entry, existing_entries):
    """Check if new_entry already exists by URL or name similarity."""
    new_url = new_entry.get('url', '').rstrip('/')
    new_name = new_entry.get('name', '').lower().strip()
    for ex in existing_entries:
        ex_url = ex.get('url', '').rstrip('/')
        ex_name = ex.get('name', '').lower().strip()
        # Match by URL (strongest signal)
        if new_url and ex_url and new_url == ex_url:
            return True
        # Match by name prefix (handles truncated titles)
        if new_name and ex_name:
            if new_name[:40] == ex_name[:40]:
                return True
    return False

def filter_duplicates(new_entries, existing_entries):
    """Remove entries that already exist. Returns (kept, skipped_count)."""
    kept = []
    skipped = 0
    for entry in new_entries:
        if is_duplicate(entry, existing_entries):
            skipped += 1
        else:
            kept.append(entry)
    return kept, skipped

def resolve_pending_arxiv(entry, groq_url=None):
    """
    Try to resolve a PENDING arXiv entry:
    1. Fetch the arXiv abstract page
    2. Look for linked data / code repos
    3. If we find a CSV, download + fit S2
    4. Update the entry with results
    Returns the updated entry (or None if still unresolved).
    """
    url = entry.get('url', '')
    if not url or 'arxiv.org' not in url:
        return None

    # Extract arXiv ID from URL
    m = re.search(r'(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})', url)
    if not m:
        return None
    arxiv_id = m.group(1)

    # Fetch abstract page
    abs_url = f'https://arxiv.org/abs/{arxiv_id}'
    print(f'    🔍 Resolving arXiv {arxiv_id}...')
    data = fetch_url(abs_url, timeout=15)
    if not data:
        return None
    html = data.decode('utf-8', errors='ignore')

    # Look for linked Zenodo / GitHub / data URLs in the abstract page
    data_urls = []
    for pattern in [
        r'href="(https://zenodo\.org/record/\d+)"',
        r'href="(https://doi\.org/10\.5281/zenodo\.\d+)"',
        r'href="(https://github\.com/[^"]+)"',
    ]:
        for m in re.finditer(pattern, html):
            data_urls.append(m.group(1))

    if not data_urls:
        print(f'    ✗ No linked datasets found')
        return None

    # Try each linked URL — look for CSVs
    for durl in data_urls[:2]:  # limit to 2 attempts
        print(f'    → Checking {durl[:60]}...')
        if 'zenodo.org' in durl:
            # Use Zenodo API to list files
            record_id = re.search(r'zenodo\.(\d+)', durl)
            if record_id:
                api_url = f'https://zenodo.org/api/records/{record_id.group(1)}'
                api_data = fetch_url(api_url, timeout=15)
                if api_data:
                    try:
                        rec = json.loads(api_data)
                        files = rec.get('files', [])
                        for f in files:
                            fname = f.get('key', '')
                            if fname.endswith(('.csv', '.tsv', '.txt')):
                                furl = f.get('links', {}).get('self', '')
                                print(f'      Found CSV: {fname}')
                                csv_data = fetch_url(furl, timeout=30)
                                if csv_data:
                                    fpath = os.path.join(OUT_DIR, f'arxiv_{arxiv_id}_{fname}')
                                    with open(fpath, 'wb') as fp:
                                        fp.write(csv_data)
                                    fit = analyze_csv_timeseries(fpath, f'arXiv {arxiv_id}: {fname}')
                                    if fit and 'D' in fit:
                                        narrative = groq_narrate(fit, groq_url if groq_url else None)
                                        print(f'      ✓ RESOLVED: D={fit["D"]:.3f} R²={fit["r2"]:.4f}')
                                        return {
                                            'D': fit['D'],
                                            'r2': fit['r2'],
                                            'verdict': fit['verdict'],
                                            'narrative': narrative,
                                            'name': f'arXiv {arxiv_id} ({fname})',
                                        }
                    except:
                        pass
    print(f'    ✗ Could not extract fitable data')
    return None

# ═══════════════════════════════════════════════════════════════════════
# UPDATE TESTS.HTML
# ═══════════════════════════════════════════════════════════════════════

def update_tests_html(new_entries, html_path):
    """Append new entries to the TESTS array in tests.html."""
    with open(html_path) as f:
        html = f.read()
    
    # Update the LAST_REFRESH timestamp (EST/EDT = America/New_York)
    from datetime import timezone, timedelta
    # America/New_York: EST=-5, EDT=-4. Approximate: check if DST (March-November)
    now_utc = datetime.utcnow()
    # Simple DST check: if month is 3-11 and not (month==3 and day<13) and not (month==11 and day>=7)
    is_dst = now_utc.month >= 3 and now_utc.month <= 11 and not (now_utc.month == 3 and now_utc.day < 13) and not (now_utc.month == 11 and now_utc.day >= 7)
    est_offset = timedelta(hours=-4 if is_dst else -5)
    now_est = now_utc + est_offset
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    refresh_str = f"{now_est.day} {months[now_est.month-1]} {now_est.year} {now_est.hour:02d}:{now_est.minute:02d} EST"
    refresh_pattern = re.compile(r'const LAST_REFRESH\s*=\s*"[^"]*"')
    if refresh_pattern.search(html):
        html = refresh_pattern.sub(f'const LAST_REFRESH = "{refresh_str}"', html)
    else:
        # Add it if not present
        html = html.replace('<script>\n', f'<script>\nconst LAST_REFRESH = "{refresh_str}";\n', 1)
    
    # Find the closing ]; of TESTS array (re is imported at module level)
    match = re.search(r'\n\];', html)
    if not match:
        print('  ✗ Could not find TESTS array closing')
        return
    
    insert_pos = match.start()
    
    # Build new entry strings
    today = datetime.now().strftime('%Y-%m-%d')
    new_js = ''

    def js_str(s):
        """Escape a Python string for safe embedding in a JS double-quoted string literal."""
        if s is None:
            return ''
        s = str(s)
        s = s.replace("\\", "\\\\")
        s = s.replace("\"", "\\\"")
        s = s.replace("\n", " ")
        s = s.replace("\r", " ")
        s = s.replace("\t", " ")
        return s
    for entry in new_entries:
        D_val = f'{entry["D"]:.4f}' if entry.get('D') else 'null'
        r2_val = f'{entry["r2"]:.4f}' if entry.get('r2') else 'null'
        url_val = '"' + js_str(entry.get('url', '')) + '"' if entry.get('url') else 'null'
        eid = js_str(entry.get('id', ''))
        name = js_str(entry.get('name', ''))
        domain = js_str(entry.get('domain', ''))
        verdict = js_str(entry.get('verdict', ''))
        narr = js_str(entry.get('narrative', ''))
        new_js += f'\n  ,{{id:\"auto-{today}-{eid}\",name:\"{name}\",domain:\"{domain}\",D:{D_val},r2:{r2_val},verdict:\"{verdict}\",narrative:\"{narr}\",source:\"auto-scan {today}\",date:\"{today}\",url:{url_val},image:null}}'
    
    html = html[:insert_pos] + new_js + html[insert_pos:]
    
    with open(html_path, 'w') as f:
        f.write(html)
    print(f'  ✓ Added {len(new_entries)} entries to {html_path}')

def update_existing_entry(html_path, entry_id, updates):
    """Update an existing entry in tests.html by id. Replaces D, r2, verdict, narrative, name."""
    if not os.path.exists(html_path) or not entry_id:
        return
    with open(html_path) as f:
        html = f.read()
    # Find the entry by id
    # Entry id format: id:"auto-2026-07-16-arxiv-3" or similar
    id_pattern = re.compile(r'(id:"' + re.escape(entry_id) + r'",.*?)(?=\n  ,\{|\n\];)', re.DOTALL)
    m = id_pattern.search(html)
    if not m:
        return
    old_entry = m.group(1)
    # Build new entry by replacing fields
    new_entry = old_entry
    if 'D' in updates:
        d_val = f'{updates["D"]:.4f}'
        new_entry = re.sub(r'D:.*?,', f'D:{d_val},', new_entry, count=1)
    if 'r2' in updates:
        r2_val = f'{updates["r2"]:.4f}'
        new_entry = re.sub(r'r2:.*?,', f'r2:{r2_val},', new_entry, count=1)
    if 'verdict' in updates:
        v = updates['verdict'].replace('"', '\\"')
        new_entry = re.sub(r'verdict:".*?"', f'verdict:"{v}"', new_entry, count=1)
    if 'narrative' in updates:
        n = updates['narrative'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
        new_entry = re.sub(r'narrative:".*?"', f'narrative:"{n}"', new_entry, count=1)
    if 'name' in updates:
        nm = updates['name'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
        new_entry = re.sub(r'name:".*?"', f'name:"{nm}"', new_entry, count=1)
    html = html.replace(old_entry, new_entry, 1)
    with open(html_path, 'w') as f:
        f.write(html)
    print(f'    ✓ Updated entry {entry_id}: D={updates.get("D","?")}, verdict={updates.get("verdict","?")}')

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
                        'model_verdict': fit.get('model_verdict'),
                        'model_note': fit.get('model_note', ''),
                        'delta_aicc': fit.get('delta_aicc'),
                        'best_alt': fit.get('best_alt'),
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
                        'model_verdict': fit.get('model_verdict'),
                        'model_note': fit.get('model_note', ''),
                        'delta_aicc': fit.get('delta_aicc'),
                        'best_alt': fit.get('best_alt'),
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
                        'model_verdict': fit.get('model_verdict'),
                        'model_note': fit.get('model_note', ''),
                        'delta_aicc': fit.get('delta_aicc'),
                        'best_alt': fit.get('best_alt'),
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
                        'model_verdict': fit.get('model_verdict'),
                        'model_note': fit.get('model_note', ''),
                        'delta_aicc': fit.get('delta_aicc'),
                        'best_alt': fit.get('best_alt'),
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
    s2_wins = sum(1 for r in all_results if r.get('model_verdict') == 'S2_WINS')
    s2_ties = sum(1 for r in all_results if r.get('model_verdict') == 'S2_TIES')
    print(f'  EXTRACTION (D>1): {extraction}')
    print(f'  NATURAL (D<0.8): {natural}')
    print(f'  PENDING: {pending}')
    print(f'  S2 WINS vs alternatives: {s2_wins}')
    print(f'  S2 TIES (within ±2 AICc): {s2_ties}')
    
    # 9. Save results JSON
    results_path = os.path.join(OUT_DIR, 'scan_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'total': len(all_results),
            'results': all_results,
        }, f, indent=2)
    print(f'\n✓ Results: {results_path}')
    
    # 10. DEDUPLICATE against existing tests.html
    print('\n🔍 Checking for duplicates...')
    existing_en = load_existing_tests(tests_html)
    existing_ru = load_existing_tests(tests_html.replace('en/', 'ru/'))
    all_existing = existing_en + existing_ru
    print(f'  Found {len(existing_en)} existing EN entries, {len(existing_ru)} RU entries')

    kept_results, skipped_count = filter_duplicates(all_results, all_existing)
    print(f'  Kept {len(kept_results)} new entries, skipped {skipped_count} duplicates')

    # 10b. RESOLVE PENDING entries from previous runs
    print('\n🔄 Resolving PENDING entries from previous runs...')
    resolved_count = 0
    for ex in existing_en:
        if ex.get('verdict') == 'PENDING' and ex.get('url', '').startswith('http'):
            print(f'  → {ex.get("name", "?")[:50]}')
            resolved = resolve_pending_arxiv(ex, groq_url if groq_url else None)
            if resolved:
                # Update the existing entry in-place in tests.html
                update_existing_entry(tests_html, ex.get('id', ''), resolved)
                update_existing_entry(tests_html.replace('en/', 'ru/'), ex.get('id', ''), resolved)
                resolved_count += 1
    if resolved_count:
        print(f'  ✓ Resolved {resolved_count} PENDING entries')
    else:
        print(f'  (no PENDING entries could be resolved this run)')

    # 10c. Update tests.html with ONLY new (non-duplicate) entries
    if os.path.exists(tests_html):
        update_tests_html(kept_results, tests_html)
    else:
        print(f'  tests.html not found at {tests_html} — skipping update')

    # 11. Also update RU tests.html
    ru_html = tests_html.replace('en/', 'ru/')
    if os.path.exists(ru_html):
        update_tests_html(kept_results, ru_html)
    
    return all_results

if __name__ == '__main__':
    results = main()
    sys.exit(0 if results else 1)
