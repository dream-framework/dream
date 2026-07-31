#!/usr/bin/env python3
"""
Backfill model_verdict, delta_aicc, best_alt for entries that have D but no
model comparison fields.

These entries were added by update_tests_html() before it was fixed to write
model_verdict fields. The D and r² are correct, but the structured comparison
data was lost. This script re-fetches the data, re-runs the S2 fit WITH model
comparison, and patches the missing fields into tests.html.

Only touches entries that have D but no model_verdict.
"""

import os, sys, re, json, csv, io, urllib.request, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from s2_model_compare import compare as s2_compare
from dream_auto_scanner import retention_curve, fit_s2, fetch_url, analyze_csv_timeseries, analyze_json_values

def parse_entries(html_path):
    """Parse all entries from tests.html."""
    with open(html_path) as f:
        html = f.read()
    m = re.search(r'const\s+TESTS\s*=\s*\[(.*?)\n\];', html, re.DOTALL)
    if not m:
        return [], html
    body = m.group(1)
    
    entries = []
    i = 0
    while i < len(body):
        brace = body.find('{', i)
        if brace < 0:
            break
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
        
        entry = {}
        for field in ['id', 'name', 'domain', 'verdict', 'model_verdict', 'narrative',
                       'source', 'date', 'url', 'image']:
            fm = re.search(r'(?:^|,)\s*' + field + r'\s*:\s*"((?:[^"\\]|\\.)*)"', entry_str)
            if fm:
                entry[field] = fm.group(1).replace('\\"', '"').replace('\\\\', '\\')
        for field in ['D', 'r2', 'delta_aicc']:
            fm = re.search(r'(?:^|,)\s*' + field + r'\s*:\s*(-?[\d]+(?:\.[\d]+)?)', entry_str)
            if fm:
                try:
                    entry[field] = float(fm.group(1))
                except:
                    entry[field] = None
            elif re.search(r'(?:^|,)\s*' + field + r'\s*:\s*null', entry_str):
                entry[field] = None
        
        entry['_raw'] = entry_str
        entry['_start'] = brace
        entry['_end'] = j + 1
        entries.append(entry)
        i = j + 1
    
    return entries, html


def refit_entry(entry):
    """Re-fetch data for an entry and re-run the S2 fit with model comparison.
    Returns (model_verdict, delta_aicc, best_alt, model_note) or None."""
    url = entry.get('url', '')
    name = entry.get('name', '')
    D = entry.get('D')
    
    if not url or D is None:
        return None
    
    print(f'  Refitting: {name[:50]} (D={D:.3f})')
    
    values = None
    
    # Determine source type and fetch
    if 'fred.stlouisfed.org' in url:
        # FRED CSV
        data = fetch_url(url, timeout=15)
        if not data:
            return None
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        for col_idx in range(1, min(len(rows[0]) if rows else 0, 5)):
            vals = []
            for row in rows[1:]:
                if col_idx < len(row):
                    try:
                        v = float(row[col_idx])
                        if not np.isnan(v) and not np.isinf(v):
                            vals.append(v)
                    except:
                        pass
            if len(vals) >= 20:
                values = vals
                break
    elif 'api.binance.com' in url:
        # Binance klines
        data = fetch_url(url, timeout=15)
        if not data:
            return None
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        arr = json.loads(text)
        values = [float(candle[4]) for candle in arr if len(candle) > 4]  # close price
    elif 'api.worldbank.org' in url:
        data = fetch_url(url, timeout=15)
        if not data:
            return None
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        obj = json.loads(text)
        if len(obj) > 1 and isinstance(obj[1], list):
            values = [float(o['value']) for o in obj[1] if o.get('value') is not None]
    elif 'open-meteo' in url:
        data = fetch_url(url, timeout=15)
        if not data:
            return None
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        obj = json.loads(text)
        if 'daily' in obj:
            for key in obj['daily']:
                if key.endswith('_mean') or key.endswith('_sum') or key.endswith('_max'):
                    values = obj['daily'][key]
                    break
    elif 'zenodo' in url or '10.5281' in url:
        # Zenodo — try to download the CSV
        # URL might be a DOI, need to resolve
        return None  # Skip Zenodo for now — too complex to refetch
    elif 'data.giss.nasa.gov' in url:
        data = fetch_url(url, timeout=15)
        if not data:
            return None
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        # GISS format: skip header rows, monthly data
        for col_idx in range(1, 13):
            vals = []
            for row in rows[1:]:
                if col_idx < len(row):
                    try:
                        v = float(row[col_idx])
                        if v != -9999 and not np.isnan(v):
                            vals.append(v)
                    except:
                        pass
            if len(vals) >= 20:
                values = vals
                break
    elif 'CSSEGISandData' in url or 'covid' in url.lower():
        data = fetch_url(url, timeout=15)
        if not data:
            return None
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        # COVID: daily cumulative counts — take daily new = diff
        if len(rows) > 1 and len(rows[0]) > 4:
            cumul = []
            for row in rows[1:]:
                try:
                    v = float(row[-1])
                    cumul.append(v)
                except:
                    pass
            if len(cumul) > 20:
                values = [max(0, cumul[i] - cumul[i-1]) for i in range(1, len(cumul))]
    
    if not values or len(values) < 20:
        print(f'    ✗ Could not fetch/parse data from {url[:60]}')
        return None
    
    # Compute ACF
    taus, acf = retention_curve(values)
    if taus is None or acf is None:
        return None
    
    # Run model comparison
    cmp = s2_compare(taus, acf, name[:40])
    if not cmp or cmp.get('verdict') == 'NO_FIT' or cmp.get('s2') is None:
        return None
    
    return {
        'model_verdict': cmp['verdict'],
        'delta_aicc': cmp.get('delta_aicc'),
        'best_alt': cmp.get('best_alt_name', ''),
        'model_note': f'S2 {cmp["verdict"].replace("S2_","").lower()} {cmp.get("best_alt_name","")} (ΔAICc={cmp.get("delta_aicc",0):.2f}).' if cmp.get('delta_aicc') is not None else '',
        'D_new': cmp['s2']['D'],
        'r2_new': cmp['s2']['r2'],
    }


def patch_entry_in_html(html_path, entry, new_fields):
    """Patch an entry's raw string in tests.html to add model comparison fields."""
    with open(html_path) as f:
        html = f.read()
    
    # Find the entry by its unique URL + name
    name = entry.get('name', '')
    url = entry.get('url', '')
    
    # Build a search pattern — find the entry by name
    # Insert model_verdict fields after verdict:"..."
    old_raw = entry['_raw']
    
    # Build the new fields string
    model_fields = ''
    if new_fields.get('model_verdict'):
        model_fields += f',model_verdict:"{new_fields["model_verdict"]}"'
    if new_fields.get('delta_aicc') is not None:
        model_fields += f',delta_aicc:{new_fields["delta_aicc"]:.4f}'
    if new_fields.get('best_alt'):
        model_fields += f',best_alt:"{new_fields["best_alt"]}"'
    if new_fields.get('model_note'):
        # Escape the note
        note = new_fields['model_note'].replace('\\', '\\\\').replace('"', '\\"')
        model_fields += f',model_note:"{note}"'
    
    # Insert after verdict:"..." 
    # The raw string has: ...verdict:"EXTRACTION",narrative:...
    # We want:             ...verdict:"EXTRACTION",model_verdict:"S2_WINS",...,narrative:...
    new_raw = re.sub(
        r'(verdict:"[^"]*")',
        r'\1' + model_fields,
        old_raw,
        count=1
    )
    
    if new_raw == old_raw:
        print(f'    ✗ Could not patch (verdict field not found)')
        return False
    
    # Replace in HTML
    html = html.replace(old_raw, new_raw, 1)
    
    with open(html_path, 'w') as f:
        f.write(html)
    
    return True


def main():
    print('=== Backfill Model Comparison Fields ===')
    
    for lang in ('en', 'ru'):
        html_path = f'{lang}/tests.html'
        print(f'\n--- {lang.upper()} ---')
        
        entries, html = parse_entries(html_path)
        print(f'Parsed {len(entries)} entries')
        
        # Find entries with D but no model_verdict
        needs_backfill = [e for e in entries 
                         if e.get('D') is not None 
                         and not e.get('model_verdict')]
        print(f'Need backfill: {len(needs_backfill)}')
        
        if not needs_backfill:
            continue
        
        success = 0
        for entry in needs_backfill:
            name = entry.get('name', '?')[:55]
            D = entry.get('D', 0)
            print(f'\n  {name} (D={D:.3f})')
            
            result = refit_entry(entry)
            if result:
                print(f'    ✓ model_verdict={result["model_verdict"]}, '
                      f'ΔAICc={result["delta_aicc"]:.2f}, best_alt={result["best_alt"]}')
                if patch_entry_in_html(html_path, entry, result):
                    success += 1
            else:
                print(f'    ✗ Could not refit')
        
        print(f'\n  Backfilled: {success}/{len(needs_backfill)}')
    
    # Re-run reconciler to fix narratives
    print('\n=== Re-running reconciler ===')
    r = subprocess.run(['python3', 'scripts/registry_integrity_reconcile.py'],
                     capture_output=True, text=True, timeout=60)
    lines = r.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f'  {line}')


if __name__ == '__main__':
    main()
