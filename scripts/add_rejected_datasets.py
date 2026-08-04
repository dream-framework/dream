#!/usr/bin/env python3
"""
Add previously rejected datasets (S2_LOSES) to the registry.
These were hidden by require_wins=True. Now we record them honestly.
"""
import sys, os, json, csv, io, urllib.request, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from dream_auto_scanner import fetch_url, retention_curve, fit_s2, load_existing_tests, update_tests_html
from s2_model_compare import compare as s2_compare

def fetch_fred(sid):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
    data = fetch_url(url, timeout=15)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    vals = []
    for row in rows[1:]:
        if len(row) >= 2 and row[1] != '.':
            try:
                v = float(row[1])
                if not np.isnan(v) and not np.isinf(v): vals.append(v)
            except: pass
    return vals if len(vals) >= 20 else None

def main():
    print('=== Adding previously rejected datasets to registry ===\n')
    
    # All datasets that were previously rejected (S2 lost)
    # We re-fetch, re-fit, and add them with model_verdict=S2_LOSES
    datasets = []
    
    # FRED series that lost
    for sid, name in [('VIXCLS','VIX Volatility Index'), ('T10YIE','10-Year Breakeven Inflation'), ('DGS10','10-Year Treasury Rate')]:
        vals = fetch_fred(sid)
        if vals:
            datasets.append((f'FRED {sid} ({name})', 'financial', f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}', vals))
            print(f'  Fetched FRED {sid}: {len(vals)} obs')
    
    # NASA GISS
    url = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'
    data = fetch_url(url, timeout=15)
    if data:
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        lines = text.strip().split('\n')
        vals = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 13:
                for m in range(1, 13):
                    try:
                        v = float(parts[m])
                        if v != -9999 and not np.isnan(v): vals.append(v)
                    except: pass
        if len(vals) >= 50:
            datasets.append(('NASA GISS Global Temperature (monthly)', 'environmental', url, vals))
            print(f'  Fetched NASA GISS: {len(vals)} obs')
    
    # HadCRUT
    url = 'https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv'
    data = fetch_url(url, timeout=15)
    if data:
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        vals = []
        for row in rows[1:]:
            if len(row) >= 2:
                try: vals.append(float(row[1]))
                except: pass
        if len(vals) >= 50:
            datasets.append(('Global Temperature (HadCRUT monthly)', 'environmental', url, vals))
            print(f'  Fetched HadCRUT: {len(vals)} obs')
    
    # COVID
    for kind in ['confirmed', 'deaths']:
        url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{kind}_global.csv'
        data = fetch_url(url, timeout=15)
        if data:
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = list(csv.reader(io.StringIO(text)))
            daily_sums = []
            for col in range(4, len(rows[0])):
                total = 0
                for row in rows[1:]:
                    if col < len(row):
                        try: total += int(row[col])
                        except: pass
                daily_sums.append(total)
            daily_new = [max(0, daily_sums[i] - daily_sums[i-1]) for i in range(1, len(daily_sums))]
            if len(daily_new) >= 50:
                datasets.append((f'COVID-19 {kind} (daily new, global)', 'ecological', url, daily_new))
                print(f'  Fetched COVID {kind}: {len(daily_new)} obs')
    
    # NDBC buoys
    for buoy, name in [('41001','E Hatteras'), ('51001','NW Hawaiian')]:
        url = f'https://www.ndbc.noaa.gov/data/realtime2/{buoy}.txt'
        data = fetch_url(url, timeout=15)
        if data:
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
            if len(vals) >= 50:
                datasets.append((f'NDBC Buoy {buoy} ({name}) wave height', 'oceanography', url, vals))
                print(f'  Fetched NDBC {buoy}: {len(vals)} obs')
    
    # Energy-Charts
    for zone, name in [('DE-LU','Germany-Luxembourg'), ('FR','France'), ('NL','Netherlands'), ('CH','Switzerland')]:
        url = f'https://api.energy-charts.info/price?bzn={zone}&start=2023-01-01T00%3A00%2B01%3A00&end=2023-06-30T00%3A00%2B01%3A00'
        data = fetch_url(url, timeout=15)
        if data:
            try:
                obj = json.loads(data)
                if 'price' in obj:
                    prices = [float(p) for p in obj['price'] if p is not None]
                    if len(prices) >= 50:
                        datasets.append((f'Energy-Charts {name} electricity spot price', 'energy', url, prices))
                        print(f'  Fetched Energy {zone}: {len(prices)} obs')
            except: pass
    
    # Wikipedia pageviews (the ones that lost)
    for title in ['Albert_Einstein', 'COVID-19_pandemic', 'Climate_change', 'Artificial_intelligence']:
        url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{title}/daily/2020010100/2025123100'
        data = fetch_url(url, timeout=15)
        if data:
            try:
                obj = json.loads(data)
                if 'items' in obj:
                    vals = [item['views'] for item in obj['items']]
                    if len(vals) >= 50:
                        datasets.append((f'Wikipedia: {title.replace("_"," ")} pageviews', 'cultural', url, vals))
                        print(f'  Fetched Wikipedia {title}: {len(vals)} obs')
            except: pass
    
    # NASA POWER
    for lat, lon, name in [(40.0,-105.0,'Colorado'), (52.5,13.4,'Berlin'), (-33.9,18.4,'Cape Town')]:
        url = f'https://power.larc.nasa.gov/api/temporal/hourly/point?start=20230101&end=20231231&parameters=ALLSKY_SFC_SW_DWN&longitude={lon}&latitude={lat}&community=RE&format=JSON'
        data = fetch_url(url, timeout=15)
        if data:
            try:
                obj = json.loads(data)
                if 'properties' in obj and 'parameter' in obj['properties']:
                    vals = [float(v) for v in obj['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values() if v is not None and v >= 0]
                    if len(vals) >= 50:
                        datasets.append((f'NASA POWER {name} solar irradiance', 'solar_energy', url, vals))
                        print(f'  Fetched NASA POWER {name}: {len(vals)} obs')
            except: pass
    
    # Water Quality
    url = 'https://www.waterqualitydata.us/data/Result/search?siteid=USGS-01594440&characteristicName=Nitrate&mimeType=csv&zip=no'
    data = fetch_url(url, timeout=20)
    if data:
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) > 5:
            header = rows[0]
            val_idx = -1
            for i, h in enumerate(header):
                if h == 'ResultMeasureValue': val_idx = i; break
            if val_idx >= 0:
                vals = []
                for row in rows[1:]:
                    if val_idx < len(row):
                        try:
                            v = float(row[val_idx])
                            if not np.isnan(v) and v >= 0: vals.append(v)
                        except: pass
                if len(vals) >= 20:
                    datasets.append(('Water Quality Portal: Nitrate at USGS-01594440', 'chemistry', url, vals))
                    print(f'  Fetched Water Quality: {len(vals)} obs')
    
    # SWPC X-ray
    url = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json'
    data = fetch_url(url, timeout=15)
    if data:
        try:
            obj = json.loads(data)
            vals = [float(item['flux']) for item in obj if isinstance(item, dict) and item.get('energy')=='0.1-0.8nm' and item.get('flux') and float(item['flux'])>0]
            if len(vals) >= 50:
                datasets.append(('SWPC GOES X-ray Flux (0.1-0.8nm band)', 'solar_physics', url, vals))
                print(f'  Fetched SWPC X-ray: {len(vals)} obs')
        except: pass
    
    # World Bank Unemployment
    url = 'https://api.worldbank.org/v2/country/US/indicator/SL.UEM.TOTL.ZS?format=json&per_page=100'
    data = fetch_url(url, timeout=15)
    if data:
        try:
            obj = json.loads(data)
            if len(obj) > 1 and obj[1]:
                vals = [d['value'] for d in obj[1] if d.get('value') is not None]
                if len(vals) >= 20:
                    datasets.append(('World Bank: USA Unemployment (annual)', 'financial', url, vals))
                    print(f'  Fetched WB Unemployment: {len(vals)} obs')
        except: pass
    
    print(f'\nTotal datasets to add: {len(datasets)}')
    
    # Fit S2 to each and build entries
    new_entries = []
    for name, domain, url, vals in datasets:
        taus, acf = retention_curve(vals)
        if taus is None or acf is None:
            print(f'  ✗ {name[:50]}: ACF failed')
            continue
        fit = fit_s2(taus, acf, name[:60])
        if fit is None:
            print(f'  ✗ {name[:50]}: fit failed')
            continue
        
        entry = {
            'id': f'rejected-{len(new_entries)}',
            'name': f'{name} (ACF retention)',
            'domain': domain,
            'D': fit['D'],
            'r2': fit['r2'],
            'verdict': fit['verdict'],
            'model_verdict': fit['model_verdict'],
            'model_note': fit['model_note'],
            'delta_aicc': fit['delta_aicc'],
            'best_alt': fit['best_alt'],
            'narrative': f'{name}. {fit["model_note"]}',
            'url': url,
        }
        new_entries.append(entry)
        mv = fit['model_verdict']
        print(f'  {"⚠" if mv=="S2_LOSES" else "✓"} {name[:50]}: D={fit["D"]:.3f}, {mv}, loses to {fit["best_alt"]}')
    
    # Dedup against existing
    existing = load_existing_tests('en/tests.html')
    truly_new = []
    for entry in new_entries:
        is_dup = False
        for ex in existing:
            if entry['name'][:30] in ex.get('name','') or ex.get('name','')[:30] in entry['name']:
                is_dup = True
                break
        if not is_dup:
            truly_new.append(entry)
    
    print(f'\nTruly new (not duplicate): {len(truly_new)}')
    
    if truly_new:
        update_tests_html(truly_new, 'en/tests.html', is_ru=False)
        ru_entries = []
        for e in truly_new:
            e_ru = dict(e)
            e_ru['name'] = e['name']  # keep English names for now
            ru_entries.append(e_ru)
        update_tests_html(ru_entries, 'ru/tests.html', is_ru=True)
    
    # Export
    import subprocess
    subprocess.run(['python3', 'scripts/export_tests_json.py', '.'], capture_output=True, text=True, timeout=30)
    
    # Reconcile
    subprocess.run(['python3', 'scripts/registry_integrity_reconcile.py'], capture_output=True, text=True, timeout=60)
    
    # Update meta-s2
    from dream_auto_scanner import update_meta_s2_article
    update_meta_s2_article('en/tests.html', is_ru=False)
    update_meta_s2_article('ru/tests.html', is_ru=True)
    
    # Final count
    existing = load_existing_tests('en/tests.html')
    from collections import Counter
    mv = Counter(e.get('model_verdict','(none)') for e in existing)
    print(f'\n=== FINAL REGISTRY ===')
    print(f'Total: {len(existing)}')
    for v, n in mv.most_common():
        print(f'  {v}: {n}')
    
    return truly_new

if __name__ == '__main__':
    main()
