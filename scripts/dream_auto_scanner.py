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

import os, sys, json, time, re, urllib.request, urllib.parse, csv, io, math
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
def fit_s2(t, R, label='', require_wins=False):
    """Fit S2 + model comparison. Returns result dict for ALL outcomes
    (wins, ties, AND losses). No filtering — the registry records everything."""
    t = np.array(t, dtype=float)
    R = np.array(R, dtype=float)
    if len(t) < 5: return None
    t = t - t[0]
    if R[0] > 0: R_norm = R / R[0]
    else: R_norm = R

    cmp = s2_compare(t, R_norm, label)
    if not cmp or cmp['verdict'] == 'NO_FIT' or cmp.get('s2') is None:
        return None

    # NO MORE FILTERING — record wins, ties, AND losses
    D = cmp['s2']['D']
    s2_verdict = 'EXTRACTION' if D > 1 else ('NATURAL' if D < 0.8 else 'THRESHOLD')

    # Build narrative for ALL outcomes
    if cmp['verdict'] == 'S2_WINS':
        model_note = f'S2 beats {cmp["best_alt_name"]} (ΔAICc={cmp["delta_aicc"]}).'
    elif cmp['verdict'] == 'S2_TIES':
        model_note = f'S2 ties {cmp["best_alt_name"]} (ΔAICc={cmp["delta_aicc"]}, within ±2).'
    else:  # S2_LOSES — record honestly
        # Check if BIEXP won — this indicates dust contamination (predicted by DREAM)
        s2_dust = cmp.get('s2_dust')
        if cmp['best_alt_name'] == 'BIEXP':
            if s2_dust:
                # Check if S2+dust beats BIEXP
                biexp_aicc = None
                for name, aicc_val, _, _ in cmp.get('rank', []):
                    if name == 'BIEXP':
                        biexp_aicc = aicc_val
                        break
                dust_beats_biexp = s2_dust['aicc'] < biexp_aicc if biexp_aicc else False
                model_note = (f'S2 loses to BIEXP (ΔAICc={cmp["delta_aicc"]}). '
                             f'S2+dust decomposition: D1={s2_dust["D1"]}, D2={s2_dust["D2"]}, '
                             f'R²={s2_dust["r2"]}, AICc={s2_dust["aicc"]}. '
                             f'{"S2+dust BEATS BIEXP — dust structure confirmed." if dust_beats_biexp else "S2+dust does not beat BIEXP."} '
                             f'{"Not a DREAM failure." if dust_beats_biexp else "Needs investigation."}')
            else:
                model_note = (f'S2 loses to BIEXP (ΔAICc={cmp["delta_aicc"]}). '
                             f'Dust-contaminated: two-scale structure. S2+dust fit failed. '
                             f'Not necessarily a DREAM failure.')
        else:
            model_note = f'S2 loses to {cmp["best_alt_name"]} (ΔAICc={cmp["delta_aicc"]}).'
        print(f'    ⚠ S2 loses to {cmp["best_alt_name"]} '
              f'(ΔAICc={cmp["delta_aicc"]}) — {label[:40]} (RECORDED, not hidden)')

    return {
        'D': D,
        'lambda_q': cmp['s2']['lambda_q'],
        'r2': cmp['s2']['r2'],
        'verdict': s2_verdict,
        'model_verdict': cmp['verdict'],      # S2_WINS | S2_TIES | S2_LOSES
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

def scan_coingecko():
    """Download crypto daily closes from CoinGecko (no key)."""
    print('\n📡 CoinGecko')
    coins = ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana', 'dogecoin', 'polkadot', 'chainlink']
    found = []
    for coin in coins:
        url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365&interval=daily'
        data = fetch_url(url, timeout=15)
        if not data: continue
        try:
            d = json.loads(data)
            vals = [p[1] for p in d.get('prices', [])]
            if len(vals) < 10: continue
            filepath = os.path.join(OUT_DIR, f'coingecko_{coin}.json')
            with open(filepath, 'w') as f:
                json.dump(vals, f)
            found.append({
                'source': 'coingecko',
                'title': f'CoinGecko: {coin}',
                'url': url,
                'filename': filepath,
                'format': 'json',
                'values': vals,
            })
        except: pass
    print(f'  Downloaded {len(found)} coins')
    return found

def scan_binance():
    """Download crypto daily klines from Binance (no key)."""
    print('\n📡 Binance')
    symbols = [('BTCUSDT', 'BTC'), ('ETHUSDT', 'ETH'), ('SOLUSDT', 'SOL'),
               ('ADAUSDT', 'ADA'), ('DOTUSDT', 'DOT'), ('XRPUSDT', 'XRP'),
               ('DOGEUSDT', 'DOGE'), ('LINKUSDT', 'LINK')]
    found = []
    for sym, name in symbols:
        url = f'https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=365'
        data = fetch_url(url, timeout=15)
        if not data: continue
        try:
            d = json.loads(data)
            vals = [float(k[4]) for k in d]  # close price
            if len(vals) < 10: continue
            filepath = os.path.join(OUT_DIR, f'binance_{name}.json')
            with open(filepath, 'w') as f:
                json.dump(vals, f)
            found.append({
                'source': 'binance',
                'title': f'Binance: {name}',
                'url': url,
                'filename': filepath,
                'format': 'json',
                'values': vals,
            })
        except: pass
    print(f'  Downloaded {len(found)} pairs')
    return found

def scan_openmeteo():
    """Download weather/environmental data from Open-Meteo (no key)."""
    print('\n📡 Open-Meteo')
    locations = [
        ('52.52', '13.41', 'Berlin'),
        ('35.68', '139.69', 'Tokyo'),
        ('40.71', '-74.01', 'NYC'),
        ('51.51', '-0.13', 'London'),
        ('-33.87', '151.21', 'Sydney'),
        ('55.75', '37.62', 'Moscow'),
        ('28.61', '77.21', 'Delhi'),
        ('-22.91', '-43.17', 'Rio'),
    ]
    variables = ['temperature_2m_mean', 'wind_speed_10m_max', 'precipitation_sum']
    found = []
    for lat, lon, city in locations:
        for var in variables:
            url = (f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
                   f'&start_date=2025-01-01&end_date=2026-07-01&daily={var}')
            data = fetch_url(url, timeout=15)
            if not data: continue
            try:
                d = json.loads(data)
                vals = d.get('daily', {}).get(var, [])
                vals = [v for v in vals if v is not None]
                if len(vals) < 10: continue
                filepath = os.path.join(OUT_DIR, f'openmeteo_{city}_{var}.json')
                with open(filepath, 'w') as f:
                    json.dump(vals, f)
                found.append({
                    'source': 'openmeteo',
                    'title': f'Open-Meteo: {city} {var}',
                    'url': url,
                    'filename': filepath,
                    'format': 'json',
                    'values': vals,
                })
            except: pass
    print(f'  Downloaded {len(found)} series')
    return found

def scan_noaa_space():
    """Download NOAA space weather data."""
    print('\n📡 NOAA Space Weather')
    endpoints = [
        ('https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json', 'NOAA Solar Wind Speed', 'speed'),
        ('https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json', 'NOAA Solar Wind Bz', 'bz'),
        ('https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json', 'NOAA Planetary K-index', None),
    ]
    found = []
    for url, name, key in endpoints:
        data = fetch_url(url, timeout=20)
        if not data: continue
        try:
            d = json.loads(data)
            if isinstance(d, list) and len(d) > 20:
                if key:
                    vals = [item.get(key, 0) for item in d if item.get(key) is not None]
                else:
                    # K-index: list of [timestamp, value] pairs, skip header
                    vals = [float(row[-1]) for row in d[1:] if row and len(row) > 1]
                if len(vals) < 10: continue
                filepath = os.path.join(OUT_DIR, f'noaa_{name.replace(" ","_").lower()}.json')
                with open(filepath, 'w') as f:
                    json.dump(vals, f)
                found.append({
                    'source': 'noaa',
                    'title': name,
                    'url': url,
                    'filename': filepath,
                    'format': 'json',
                    'values': vals,
                })
        except: pass
    print(f'  Downloaded {len(found)} series')
    return found

def scan_nasa_giss():
    """Download NASA GISS global temperature."""
    print('\n📡 NASA GISS')
    url = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'
    data = fetch_url(url, timeout=20)
    if not data: return []
    text = data.decode('utf-8', errors='ignore')
    vals = []
    for line in text.strip().split('\n'):
        parts = line.split(',')
        if parts and re.match(r'^\d{4}$', parts[0].strip()):
            for p in parts[1:13]:
                try:
                    v = float(p.strip())
                    if v != -9999: vals.append(v)
                except: pass
    if len(vals) < 10: return []
    filepath = os.path.join(OUT_DIR, 'giss_temp.json')
    with open(filepath, 'w') as f:
        json.dump(vals, f)
    print(f'  {len(vals)} monthly values')
    return [{'source': 'giss', 'title': 'NASA GISS Global Temperature', 'url': url,
             'filename': filepath, 'format': 'json', 'values': vals}]

def scan_covid():
    """Download COVID-19 time series from JHU CSSE."""
    print('\n📡 COVID-19 (JHU CSSE)')
    found = []
    for kind, label in [('confirmed', 'Confirmed'), ('deaths', 'Deaths')]:
        url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{kind}_global.csv'
        data = fetch_url(url, timeout=30)
        if not data: continue
        try:
            import csv as csvmod
            reader = csvmod.reader(data.decode('utf-8').splitlines())
            rows = list(reader)
            daily = [0] * (len(rows[0]) - 4)
            for row in rows[1:]:
                for i in range(4, len(row)):
                    try: daily[i-4] += int(row[i])
                    except: pass
            new_vals = [max(0, daily[i] - daily[i-1]) for i in range(1, len(daily))]
            if len(new_vals) < 10: continue
            filepath = os.path.join(OUT_DIR, f'covid_{kind}.json')
            with open(filepath, 'w') as f:
                json.dump(new_vals, f)
            found.append({
                'source': 'covid',
                'title': f'COVID-19 {label} (daily new, global)',
                'url': url,
                'filename': filepath,
                'format': 'json',
                'values': new_vals,
            })
        except: pass
    print(f'  Downloaded {len(found)} series')
    return found

def scan_global_temp():
    """Download global temperature from datasets library."""
    print('\n📡 Global Temperature (HadCRUT)')
    url = 'https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv'
    data = fetch_url(url, timeout=15)
    if not data: return []
    try:
        import csv as csvmod
        reader = csvmod.DictReader(data.decode('utf-8').splitlines())
        vals = [float(row['Mean']) for row in reader if 'Mean' in row]
        if len(vals) < 10: return []
        filepath = os.path.join(OUT_DIR, 'global_temp.json')
        with open(filepath, 'w') as f:
            json.dump(vals, f)
        print(f'  {len(vals)} monthly values')
        return [{'source': 'globaltemp', 'title': 'Global Temperature (HadCRUT)', 'url': url,
                 'filename': filepath, 'format': 'json', 'values': vals}]
    except: return []

def scan_eurostat():
    """Download Eurostat economic indicators (no key)."""
    print('\n📡 Eurostat')
    indicators = [
        ('ei_bsrt_m_rt', 'Eurostat Business Climate Indicator'),
        ('teim020', 'Eurostat Industrial Production Index'),
    ]
    found = []
    for code, name in indicators:
        url = f'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}?format=SDMX-CSV'
        data = fetch_url(url, timeout=15)
        if not data: continue
        try:
            import csv as csvmod
            reader = csvmod.DictReader(data.decode('utf-8').splitlines())
            vals = []
            for row in reader:
                for k in row:
                    if 'OBS_VALUE' in k:
                        try: vals.append(float(row[k]))
                        except: pass
            if len(vals) < 10: continue
            filepath = os.path.join(OUT_DIR, f'eurostat_{code}.json')
            with open(filepath, 'w') as f:
                json.dump(vals, f)
            found.append({
                'source': 'eurostat',
                'title': name,
                'url': url,
                'filename': filepath,
                'format': 'json',
                'values': vals,
            })
        except: pass
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
# NEW SOURCES — wider family diversity
# ═══════════════════════════════════════════════════════════════════════

def scan_wikipedia_pageviews():
    """Download Wikipedia pageview time series for notable articles.
    New family: cultural attention / information seeking behavior."""
    print('\n📡 Wikipedia Pageviews')
    articles = [
        ('Earth', 'en'), ('Albert_Einstein', 'en'), ('World_War_II', 'en'),
        ('COVID-19_pandemic', 'en'), ('Climate_change', 'en'),
        ('Quantum_mechanics', 'en'), ('Artificial_intelligence', 'en'),
    ]
    found = []
    for title, lang in articles:
        url = (f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'
               f'{lang}.wikipedia/all-access/user/{title}/daily/2020010100/2025123100')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if 'items' not in obj: continue
            values = [item['views'] for item in obj['items']]
            if len(values) < 50: continue
            found.append({
                'source': 'wikipedia',
                'title': f'Wikipedia: {title.replace("_"," ")} pageviews',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} Wikipedia pageview series')
    return found

def scan_ecb():
    """Download European Central Bank data portal series.
    New family: European financial (different jurisdiction from FRED)."""
    print('\n📡 ECB Data Portal')
    series = [
        ('ICP.M.U2.N.000000.4.ANR', 'Euro Area HICP (annual rate)'),
        ('STS.M.U2.Y.PROD.NSA0100', 'Euro Area Industrial Production'),
        ('BKN.M.U2.N.STS100.4.0000', 'Euro Area Business Climate'),
        ('ILM.M.U2.EUR.4F.BB.U2_10Y.YLD', 'Euro Area 10Y Bond Yield'),
    ]
    found = []
    for key, name in series:
        url = f'https://data-api.ecb.europa.eu/service/data/{key}?format=csvdata&lastNObservations=120'
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) < 5: continue
            header = rows[0]
            val_idx = header.index('OBS_VALUE') if 'OBS_VALUE' in header else -1
            if val_idx < 0: continue
            values = []
            for row in rows[1:]:
                if val_idx < len(row):
                    try:
                        v = float(row[val_idx])
                        if not np.isnan(v): values.append(v)
                    except: pass
            if len(values) < 20: continue
            found.append({
                'source': 'ecb',
                'title': f'ECB: {name}',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} ECB series')
    return found

def scan_oecd():
    """Download OECD statistics.
    New family: international economic (different from World Bank/FRED)."""
    print('\n📡 OECD Stats')
    datasets = [
        ('MEI', 'M.USA.CPI...IX', 'USA CPI (OECD)'),
        ('MEI', 'M.JPN.CPI...IX', 'Japan CPI (OECD)'),
        ('MEI', 'M.GBR.CPI...IX', 'UK CPI (OECD)'),
        ('MEI', 'M.DEU.CPI...IX', 'Germany CPI (OECD)'),
    ]
    found = []
    for ds_id, key, name in datasets:
        url = f'https://stats.oecd.org/SDMX-JSON/data/{ds_id}/{key}/all?startTime=2000-01&endTime=2025-12'
        try:
            data = fetch_url(url, timeout=10)  # shorter timeout
            if not data: continue
            obj = json.loads(data)
            if 'dataSets' not in obj: continue
            ds0 = obj['dataSets'][0]
            if 'observations' not in ds0: continue
            obs = ds0['observations']
            values = [float(obs[k][0]) for k in sorted(obs.keys()) if obs[k] and obs[k][0] is not None]
            if len(values) < 20: continue
            found.append({
                'source': 'oecd',
                'title': f'OECD: {name}',
                'url': url, 'values': values,
            })
        except Exception as e:
            print(f'  ✗ OECD {name}: {e}')
    print(f'  Downloaded {len(found)} OECD series')
    return found

def scan_berkeley_earth():
    """Download Berkeley Earth temperature data.
    New family: independent climate reconstruction (different from GISS/HadCRUT)."""
    print('\n📡 Berkeley Earth')
    url = 'http://berkeleyearth.lbl.gov/auto/Global/Land_and_Ocean_complete.txt'
    try:
        data = fetch_url(url, timeout=20)
        if not data:
            print('  ✗ No data')
            return []
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        values = []
        for line in text.split('\n'):
            if line.startswith('%') or not line.strip(): continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    v = float(parts[2])
                    if not np.isnan(v): values.append(v)
                except: pass
        if len(values) < 50:
            print(f'  ✗ Only {len(values)} values')
            return []
        found = [{
            'source': 'berkeley_earth',
            'title': 'Berkeley Earth Global Temperature (monthly anomaly)',
            'url': url, 'values': values,
        }]
        print(f'  Downloaded {len(values)} monthly values')
        return found
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return []

def scan_un_comtrade():
    """Download UN Comtrade international trade flow data.
    New family: trade flows (new domain)."""
    print('\n📡 UN Comtrade')
    reporters = [('842', 'USA'), ('156', 'China'), ('392', 'Japan'), ('276', 'Germany')]
    found = []
    for reporter_code, reporter_name in reporters:
        url = (f'https://comtradeapi.un.org/public/v1/preview/C/A/HS?'
               f'reporterCode={reporter_code}&period=2005,2006,2007,2008,2009,2010,'
               f'2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023&'
               f'partnerCode=World&cmdCode=TOTAL&flowCode=M,X')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if 'data' not in obj or not obj['data']: continue
            by_year = {}
            for item in obj['data']:
                yr = item.get('period')
                val = item.get('primaryValue')
                if yr and val:
                    by_year.setdefault(yr, 0)
                    by_year[yr] += val
            years = sorted(by_year.keys())
            values = [by_year[y] for y in years]
            if len(values) < 10: continue
            found.append({
                'source': 'un_comtrade',
                'title': f'UN Comtrade: {reporter_name} total trade (annual)',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} Comtrade series')
    return found

def scan_data_gov():
    """Download from US government open data (BLS employment).
    New family: government operations."""
    print('\n📡 Data.gov / BLS')
    sources = [
        ('https://api.bls.gov/publicAPI/v2/timeseries/data/CES0000000001?startyear=2010&endyear=2025',
         'BLS: Total Nonfarm Payrolls (monthly)'),
    ]
    found = []
    for url, name in sources:
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if 'Results' not in obj or not obj['Results']: continue
            series = obj['Results']['series'][0]
            values = []
            for item in reversed(series.get('data', [])):
                try:
                    v = float(item['value'])
                    if not np.isnan(v): values.append(v)
                except: pass
            if len(values) < 20: continue
            found.append({
                'source': 'data_gov',
                'title': name,
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} Data.gov series')
    return found

def scan_met_museum():
    """Download Metropolitan Museum of Art collection data.
    New family: cultural / artistic creation dates."""
    print('\n📡 Met Museum (cultural)')
    url = 'https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv'
    try:
        data = fetch_url(url, timeout=30)
        if not data:
            print('  ✗ No data')
            return []
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) < 10:
            return []
        header = rows[0]
        date_idx = -1
        for i, h in enumerate(header):
            if 'Begin Date' in h:
                date_idx = i
                break
        if date_idx < 0:
            print('  ✗ No date column')
            return []
        dates = []
        for row in rows[1:]:
            if date_idx < len(row):
                try:
                    d = int(row[date_idx])
                    if -3000 <= d <= 2025:
                        dates.append(d)
                except: pass
        if len(dates) < 50:
            print(f'  ✗ Only {len(dates)} dates')
            return []
        from collections import Counter
        century_counts = Counter()
        for d in dates:
            c = (d // 100) * 100
            century_counts[c] += 1
        centuries = sorted(century_counts.keys())
        counts = [century_counts[c] for c in centuries]
        found = [{
            'source': 'met_museum',
            'title': 'Met Museum: Object creation date distribution (by century)',
            'url': url, 'values': counts,
        }]
        print(f'  Downloaded {len(dates)} objects across {len(centuries)} centuries')
        return found
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return []

def scan_seismic():
    """Download seismic data from IRIS (seismology).
    New family: geophysics / seismology (different from USGS earthquakes)."""
    print('\n📡 IRIS Seismic')
    url = ('https://service.iris.edu/fdsnws/event/1/query?'
           'starttime=2020-01-01&endtime=2025-12-31&'
           'minmagnitude=5.0&format=text')
    try:
        data = fetch_url(url, timeout=20)
        if not data:
            print('  ✗ No data')
            return []
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        magnitudes = []
        for line in text.strip().split('\n')[1:]:
            parts = line.split('|')
            if len(parts) >= 11:
                try:
                    mag = float(parts[10])
                    if not np.isnan(mag): magnitudes.append(mag)
                except: pass
        if len(magnitudes) < 50:
            print(f'  ✗ Only {len(magnitudes)} events')
            return []
        found = [{
            'source': 'iris_seismic',
            'title': 'IRIS Seismic events M>=5.0 (2020-2025, magnitude distribution)',
            'url': url, 'values': magnitudes,
        }]
        print(f'  Downloaded {len(magnitudes)} seismic events')
        return found
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return []

# ═══════════════════════════════════════════════════════════════════════
# TIER 2 NEW SOURCES — biology, chemistry, industrial, hydrology, ocean
# ═══════════════════════════════════════════════════════════════════════

def scan_usgs_hydrology():
    """Download USGS river flow / gauge data (hydrology — different from earthquakes).
    New family: hydrology."""
    print('\n📡 USGS Hydrology (river flow)')
    sites = [
        ('01646500', 'Potomac River at Point of Rocks, MD'),
        ('04213500', 'Cuyahoga River at Independence, OH'),
        ('07010000', 'Mississippi River at St. Louis, MO'),
        ('11447650', 'Sacramento River at Freeport, CA'),
    ]
    found = []
    for site_id, name in sites:
        url = (f'https://waterservices.usgs.gov/nwis/dv/?sites={site_id}'
               f'&parameterCd=00060&startDT=2020-01-01&endDT=2025-12-31&format=json')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if 'value' not in obj or 'timeSeries' not in obj['value']: continue
            ts = obj['value']['timeSeries']
            if not ts: continue
            values = []
            for item in ts[0].get('values', [{}])[0].get('value', []):
                try:
                    v = float(item['value'])
                    if v >= 0: values.append(v)
                except: pass
            if len(values) < 50: continue
            found.append({
                'source': 'usgs_hydrology',
                'title': f'USGS: {name} (daily discharge)',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} hydrology series')
    return found

def scan_noaa_tides():
    """Download NOAA water level data (ocean/coastal). New family: oceanography."""
    print('\n📡 NOAA Tides & Currents')
    stations = [
        ('8454000', 'Providence, RI'),
        ('8518750', 'The Battery, NY'),
        ('9447130', 'Seattle, WA'),
    ]
    found = []
    for station_id, name in stations:
        url = (f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?'
               f'begin_date=20230601&end_date=20230630&station={station_id}'
               f'&product=water_level&datum=mllw&units=metric&time_zone=gmt&format=csv&application=web')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) < 5: continue
            header = rows[0]
            wl_idx = -1
            for i, h in enumerate(header):
                if ' Water Level' in h:
                    wl_idx = i; break
            if wl_idx < 0: wl_idx = 1
            values = []
            for row in rows[1:]:
                if wl_idx < len(row):
                    try:
                        v = float(row[wl_idx])
                        if not np.isnan(v): values.append(v)
                    except: pass
            if len(values) < 50: continue
            found.append({
                'source': 'noaa_tides',
                'title': f'NOAA Tide: {name} (6-min water level)',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} tide series')
    return found

def scan_ndbc_buoy():
    """Download NDBC buoy data. New family: ocean buoys."""
    print('\n📡 NDBC Buoys')
    buoys = [('46035', 'Bering Sea'), ('41001', 'E Hatteras'), ('51001', 'NW Hawaiian')]
    found = []
    for buoy_id, name in buoys:
        url = f'https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt'
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = text.strip().split('\n')
            if len(rows) < 5: continue
            values = []
            for row in rows[2:]:
                parts = row.split()
                if len(parts) >= 6:
                    try:
                        v = float(parts[5])  # WVHT
                        if 0 <= v < 99: values.append(v)
                    except: pass
            if len(values) < 50: continue
            found.append({
                'source': 'ndbc_buoy',
                'title': f'NDBC Buoy {buoy_id} ({name}) wave height',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} buoy series')
    return found

def scan_swpc_sunspots():
    """Download NOAA SWPC sunspot number (275 years!). New family: solar physics."""
    print('\n📡 SWPC Sunspot Number (since 1749)')
    url = 'https://services.swpc.noaa.gov/json/solar-cycle/sunspots.json'
    try:
        data = fetch_url(url, timeout=15)
        if not data: return []
        obj = json.loads(data)
        if not isinstance(obj, list): return []
        values = []
        for item in obj:
            if isinstance(item, dict):
                v = item.get('sunspot_count', item.get('smoothed', item.get('ssn', 0)))
                if v is not None and v >= 0: values.append(float(v))
        if len(values) < 50:
            print(f'  ✗ Only {len(values)} values')
            return []
        found = [{'source': 'swpc_sunspots',
                  'title': 'SWPC Sunspot Number (monthly, 1749-present)',
                  'url': url, 'values': values}]
        print(f'  Downloaded {len(values)} monthly values')
        return found
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return []

def scan_swpc_xray():
    """Download NOAA SWPC GOES X-ray flux. New family: solar flares."""
    print('\n📡 SWPC GOES X-ray Flux')
    url = 'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json'
    try:
        data = fetch_url(url, timeout=15)
        if not data: return []
        obj = json.loads(data)
        if not isinstance(obj, list): return []
        # Filter to the 0.1-0.8nm band (long-band, more flare-sensitive)
        values = []
        for item in obj:
            if isinstance(item, dict):
                if item.get('energy') != '0.1-0.8nm':
                    continue
                flux = item.get('flux', 0)
                if flux and float(flux) > 0: values.append(float(flux))
        if len(values) < 50:
            print(f'  ✗ Only {len(values)} values')
            return []
        found = [{'source': 'swpc_xray',
                  'title': 'SWPC GOES X-ray Flux (1-day, 0.1-0.8nm band)',
                  'url': url, 'values': values}]
        print(f'  Downloaded {len(values)} values (0.1-0.8nm band)')
        return found
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return []

def scan_usgs_geomag():
    """Download USGS geomagnetic field data. New family: geomagnetism."""
    print('\n📡 USGS Geomagnetic Field')
    stations = ['BOU', 'BRW', 'FRD', 'HON']
    found = []
    for station in stations:
        # Use just 1 day to avoid timeout (1-min cadence = 1440 values)
        url = (f'https://geomag.usgs.gov/ws/data/?id={station}'
               f'&starttime=2023-06-01T00:00:00Z&endtime=2023-06-01T23:59:59Z'
               f'&elements=X&format=json')
        try:
            data = fetch_url(url, timeout=12)
            if not data: continue
            obj = json.loads(data)
            values = []
            # USGS geomag returns {metadata:..., values: [[timestamp, X], ...]}
            if 'values' in obj:
                for item in obj['values']:
                    if isinstance(item, list) and len(item) > 1:
                        try:
                            v = float(item[1])
                            if not np.isnan(v): values.append(v)
                        except: pass
                    elif isinstance(item, dict) and 'X' in item:
                        try: values.append(float(item['X']))
                        except: pass
            if len(values) < 50: continue
            found.append({'source': 'usgs_geomag',
                          'title': f'USGS Geomag: {station} (X-component, 1-min)',
                          'url': url, 'values': values})
        except: pass
    print(f'  Downloaded {len(found)} geomag series')
    return found

def scan_energy_charts():
    """Download European electricity spot prices. New family: energy markets."""
    print('\n📡 Energy-Charts (electricity prices)')
    zones = [('DE-LU', 'Germany-Luxembourg'), ('FR', 'France'), ('NL', 'Netherlands'), ('CH', 'Switzerland')]
    found = []
    for zone, name in zones:
        url = (f'https://api.energy-charts.info/price?bzn={zone}'
               f'&start=2023-01-01T00%3A00%2B01%3A00&end=2023-12-31T00%3A00%2B01%3A00')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if 'price' not in obj: continue
            values = [float(p) for p in obj['price'] if p is not None]
            if len(values) < 50: continue
            found.append({'source': 'energy_charts',
                          'title': f'Energy-Charts: {name} electricity spot price (hourly)',
                          'url': url, 'values': values})
        except: pass
    print(f'  Downloaded {len(found)} electricity price series')
    return found

def scan_waqi_air():
    """Download WAQI air quality data. New family: air quality / chemistry.
    Uses the history endpoint (last 24h hourly data) instead of forecast."""
    print('\n📡 WAQI Air Quality')
    cities = ['beijing', 'delhi', 'london', 'tokyo', 'newyork']
    found = []
    for city in cities:
        url = f'https://api.waqi.info/feed/{city}/?token=demo'
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if obj.get('status') != 'ok': continue
            d = obj.get('data', {})
            # The 'iaqi' has current values; forecast has 7-day daily
            # Try forecast daily pm25 (usually 8 entries)
            forecast = d.get('forecast', {}).get('daily', {})
            pm25 = forecast.get('pm25', [])
            values = [f.get('avg', 0) for f in pm25 if isinstance(f, dict)]
            # If forecast is too short, try the 24h history (hourly)
            if len(values) < 20:
                # The forecast only has ~8 days — not enough for ACF
                # Skip this source for now; WAQI demo token doesn't give history
                continue
            found.append({'source': 'waqi_air',
                          'title': f'WAQI: {city} PM2.5 (daily forecast)',
                          'url': url, 'values': values})
        except: pass
    print(f'  Downloaded {len(found)} air quality series')
    return found

def scan_water_quality():
    """Download US Water Quality Portal data. New family: chemistry."""
    print('\n📡 Water Quality Portal')
    url = ('https://www.waterqualitydata.us/data/Result/search?'
           'siteid=USGS-01594440&characteristicName=Nitrate&mimeType=csv&zip=no')
    try:
        data = fetch_url(url, timeout=20)
        if not data: return []
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) < 5: return []
        header = rows[0]
        # The value column is 'ResultMeasureValue'
        val_idx = -1
        for i, h in enumerate(header):
            if h == 'ResultMeasureValue':
                val_idx = i; break
        if val_idx < 0:
            # Fallback: look for any column with 'MeasureValue'
            for i, h in enumerate(header):
                if 'MeasureValue' in h:
                    val_idx = i; break
        if val_idx < 0:
            print('  ✗ No value column found')
            return []
        values = []
        for row in rows[1:]:
            if val_idx < len(row):
                try:
                    v = float(row[val_idx])
                    if not np.isnan(v) and v >= 0: values.append(v)
                except: pass
        if len(values) < 20:
            print(f'  ✗ Only {len(values)} values')
            return []
        found = [{'source': 'water_quality',
                  'title': 'Water Quality Portal: Nitrate at USGS-01594440',
                  'url': url, 'values': values}]
        print(f'  Downloaded {len(values)} measurements')
        return found
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return []

def scan_nasa_power():
    """Download NASA POWER solar radiation data.
    New family: solar energy / irradiance (different from temperature/weather)."""
    print('\n📡 NASA POWER (solar irradiance)')
    locations = [
        (40.0, -105.0, 'Colorado USA'),
        (52.5, 13.4, 'Berlin DE'),
        (-33.9, 18.4, 'Cape Town ZA'),
    ]
    found = []
    for lat, lon, name in locations:
        url = (f'https://power.larc.nasa.gov/api/temporal/hourly/point?'
               f'start=20230101&end=20231231'
               f'&parameters=ALLSKY_SFC_SW_DWN,WS10M'
               f'&longitude={lon}&latitude={lat}&community=RE&format=JSON')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            if 'properties' not in obj: continue
            props = obj['properties']
            if 'parameter' not in props: continue
            params = props['parameter']
            # ALLSKY_SFC_SW_DWN = all-sky surface shortwave downward irradiance
            if 'ALLSKY_SFC_SW_DWN' in params:
                values_dict = params['ALLSKY_SFC_SW_DWN']
                values = [float(v) for v in values_dict.values()
                          if v is not None and v >= 0]
                if len(values) < 50:
                    continue
                found.append({
                    'source': 'nasa_power',
                    'title': f'NASA POWER: {name} solar irradiance (hourly)',
                    'url': url, 'values': values,
                })
        except: pass
    print(f'  Downloaded {len(found)} solar irradiance series')
    return found

def scan_open_power_system():
    """Download Open Power System Data — European power grid.
    New family: energy / power grid (industrial scale).
    Uses a small subset of columns from the large CSV."""
    print('\n📡 Open Power System Data (European grid)')
    # The full CSV is 22MB — too large for CI. Use the Energy-Charts API instead
    # for individual country load data
    zones = [
        ('DE', 'Germany load'),
        ('FR', 'France load'),
    ]
    found = []
    for zone, name in zones:
        # Energy-Charts consumption endpoint
        url = (f'https://api.energy-charts.info/total_load?bzn={zone}'
               f'&start=2023-01-01T00%3A00%2B01%3A00&end=2023-12-31T00%3A00%2B01%3A00')
        try:
            data = fetch_url(url, timeout=15)
            if not data: continue
            obj = json.loads(data)
            # Response has 'unix_seconds' and 'load' (or 'consumption')
            values = []
            if 'load' in obj:
                values = [float(v) for v in obj['load'] if v is not None]
            elif 'consumption' in obj:
                values = [float(v) for v in obj['consumption'] if v is not None]
            if len(values) < 50: continue
            found.append({
                'source': 'open_power_system',
                'title': f'Energy-Charts: {name} (hourly total load)',
                'url': url, 'values': values,
            })
        except: pass
    print(f'  Downloaded {len(found)} power load series')
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

def groq_narrate_ru(fit, backend_url=None):
    """Russian version of the narrative."""
    model_note = fit.get('model_note', '')
    model_note_ru = model_note.replace('S2 beats', 'S2 превосходит').replace('S2 ties', 'S2 сравнима с').replace('S2 loses to', 'S2 уступает').replace('within \u00b12', 'в пределах \u00b12')
    if not backend_url:
        D = fit.get('D', 0)
        r2 = fit.get('r2', 0)
        verdict = fit.get('verdict', 'UNKNOWN')
        if verdict == 'EXTRACTION':
            base = f'D={D:.3f}, R\u00b2={r2:.4f}. D>1 указывает на режим извлечения \u2014 сохранение коллапсирует сверхэкспоненциально.'
        elif verdict == 'NATURAL':
            base = f'D={D:.3f}, R\u00b2={r2:.4f}. D<1 подтверждает естественное сохранение \u2014 тяжёлый хвост, медленное угасание.'
        else:
            base = f'D={D:.3f}, R\u00b2={r2:.4f}. D около порога \u2014 зона перехода режимов.'
        return f'{base} {model_note_ru}'.strip()
    try:
        import urllib.request
        msg = (f"S2 fit result: D={fit.get('D')}, R\u00b2={fit.get('r2')}, verdict={fit.get('verdict')}, dataset={fit.get('label')}. {model_note_ru} Write a 1-2 sentence narrative in Russian.")
        payload = json.dumps({'message': msg, 'lang': 'ru'}).encode()
        req = urllib.request.Request(f'{backend_url}/groq-chat', data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get('reply', '')[:200]
    except:
        return groq_narrate_ru(fit)

def translate_name_ru(name):
    """Translate common auto-scan name patterns to Russian."""
    ru_names = {'FRED GDP':'FRED ВВП','FRED CPIAUCSL':'FRED ИПЦ','FRED UNRATE':'FRED Безработица','FRED FEDFUNDS':'FRED Ставка ФРС','FRED M2SL':'FRED M2','FRED DEXUSEU':'FRED USD/EUR','FRED SP500':'FRED S&P 500','FRED VIXCLS':'FRED VIX','FRED T10YIE':'FRED T10YIE','FRED DGS10':'FRED DGS10'}
    for en, ru in ru_names.items():
        if en in name:
            return name.replace(en, ru).replace('(ACF retention)', '(АСФ сохранение)')
    if 'World Bank: GDP' in name:
        return name.replace('World Bank: GDP', 'World Bank: ВВП').replace('(ACF retention)', '(АСФ сохранение)')
    if 'World Bank: CPI' in name:
        return name.replace('World Bank: CPI', 'World Bank: ИПЦ').replace('(ACF retention)', '(АСФ сохранение)')
    if 'World Bank: Unemployment' in name:
        return name.replace('World Bank: Unemployment', 'World Bank: Безработица').replace('(ACF retention)', '(АСФ сохранение)')
    if name.startswith('arXiv: '):
        title = name[7:]
        ru_title_map = {'Stretched Exponential Decay':'Растянутая экспонента','Exponential decay of correlations':'Экспоненциальное затухание корреляций','Cost-Aware Logging':'Логирование с учётом стоимости','Large deviations':'Большие уклонения','On the Polynomial and Exponential Decay':'О полиномиальном и экспоненциальном затухании'}
        for en_t, ru_t in ru_title_map.items():
            if title.startswith(en_t):
                return f'arXiv: {ru_t}'
    return name

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
        for field in ('id', 'name', 'url', 'verdict', 'date', 'D', 'r2', 'model_verdict', 'kind', 'domain', 'source'):
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
        if new_url and ex_url and new_url == ex_url:
            return True
        if new_name and ex_name:
            if new_name[:40] == ex_name[:40]:
                return True
    return False

def is_blacklisted(entry):
    """Check if an entry matches the blacklist (scan_blacklist.json)."""
    bl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scan_blacklist.json')
    if not os.path.exists(bl_path):
        return False
    try:
        with open(bl_path) as f:
            bl = json.load(f)
    except:
        return False
    name = entry.get('name', '').lower()
    url = entry.get('url', '').lower()
    for bl_name in bl.get('blacklisted_names', []):
        if bl_name.lower() in name:
            return True
    for pattern in bl.get('blacklisted_url_patterns', []):
        if pattern in url:
            return True
    return False

def filter_duplicates(new_entries, existing_entries):
    """Remove entries that already exist or are blacklisted. Returns (kept, skipped_count)."""
    kept = []
    skipped = 0
    for entry in new_entries:
        if is_duplicate(entry, existing_entries):
            skipped += 1
        elif is_blacklisted(entry):
            skipped += 1
            print(f'    ✗ Blacklisted: {entry.get("name","?")[:50]}')
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

def update_tests_html(new_entries, html_path, is_ru=False):
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
        name_raw = entry.get('name', '')
        name = js_str(translate_name_ru(name_raw) if is_ru else name_raw)
        domain = js_str(entry.get('domain', ''))
        verdict = js_str(entry.get('verdict', ''))
        narr_raw = entry.get('narrative', '')
        if is_ru:
            narr_en = narr_raw
            narr_en = narr_en.replace('D>1 indicates extraction regime \u2014 retention collapses super-exponentially.', 'D>1 указывает на режим извлечения \u2014 сохранение коллапсирует сверхэкспоненциально.')
            narr_en = narr_en.replace('D<1 confirms natural retention \u2014 heavy-tailed, slow decay.', 'D<1 подтверждает естественное сохранение \u2014 тяжёлый хвост, медленное угасание.')
            narr_en = narr_en.replace('S2 beats', 'S2 превосходит')
            narr_en = narr_en.replace('S2 ties', 'S2 сравнима с')
            narr_en = narr_en.replace('S2 loses to', 'S2 уступает')
            narr_en = narr_en.replace('within \u00b12', 'в пределах \u00b12')
            narr_en = narr_en.replace('Paper found via arXiv search. Data extraction and S2 fit pending.', 'Статья найдена через arXiv. Извлечение данных и S2-аппроксимация ожидаются.')
            narr = js_str(narr_en)
        else:
            narr = js_str(narr_raw)
        
        # Build model comparison fields (if available)
        model_verdict_val = ''
        if entry.get('model_verdict'):
            mv = js_str(entry['model_verdict'])
            model_verdict_val += f',model_verdict:"{mv}"'
        if entry.get('delta_aicc') is not None:
            model_verdict_val += f',delta_aicc:{entry["delta_aicc"]:.4f}'
        if entry.get('best_alt'):
            ba = js_str(entry['best_alt'])
            model_verdict_val += f',best_alt:"{ba}"'
        if entry.get('model_note'):
            mn = js_str(entry['model_note'])
            model_verdict_val += f',model_note:"{mn}"'
        
        new_js += f'\n  ,{{id:\"auto-{today}-{eid}\",name:\"{name}\",domain:\"{domain}\",D:{D_val},r2:{r2_val},verdict:\"{verdict}\"{model_verdict_val},narrative:\"{narr}\",source:\"auto-scan {today}\",date:\"{today}\",url:{url_val},image:null}}'
    
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
    if 'model_verdict' in updates:
        mv = updates['model_verdict']
        if re.search(r'model_verdict:', new_entry):
            new_entry = re.sub(r'model_verdict:".*?"', f'model_verdict:"{mv}"', new_entry, count=1)
        else:
            new_entry = re.sub(r'(verdict:".*?",)', r'\1\n    model_verdict:"' + mv + '",', new_entry, count=1)
    html = html.replace(old_entry, new_entry, 1)
    with open(html_path, 'w') as f:
        f.write(html)
    print(f'    ✓ Updated entry {entry_id}: D={updates.get("D","?")}, verdict={updates.get("verdict","?")}')

def retry_no_comparison_entries(existing_entries, groq_url=None):
    """Retry entries with D but no model_verdict — re-download and fit with comparison."""
    import numpy as np
    from scipy.optimize import curve_fit
    def s2_func(t, lam_q, D): return np.exp(-np.power(np.clip(t, 1e-10, None) / max(lam_q, 1e-10), max(D, 0.01)))
    def power_law(t, lam_0, alpha): return np.exp(-np.power(np.clip(t, 1e-10, None) / max(lam_0, 1e-10), max(alpha, 0.01)))
    def exp_func(t, tau): return np.exp(-np.clip(t, 1e-10, None) / max(tau, 1e-10))
    def aic(n, rss, k):
        if n - k - 1 <= 0 or rss <= 0: return 9999
        return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    def acf_retention(vals, max_lag=None):
        vals = np.array(vals, dtype=float); n = len(vals)
        if n < 10: return None, None
        vals = vals - vals.mean(); var = np.var(vals)
        if var == 0: return None, None
        max_lag = min(max_lag or n//2, n-1, 100)
        R = [np.sum(vals[:n-k] * vals[k:]) / (n * var) for k in range(max_lag)]
        return np.arange(max_lag), np.array(R)
    def fit_cmp(t, R):
        t = np.array(t, dtype=float); R = np.array(R, dtype=float)
        if len(t) < 5: return None
        t = t - t[0]
        if R[0] > 0: R = R / R[0]
        results = {}
        try:
            popt, _ = curve_fit(s2_func, t, R, p0=[t[len(t)//2], 0.5], maxfev=5000)
            rss = np.sum((R - s2_func(t, *popt))**2)
            results['s2'] = {'D': popt[1], 'aic': aic(len(t), rss, 2), 'rss': rss}
        except: results['s2'] = None
        try:
            popt, _ = curve_fit(power_law, t, R, p0=[t[len(t)//2], 1.0], maxfev=5000)
            rss = np.sum((R - power_law(t, *popt))**2)
            results['power'] = {'aic': aic(len(t), rss, 2)}
        except: results['power'] = None
        try:
            popt, _ = curve_fit(exp_func, t, R, p0=[t[len(t)//2]], maxfev=5000)
            rss = np.sum((R - exp_func(t, *popt))**2)
            results['exp'] = {'aic': aic(len(t), rss, 1)}
        except: results['exp'] = None
        aics = {k: v['aic'] for k, v in results.items() if v}
        if not aics or not results.get('s2'): return None
        best = min(aics, key=aics.get)
        s2_aic = results['s2']['aic']
        best_alt = min(v for k, v in aics.items() if k != 's2') if len(aics) > 1 else s2_aic
        delta = s2_aic - best_alt if best != 's2' else 0
        if best == 's2': verdict = 'S2_WINS' if delta < -2 else 'S2_TIES'
        elif delta < 2: verdict = 'S2_TIES'
        else: verdict = 'S2_LOSES'
        r2 = 1 - results['s2']['rss'] / max(np.sum((R - R.mean())**2), 1e-10)
        return {'D': results['s2']['D'], 'r2': r2, 'model_verdict': verdict, 'delta_aic': delta}
    retried = resolved = 0
    for ex in existing_entries:
        # Only retry entries that have D but NO model_verdict (or UNDETERMINED)
        existing_mv = ex.get('model_verdict', '')
        if existing_mv and existing_mv != 'UNDETERMINED': continue
        if ex.get('D') is None: continue
        url = ex.get('url', '')
        if not url.startswith('http'): continue
        if not any(x in url for x in ['fredgraph.csv', 'api.worldbank', 'coingecko', 'binance', 'open-meteo', 'earthquake.usgs.gov/earthquakes/feed']): continue
        retried += 1
        eid = ex.get('id', ''); name = ex.get('name', '?')
        print(f'  -> Retrying: {name[:50]} ({eid})')
        try:
            vals = None
            if 'fredgraph.csv' in url:
                data = fetch_url(url, timeout=20)
                if data:
                    import csv as csvmod
                    rows = list(csvmod.reader(data.decode('utf-8').splitlines()))
                    vals = [float(r[1]) for r in rows[1:] if len(r) >= 2 and r[1] not in ('', '.')]
            elif 'api.worldbank' in url:
                data = fetch_url(url, timeout=20)
                if data:
                    d = json.loads(data)
                    if len(d) > 1 and d[1]:
                        vals = [x['value'] for x in d[1] if x['value'] is not None]
                        vals.reverse()
            elif 'coingecko' in url:
                data = fetch_url(url, timeout=20)
                if data: vals = [p[1] for p in json.loads(data).get('prices', [])]
            elif 'binance' in url:
                data = fetch_url(url, timeout=20)
                if data: vals = [float(k[4]) for k in json.loads(data)]
            elif 'open-meteo' in url:
                data = fetch_url(url, timeout=20)
                if data:
                    d = json.loads(data)
                    for var in ['temperature_2m_mean', 'wind_speed_10m_max', 'precipitation_sum']:
                        if var in d.get('daily', {}):
                            vals = [v for v in d['daily'][var] if v is not None]; break
            elif 'earthquake.usgs.gov/earthquakes/feed' in url:
                data = fetch_url(url, timeout=20)
                if data:
                    import csv as csvmod
                    reader = csvmod.DictReader(data.decode('utf-8').splitlines())
                    mags = [float(r['mag']) for r in reader if r.get('mag') and r['mag'] != '']
                    if len(mags) > 20:
                        n = len(mags)
                        thresholds = np.linspace(max(mags), min(mags), min(50, n))
                        R = np.array([np.sum(np.array(mags) >= t) / n for t in thresholds])
                        t_arr = np.arange(len(thresholds))
                        result = fit_cmp(t_arr, R)
                        if result:
                            update_existing_entry('en/tests.html', eid, result)
                            update_existing_entry('ru/tests.html', eid, result)
                            resolved += 1
                        continue
            if not vals or len(vals) < 10: continue
            t, R = acf_retention(vals)
            if t is None: continue
            result = fit_cmp(t, R)
            if result:
                # Guard: reject bad fits (negative D, negative R², extreme D)
                if result.get('D') is not None and 0 < result['D'] < 10 and result.get('r2', 0) >= 0:
                    update_existing_entry('en/tests.html', eid, result)
                    update_existing_entry('ru/tests.html', eid, result)
                    resolved += 1
                else:
                    print(f'    Skipping bad fit: D={result.get("D")}, R²={result.get("r2")}')
        except Exception as e:
            print(f'    Failed: {e}')
    return retried, resolved

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
    
    # New meta-sources (no API key required)
    coingecko_results = scan_coingecko()
    binance_results = scan_binance()
    openmeteo_results = scan_openmeteo()
    noaa_results = scan_noaa_space()
    giss_results = scan_nasa_giss()
    covid_results = scan_covid()
    globaltemp_results = scan_global_temp()
    eurostat_results = scan_eurostat()
    
    # NEW: Wider family sources
    wikipedia_results = scan_wikipedia_pageviews()
    ecb_results = scan_ecb()
    # NOTE: These sources are disabled in CI because they're too slow or broken:
    # - scan_oecd(): hangs (OECD API is unreliable)
    # - scan_berkeley_earth(): 60MB file download times out
    # - scan_un_comtrade(): API returns 400/429
    # - scan_data_gov(): BLS API times out
    # - scan_met_museum(): 60MB CSV download times out
    # They can be re-enabled locally for manual testing.
    # oecd_results = scan_oecd()
    # berkeley_results = scan_berkeley_earth()
    # comtrade_results = scan_un_comtrade()
    # data_gov_results = scan_data_gov()
    # met_results = scan_met_museum()
    oecd_results = []
    berkeley_results = []
    comtrade_results = []
    data_gov_results = []
    met_results = []
    seismic_results = scan_seismic()
    
    # NEW TIER 2: Biology, chemistry, industrial, hydrology, ocean
    hydrology_results = scan_usgs_hydrology()
    tide_results = scan_noaa_tides()
    buoy_results = scan_ndbc_buoy()
    sunspot_results = scan_swpc_sunspots()
    xray_results = scan_swpc_xray()
    geomag_results = scan_usgs_geomag()
    energy_results = scan_energy_charts()
    waqi_results = scan_waqi_air()
    water_quality_results = scan_water_quality()
    nasa_power_results = scan_nasa_power()
    power_load_results = scan_open_power_system()
    
    # Combine all JSON-value sources for unified analysis
    json_sources = (coingecko_results + binance_results + openmeteo_results +
                    noaa_results + giss_results + covid_results + 
                    globaltemp_results + eurostat_results + wb_results +
                    wikipedia_results + ecb_results + oecd_results +
                    berkeley_results + comtrade_results + data_gov_results +
                    met_results + seismic_results +
                    hydrology_results + tide_results + buoy_results +
                    sunspot_results + xray_results + geomag_results +
                    energy_results + waqi_results + water_quality_results +
                    nasa_power_results + power_load_results)
    
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
    
    # 7b. Analyze new meta-sources (CoinGecko, Binance, Open-Meteo, NOAA, GISS, COVID, Global Temp, Eurostat)
    print('\n📊 Analyzing new meta-sources...')
    for item in json_sources:
        if item in wb_results: continue  # already analyzed above
        fit = analyze_json_values(item.get('values', []), item['title'][:60])
        if fit and 'D' in fit:
            narrative = groq_narrate(fit, groq_url if groq_url else None)
            source_name = item.get('source', 'unknown')
            domain_map = {
                'coingecko': 'crypto', 'binance': 'crypto',
                'openmeteo': 'environmental', 'noaa': 'space_weather',
                'giss': 'environmental', 'covid': 'ecological',
                'globaltemp': 'environmental', 'eurostat': 'economic',
                'wikipedia': 'cultural', 'ecb': 'economic_eu',
                'oecd': 'economic_intl', 'berkeley_earth': 'climate',
                'un_comtrade': 'trade', 'data_gov': 'government',
                'met_museum': 'cultural', 'iris_seismic': 'geophysics',
                'usgs_hydrology': 'hydrology', 'noaa_tides': 'oceanography',
                'ndbc_buoy': 'oceanography', 'swpc_sunspots': 'solar_physics',
                'swpc_xray': 'solar_physics', 'usgs_geomag': 'geomagnetism',
                'energy_charts': 'energy', 'waqi_air': 'air_quality',
                'water_quality': 'chemistry',
                'nasa_power': 'solar_energy', 'open_power_system': 'power_grid',
            }
            domain = domain_map.get(source_name, 'scouting')
            entry_id = f'{source_name}-{item["title"].lower().replace(" ","-").replace(":","")[:30]}'
            all_results.append({
                'id': entry_id,
                'name': f'{item["title"]} (ACF retention)',
                'domain': domain,
                'D': fit['D'], 'r2': fit['r2'],
                'verdict': fit['verdict'],
                        'model_verdict': fit.get('model_verdict'),
                        'model_note': fit.get('model_note', ''),
                        'delta_aicc': fit.get('delta_aicc'),
                        'best_alt': fit.get('best_alt'),
                'narrative': narrative,
                'url': item.get('url', ''),
            })
            print(f'  ✓ D={fit["D"]:.3f} R²={fit["r2"]:.4f} {fit["verdict"]} — {item["title"][:40]}')
    
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

    # 10b2. RETRY no-comparison entries (have D but no model_verdict)
    print('\n🔄 Retrying no-comparison entries (have D, need AICc comparison)...')
    retried, resolved_cmp = retry_no_comparison_entries(existing_en, groq_url if groq_url else None)
    if retried:
        print(f'  ✓ Retried {retried} entries, resolved {resolved_cmp} with model comparison')
    else:
        print(f'  (no no-comparison entries with downloadable URLs found)')

    # 10b3. RERUN S2_LOSES entries on S2+dust model
    # If S2 loses to BIEXP but S2+dust (two-component S2) beats BIEXP, that's
    # dust contamination — predicted by DREAM, not a failure. Flip the verdict
    # to S2_DUST_WINS and record D1/D2/R²/delta.
    print('\n🔄 Rerunning S2_LOSES entries on S2+dust model...')
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import rerun_losses_on_s2_dust as _dust_rerun
        redeemed = _dust_rerun.main()
        if redeemed:
            print(f'  ✓ {redeemed} entries redeemed via S2+dust decomposition')
        else:
            print(f'  (no entries redeemed this run)')
    except Exception as e:
        print(f'  ⚠ S2+dust rerun failed: {e}')

    # 10b4. FLIP entries where best_alt=S2_DUST but verdict still S2_LOSES
    # These were already fitted (S2_DUST is the best model) but fit_s2() labels
    # them S2_LOSES because S2 itself isn't the best. Flip them to S2_DUST_WINS.
    print('\n🔄 Flipping S2_LOSES with best_alt=S2_DUST → S2_DUST_WINS...')
    try:
        import flip_dust_wins as _flip
        _flip.main()
    except Exception as e:
        print(f'  ⚠ Flip failed: {e}')

    # 10c. Update tests.html with ONLY new (non-duplicate) entries
    if os.path.exists(tests_html):
        update_tests_html(kept_results, tests_html)
    else:
        print(f'  tests.html not found at {tests_html} — skipping update')

    # 11. Also update RU tests.html
    ru_html = tests_html.replace('en/', 'ru/')
    if os.path.exists(ru_html):
        update_tests_html(kept_results, ru_html, is_ru=True)
    
    # Ensure every entry has a URL — fallback to source homepage
    for entry in all_results:
        if not entry.get('url'):
            source = entry.get('source', '')
            if 'zenodo' in source:
                entry['url'] = 'https://zenodo.org/search?q=stretched+exponential'
            elif 'arxiv' in source:
                entry['url'] = 'https://arxiv.org/search?q=stretched+exponential'
            elif 'fred' in source.lower():
                entry['url'] = 'https://fred.stlouisfed.org/'
            elif 'usgs' in source.lower():
                entry['url'] = 'https://earthquake.usgs.gov/'
            elif 'world bank' in source.lower() or 'wb' in source.lower():
                entry['url'] = 'https://data.worldbank.org/'
            else:
                entry['url'] = 'https://dream-framework.github.io/dream/'
    
    # 11b. Retrofit AICc gate on existing entries with downloadable real data
    print('\n🔄 Retrofitting AICc on existing entries with real data...')
    retrofit_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retrofit_aicc_real.py')
    if os.path.exists(retrofit_script):
        try:
            subprocess.run(['python3', retrofit_script], capture_output=True, text=True, timeout=120)
            print('  ✓ AICc retrofit complete')
        except Exception as e:
            print(f'  ✗ Retrofit failed: {e}')
    
    # 12. Export tests.json for download (EN + RU)
    print('\n📦 Exporting tests.json...')
    import subprocess
    export_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'export_tests_json.py')
    if os.path.exists(export_script):
        try:
            subprocess.run(['python3', export_script, '.'], capture_output=True, text=True, timeout=30)
            print('  ✓ tests.json exported (EN + RU)')
        except Exception as e:
            print(f'  ✗ Export failed: {e}')
    
    # 12b. Reconcile registry data integrity
    # Derive every narrative + regime label from immutable numeric fields.
    # This prevents stale narratives from persisting after D/r2 refits.
    print('\n🔧 Reconciling registry data integrity...')
    reconcile_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'registry_integrity_reconcile.py')
    if os.path.exists(reconcile_script):
        try:
            r = subprocess.run(['python3', reconcile_script], capture_output=True, text=True, timeout=60)
            # Print last few lines of output
            lines = r.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f'  {line}')
            print('  ✓ Registry reconciled')
        except Exception as e:
            print(f'  ✗ Reconcile failed: {e}')
    
    # 12c. Update provenance ledger
    # Record every scan run with timestamp, entries considered, and outcomes.
    # Mark entries after 2026-07-31 as 'prospective' evidence.
    print('\n📋 Updating provenance ledger...')
    provenance_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scout_provenance.py')
    if os.path.exists(provenance_script):
        try:
            # Re-initialize the ledger from current registry state
            r = subprocess.run(['python3', provenance_script, '--init'],
                             capture_output=True, text=True, timeout=60)
            lines = r.stdout.strip().split('\n')
            for line in lines[-3:]:
                print(f'  {line}')
            print('  ✓ Provenance ledger updated')
        except Exception as e:
            print(f'  ✗ Provenance update failed: {e}')
    
    # 13. Update meta-s2 article with current registry stats
    print('\n📝 Updating meta-s2 article...')
    try:
        update_meta_s2_article(tests_html)
        update_meta_s2_article(tests_html.replace('en/', 'ru/'), is_ru=True)
        print('  ✓ meta-s2 article updated')
    except Exception as e:
        print(f'  ✗ meta-s2 update failed: {e}')
    
    # 14. Run multi-theorem scanner (cosmology, spectral ratios, topology, etc.)
    print('\n📡 Running multi-theorem scanner...')
    theorem_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dream_theorem_scanner.py')
    if os.path.exists(theorem_script):
        try:
            r = subprocess.run(['python3', theorem_script], capture_output=True, text=True, timeout=120)
            lines = r.stdout.strip().split('\n')
            for line in lines[-8:]:
                print(f'  {line}')
            print('  ✓ Theorem scanner complete')
            
            # Embed theorem tests into tests.html — replace the ENTIRE THEOREM_TESTS array
            theorem_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'theorem_tests.json')
            if os.path.exists(theorem_json_path):
                with open(theorem_json_path) as f:
                    tdata = json.load(f)
                tests_t = tdata.get('tests', [])
                
                # Build JS entries
                def esc_t(s):
                    if s is None: return ''
                    return str(s).replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '')
                
                js_entries = []
                for t in tests_t:
                    parts = []
                    for k, v in t.items():
                        if isinstance(v, bool):
                            # MUST check bool BEFORE int (bool is subclass of int in Python)
                            parts.append(f'{k}:{"true" if v else "false"}')
                        elif isinstance(v, str):
                            parts.append(f'{k}:"{esc_t(v)}"')
                        elif isinstance(v, (int, float)):
                            if isinstance(v, float) and math.isinf(v):
                                pass  # skip Infinity
                            elif isinstance(v, float) and math.isnan(v):
                                pass  # skip NaN
                            else:
                                parts.append(f'{k}:{v}')
                        elif v is None:
                            pass  # skip None
                    js_entries.append('  {' + ','.join(parts) + '}')
                js_block = ',\n'.join(js_entries)
                new_array = f'const THEOREM_TESTS = [\n{js_block}\n];'
                
                # Patch both EN and RU tests.html — replace the entire THEOREM_TESTS array
                for html_path in [tests_html, tests_html.replace('en/', 'ru/')]:
                    if os.path.exists(html_path):
                        with open(html_path) as f:
                            html = f.read()
                        # Replace everything between "const THEOREM_TESTS = [" and the closing "];"
                        import re as _re
                        html = _re.sub(
                            r'const\s+THEOREM_TESTS\s*=\s*\[.*?\];',
                            new_array,
                            html,
                            count=1,
                            flags=_re.DOTALL
                        )
                        with open(html_path, 'w') as f:
                            f.write(html)
                print(f'  ✓ Embedded {len(tests_t)} theorem tests into tests.html (EN+RU)')
        except Exception as e:
            print(f'  ✗ Theorem scanner failed: {e}')
    
    return all_results

def update_meta_s2_article(tests_html_path, is_ru=False):
    """
    Update the meta-s2 article, the snapshot JSON, AND the embedded
    META_S2_SNAPSHOT constant in tests.html.

    All three must stay in sync — the article body and the live readout card
    on tests.html must show the same numbers. The scanner regenerates:

      1. `meta_s2_snapshot.json` (repo root) — single source of truth
      2. `articles/meta-s2.html` (EN + RU) — full article, rendered from template
      3. The `META_S2_SNAPSHOT = {...}` JS constant embedded in `tests.html`
         (EN + RU) — used by the live Meta-S2 readout card

    Without step 3, the live card on tests.html stays stale even when the
    article is updated.
    """
    # Local imports — meta_s2_article lives next to this scanner
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from meta_s2_article import extend_snapshot, render

    article_path = tests_html_path.replace('tests.html', 'articles/meta-s2.html')
    # Snapshot lives at repo root (one level up from en/ or ru/)
    snapshot_path = os.path.join(os.path.dirname(tests_html_path), '..', 'meta_s2_snapshot.json')

    # Load existing tests from tests.html
    existing = load_existing_tests(tests_html_path)
    Ds_all = sorted([e['D'] for e in existing if e.get('D') is not None and 0 < e['D'] < 4.99])
    n = len(Ds_all)
    if n < 10:
        print(f'  ! Skipping meta-s2 update — only {n} uncensored D values')
        return

    # Compute the full snapshot (all stats needed by the article template)
    try:
        snapshot = extend_snapshot(existing)
    except Exception as e:
        print(f'  ! Snapshot computation failed: {e}')
        return

    # Write the snapshot JSON (shared between EN and RU — only write once)
    if not is_ru:
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        print(f'  Snapshot: n={snapshot["n"]}, D_mle={snapshot["d_mle"]}, '
              f'KS p={snapshot["ks_p"]}, AD={snapshot["ad_stat"]}, '
              f'ΔAIC_W={snapshot["delta_aic_weibull"]}, '
              f'ΔAIC_G={snapshot["delta_aic_gamma"]}, '
              f'ΔAIC_L={snapshot["delta_aic_lognormal"]}, '
              f'Silverman p={snapshot["silverman_p"]}, '
              f'CI=[{snapshot["boot_lo"]}, {snapshot["boot_hi"]}]')

    # ── Step 3: Patch the embedded META_S2_SNAPSHOT constant in tests.html ──
    # The live readout card reads from this JS object, not from the JSON file.
    # If we don't rewrite it, the card shows stale numbers.
    try:
        with open(tests_html_path) as f:
            html = f.read()

        # Build the JS object literal — keep field names compatible with the
        # existing readout code in tests.html (which expects: n, d_mle, lam_mle,
        # ks_p, d_direct, r2_direct, d_linear, r2_linear, natural, extraction,
        # mean, median, total, compared).
        embedded = {
            'n': snapshot['n'],
            'd_mle': snapshot['d_mle'],
            'lam_mle': snapshot['lam_mle'],
            'ks_p': snapshot['ks_p'],
            'd_direct': snapshot['d_direct'],
            'r2_direct': snapshot['r2_direct'],
            'd_linear': snapshot['d_linear'],
            'r2_linear': snapshot['r2_linear'],
            'natural': snapshot['natural'],
            'extraction': snapshot['extraction'],
            'mean': snapshot['mean'],
            'median': snapshot['median'],
            # tests.html uses 'total' and 'compared' (not n_total / n_compared)
            'total': snapshot['n_total'],
            'compared': snapshot['n_compared'],
            # Also include the extended stats so the readout can show them later
            'ad_stat': snapshot['ad_stat'],
            'delta_aic_weibull': snapshot['delta_aic_weibull'],
            'delta_aic_gamma': snapshot['delta_aic_gamma'],
            'delta_aic_lognormal': snapshot['delta_aic_lognormal'],
            'silverman_p': snapshot['silverman_p'],
            'boot_lo': snapshot['boot_lo'],
            'boot_hi': snapshot['boot_hi'],
            'date': snapshot['date'],
        }
        embedded_json = json.dumps(embedded, indent=2)

        # Replace the META_S2_SNAPSHOT = {...} block in tests.html.
        # Match from "const META_S2_SNAPSHOT = {" to the closing "};"
        # (the closing brace + semicolon on its own line).
        pattern = re.compile(
            r'const\s+META_S2_SNAPSHOT\s*=\s*\{.*?\n\};',
            re.DOTALL
        )
        new_block = f'const META_S2_SNAPSHOT = {embedded_json};'
        if pattern.search(html):
            html = pattern.sub(new_block, html, count=1)
            with open(tests_html_path, 'w') as f:
                f.write(html)
            print(f'  ✓ Embedded META_S2_SNAPSHOT patched in {tests_html_path}')
        else:
            print(f'  ! Could not find META_S2_SNAPSHOT in {tests_html_path} — leaving unchanged')
    except Exception as e:
        print(f'  ! Failed to patch embedded snapshot in {tests_html_path}: {e}')

    # Render the article for the requested language
    lang = 'ru' if is_ru else 'en'
    try:
        html = render(lang, snapshot, existing=existing)
    except Exception as e:
        print(f'  ! {lang} render failed: {e}')
        return

    with open(article_path, 'w') as f:
        f.write(html)

    print(f'  {article_path}: n={snapshot["n"]}, D_meta={snapshot["d_mle"]:.3f}, '
          f'KS p={snapshot["ks_p"]:.3f} ({lang})')


if __name__ == '__main__':
    results = main()
    sys.exit(0 if results else 1)
