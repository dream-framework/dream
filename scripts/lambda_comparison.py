#!/usr/bin/env python3
"""
Lambda comparison: dust-dominated datasets where S2 loses.

For each S2_LOSES entry with downloadable data, refit S2 and the best
alternative (usually BIEXP), then compare characteristic scales:
  - S2: lambda_q (single characteristic decay scale)
  - BIEXP: lambda_1, lambda_2 (two decay scales)
  - EXP: lambda
  - GAUSS: sigma
  - POWER: lambda_0

If S2's lambda_q is off by orders of magnitude from the alternative's
characteristic scale, that's evidence S2 is fitting the wrong region
of the curve — confirming our 'dust-dominated' verdict is honest, not
wishful thinking.

Outputs a table: name | S2 lambda_q | best_alt scale(s) | ratio | orders of magnitude
"""
import os, sys, json, csv, io, re
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve, load_existing_tests
from s2_model_compare import compare as s2_compare, fit_all_models


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

def refetch_energy_charts(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'price' in obj:
            return [float(p) for p in obj['price'] if p is not None]
    except: pass
    return None

def refetch_wikipedia(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'items' in obj:
            return [item['views'] for item in obj['items']]
    except: pass
    return None

def refetch_nasa_power(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'properties' in obj and 'parameter' in obj['properties']:
            for key, vals_dict in obj['properties']['parameter'].items():
                vals = [float(v) for v in vals_dict.values() if v is not None and v >= 0]
                if len(vals) >= 50:
                    if len(vals) % 24 == 0:
                        hour_means = np.zeros(24)
                        for h in range(24):
                            hour_means[h] = np.mean(vals[h::24])
                        detrended = np.array([vals[i] - hour_means[i % 24] for i in range(len(vals))])
                        return detrended.tolist()
                    return vals
    except: pass
    return None

def refetch_open_meteo(url):
    data = fetch_url(url, timeout=25)
    if not data: return None
    try:
        d = json.loads(data)
        daily = d.get('daily', {})
        for var in ['wind_speed_10m_max', 'precipitation_sum', 'temperature_2m_mean']:
            if var in daily:
                vals = [v for v in daily[var] if v is not None]
                if len(vals) >= 50: return vals
    except: pass
    return None

def refetch_usgs_river(url):
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
    return vals if len(vals) >= 50 else None

def refetch_values(url, name=''):
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url:           return refetch_fred(url)
        if 'CSSEGISandData' in url:          return refetch_covid(url)
        if 'ndbc.noaa.gov' in url:           return refetch_ndbc(url)
        if 'energy-charts.info' in url:      return refetch_energy_charts(url)
        if 'wikimedia.org' in url:           return refetch_wikipedia(url)
        if 'power.larc.nasa.gov' in url:     return refetch_nasa_power(url)
        if 'open-meteo' in url:              return refetch_open_meteo(url)
        if 'waterservices.usgs.gov' in url:  return refetch_usgs_river(url)
    except Exception as e:
        print(f'    Refetch error: {e}')
    return None


def orders_of_magnitude(ratio):
    """Return log10 of ratio, rounded. 0 = same order, 1 = ~10x off, 2 = ~100x off."""
    if ratio <= 0 or not np.isfinite(ratio): return float('nan')
    return round(np.log10(ratio))


def main():
    print('=' * 90)
    print('LAMBDA COMPARISON: dust-dominated datasets where S2 loses')
    print('=' * 90)
    print()
    print('Question: Is S2 fitting the wrong region of the curve?')
    print('If S2 λ_q is off by 1+ orders of magnitude from the best alternative')
    print('characteristic scale, S2 is fitting a different feature, not just losing')
    print('on AICc. That would confirm "dust-dominated" is honest, not wishful thinking.')
    print()

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    losses = [t for t in data['tests'] if t.get('model_verdict') == 'S2_LOSES']
    print(f'S2_LOSES entries: {len(losses)}')
    print()

    # Process those with downloadable URLs
    results = []
    for entry in losses:
        url = entry.get('url', '')
        name = entry['name']
        best_alt = entry.get('best_alt', '?')

        vals = refetch_values(url, name)
        if not vals or len(vals) < 50:
            continue

        taus, acf = retention_curve(vals)
        if taus is None or len(taus) < 5:
            continue

        # Fit all models
        t_arr = np.array(taus, dtype=float)
        R_arr = np.array(acf, dtype=float)
        t_arr = t_arr - t_arr[0]
        if R_arr[0] > 0: R_arr = R_arr / R_arr[0]

        fits = fit_all_models(t_arr, R_arr)
        if not fits or 'S2' not in fits:
            continue

        s2_lambda = fits['S2']['lambda_q']
        s2_D = fits['S2']['D']

        # Extract best alternative's characteristic scale
        alt_scale = None
        alt_name = None
        if best_alt == 'BIEXP' and 'BIEXP' in fits:
            l1, l2 = fits['BIEXP']['popt'][1], fits['BIEXP']['popt'][3]
            # Use the dominant (larger amplitude) component — but we don't have amplitudes separated here
            # Use geometric mean of the two scales
            alt_scale = np.sqrt(l1 * l2) if l1 > 0 and l2 > 0 else max(l1, l2)
            alt_name = f'BIEXP λ_geo({l1:.1f}, {l2:.1f})'
        elif best_alt == 'EXP' and 'EXP' in fits:
            alt_scale = fits['EXP']['popt'][1]
            alt_name = 'EXP λ'
        elif best_alt == 'GAUSS' and 'GAUSS' in fits:
            alt_scale = fits['GAUSS']['popt'][1]
            alt_name = 'GAUSS σ'
        elif best_alt == 'POWER' and 'POWER' in fits:
            alt_scale = 1.0  # POWER has no characteristic scale
            alt_name = 'POWER (scale-free)'
        elif best_alt == 'LOGNORM' and 'LOGNORM' in fits:
            alt_scale = np.exp(fits['LOGNORM']['popt'][1])  # median
            alt_name = 'LOGNORM median'
        elif best_alt == 'S2_DUST' and 'S2_DUST' in fits:
            l1, l2 = fits['S2_DUST']['popt'][1], fits['S2_DUST']['popt'][4]
            alt_scale = np.sqrt(l1 * l2) if l1 > 0 and l2 > 0 else max(l1, l2)
            alt_name = f'S2_DUST λ_geo({l1:.1f}, {l2:.1f})'

        if alt_scale is None or alt_scale <= 0 or s2_lambda <= 0:
            continue

        ratio = s2_lambda / alt_scale
        oom = orders_of_magnitude(ratio)

        results.append({
            'name': name,
            'D': s2_D,
            'best_alt': best_alt,
            's2_lambda': s2_lambda,
            'alt_scale': alt_scale,
            'alt_name': alt_name,
            'ratio': ratio,
            'oom': oom,
        })

    # Sort by orders of magnitude (largest deviation first)
    results.sort(key=lambda r: abs(r['oom']) if not np.isnan(r['oom']) else 0, reverse=True)

    print(f'{"Name":<50} {"D":>5} {"best_alt":>10} {"S2 λ_q":>8} {"alt scale":>12} {"ratio":>8} {"OOM":>5}')
    print('-' * 110)
    for r in results:
        oom_str = f'{r["oom"]:+d}' if not np.isnan(r['oom']) else '?'
        print(f'  {r["name"][:48]:<50} {r["D"]:>5.2f} {r["best_alt"]:>10} {r["s2_lambda"]:>8.2f} {r["alt_scale"]:>10.2f}   {r["ratio"]:>6.2f}x {oom_str:>5}')

    print()
    print(f'{"=" * 90}')
    print('SUMMARY')
    print(f'{"=" * 90}')
    print(f'Total dust-dominated entries with downloadable data: {len(results)}')
    if not results:
        return

    oom_values = [abs(r['oom']) for r in results if not np.isnan(r['oom'])]
    print(f'  |OOM| = 0 (same order):      {sum(1 for o in oom_values if o == 0)}')
    print(f'  |OOM| = 1 (~10x off):        {sum(1 for o in oom_values if o == 1)}')
    print(f'  |OOM| = 2 (~100x off):       {sum(1 for o in oom_values if o == 2)}')
    print(f'  |OOM| ≥ 3 (~1000x+ off):     {sum(1 for o in oom_values if o >= 3)}')
    print()
    print(f'  Mean |OOM|: {np.mean(oom_values):.2f}')
    print(f'  Median |OOM|: {np.median(oom_values):.2f}')
    print(f'  Max |OOM|: {max(oom_values)}')
    print()

    # Interpretation
    n_off = sum(1 for o in oom_values if o >= 1)
    pct_off = 100 * n_off / len(oom_values)
    print(f'  {n_off}/{len(oom_values)} ({pct_off:.0f}%) have S2 λ_q off by 1+ order of magnitude')
    print()
    if pct_off > 50:
        print('VERDICT: S2 is fitting the wrong region in most dust-dominated cases.')
        print('The "dust-dominated" label is honest — S2 genuinely fails to capture the')
        print('characteristic scale, not just loses on AICc.')
    elif pct_off > 25:
        print('VERDICT: S2 is off by 1+ OOM in a substantial minority of cases.')
        print('Mixed evidence — some are real failures, some are close calls.')
    else:
        print('VERDICT: S2 λ_q is mostly within 1 order of the best alternative.')
        print('The losses are mostly about curve shape (dust structure), not about')
        print('getting the characteristic scale wrong. "Wishful thinking" concern is mild.')


if __name__ == '__main__':
    main()
