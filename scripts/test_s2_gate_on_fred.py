#!/usr/bin/env python3
"""
Test the S2 model-comparison gate on real FRED economic time series.
Downloads 8 series, computes the ACF retention curve, fits all 6 candidate
models, and shows the AICc ranking + verdict for each.
"""
import os, sys, csv, io, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_model_compare import compare
from dream_auto_scanner import retention_curve, fetch_url

SERIES = [
    ('GDP',      'US GDP quarterly'),
    ('CPIAUCSL', 'CPI all urban consumers'),
    ('UNRATE',   'Unemployment rate'),
    ('FEDFUNDS', 'Federal funds rate'),
    ('M2SL',     'M2 money supply'),
    ('SP500',    'S&P 500 index'),
    ('VIXCLS',   'VIX volatility index'),
    ('DEXUSEU',  'USD/EUR exchange rate'),
]

def load_fred(sid):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}'
    data = fetch_url(url, timeout=20)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2: return None
    # find first numeric column
    for col in range(len(rows[0])):
        vals = []
        for r in rows[1:]:
            if col < len(r):
                try:
                    v = float(r[col])
                    if not np.isnan(v) and not np.isinf(v):
                        vals.append(v)
                except: pass
        if len(vals) >= 30:
            return np.array(vals)
    return None

def main():
    print('=' * 78)
    print('S2 vs ALTERNATIVE MODELS — AICc GATE TEST ON FRED DATA')
    print('=' * 78)
    print(f'{"Series":<10} {"n":>5} {"D":>7} {"R²":>7} {"verdict":<10} '
          f'{"ΔAICc":>8} {"best_alt":<10} {"model_verdict":<14}')
    print('-' * 78)

    promoted = 0
    skipped = 0
    for sid, name in SERIES:
        vals = load_fred(sid)
        if vals is None:
            print(f'{sid:<10}  --- could not download ---')
            continue
        t, acf = retention_curve(vals)
        if t is None or len(t) < 10:
            print(f'{sid:<10}  --- ACF too short ---')
            continue
        cmp = compare(t, acf, name)
        if cmp['verdict'] == 'NO_FIT' or cmp.get('s2') is None:
            print(f'{sid:<10}  --- no fit ---')
            continue
        s2 = cmp['s2']
        s2_verdict = ('EXTRACTION' if s2['D'] > 1 else
                      'NATURAL' if s2['D'] < 0.8 else 'THRESHOLD')
        print(f'{sid:<10} {cmp["n"]:>5} {s2["D"]:>7.3f} {s2["r2"]:>7.4f} '
              f'{s2_verdict:<10} {cmp["delta_aicc"]:>+8.2f} '
              f'{cmp["best_alt_name"] or "-":<10} {cmp["verdict"]:<14}')
        if cmp['verdict'] == 'S2_WINS':
            promoted += 1
        elif cmp['verdict'] == 'S2_LOSES':
            skipped += 1

        # Show full ranking for S2_WINS cases
        if cmp['verdict'] in ('S2_WINS', 'S2_TIES'):
            for mname, aicc, r2, k in cmp['rank']:
                mark = ' ★' if mname == 'S2' else '  '
                print(f'            {mark}{mname:<10}  AICc={aicc:>8.2f}  R²={r2:.4f}  k={k}')

    print('-' * 78)
    print(f'PROMOTED to registry (S2 wins): {promoted}/{len(SERIES)}')
    print(f'SKIPPED (S2 loses to simpler model): {skipped}/{len(SERIES)}')
    print()
    print('Interpretation: a promoted entry means S2 with its 3 parameters')
    print('beat every simpler model (EXP, POWER, GAUSS, BIEXP, LOGNORM) by')
    print('ΔAICc ≤ −2 — i.e. the extra D parameter earns its keep.')

if __name__ == '__main__':
    main()
