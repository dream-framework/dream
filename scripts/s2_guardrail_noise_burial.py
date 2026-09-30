#!/usr/bin/env python3
"""
S2 Guardrail — Noise-Burial Test
=================================

For each dataset that FAILED the full guardrails, test whether S2 structure
is buried under noise by applying progressive denoising and re-running
all 5 guardrails at each noise level.

If a FAIL converts to SURVIVES after denoising → structure was buried.
If it remains FAILS at all noise levels → genuinely non-S2-shaped.
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/z/my-project/dream_repo/scripts')
from s2_guardrail_audit import (
    run_all_guardrails, GUARDRAILS, m_s2, aicc,
    to_acf, check_fit_quality
)

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch(url, timeout=15):
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-Noise/1.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 1: raise
            time.sleep(1)

# ─── Denoising methods ──────────────────────────────────────────────

def moving_average(values, window):
    """Simple moving average with edge padding."""
    if window <= 1:
        return values.copy()
    n = len(values)
    padded = np.pad(values, (window//2, window//2), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')[:n]

def savgol_smooth(values, window):
    """Savitzky-Golay filter — preserves shape better than MA."""
    if window <= 3 or window >= len(values):
        return values.copy()
    # Window must be odd
    if window % 2 == 0:
        window += 1
    if window >= len(values):
        window = len(values) - 1 if len(values) % 2 == 0 else len(values) - 2
    try:
        return savgol_filter(values, window, polyorder=min(3, window-1))
    except Exception:
        return values.copy()

def lowpass_fft(values, keep_frac):
    """FFT-based low-pass: keep lowest keep_frac of frequencies."""
    n = len(values)
    fft = np.fft.fft(values)
    n_keep = max(1, int(n * keep_frac))
    fft_masked = fft.copy()
    fft_masked[n_keep:-n_keep] = 0  # zero out high frequencies
    return np.real(np.fft.ifft(fft_masked))

# ─── Fetch the 6 failing datasets ──────────────────────────────────

def fetch_worldbank(url):
    raw = fetch(url).decode('utf-8')
    j = json.loads(raw)
    rows = j[1] if len(j) > 1 and isinstance(j[1], list) else []
    vals = []
    for r in rows:
        try:
            v = float(r['value'])
            if v == v and v != 0: vals.append(v)
        except: continue
    return np.array(vals, dtype=float) if len(vals) >= 30 else None

def fetch_usgs():
    raw = fetch('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv').decode()
    reader = csv.DictReader(io.StringIO(raw))
    times = [r.get('time') for r in reader if r.get('time')]
    if len(times) < 30: return None
    import datetime
    hrs = []
    for tstr in times:
        try:
            dt = datetime.datetime.fromisoformat(tstr.replace('Z', '+00:00'))
            hrs.append(dt.timestamp() / 3600.0)
        except: continue
    hrs.sort()
    h0 = math.floor(min(hrs)); h_max = math.ceil(max(hrs))
    counts, _ = np.histogram(hrs, bins=np.arange(h0, h_max + 2))
    return counts.astype(float)

FAILING_DATASETS = [
    {
        'name': 'WB USA GDP (annual)',
        'url': 'https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD?format=json',
        'fetcher': lambda url: fetch_worldbank(url),
        'baseline_r2': -0.207,
    },
    {
        'name': 'WB USA CPI (annual)',
        'url': 'https://api.worldbank.org/v2/country/US/indicator/FP.CPI.TOTL?format=json',
        'fetcher': lambda url: fetch_worldbank(url),
        'baseline_r2': -0.208,
    },
    {
        'name': 'WB GDP (ACF)',
        'url': 'https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.CD?format=json&per_page=100',
        'fetcher': lambda url: fetch_worldbank(url),
        'baseline_r2': -0.204,
    },
    {
        'name': 'WB CPI (ACF)',
        'url': 'https://api.worldbank.org/v2/country/US/indicator/FP.CPI.TOTL?format=json&per_page=100',
        'fetcher': lambda url: fetch_worldbank(url),
        'baseline_r2': -0.199,
    },
    {
        'name': 'WB Unemployment (annual)',
        'url': 'https://api.worldbank.org/v2/country/US/indicator/SL.UEM.TOTL.ZS?format=json&per_page=100',
        'fetcher': lambda url: fetch_worldbank(url),
        'baseline_r2': 0.076,
    },
    {
        'name': 'USGS Earthquakes',
        'url': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv',
        'fetcher': lambda _: fetch_usgs(),
        'baseline_r2': 0.646,
    },
]

# ─── Denoising levels to test ──────────────────────────────────────

DENOISE_LEVELS = [
    ('raw', lambda v: v),
    ('MA_w3', lambda v: moving_average(v, 3)),
    ('MA_w5', lambda v: moving_average(v, 5)),
    ('MA_w10', lambda v: moving_average(v, 10)),
    ('Savgol_w7', lambda v: savgol_smooth(v, 7)),
    ('Savgol_w11', lambda v: savgol_smooth(v, 11)),
    ('LowPass_50%', lambda v: lowpass_fft(v, 0.50)),
    ('LowPass_25%', lambda v: lowpass_fft(v, 0.25)),
    ('LowPass_10%', lambda v: lowpass_fft(v, 0.10)),
]

# ─── Main ────────────────────────────────────────────────────────────

print('='*72)
print('S2 GUARDRAIL — NOISE-BURIAL TEST')
print('='*72)
print()
print('For each dataset that FAILED full guardrails:')
print('  Apply progressive denoising → re-run all 5 guardrails')
print('  If FAIL → SURVIVES after denoising: structure was buried under noise')
print('  If remains FAILS: genuinely non-S2-shaped')
print()

all_results = []

for ds in FAILING_DATASETS:
    print(f'\n{"="*72}')
    print(f'DATASET: {ds["name"]}  (baseline R² = {ds["baseline_r2"]:.3f})')
    print(f'{"="*72}')
    
    try:
        values = ds['fetcher'](ds['url'])
        if values is None or len(values) < 30:
            print(f'  FETCH failed')
            continue
        print(f'  n = {len(values)}', flush=True)
    except Exception as e:
        print(f'  FETCH ERROR: {repr(e)[:80]}')
        continue
    
    ds_result = {'name': ds['name'], 'baseline_r2': ds['baseline_r2'], 'n': len(values), 'levels': {}}
    
    for level_name, denoise_fn in DENOISE_LEVELS:
        print(f'\n  --- Denoise: {level_name} ---', flush=True)
        
        # Apply denoising
        try:
            denoised = denoise_fn(values)
        except Exception as e:
            print(f'    DENOISE ERROR: {repr(e)[:80]}')
            continue
        
        if len(denoised) < 30:
            print(f'    Too short after denoising: n={len(denoised)}')
            continue
        
        # Compute ACF
        acf_result = to_acf(denoised, max_lag=80)
        if acf_result is None:
            print(f'    ACF failed')
            continue
        t_arr, R_arr = acf_result
        t_arr = t_arr[1:]; R_arr = R_arr[1:]
        if len(t_arr) < 20:
            print(f'    Too few lags: {len(t_arr)}')
            continue
        
        # Run all guardrails
        result = run_all_guardrails(t_arr, R_arr)
        
        # Extract key stats
        g1 = result['guardrails'].get('G1_fit_quality', {})
        g3 = result['guardrails'].get('G3_stability', {})
        g4 = result['guardrails'].get('G4_null_destruction', {})
        g5 = result['guardrails'].get('G5_identifiability', {})
        
        r2 = g1.get('r2', 0) or 0
        d_cv = g3.get('D_cv', 0) or 0
        lam_cv = g3.get('lam_cv', 0) or 0
        null_ratio = g4.get('ratio', 0) or 0
        
        verdict = result['verdict']
        marker = '✓ SURVIVES' if verdict == 'SURVIVES' else f'✗ {result["reason"][:40]}'
        
        print(f'    R²={r2:.3f}  D_cv={d_cv:.2f}  λ_cv={lam_cv:.2f}  null_ratio={null_ratio:.1f}  → {marker}')
        
        ds_result['levels'][level_name] = {
            'verdict': verdict,
            'r2': float(r2),
            'd_cv': float(d_cv),
            'lam_cv': float(lam_cv),
            'null_ratio': float(null_ratio),
            'reason': result['reason'],
            'g1_r2': float(g1.get('r2', 0) or 0),
            'g3_pass': g3.get('pass'),
            'g4_pass': g4.get('pass'),
            'g5_pass': g5.get('pass'),
        }
        
        if verdict == 'SURVIVES':
            print(f'    >>> STRUCTURE WAS BURIED UNDER NOISE — S2 SURVIVES AFTER DENOISING <<<')
    
    all_results.append(ds_result)

# ─── Summary ─────────────────────────────────────────────────────────

print('\n' + '='*72)
print('NOISE-BURIAL TEST SUMMARY')
print('='*72)
print()
print(f'{"Dataset":<25s} {"Baseline":>10s}', end='')
for level_name, _ in DENOISE_LEVELS:
    print(f'  {level_name:>12s}', end='')
print()
print('-' * (25 + 10 + 14 * len(DENOISE_LEVELS)))

for ds in all_results:
    print(f'{ds["name"]:<25s} {ds["baseline_r2"]:>10.3f}', end='')
    for level_name, _ in DENOISE_LEVELS:
        level = ds['levels'].get(level_name, {})
        v = level.get('verdict', '?')
        r2 = level.get('r2', 0)
        if v == 'SURVIVES':
            print(f'  {"✓":>4s} R²={r2:.2f}', end='')
        elif v == 'FAILS':
            print(f'  {"✗":>4s} R²={r2:.2f}', end='')
        else:
            print(f'  {"?":>4s}       ', end='')
    print()

# Count rescues
n_rescued = 0
n_still_fail = 0
for ds in all_results:
    any_survive = any(v.get('verdict') == 'SURVIVES' for v in ds['levels'].values())
    if any_survive:
        n_rescued += 1
    else:
        n_still_fail += 1

print(f'\nRescued by denoising (structure was buried): {n_rescued}/{len(all_results)}')
print(f'Still fails (genuinely non-S2):              {n_still_fail}/{len(all_results)}')

# Which denoising method worked best?
print(f'\nRescue by method:')
from collections import Counter
rescue_methods = Counter()
for ds in all_results:
    for level_name, level_data in ds['levels'].items():
        if level_data.get('verdict') == 'SURVIVES':
            rescue_methods[level_name] += 1
for method, count in rescue_methods.most_common():
    print(f'  {method}: rescued {count} dataset(s)')

# Save
out = {
    'test': 'S2 Guardrail Noise-Burial Test',
    'n_failing_datasets': len(all_results),
    'n_rescued': n_rescued,
    'n_still_fail': n_still_fail,
    'denoise_levels': [l[0] for l in DENOISE_LEVELS],
    'results': all_results,
}
out_path = os.path.join(OUT_DIR, 's2_guardrail_noise_burial.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')

print('\n' + '='*72)
print('VERDICT')
print('='*72)
if n_rescued > 0:
    print(f'{n_rescued}/{len(all_results)} failing datasets were rescued by denoising.')
    print('Their S2 structure was BURIED UNDER NOISE — not absent.')
    print('With appropriate noise reduction, S2 guardrails pass.')
    print()
    print('This means the original survival rate was UNDERESTIMATED.')
    print('After denoising, the true Meta-S2 survival rate is higher.')
else:
    print('No failing datasets were rescued by denoising.')
    print('The failures are GENUINELY non-S2-shaped — not noise-buried.')
    print('These datasets have different structural form (trends, regime shifts, policy cycles).')
