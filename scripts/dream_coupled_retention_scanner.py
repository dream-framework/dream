#!/usr/bin/env python3
"""
DREAM Coupled Retention Scanner (T7)

Computes the cross-retention matrix R_ij(λ) for multi-observable datasets
and tests whether the principal eigenmode fits S2 better than raw single curves.

For each dataset with multiple observables:
  1. Compute autocorrelation ACF_i(λ) for each observable (self-retention)
  2. Compute cross-correlation CCF_ij(λ) for each pair (cross-retention)
  3. Build retention matrix R_ij(λ) at each scale λ
  4. Find principal eigenvalue at each λ
  5. Fit S2 to: (a) raw single curves, (b) principal eigenmode
  6. Compare R²

If principal eigenmode fits S2 better than raw curves, this supports
the coupled retention hypothesis (T7): S2 governs the dominant
retention mode, not individual observables.

Datasets used (already cached from S2 scanner):
  - FRED: GDP, CPI, UNRATE, FEDFUNDS, M2, SP500 (6 coupled economic observables)
  - Binance: BTC/ETH/SOL close+volume (coupled crypto observables)
  - NASA POWER: irradiance, wind, temperature (coupled environmental)
  - NOAA Tides: water level at multiple stations (coupled oceanographic)
"""

import os, sys, json, csv, io, urllib.request, math
import numpy as np
from datetime import datetime, timezone
from scipy.optimize import curve_fit
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.environ.get('SCAN_OUT', '/tmp/dream_scan')
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_url(url, timeout=15):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except:
        return None

def safe_float(v, default=None):
    try:
        return float(v)
    except:
        return default


# ═════════════════════════════════════════════════════════════════════
# CORE: Cross-retention matrix computation
# ═════════════════════════════════════════════════════════════════════

def compute_acf(values, max_lag=None):
    """Compute autocorrelation function R(λ) for a single observable."""
    v = np.array(values, dtype=float)
    n = len(v)
    if n < 10:
        return None, None
    v = v - v.mean()
    var = float(np.var(v))
    if var <= 0:
        return None, None
    if max_lag is None:
        max_lag = min(n // 3, 40)
    lags = list(range(1, max_lag + 1))
    R = []
    for k in lags:
        if k >= n:
            break
        c = float(np.sum(v[:n-k] * v[k:]) / n) / var
        R.append(max(c, 1e-6))
    return np.array(lags, dtype=float), np.array(R, dtype=float)


def compute_ccf(values_i, values_j, max_lag=None):
    """Compute cross-correlation function CCF_ij(λ) between two observables.
    
    CCF_ij(λ) = correlation between observable_i at time t 
                and observable_j at time t+λ
    """
    vi = np.array(values_i, dtype=float)
    vj = np.array(values_j, dtype=float)
    n = min(len(vi), len(vj))
    if n < 10:
        return None, None
    vi = vi[:n] - vi.mean()
    vj = vj[:n] - vj.mean()
    si = float(np.std(vi))
    sj = float(np.std(vj))
    if si <= 0 or sj <= 0:
        return None, None
    if max_lag is None:
        max_lag = min(n // 3, 40)
    lags = list(range(0, max_lag + 1))  # include lag 0 for cross-corr
    C = []
    for k in lags:
        if k >= n:
            break
        if k == 0:
            c = float(np.sum(vi * vj) / n) / (si * sj)
        else:
            c = float(np.sum(vi[:n-k] * vj[k:]) / n) / (si * sj)
        C.append(c)
    return np.array(lags, dtype=float), np.array(C, dtype=float)


def detrend_periodic(values, period=24):
    """Remove periodic component (e.g., diurnal cycle for hourly data).
    
    NASA POWER and Open-Meteo provide hourly data with strong 24-hour cycles.
    The irradiance ACF goes negative at lag 6 (nighttime) — this is NOT
    retention decay, it's the diurnal cycle. We remove it before computing
    retention.
    
    Method: subtract the mean for each hour-of-day.
    """
    v = np.array(values, dtype=float)
    n = len(v)
    if n < period * 3:
        return v  # not enough data for detrending
    
    # Compute mean for each phase of the cycle
    residual = np.copy(v)
    for phase in range(period):
        mask = np.arange(n) % period == phase
        if mask.sum() > 0:
            phase_mean = np.mean(v[mask])
            residual[mask] = v[mask] - phase_mean
    
    return residual


def build_retention_matrix(observables, max_lag=30, detrend_period=None):
    """Build cross-retention matrix R_ij(λ) for a set of coupled observables.
    
    observables: dict of {name: values_array}
    detrend_period: if set (e.g., 24 for hourly data), removes periodic component
    Returns: dict with lags, self_retention, cross_retention, eigenvalues, eigenmodes
    """
    names = list(observables.keys())
    n_obs = len(names)
    if n_obs < 2:
        return None
    
    # Align all observables to the same length
    min_len = min(len(v) for v in observables.values() if v is not None)
    if min_len < 20:
        return None
    
    # Truncate and optionally detrend
    processed = {}
    for name in names:
        vals = observables[name][:min_len]
        if detrend_period is not None:
            vals = detrend_periodic(vals, period=detrend_period)
        processed[name] = vals
    
    # Compute self-retention (diagonal) and cross-retention (off-diagonal)
    lags = None
    R_matrix = {}  # R_matrix[(i,j)] = array of R_ij(λ) values
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                # Self-retention (ACF)
                l, R = compute_acf(processed[name_i], max_lag=max_lag)
            else:
                # Cross-retention (CCF)
                l, R = compute_ccf(processed[name_i], processed[name_j], max_lag=max_lag)
            
            if l is not None and R is not None:
                R_matrix[(i, j)] = R
                if lags is None:
                    lags = l
            else:
                R_matrix[(i, j)] = np.zeros(max_lag)
    
    if lags is None:
        return None
    
    # At each lag λ, build the n×n matrix and find eigenvalues
    n_lags = len(lags)
    eigenvalues_per_lag = []
    eigenvectors_per_lag = []
    
    for lag_idx in range(n_lags):
        # Build n_obs × n_obs matrix at this lag
        mat = np.zeros((n_obs, n_obs))
        for i in range(n_obs):
            for j in range(n_obs):
                if (i, j) in R_matrix and lag_idx < len(R_matrix[(i, j)]):
                    mat[i, j] = R_matrix[(i, j)][lag_idx]
        
        # For self-retention (diagonal), values should be positive
        # For cross-retention, can be negative — take absolute value for eigenvalue analysis
        # Use the absolute value matrix (retention magnitude)
        mat_abs = np.abs(mat)
        
        try:
            eigenvalues = np.linalg.eigvalsh(mat_abs)
            eigenvalues_per_lag.append(eigenvalues)
            
            # Also get eigenvectors for the principal mode
            eigenvalues_full, eigenvectors = np.linalg.eigh(mat_abs)
            # Principal eigenvector = last column (largest eigenvalue)
            eigenvectors_per_lag.append(eigenvectors[:, -1])
        except:
            eigenvalues_per_lag.append(np.zeros(n_obs))
            eigenvectors_per_lag.append(np.zeros(n_obs))
    
    eigenvalues_per_lag = np.array(eigenvalues_per_lag)  # shape: (n_lags, n_obs)
    
    # Principal eigenvalue at each lag
    principal_eigenvalue = eigenvalues_per_lag[:, -1]  # largest eigenvalue
    
    # Normalize to [0,1] for S2 fitting: R(λ) = eigenvalue(λ) / eigenvalue(0)
    # The eigenvalue at lag 0 (or lag 1 for ACF) represents full retention
    # Subsequent lags show decay — this is the retention curve
    max_eig = float(np.max(principal_eigenvalue))
    if max_eig > 0:
        principal_eigenvalue = principal_eigenvalue / max_eig
    # Clip to (0, 1] — must be positive and ≤ 1 for S2 fit
    principal_eigenvalue = np.clip(principal_eigenvalue, 1e-6, 1.0)
    
    # Self-retention curves (diagonal)
    self_retention = {}
    for i, name in enumerate(names):
        if (i, i) in R_matrix:
            self_retention[name] = R_matrix[(i, i)]
    
    # Cross-retention summary (average off-diagonal magnitude)
    cross_retention_avg = []
    for lag_idx in range(n_lags):
        off_diag = []
        for i in range(n_obs):
            for j in range(n_obs):
                if i != j and (i, j) in R_matrix and lag_idx < len(R_matrix[(i, j)]):
                    off_diag.append(abs(R_matrix[(i, j)][lag_idx]))
        cross_retention_avg.append(np.mean(off_diag) if off_diag else 0)
    
    return {
        'lags': lags,
        'n_observables': n_obs,
        'observable_names': names,
        'self_retention': self_retention,
        'cross_retention_avg': np.array(cross_retention_avg),
        'principal_eigenvalue': principal_eigenvalue,
        'all_eigenvalues': eigenvalues_per_lag,
        'coupling_strength': float(np.mean(cross_retention_avg[:5])),  # avg coupling at small lags
    }


def fit_s2(t, R):
    """Fit S2: R(λ) = exp[-(λ/λ_q)^D]. Returns (D, lam_q, r2)."""
    if t is None or R is None or len(t) < 5:
        return None, None, None
    mask = (R > 0) & (R < 1)
    if mask.sum() < 3:
        return None, None, None
    ln_t = np.log(t[mask])
    ln_neg_ln_R = np.log(-np.log(R[mask]))
    N = len(ln_t)
    sx, sy = ln_t.sum(), ln_neg_ln_R.sum()
    sxy = float(np.sum(ln_t * ln_neg_ln_R))
    sx2 = float(np.sum(ln_t ** 2))
    slope = (N * sxy - sx * sy) / (N * sx2 - sx ** 2)
    intercept = (sy - slope * sx) / N
    D = float(slope)
    if D <= 0 or D > 10:
        return None, None, None
    pred = slope * ln_t + intercept
    ss_res = float(np.sum((ln_neg_ln_R - pred) ** 2))
    ss_tot = float(np.sum((ln_neg_ln_R - ln_neg_ln_R.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    lam_q = float(np.exp(-intercept / slope)) if slope != 0 else 1.0
    return D, lam_q, r2


# ═════════════════════════════════════════════════════════════════════
# DATASET FETCHERS
# ═════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id):
    """Fetch a FRED series as a list of floats."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    data = fetch_url(url, timeout=15)
    if not data:
        return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 5:
        return None
    values = []
    for row in rows[1:]:
        if len(row) >= 2 and row[1] != '.':
            try:
                v = float(row[1])
                if not np.isnan(v) and not np.isinf(v):
                    values.append(v)
            except:
                pass
    return values if len(values) >= 20 else None


def fetch_binance_ohlcv(symbol='BTCUSDT'):
    """Fetch Binance daily close + volume."""
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=365'
    data = fetch_url(url, timeout=15)
    if not data:
        return None, None
    arr = json.loads(data)
    closes = [float(c[4]) for c in arr]  # close price
    volumes = [float(c[5]) for c in arr]  # volume
    return closes, volumes


def fetch_nasa_power_multi():
    """Fetch NASA POWER with multiple parameters for one location."""
    url = ('https://power.larc.nasa.gov/api/temporal/hourly/point?'
           'start=20230101&end=20231231'
           '&parameters=ALLSKY_SFC_SW_DWN,WS10M,T2M,RH2M,PRECTOTCORR,PS'
           '&longitude=-105.0&latitude=40.0&community=RE&format=JSON')
    data = fetch_url(url, timeout=15)
    if not data:
        return None
    obj = json.loads(data)
    if 'properties' not in obj or 'parameter' not in obj['properties']:
        return None
    params = obj['properties']['parameter']
    result = {}
    for p in ['ALLSKY_SFC_SW_DWN', 'WS10M', 'T2M', 'RH2M', 'PRECTOTCORR', 'PS']:
        if p in params:
            vals = [float(v) for v in params[p].values() if v is not None and v >= 0]
            if len(vals) >= 50:
                result[p] = vals
    return result if len(result) >= 3 else None


def fetch_openmeteo_multi():
    """Fetch Open-Meteo with multiple parameters for one city."""
    url = ('https://archive-api.open-meteo.com/v1/archive?latitude=40.42&longitude=-74.0'
           '&start_date=2023-01-01&end_date=2023-12-31'
           '&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m')
    data = fetch_url(url, timeout=15)
    if not data:
        return None
    obj = json.loads(data)
    if 'hourly' not in obj:
        return None
    hourly = obj['hourly']
    result = {}
    for p in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']:
        if p in hourly:
            vals = [float(v) for v in hourly[p] if v is not None]
            if len(vals) >= 50:
                result[p] = vals
    return result if len(result) >= 2 else None


# ═════════════════════════════════════════════════════════════════════
# COUPLED RETENTION TESTS
# ═════════════════════════════════════════════════════════════════════

def test_coupled_retention():
    """Test T7: does the principal eigenmode of coupled observables fit S2 better?"""
    print('\n📡 COUPLED RETENTION (T7): Cross-retention matrix analysis')
    results = []
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    datasets = []
    
    # 1. FRED economic (6 coupled observables)
    print('\n  Fetching FRED economic data...')
    fred_series = {
        'GDP': 'GDP', 'CPI': 'CPIAUCSL', 'UNRATE': 'UNRATE',
        'FEDFUNDS': 'FEDFUNDS', 'M2': 'M2SL', 'SP500': 'SP500',
    }
    fred_data = {}
    for name, sid in fred_series.items():
        vals = fetch_fred_series(sid)
        if vals and len(vals) >= 50:
            fred_data[name] = vals
            print(f'    {name}: {len(vals)} observations')
        else:
            print(f'    {name}: insufficient data')
    if len(fred_data) >= 3:
        datasets.append(('FRED Economic', fred_data, 'economic'))
    
    # 2. Binance crypto (close + volume for 3 coins)
    print('\n  Fetching Binance crypto data...')
    binance_data = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        closes, volumes = fetch_binance_ohlcv(symbol)
        if closes and len(closes) >= 50:
            short = symbol.replace('USDT', '')
            binance_data[f'{short}_price'] = closes
            binance_data[f'{short}_volume'] = volumes
            print(f'    {short}: {len(closes)} days (price+volume)')
    if len(binance_data) >= 4:
        datasets.append(('Binance Crypto', binance_data, 'crypto'))
    
    # 3. NASA POWER environmental
    print('\n  Fetching NASA POWER environmental data...')
    nasa_data = fetch_nasa_power_multi()
    if nasa_data:
        print(f'    Parameters: {list(nasa_data.keys())}')
        datasets.append(('NASA POWER Environmental', nasa_data, 'environmental'))
    
    # 4. Open-Meteo weather
    print('\n  Fetching Open-Meteo weather data...')
    meteo_data = fetch_openmeteo_multi()
    if meteo_data:
        print(f'    Parameters: {list(meteo_data.keys())}')
        datasets.append(('Open-Meteo Weather', meteo_data, 'environmental'))
    
    # 5. COVID-19 (S2 "failure" dataset — lost to BIEXP in single-curve fit)
    print('\n  Fetching COVID-19 data (S2 failure dataset)...')
    covid_data = {}
    for kind in ['confirmed', 'deaths']:
        url = f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{kind}_global.csv'
        data = fetch_url(url, timeout=15)
        if data:
            text = data.decode('utf-8') if isinstance(data, bytes) else data
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) > 1:
                # Sum all regions per day
                daily_sums = []
                for col in range(4, len(rows[0])):
                    total = 0
                    for row in rows[1:]:
                        if col < len(row):
                            try:
                                total += int(row[col])
                            except:
                                pass
                    daily_sums.append(total)
                # Convert to daily new (diff)
                daily_new = [max(0, daily_sums[i] - daily_sums[i-1]) for i in range(1, len(daily_sums))]
                if len(daily_new) >= 50:
                    covid_data[f'COVID_{kind}'] = daily_new
                    print(f'    COVID {kind}: {len(daily_new)} days')
    if len(covid_data) >= 2:
        datasets.append(('COVID-19 (S2 failure)', covid_data, 'epidemiological'))
    
    # 6. Energy-Charts electricity prices (S2 "failure" — lost to BIEXP)
    print('\n  Fetching Energy-Charts prices (S2 failure dataset)...')
    energy_data = {}
    for zone, name in [('DE-LU', 'DE'), ('FR', 'FR'), ('NL', 'NL'), ('CH', 'CH')]:
        url = (f'https://api.energy-charts.info/price?bzn={zone}'
               f'&start=2023-01-01T00%3A00%2B01%3A00&end=2023-06-30T00%3A00%2B01%3A00')
        data = fetch_url(url, timeout=15)
        if data:
            try:
                obj = json.loads(data)
                if 'price' in obj:
                    prices = [float(p) for p in obj['price'] if p is not None]
                    if len(prices) >= 50:
                        energy_data[f'{name}_price'] = prices
                        print(f'    {name}: {len(prices)} hours')
            except:
                pass
    if len(energy_data) >= 3:
        datasets.append(('Energy Prices (S2 failure)', energy_data, 'energy'))
    
    # Analyze each dataset
    for ds_name, observables, domain in datasets:
        print(f'\n  Analyzing: {ds_name} ({len(observables)} observables)')
        
        # Detect if data is hourly (need diurnal detrending + more lags)
        min_len = min(len(v) for v in observables.values())
        is_hourly = min_len > 1000
        detrend = 24 if is_hourly else None
        max_lag = 50 if is_hourly else 25  # hourly data needs more lags to see decay
        
        result = build_retention_matrix(observables, max_lag=max_lag, detrend_period=detrend)
        if result is None:
            print(f'    ✗ Could not build retention matrix')
            continue
        
        lags = result['lags']
        principal = result['principal_eigenvalue']
        coupling = result['coupling_strength']
        
        # Fit S2 to principal eigenmode
        D_principal, lam_principal, r2_principal = fit_s2(lags, principal)
        
        # Fit S2 to each individual self-retention curve
        single_fits = {}
        best_single_r2 = -1
        worst_single_r2 = 2
        for name, R in result['self_retention'].items():
            D, lam, r2 = fit_s2(lags[:len(R)], R)
            if D is not None:
                single_fits[name] = {'D': D, 'lam_q': lam, 'r2': r2}
                if r2 > best_single_r2:
                    best_single_r2 = r2
                if r2 < worst_single_r2:
                    worst_single_r2 = r2
        
        avg_single_r2 = np.mean([f['r2'] for f in single_fits.values()]) if single_fits else 0
        
        # Verdict: refined logic based on improvement magnitude and coupling strength
        eigenmode_better = r2_principal is not None and r2_principal > avg_single_r2
        improvement = (r2_principal - avg_single_r2) if r2_principal is not None else 0
        
        # Coupling complexity
        if coupling < 0.1:
            coupling_level = 'Low (single mode dominates)'
        elif coupling < 0.3:
            coupling_level = 'Medium (moderate coupling)'
        else:
            coupling_level = 'High (strong coupling)'
        
        # REFINED VERDICT LOGIC:
        # SUPPORTED: improvement > 0.05 AND coupling > 0.15 (meaningful improvement in coupled system)
        # PARTIAL: eigenmode fits but marginal (<0.05 improvement) OR coupling too low to matter
        # INCONSISTENT: eigenmode worse than single curves OR fit failed
        if r2_principal is None:
            verdict = 'INCONSISTENT'
            verdict_note = 'Could not fit S2 to principal eigenmode. May need more observables or different lag range.'
        elif improvement > 0.05 and coupling > 0.15:
            verdict = 'SUPPORTED'
            verdict_note = (
                f'SUPPORTED: eigenmode improves R² by +{improvement:.3f} over single curves '
                f'with coupling={coupling:.3f}. T7 applies: the principal retention mode '
                f'captures structure that individual observables miss.'
            )
        elif eigenmode_better:
            verdict = 'PARTIAL'
            verdict_note = (
                f'PARTIAL (marginal): eigenmode improves R² by only +{improvement:.3f}. '
                f'This is below the 0.05 threshold for meaningful improvement. '
                f'Coupling={coupling:.3f} may be too {"low" if coupling < 0.15 else "moderate"} '
                f'for T7 to add significant value. Single curves already fit well.'
            )
        else:
            verdict = 'INCONSISTENT'
            verdict_note = (
                f'Eigenmode R²={r2_principal:.4f} is WORSE than single avg R²={avg_single_r2:.4f}. '
                f'T7 does not help for this dataset.'
            )
        
        # Build narrative
        if D_principal is not None:
            narrative = (
                f'{ds_name}: {len(observables)} coupled observables analyzed. '
                f'Cross-retention matrix R_ij(λ) computed with {len(lags)} lags. '
                f'Principal eigenvalue fitted to S2: D_eff={D_principal:.4f}, '
                f'λ_q={lam_principal:.4f}, R²={r2_principal:.4f}. '
                f'Average single-curve R²={avg_single_r2:.4f}. '
                f'Improvement: +{improvement:.4f} ({"+".replace("+","+") if improvement > 0 else ""}{improvement*100:.1f}%). '
                f'Coupling strength={coupling:.3f} ({coupling_level}). '
                f'{verdict_note}'
            )
        else:
            narrative = (
                f'{ds_name}: {len(observables)} coupled observables. '
                f'Could not fit S2 to principal eigenmode. '
                f'Coupling strength={coupling:.3f} ({coupling_level}). '
                f'Average single-curve R²={avg_single_r2:.4f}.'
            )
        
        results.append({
            'theorem': 'T7',
            'category': 'coupled_retention',
            'name': f'{ds_name}: coupled retention matrix ({len(observables)} observables)',
            'source': f'Dynamic fetch ({today})',
            'url': '',
            'date': today,
            'data_points': len(lags),
            'n_observables': len(observables),
            'observable_names': ', '.join(observables.keys()),
            'D_eff_principal': round(float(D_principal), 4) if D_principal else None,
            'r2_principal': round(float(r2_principal), 4) if r2_principal else None,
            'avg_r2_single': round(float(avg_single_r2), 4),
            'best_r2_single': round(float(best_single_r2), 4) if best_single_r2 >= 0 else None,
            'improvement': round(float(improvement), 4),
            'coupling_strength': round(float(coupling), 4),
            'coupling_level': coupling_level,
            'eigenmode_better': bool(eigenmode_better),
            'verdict': verdict,
            'verdict_note': verdict_note,
            'narrative': narrative,
        })
        
        print(f'    Principal: D={D_principal:.4f}, R²={r2_principal:.4f}' if D_principal else '    Principal: fit failed')
        print(f'    Single avg: R²={avg_single_r2:.4f}, best: {best_single_r2:.4f}')
        print(f'    Coupling: {coupling:.3f} ({coupling_level})')
        print(f'    Eigenmode better: {eigenmode_better}')
    
    return results


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('DREAM Coupled Retention Scanner (T7)')
    print(f'Date: {datetime.now(timezone.utc).isoformat()}')
    print('=' * 60)
    
    results = test_coupled_retention()
    
    print(f'\n{"=" * 60}')
    print(f'COUPLED RETENTION SCAN COMPLETE')
    print(f'{"=" * 60}')
    print(f'Datasets analyzed: {len(results)}')
    
    from collections import Counter
    verdicts = Counter(r['verdict'] for r in results)
    print(f'Verdicts: {dict(verdicts)}')
    
    output = {
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'total_tests': len(results),
        'tests': results,
    }
    
    output_path = os.path.join(OUT_DIR, 'coupled_retention_tests.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    repo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coupled_retention_tests.json')
    with open(repo_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f'\n✓ Written to {output_path}')
    print(f'✓ Written to {repo_path}')
    
    return results

if __name__ == '__main__':
    main()
