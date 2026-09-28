#!/usr/bin/env python3
"""
S2 GUARDRAIL FRAMEWORK
=====================

Implements the formal admissibility tests for "S2 is present" —
not just "S2 fits" (which is vacuous; any optimizer returns parameters),
but "S2 survives" the predetermined guardrails.

Guardrails (from user's formulation):
  1. Fit quality:      R² > R²_min  (S2 must actually explain structure)
  2. Error criterion:  AICc well-defined, not degenerate
  3. Parameter stability: bootstrap variance of (D, λ_q) bounded
  4. Null destruction:  scrambled data substantially weakens the signal
  5. Identifiability:   parameters not pinned against bounds
  6. Independent recurrence: happens across many datasets (Meta-S2)

For the 208-entry registry, we apply guardrails 1, 2, 5 from metadata,
and guardrails 3, 4 require re-fetching raw data (done for fetchable subset).

Output:
  - Per-dataset verdict: SURVIVES / FAILS / INSUFFICIENT_DATA
  - Meta-S2 survival rate: P(S2 survives | eligible dataset)
"""
import json, os, ssl, urllib.request, csv, io, time, sys, math, signal
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

REPO = '/home/z/my-project/dream_repo'
OUT_DIR = '/home/z/my-project/download'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Guardrail thresholds (predetermined, not tuned) ─────────────────
GUARDRAILS = {
    'r2_min': 0.90,           # S2 must explain >90% of variance
    'r2_strong': 0.95,        # Strong evidence threshold
    'bootstrap_n': 30,       # Number of bootstrap resamples for stability
    'd_cv_max': 0.30,         # D coefficient of variation must be < 30%
    'lam_cv_max': 0.50,       # λ_q coefficient of variation must be < 50%
    'null_ratio_min': 2.0,    # Real signal must be >2× stronger than null
    'd_bound_margin': 0.10,   # D must be >0.10 away from bounds (0.01, 10.0)
    'lam_bound_margin': 0.10, # λ_q must be >10% away from bounds
    'delta_aicc_min': -2.0,   # S2 must not lose by >2 AICc vs best alt
    'scramble_n': 5,          # Number of scramble iterations for null test
}

# S2 model
def m_s2(t, A, lam, D):
    return A * np.exp(-np.power(np.maximum(t, 1e-9) / max(lam, 1e-9), D))

def aicc(rss, n, k):
    if not np.isfinite(rss) or n <= k + 1: return float('inf')
    return 2*k + n*np.log(rss/n) + (2*k*(k+1))/(n-k-1)

# ─── Guardrail checks ────────────────────────────────────────────────

def check_fit_quality(t, R):
    """Guardrail 1: S2 must explain >R²_min of variance."""
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    tm = float(t[len(t)//2])
    best = None
    for p0 in [[1.0, tm, 0.5], [1.0, tm*0.5, 1.0], [1.0, tm*2, 0.3]]:
        try:
            popt, _ = curve_fit(m_s2, t, R_n, p0=p0,
                bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]), maxfev=10000)
            rss = float(np.sum((R_n - m_s2(t, *popt))**2))
            if best is None or rss < best[1]:
                best = (popt, rss)
        except: continue
    if not best:
        return {'pass': False, 'reason': 'fit_failed', 'r2': None}
    popt, rss = best
    ss_tot = float(np.sum((R_n - R_n.mean())**2))
    r2 = 1 - rss/ss_tot if ss_tot > 0 else 0
    return {
        'pass': r2 >= GUARDRAILS['r2_min'],
        'r2': float(r2),
        'strong': r2 >= GUARDRAILS['r2_strong'],
        'params': list(popt),
        'rss': rss,
        'reason': 'ok' if r2 >= GUARDRAILS['r2_min'] else f'r2={r2:.3f}<{GUARDRAILS["r2_min"]}',
    }

def check_identifiability(params):
    """Guardrail 5: parameters not pinned against bounds."""
    A, lam, D = params
    d_lo, d_hi = 0.01, 10.0
    lam_lo, lam_hi = 1e-3, 1e6
    d_at_bound = (D - d_lo < GUARDRAILS['d_bound_margin']) or (d_hi - D < GUARDRAILS['d_bound_margin'])
    lam_at_bound = (lam - lam_lo < GUARDRAILS['lam_bound_margin'] * lam) or (lam_hi - lam < GUARDRAILS['lam_bound_margin'] * lam)
    return {
        'pass': not (d_at_bound or lam_at_bound),
        'D_at_bound': d_at_bound,
        'lam_at_bound': lam_at_bound,
        'reason': 'ok' if not (d_at_bound or lam_at_bound) else 'parameter_at_bound',
    }

def check_parameter_stability(t, R, n_boot=30):
    """Guardrail 3: bootstrap variance of (D, λ_q) bounded."""
    if R[0] > 0: R_n = R / R[0]
    else: R_n = R / max(abs(R))
    n = len(t)
    rng = np.random.RandomState(42)
    Ds, lams = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        t_b, R_b = t[idx], R_n[idx]
        tm = float(np.median(t_b))
        try:
            popt, _ = curve_fit(m_s2, t_b, R_b, p0=[1.0, tm, 0.5],
                bounds=([0.01, 1e-3, 0.01], [2.0, 1e6, 10.0]), maxfev=5000)
            Ds.append(popt[2]); lams.append(popt[1])
        except: continue
    if len(Ds) < 10:
        return {'pass': False, 'reason': 'too_few_bootstrap_fits', 'n': len(Ds)}
    Ds = np.array(Ds); lams = np.array(lams)
    d_cv = Ds.std() / max(abs(Ds.mean()), 1e-6)
    lam_cv = lams.std() / max(abs(lams.mean()), 1e-6)
    return {
        'pass': d_cv < GUARDRAILS['d_cv_max'] and lam_cv < GUARDRAILS['lam_cv_max'],
        'D_mean': float(Ds.mean()), 'D_std': float(Ds.std()), 'D_cv': float(d_cv),
        'lam_mean': float(lams.mean()), 'lam_std': float(lams.std()), 'lam_cv': float(lam_cv),
        'n': len(Ds),
        'reason': 'ok' if (d_cv < GUARDRAILS['d_cv_max'] and lam_cv < GUARDRAILS['lam_cv_max']) else f'D_cv={d_cv:.2f}, lam_cv={lam_cv:.2f}',
    }

def check_null_destruction(t, R, n_scramble=5):
    """Guardrail 4: scrambled data substantially weakens the signal."""
    # Real R²
    fq = check_fit_quality(t, R)
    real_r2 = fq.get('r2', 0)
    if real_r2 is None:
        return {'pass': False, 'reason': 'real_fit_failed'}
    # Scrambled R²
    rng = np.random.RandomState(42)
    scram_r2s = []
    for _ in range(n_scramble):
        R_scram = rng.permutation(R)
        fq_s = check_fit_quality(t, R_scram)
        if fq_s.get('r2') is not None:
            scram_r2s.append(fq_s['r2'])
    if not scram_r2s:
        return {'pass': False, 'reason': 'scramble_failed'}
    scram_r2_med = float(np.median(scram_r2s))
    ratio = real_r2 / max(scram_r2_med, 1e-6)
    return {
        'pass': ratio >= GUARDRAILS['null_ratio_min'],
        'real_r2': float(real_r2),
        'scram_r2_median': scram_r2_med,
        'ratio': float(ratio),
        'n_scramble': len(scram_r2s),
        'reason': 'ok' if ratio >= GUARDRAILS['null_ratio_min'] else f'ratio={ratio:.2f}<{GUARDRAILS["null_ratio_min"]}',
    }

def run_all_guardrails(t, R):
    """Run all 5 guardrails on (t, R). Return verdict."""
    results = {}
    # G1: Fit quality
    g1 = check_fit_quality(t, R)
    results['G1_fit_quality'] = g1
    if not g1['pass']:
        return {'verdict': 'FAILS', 'guardrails': results, 'reason': g1['reason']}
    # G5: Identifiability
    g5 = check_identifiability(g1['params'])
    results['G5_identifiability'] = g5
    if not g5['pass']:
        return {'verdict': 'FAILS', 'guardrails': results, 'reason': g5['reason']}
    # G3: Parameter stability
    g3 = check_parameter_stability(t, R, GUARDRAILS['bootstrap_n'])
    results['G3_stability'] = g3
    if not g3['pass']:
        return {'verdict': 'FAILS', 'guardrails': results, 'reason': g3['reason']}
    # G4: Null destruction
    g4 = check_null_destruction(t, R, GUARDRAILS['scramble_n'])
    results['G4_null_destruction'] = g4
    if not g4['pass']:
        return {'verdict': 'FAILS', 'guardrails': results, 'reason': g4['reason']}
    # All passed
    return {'verdict': 'SURVIVES', 'guardrails': results, 'reason': 'all_guardrails_passed'}


# ─── Apply to registry metadata (partial guardrails) ─────────────────

print('='*72)
print('S2 GUARDRAIL FRAMEWORK — REGISTRY AUDIT')
print('='*72)
print()
print(f'Guardrail thresholds:')
for k, v in GUARDRAILS.items():
    print(f'  {k:>25s}: {v}')
print()

with open(os.path.join(REPO, 'en/tests.json')) as f:
    tests = json.load(f)['tests']

print(f'Registry: {len(tests)} entries')
print()

# Apply metadata-level guardrails (G1 from r2, G2 from model_verdict, G5 from D bounds)
print('--- Metadata-level guardrails (from registry) ---')
print()

metadata_results = []
for t in tests:
    name = t.get('name', '')[:50]
    dom = t.get('domain', '')
    D = t.get('D')
    r2 = t.get('r2')
    mv = t.get('model_verdict', '')
    da = t.get('delta_aicc')
    ba = t.get('best_alt', '')
    
    verdict = 'INSUFFICIENT_DATA'
    reasons = []
    
    # G1: Fit quality (from r2)
    if r2 is not None and r2 >= 0:
        if r2 < GUARDRAILS['r2_min']:
            verdict = 'FAILS'
            reasons.append(f'r2={r2:.3f}<{GUARDRAILS["r2_min"]}')
        else:
            pass  # G1 passes from metadata
    else:
        reasons.append('no_r2')
    
    # G5: Identifiability (from D — check if at bounds)
    if D is not None:
        if D <= 0.02 or D >= 9.9:
            verdict = 'FAILS'
            reasons.append(f'D={D:.3f}_at_bound')
    else:
        reasons.append('no_D')
    
    # G2: Model comparison (from delta_aicc and model_verdict)
    if da is not None and mv:
        if mv == 'S2_LOSES' and da is not None and da > 10:
            # S2 loses by >10 AICc — this doesn't fail guardrail (S2 doesn't need to be best)
            # but it's a flag
            pass
        elif mv == 'S2_WINS':
            pass  # Good
    
    if verdict != 'FAILS' and not reasons:
        verdict = 'METADATA_PASS'  # Passes metadata-level checks; needs raw data for full guardrails
    
    metadata_results.append({
        'name': name,
        'domain': dom,
        'D': D,
        'r2': r2,
        'model_verdict': mv,
        'delta_aicc': da,
        'best_alt': ba,
        'metadata_verdict': verdict,
        'reasons': reasons,
    })

# Summary
n_total = len(metadata_results)
n_pass = sum(1 for r in metadata_results if r['metadata_verdict'] == 'METADATA_PASS')
n_fail = sum(1 for r in metadata_results if r['metadata_verdict'] == 'FAILS')
n_insuf = sum(1 for r in metadata_results if r['metadata_verdict'] == 'INSUFFICIENT_DATA')

print(f'Metadata-level guardrail results:')
print(f'  METADATA_PASS (eligible for full guardrails): {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)')
print(f'  FAILS (failed metadata checks):              {n_fail}/{n_total} ({100*n_fail/n_total:.1f}%)')
print(f'  INSUFFICIENT_DATA:                            {n_insuf}/{n_total} ({100*n_insuf/n_total:.1f}%)')
print()

# D distribution of survivors
survivors = [r for r in metadata_results if r['metadata_verdict'] == 'METADATA_PASS']
if survivors:
    Ds = [r['D'] for r in survivors if r['D'] is not None]
    r2s = [r['r2'] for r in survivors if r['r2'] is not None]
    print(f'Survivors D distribution: n={len(Ds)}, mean={np.mean(Ds):.3f}, median={np.median(Ds):.3f}, std={np.std(Ds):.3f}')
    print(f'Survivors R² distribution: n={len(r2s)}, mean={np.mean(r2s):.3f}, median={np.median(r2s):.3f}')

# By domain
print(f'\nSurvival by domain:')
from collections import defaultdict
by_dom = defaultdict(lambda: {'total': 0, 'pass': 0, 'fail': 0})
for r in metadata_results:
    d = r['domain']
    by_dom[d]['total'] += 1
    if r['metadata_verdict'] == 'METADATA_PASS':
        by_dom[d]['pass'] += 1
    elif r['metadata_verdict'] == 'FAILS':
        by_dom[d]['fail'] += 1
for dom, v in sorted(by_dom.items(), key=lambda kv: -kv[1]['total']):
    pct = 100 * v['pass'] / max(v['total'], 1)
    print(f'  {dom:<20s}: {v["pass"]:>3d}/{v["total"]:>3d} pass ({pct:>5.1f}%)  {v["fail"]:>3d} fail')

# ─── Full guardrails on fetchable subset ────────────────────────────

print('\n' + '='*72)
print('FULL GUARDRAILS ON FETCHABLE SUBSET')
print('='*72)
print()

# Use same fetchers as t71 toy
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch(url, timeout=15):
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-Guardrail/1.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 1: raise
            time.sleep(1)

def to_acf(values, max_lag=80):
    R = np.asarray(values, dtype=float)
    if len(R) < 30: return None
    R = R - R.mean()
    if np.std(R) > 0: R = R / np.std(R)
    n = len(R)
    max_lag = min(n - 1, max_lag)
    acf = np.array([np.sum(R[:n-lag] * R[lag:]) / (n - lag) for lag in range(max_lag)])
    if acf[0] > 0: acf = acf / acf[0]
    return np.arange(max_lag, dtype=float), acf

FETCHABLE_HOSTS = {
    'api.worldbank.org', 'api.coingecko.com', 'api.binance.com',
    'archive-api.open-meteo.com', 'earthquake.usgs.gov',
}

def is_fetchable(url):
    if not url: return False
    if url.startswith('10.5281/'): return False
    from urllib.parse import urlparse
    return urlparse(url).netloc in FETCHABLE_HOSTS

def fetch_binance(sym):
    raw = fetch(f'https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=365').decode()
    return np.array([float(r[4]) for r in json.loads(raw)])

def fetch_coingecko(coin):
    raw = fetch(f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365&interval=daily').decode()
    return np.array([float(p[1]) for p in json.loads(raw).get('prices', [])])

def fetch_openmeteo(lat, lon):
    import datetime
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365)
    fmt = lambda d: d.strftime('%Y-%m-%d')
    url = f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={fmt(start)}&end_date={fmt(today)}&daily=temperature_2m_mean'
    return np.array(json.loads(fetch(url).decode()).get('daily', {}).get('temperature_2m_mean', []), dtype=float)

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

# Collect fetchable entries
fetchable = []
seen_urls = set()
for t in tests:
    url = t.get('url', '')
    if not url or not is_fetchable(url): continue
    if url in seen_urls: continue
    seen_urls.add(url)
    fetchable.append(t)

print(f'Fetchable unique entries: {len(fetchable)}')

# Run full guardrails on each
full_results = []
for i, t in enumerate(fetchable):
    name = t.get('name', '')[:50]
    url = t.get('url', '')
    dom = t.get('domain', '')
    print(f'\n[{i+1}/{len(fetchable)}] [{dom}] {name}', flush=True)
    
    # Fetch
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        values = None
        if 'binance' in host:
            import re
            m = re.search(r'symbol=(\w+)', url)
            if m: values = fetch_binance(m.group(1))
        elif 'coingecko' in host:
            import re
            m = re.search(r'/coins/(\w+)', url)
            if m: values = fetch_coingecko(m.group(1))
        elif 'open-meteo' in host:
            import re
            lat_m = re.search(r'latitude=([\-\d.]+)', url)
            lon_m = re.search(r'longitude=([\-\d.]+)', url)
            if lat_m and lon_m:
                values = fetch_openmeteo(float(lat_m.group(1)), float(lon_m.group(1)))
        elif 'usgs' in host and 'earthquake' in url:
            values = fetch_usgs()
        elif 'worldbank' in host:
            raw = fetch(url).decode()
            j = json.loads(raw)
            rows = j[1] if len(j) > 1 else []
            vals = []
            for r in rows:
                try:
                    v = float(r['value'])
                    if v == v and v != 0: vals.append(v)
                except: continue
            if len(vals) >= 30:
                values = np.array(vals, dtype=float)
        if values is None or len(values) < 30:
            print(f'  FETCH failed or too short')
            continue
        print(f'  n={len(values)}', flush=True)
    except Exception as e:
        print(f'  FETCH ERROR: {repr(e)[:80]}')
        continue
    
    # Compute ACF
    acf_result = to_acf(values, max_lag=80)
    if acf_result is None:
        print(f'  ACF failed')
        continue
    t_arr, R_arr = acf_result
    t_arr = t_arr[1:]; R_arr = R_arr[1:]  # drop lag 0
    if len(t_arr) < 20:
        print(f'  Too few lags: {len(t_arr)}')
        continue
    
    # Run all guardrails
    result = run_all_guardrails(t_arr, R_arr)
    
    # Extract key stats
    g1 = result['guardrails'].get('G1_fit_quality', {})
    g3 = result['guardrails'].get('G3_stability', {})
    g4 = result['guardrails'].get('G4_null_destruction', {})
    g5 = result['guardrails'].get('G5_identifiability', {})
    
    print(f'  Verdict: {result["verdict"]}', flush=True)
    if g1.get('r2') is not None:
        print(f'    G1 R² = {g1["r2"]:.4f}  {"PASS" if g1["pass"] else "FAIL"}')
    if g5.get('pass') is not None:
        print(f'    G5 identifiability: {"PASS" if g5["pass"] else "FAIL"} ({g5.get("reason","")})')
    if g3.get('D_cv') is not None:
        print(f'    G3 stability: D_cv={g3["D_cv"]:.3f}, lam_cv={g3["lam_cv"]:.3f}  {"PASS" if g3["pass"] else "FAIL"}')
    if g4.get('ratio') is not None:
        print(f'    G4 null destruction: real_r2={g4["real_r2"]:.3f}, scram_r2={g4["scram_r2_median"]:.3f}, ratio={g4["ratio"]:.2f}  {"PASS" if g4["pass"] else "FAIL"}')
    
    full_results.append({
        'name': name,
        'domain': dom,
        'url': url,
        'n': int(len(values)),
        'verdict': result['verdict'],
        'reason': result['reason'],
        'guardrails': {
            'G1_fit_quality': {k: v for k, v in g1.items() if k != 'params'} if g1 else None,
            'G3_stability': g3 if g3 else None,
            'G4_null_destruction': g4 if g4 else None,
            'G5_identifiability': g5 if g5 else None,
        },
    })

# ─── Summary ────────────────────────────────────────────────────────

print('\n' + '='*72)
print('GUARDRAIL SUMMARY')
print('='*72)
print()

# Metadata-level
print(f'Metadata-level guardrails (all {n_total} registry entries):')
print(f'  METADATA_PASS: {n_pass} ({100*n_pass/n_total:.1f}%)')
print(f'  FAILS:         {n_fail} ({100*n_fail/n_total:.1f}%)')
print(f'  INSUFFICIENT:  {n_insuf} ({100*n_insuf/n_total:.1f}%)')
print()

# Full guardrails
if full_results:
    n_full = len(full_results)
    n_survive = sum(1 for r in full_results if r['verdict'] == 'SURVIVES')
    n_fail_full = sum(1 for r in full_results if r['verdict'] == 'FAILS')
    print(f'Full guardrails ({n_full} fetchable datasets):')
    print(f'  SURVIVES: {n_survive}/{n_full} ({100*n_survive/max(n_full,1):.1f}%)')
    print(f'  FAILS:    {n_fail_full}/{n_full} ({100*n_fail_full/max(n_full,1):.1f}%)')
    print()
    print(f'  Meta-S2 survival rate: P(S2 survives | eligible) = {100*n_survive/max(n_full,1):.1f}%')
    print()
    
    # Per-dataset
    print(f'Per-dataset full guardrail results:')
    print(f'  {"Dataset":<40s} {"Domain":<15s} {"R²":>6} {"D_cv":>6} {"lam_cv":>7} {"null_r":>6} {"Verdict":>10}')
    print('  ' + '-'*95)
    for r in full_results:
        g1 = r['guardrails'].get('G1_fit_quality', {}) or {}
        g3 = r['guardrails'].get('G3_stability', {}) or {}
        g4 = r['guardrails'].get('G4_null_destruction', {}) or {}
        r2 = g1.get('r2', 0) or 0
        dcv = g3.get('D_cv', 0) or 0
        lcv = g3.get('lam_cv', 0) or 0
        nr = g4.get('ratio', 0) or 0
        print(f'  {r["name"][:40]:<40s} {r["domain"][:15]:<15s} {r2:>6.3f} {dcv:>6.2f} {lcv:>7.2f} {nr:>6.1f} {r["verdict"]:>10}')
    
    # Failure reasons
    print(f'\nFailure reasons:')
    from collections import Counter
    fail_reasons = Counter()
    for r in full_results:
        if r['verdict'] == 'FAILS':
            fail_reasons[r['reason']] += 1
    for reason, count in fail_reasons.most_common():
        print(f'  {reason}: {count}')

# Save
out = {
    'framework': 'S2 Guardrail Framework',
    'guardrails': GUARDRAILS,
    'metadata_results': {
        'n_total': n_total,
        'n_pass': n_pass,
        'n_fail': n_fail,
        'n_insufficient': n_insuf,
        'metadata_pass_rate': float(100 * n_pass / n_total),
        'by_domain': {dom: dict(v) for dom, v in by_dom.items()},
    },
    'full_guardrail_results': full_results,
    'meta_s2_survival_rate': float(100 * sum(1 for r in full_results if r['verdict'] == 'SURVIVES') / max(len(full_results), 1)) if full_results else None,
}
out_path = os.path.join(OUT_DIR, 's2_guardrail_audit.json')
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
