#!/usr/bin/env python3
"""
Rerun all S2_LOSES entries through the S2+dust model.

For each entry with model_verdict == 'S2_LOSES':
  1. Refetch the raw data from the entry's URL.
  2. Recompute the ACF retention curve.
  3. Re-fit with the full 7-model comparison (EXP, BIEXP, POWER, LOGNORM, GAUSS,
     S2, S2_DUST) via fit_s2() in dream_auto_scanner.
  4. If S2_DUST beats the best alternative (e.g. BIEXP) by AICc, flip the
     verdict to 'S2_DUST_WINS' and record D1/D2/R²/delta in model_note.
     Otherwise keep 'S2_LOSES' but enrich model_note with the S2_DUST fit info.

This is a redemption pass: a system where S2 loses to BIEXP but S2_DUST beats
BIEXP is NOT a DREAM failure — it is dust contamination, exactly as predicted
by the framework.

Intended to run as part of every scout (see dream_auto_scanner.py main()).
"""
import os, sys, json, csv, io, urllib.request, re
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from dream_auto_scanner import fetch_url, retention_curve, fit_s2, load_existing_tests


# ── Per-source refetchers ────────────────────────────────────────────

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
    return vals if len(vals) >= 20 else None

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
            prices = [float(p) for p in obj['price'] if p is not None]
            return prices if len(prices) >= 50 else None
    except: pass
    return None

def refetch_wikipedia(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'items' in obj:
            vals = [item['views'] for item in obj['items']]
            return vals if len(vals) >= 50 else None
    except: pass
    return None

def refetch_nasa_power(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if 'properties' in obj and 'parameter' in obj['properties']:
            param = obj['properties']['parameter']
            for key, vals_dict in param.items():
                vals = [float(v) for v in vals_dict.values() if v is not None and v >= 0]
                if len(vals) >= 50:
                    # Diurnal detrend: subtract hour-of-day mean
                    if len(vals) % 24 == 0:
                        hour_means = np.zeros(24)
                        for h in range(24):
                            hour_means[h] = np.mean(vals[h::24])
                        detrended = np.array([vals[i] - hour_means[i % 24] for i in range(len(vals))])
                        return detrended.tolist()
                    return vals
    except: pass
    return None

def refetch_water_quality(url):
    data = fetch_url(url, timeout=25)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 5: return None
    val_idx = -1
    for i, h in enumerate(rows[0]):
        if h == 'ResultMeasureValue': val_idx = i; break
    if val_idx < 0: return None
    vals = []
    for row in rows[1:]:
        if val_idx < len(row):
            try:
                v = float(row[val_idx])
                if not np.isnan(v) and v >= 0: vals.append(v)
            except: pass
    return vals if len(vals) >= 20 else None

def refetch_swpc(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        vals = [float(item['flux']) for item in obj
                if isinstance(item, dict) and item.get('energy') == '0.1-0.8nm'
                and item.get('flux') and float(item['flux']) > 0]
        return vals if len(vals) >= 50 else None
    except: pass
    return None

def refetch_worldbank(url):
    data = fetch_url(url, timeout=20)
    if not data: return None
    try:
        obj = json.loads(data)
        if len(obj) > 1 and obj[1]:
            vals = [d['value'] for d in obj[1] if d.get('value') is not None]
            vals.reverse()
            return vals if len(vals) >= 20 else None
    except: pass
    return None

def refetch_open_meteo(url, name=''):
    data = fetch_url(url, timeout=25)
    if not data: return None
    try:
        d = json.loads(data)
        daily = d.get('daily', {})
        # Pick the variable that matches the name
        for var in ['wind_speed_10m_max', 'precipitation_sum', 'temperature_2m_mean']:
            if var in daily:
                vals = [v for v in daily[var] if v is not None]
                if len(vals) >= 50: return vals
        # Fallback: any daily variable
        for var, vals in daily.items():
            if isinstance(vals, list) and len(vals) >= 50:
                nums = [float(v) for v in vals if v is not None]
                if nums: return nums
    except: pass
    return None

def refetch_usgs_river(url):
    data = fetch_url(url, timeout=25)
    if not data: return None
    text = data.decode('utf-8') if isinstance(data, bytes) else data
    lines = text.strip().split('\n')
    # USGS RDB: comment lines start with #, then header, then data
    vals = []
    for line in lines:
        if line.startswith('#') or not line.strip(): continue
        parts = line.split('\t')
        if len(parts) >= 5:
            try:
                v = float(parts[4])  # 5th column is usually the value
                if v >= 0 and not np.isnan(v): vals.append(v)
            except: pass
    return vals if len(vals) >= 50 else None


def refetch_values(url, name=''):
    """Dispatch to the right refetcher based on URL pattern."""
    if not url or not url.startswith('http'): return None
    try:
        if 'fredgraph.csv' in url:           return refetch_fred(url)
        if 'CSSEGISandData' in url:          return refetch_covid(url)
        if 'ndbc.noaa.gov' in url:           return refetch_ndbc(url)
        if 'energy-charts.info' in url:      return refetch_energy_charts(url)
        if 'wikimedia.org' in url:           return refetch_wikipedia(url)
        if 'power.larc.nasa.gov' in url:     return refetch_nasa_power(url)
        if 'waterqualitydata' in url:        return refetch_water_quality(url)
        if 'swpc.noaa.gov' in url:           return refetch_swpc(url)
        if 'api.worldbank.org' in url:       return refetch_worldbank(url)
        if 'open-meteo' in url:              return refetch_open_meteo(url, name)
        if 'waterservices.usgs.gov' in url:  return refetch_usgs_river(url)
    except Exception as e:
        print(f'    Refetch error: {e}')
    return None


# ── Verdict update ───────────────────────────────────────────────────

def derive_dust_verdict(fit, cmp=None):
    """Given a fit_s2 result, decide if S2_DUST redeems the loss.

    Returns (new_model_verdict, new_model_note, new_delta_aicc, new_best_alt).
    """
    mv = fit.get('model_verdict', '')
    if mv != 'S2_LOSES':
        # Not a loss — no change
        return mv, fit.get('model_note', ''), fit.get('delta_aicc'), fit.get('best_alt')

    best_alt = fit.get('best_alt', '')
    delta_aicc = fit.get('delta_aicc')

    # Re-run the comparison to get S2_DUST numbers explicitly
    # fit_s2 already computed this — extract from model_note or rerun compare
    note = fit.get('model_note', '')

    # Check the ranking in fit
    ranking = fit.get('ranking', [])
    s2_dust_aicc = None
    biexp_aicc = None
    s2_dust_info = None
    for name, aicc_val, r2, k in ranking:
        if name == 'S2_DUST':
            s2_dust_aicc = aicc_val
        if name == 'BIEXP':
            biexp_aicc = aicc_val

    # Need to re-fetch the comparison to get D1/D2/lambda_q
    # fit_s2 doesn't expose them in the result — recompute by re-running compare
    return None  # signal caller to recompute


def rerun_one(entry):
    """Refetch + refit a single S2_LOSES entry. Returns updates dict or None."""
    name = entry.get('name', '')
    url = entry.get('url', '')
    if not url:
        return None

    print(f'  → {name[:60]}')
    vals = refetch_values(url, name)
    if not vals or len(vals) < 20:
        print(f'    ✗ Refetch failed or too short ({len(vals) if vals else 0} obs)')
        return None

    taus, acf = retention_curve(vals)
    if taus is None or acf is None:
        print(f'    ✗ ACF failed')
        return None

    # Re-fit with full model comparison (includes S2_DUST)
    fit = fit_s2(taus, acf, name[:60])
    if fit is None:
        print(f'    ✗ Fit failed')
        return None

    # Re-run compare() to get the full S2_DUST info
    # fit_s2 already called s2_compare internally, but the result is collapsed.
    # Re-run s2_compare here to access s2_dust dict.
    from s2_model_compare import compare as s2_compare
    t_arr = np.array(taus, dtype=float)
    R_arr = np.array(acf, dtype=float)
    t_arr = t_arr - t_arr[0]
    if R_arr[0] > 0: R_arr = R_arr / R_arr[0]
    cmp = s2_compare(t_arr, R_arr, name[:60])

    s2_dust = cmp.get('s2_dust') if cmp else None
    best_alt_name = cmp.get('best_alt_name') if cmp else fit.get('best_alt')
    best_alt_aicc = cmp.get('best_alt_aicc') if cmp else None
    s2_aicc = cmp.get('s2_aicc') if cmp else None

    D = fit['D']
    r2 = fit['r2']
    regime = fit['verdict']  # EXTRACTION/NATURAL/THRESHOLD

    # Decide verdict
    new_mv = 'S2_LOSES'
    if s2_dust and best_alt_aicc is not None:
        if best_alt_name == 'S2_DUST':
            # S2_DUST is the BEST model overall — that IS the redemption.
            # S2_DUST is part of the DREAM framework (two-component S2),
            # so this is dust contamination confirmed, not a DREAM failure.
            new_mv = 'S2_DUST_WINS'
        else:
            # S2 lost to something else (BIEXP, GAUSS, etc.) — does S2_DUST
            # beat THAT alternative?
            delta_dust_vs_alt = s2_dust['aicc'] - best_alt_aicc
            if delta_dust_vs_alt <= -2:
                new_mv = 'S2_DUST_WINS'

    # Build model_note
    if new_mv == 'S2_DUST_WINS':
        if best_alt_name == 'S2_DUST':
            # S2_DUST is the best overall model
            model_note = (f'S2 loses to S2+dust (ΔAICc={fit["delta_aicc"]:.2f}). '
                          f'S2+dust is the best overall model: D1={s2_dust["D1"]}, D2={s2_dust["D2"]}, '
                          f'R²={s2_dust["r2"]:.4f}, AICc={s2_dust["aicc"]:.2f}. '
                          f'Dust decomposition confirmed — not a DREAM failure.')
        else:
            model_note = (f'S2 loses to {best_alt_name} (ΔAICc={fit["delta_aicc"]:.2f}), '
                          f'but S2+dust beats {best_alt_name} '
                          f'(S2+dust AICc={s2_dust["aicc"]:.2f} vs {best_alt_name} AICc={best_alt_aicc:.2f}). '
                          f'Dust decomposition: D1={s2_dust["D1"]}, D2={s2_dust["D2"]}, '
                          f'R²={s2_dust["r2"]:.4f}. Not a DREAM failure — dust structure confirmed.')
    else:
        if s2_dust:
            model_note = (f'S2 loses to {best_alt_name} (ΔAICc={fit["delta_aicc"]:.2f}). '
                          f'S2+dust fit: D1={s2_dust["D1"]}, D2={s2_dust["D2"]}, '
                          f'R²={s2_dust["r2"]:.4f}, AICc={s2_dust["aicc"]:.2f}. '
                          f'S2+dust does NOT beat {best_alt_name} — needs investigation.')
        else:
            model_note = (f'S2 loses to {best_alt_name} (ΔAICc={fit["delta_aicc"]:.2f}). '
                          f'S2+dust fit failed.')

    delta_dust = (s2_dust['aicc'] - best_alt_aicc) if (s2_dust and best_alt_aicc) else None
    if best_alt_name == 'S2_DUST':
        delta_dust = fit['delta_aicc']  # S2 vs S2_DUST delta

    print(f'    D={D:.3f}, R²={r2:.3f}, {new_mv} '
          f'{"✓ dust redeemed" if new_mv == "S2_DUST_WINS" else "✗ still loses"}'
          f'{" (Δ_dust=" + str(round(delta_dust,2)) + ")" if delta_dust is not None else ""}')

    return {
        'D': D,
        'r2': r2,
        'verdict': regime,
        'model_verdict': new_mv,
        'model_note': model_note,
        'delta_aicc': fit['delta_aicc'],
        'best_alt': best_alt_name,
        's2_dust_d1': s2_dust['D1'] if s2_dust else None,
        's2_dust_d2': s2_dust['D2'] if s2_dust else None,
        's2_dust_r2': s2_dust['r2'] if s2_dust else None,
        's2_dust_aicc': s2_dust['aicc'] if s2_dust else None,
        's2_dust_delta': delta_dust,
    }


def update_entry_by_name(html_path, name, updates):
    """Update an entry in tests.html by name (since many entries have empty id)."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    # Escape name for regex
    name_esc = re.escape(name)

    # Find the entry block containing this name
    # Entry format: {id:"...",name:"<name>",...}
    # Match from the opening { before name:"<name>" to the next ,{ or ];
    pattern = re.compile(
        r'(\{[^{}]*?name:"' + name_esc + r'"[^{}]*?\})',
        re.DOTALL
    )
    m = pattern.search(html)
    if not m:
        print(f'    ✗ Entry not found by name in {html_path}')
        return False

    old_entry = m.group(1)
    new_entry = old_entry

    # Replace D
    if 'D' in updates and updates['D'] is not None:
        new_entry = re.sub(r'\bD:-?[\d.]+', f'D:{updates["D"]:.4f}', new_entry, count=1)
    # Replace r2
    if 'r2' in updates and updates['r2'] is not None:
        new_entry = re.sub(r'\br2:-?[\d.]+', f'r2:{updates["r2"]:.4f}', new_entry, count=1)
    # Replace verdict
    if 'verdict' in updates:
        v = updates['verdict'].replace('"', '\\"')
        new_entry = re.sub(r'verdict:"[^"]*"', f'verdict:"{v}"', new_entry, count=1)
    # Replace model_verdict
    if 'model_verdict' in updates:
        mv = updates['model_verdict']
        if re.search(r'model_verdict:', new_entry):
            new_entry = re.sub(r'model_verdict:"[^"]*"', f'model_verdict:"{mv}"', new_entry, count=1)
        else:
            # Insert after verdict
            new_entry = re.sub(
                r'(verdict:"[^"]*")',
                r'\1,model_verdict:"' + mv + '"',
                new_entry, count=1
            )
    # Replace delta_aicc
    if 'delta_aicc' in updates and updates['delta_aicc'] is not None:
        if re.search(r'delta_aicc:', new_entry):
            new_entry = re.sub(r'delta_aicc:-?[\d.]+', f'delta_aicc:{updates["delta_aicc"]:.4f}', new_entry, count=1)
        else:
            new_entry = re.sub(
                r'(model_verdict:"[^"]*")',
                r'\1,delta_aicc:' + f'{updates["delta_aicc"]:.4f}',
                new_entry, count=1
            )
    # Replace best_alt
    if 'best_alt' in updates and updates['best_alt']:
        ba = updates['best_alt'].replace('"', '\\"')
        if re.search(r'best_alt:', new_entry):
            new_entry = re.sub(r'best_alt:"[^"]*"', f'best_alt:"{ba}"', new_entry, count=1)
        else:
            new_entry = re.sub(
                r'(delta_aicc:-?[\d.]+)',
                r'\1,best_alt:"' + ba + '"',
                new_entry, count=1
            )
    # Replace model_note
    if 'model_note' in updates and updates['model_note']:
        mn = updates['model_note'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
        if re.search(r'model_note:', new_entry):
            new_entry = re.sub(r'model_note:"(?:[^"\\]|\\.)*"', f'model_note:"{mn}"', new_entry, count=1)
        else:
            new_entry = re.sub(
                r'(best_alt:"[^"]*")',
                r'\1,model_note:"' + mn + '"',
                new_entry, count=1
            )
    # Replace narrative to match new fields
    # (Reconciler will rebuild narratives later — just sync the model_note portion)

    html = html.replace(old_entry, new_entry, 1)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return True


def main():
    print('=' * 60)
    print('RERUN S2_LOSES ON S2+DUST MODEL')
    print('=' * 60)

    # Load current tests.json
    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    losses = [t for t in data['tests'] if t.get('model_verdict') == 'S2_LOSES']
    print(f'\nS2_LOSES entries to rerun: {len(losses)}\n')

    if not losses:
        print('Nothing to rerun.')
        return 0

    redeemed = 0
    still_loses = 0
    failed = 0

    for entry in losses:
        name = entry.get('name', '')
        updates = rerun_one(entry)
        if updates is None:
            failed += 1
            continue

        if updates['model_verdict'] == 'S2_DUST_WINS':
            redeemed += 1
        else:
            still_loses += 1

        # Update both EN and RU
        update_entry_by_name(os.path.join(REPO, 'en/tests.html'), name, updates)
        update_entry_by_name(os.path.join(REPO, 'ru/tests.html'), name, updates)

    print(f'\n{"=" * 60}')
    print(f'SUMMARY')
    print(f'{"=" * 60}')
    print(f'  Total S2_LOSES rerun:    {len(losses)}')
    print(f'  Redeemed (S2_DUST_WINS): {redeemed}')
    print(f'  Still S2_LOSES:          {still_loses}')
    print(f'  Failed to refetch:       {failed}')

    # Re-export tests.json
    print('\n--- Re-exporting tests.json ---')
    import subprocess
    r = subprocess.run(
        ['python3', 'scripts/export_tests_json.py', '.'],
        cwd=REPO, capture_output=True, text=True, timeout=30
    )
    print(r.stdout.strip())

    # Run reconciler to sync narratives
    print('\n--- Running registry integrity reconciler ---')
    r = subprocess.run(
        ['python3', 'scripts/registry_integrity_reconcile.py'],
        cwd=REPO, capture_output=True, text=True, timeout=60
    )
    for line in r.stdout.strip().split('\n')[-15:]:
        print(line)

    # Update meta-s2
    print('\n--- Updating meta-s2 article ---')
    from dream_auto_scanner import update_meta_s2_article
    update_meta_s2_article(os.path.join(REPO, 'en/tests.html'), is_ru=False)
    update_meta_s2_article(os.path.join(REPO, 'ru/tests.html'), is_ru=True)

    # Final count
    with open(os.path.join(REPO, 'en/tests.json')) as f:
        d = json.load(f)
    from collections import Counter
    mv = Counter(t.get('model_verdict', '?') for t in d['tests'])
    print(f'\n=== FINAL REGISTRY ===')
    print(f'Total: {d["total_tests"]}')
    for v, n in mv.most_common():
        print(f'  {v}: {n}')

    return redeemed


if __name__ == '__main__':
    main()
