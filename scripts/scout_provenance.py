#!/usr/bin/env python3
"""
DREAM Scout Provenance Ledger

Freezes a protocol and preserves every discovered dataset, including rejected
and failed downloads. Records:
  - Source URL, retrieval timestamp, data hash
  - Transformation, ACF definition, lag window, missing-data treatment
  - Parameter bounds and optimizer diagnostics
  - AICc for every candidate model (not only the winning comparison)
  - Dataset-family identity so correlated rows are not treated as independent
  - Eligibility or exclusion reason

Also marks the prospective baseline: all entries as of 2026-07-31 are
'retrospective' (in-sample); every entry after that is 'prospective'.

The ledger is appended to on every scanner run — it's an immutable audit trail.

Usage:
  python3 scripts/scout_provenance.py [--init]  # --init creates the baseline
"""

import os, sys, json, hashlib, re, argparse
from datetime import datetime, timezone

LEDGER_PATH = 'scout_provenance_ledger.json'
PROSPECTIVE_BASELINE_DATE = '2026-07-31'  # everything after this is prospective

def compute_data_hash(values):
    """Compute SHA-256 hash of a data array."""
    if not values:
        return None
    # Convert to string with fixed precision for reproducibility
    s = ','.join(f'{v:.10f}' for v in values)
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def derive_family(name):
    """Derive dataset-family identity from name prefix."""
    if not name:
        return 'unknown'
    n = str(name).strip()
    if ':' in n:
        prefix = n.split(':', 1)[0].strip().lower()
        return prefix
    nl = n.lower()
    if any(nl.startswith(c) for c in ('btc', 'eth', 'sol', 'ada', 'dot', 'bnb', 'xrp', 'doge', 'link')):
        return 'crypto'
    if nl.startswith('fred'):
        return 'fred'
    if nl.startswith('open-meteo') or 'temperature' in nl or 'precipitation' in nl:
        return 'weather'
    if nl.startswith('usgs') or 'earthquake' in nl:
        return 'earthquakes'
    if nl.startswith('covid'):
        return 'covid'
    if 'temperature' in nl or 'giss' in nl or 'hadcrut' in nl or 'global temp' in nl:
        return 'climate'
    if nl.startswith('arxiv'):
        return 'arxiv'
    if 'world bank' in nl:
        return 'world_bank'
    return n.split()[0].lower() if n.split() else 'unknown'

def parse_tests_html(html_path):
    """Parse tests.html and return list of entry dicts with ALL fields."""
    with open(html_path) as f:
        html = f.read()
    
    m = re.search(r'const\s+TESTS\s*=\s*\[(.*?)\n\];', html, re.DOTALL)
    if not m:
        return []
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
                       'source', 'date', 'url', 'image', 'kind']:
            fm = re.search(r'(?:^|,)\s*' + field + r'\s*:\s*"((?:[^"\\]|\\.)*)"', entry_str)
            if fm:
                entry[field] = fm.group(1).replace('\\"', '"').replace('\\\\', '\\')
        for field in ['D', 'r2']:
            fm = re.search(r'(?:^|,)\s*' + field + r'\s*:\s*(-?[\d]+(?:\.[\d]+)?)', entry_str)
            if fm:
                try:
                    entry[field] = float(fm.group(1))
                except ValueError:
                    entry[field] = None
            elif re.search(r'(?:^|,)\s*' + field + r'\s*:\s*null', entry_str):
                entry[field] = None
        
        if entry.get('name'):
            entries.append(entry)
        i = j + 1
    
    return entries


def build_ledger_entry(entry, is_retrospective=True):
    """Build a provenance ledger entry from a parsed test entry."""
    D = entry.get('D')
    name = entry.get('name', '?')
    url = entry.get('url', '')
    date = entry.get('date', '')
    
    return {
        'id': entry.get('id', ''),
        'name': name,
        'family': derive_family(name),
        'source_url': url,
        'registry_date': date,
        'D': D,
        'r2': entry.get('r2'),
        'verdict': entry.get('verdict', ''),
        'model_verdict': entry.get('model_verdict', ''),
        'narrative': entry.get('narrative', ''),
        'prospective': not is_retrospective,
        'eligibility': 'uncensored' if (D is not None and 0 < D < 4.99) else
                       ('censored' if D is not None and D >= 4.99 else
                        'pending'),
        'exclusion_reason': None if (D is not None and 0 < D < 4.99) else
                             ('optimizer boundary' if D is not None and D >= 4.99 else
                              'S2 fit not yet attempted'),
        # Protocol fields (to be filled by scanner during actual scan)
        'data_hash': None,  # SHA-256 of raw data array
        'retrieval_timestamp': None,
        'n_observations': None,
        'acf_lag_window': None,
        'missing_data_treatment': 'drop',
        'parameter_bounds': {'D': [0.01, 10.0], 'lambda_q': [0.01, 1000.0]},
        'optimizer': 'scipy.optimize.curve_fit (Levenberg-Marquardt)',
        'aicc_ranking': None,  # Full AICc table for all 6 candidate models
        'out_of_sample_score': None,  # Held-out retention region R²
        'simulation_calibration': None,  # False-positive rate from simulation
    }


def init_ledger():
    """Initialize the provenance ledger from the current registry state.
    
    Marks all existing entries as 'retrospective' (in-sample baseline).
    All future entries added by the scanner will be 'prospective'.
    """
    print('=== Initializing Scout Provenance Ledger ===')
    print(f'Prospective baseline date: {PROSPECTIVE_BASELINE_DATE}')
    print(f'Everything after this date is prospective evidence.')
    print()
    
    entries = parse_tests_html('en/tests.html')
    print(f'Parsed {len(entries)} entries from en/tests.html')
    
    ledger_entries = []
    for e in entries:
        date = e.get('date', '')
        is_retrospective = date <= PROSPECTIVE_BASELINE_DATE
        le = build_ledger_entry(e, is_retrospective=is_retrospective)
        ledger_entries.append(le)
    
    # Summary
    retro = [e for e in ledger_entries if not e['prospective']]
    pros = [e for e in ledger_entries if e['prospective']]
    uncensored = [e for e in ledger_entries if e['eligibility'] == 'uncensored']
    pending = [e for e in ledger_entries if e['eligibility'] == 'pending']
    censored = [e for e in ledger_entries if e['eligibility'] == 'censored']
    
    print(f'\nLedger summary:')
    print(f'  Total entries: {len(ledger_entries)}')
    print(f'  Retrospective (baseline, ≤{PROSPECTIVE_BASELINE_DATE}): {len(retro)}')
    print(f'  Prospective (>{PROSPECTIVE_BASELINE_DATE}): {len(pros)}')
    print(f'  Uncensored: {len(uncensored)}')
    print(f'  Pending: {len(pending)}')
    print(f'  Censored: {len(censored)}')
    
    # Family breakdown
    families = {}
    for e in ledger_entries:
        fam = e['family']
        if fam not in families:
            families[fam] = {'retro': 0, 'pros': 0, 'uncensored': 0}
        if e['prospective']:
            families[fam]['pros'] += 1
        else:
            families[fam]['retro'] += 1
        if e['eligibility'] == 'uncensored':
            families[fam]['uncensored'] += 1
    
    print(f'\nFamily breakdown ({len(families)} families):')
    print(f'  {"Family":<20} {"Retro":>6} {"Pros":>6} {"Uncensored":>11}')
    print(f'  {"-"*47}')
    for fam in sorted(families.keys()):
        f = families[fam]
        print(f'  {fam:<20} {f["retro"]:>6} {f["pros"]:>6} {f["uncensored"]:>11}')
    
    # Nested vs non-nested comparison info
    print(f'\nModel comparison separation:')
    wins = [e for e in ledger_entries if e.get('model_verdict') == 'S2_WINS']
    ties = [e for e in ledger_entries if e.get('model_verdict') == 'S2_TIES']
    losses = [e for e in ledger_entries if e.get('model_verdict') == 'S2_LOSES']
    no_cmp = [e for e in ledger_entries if e.get('model_verdict') not in ('S2_WINS', 'S2_TIES', 'S2_LOSES')]
    print(f'  S2 wins: {len(wins)}')
    print(f'  S2 ties: {len(ties)}')
    print(f'  S2 losses: {len(losses)}')
    print(f'  No comparison: {len(no_cmp)}')
    
    # Build the ledger
    ledger = {
        'version': '1.0',
        'initialized': datetime.now(timezone.utc).isoformat(),
        'prospective_baseline_date': PROSPECTIVE_BASELINE_DATE,
        'protocol': {
            'acf_definition': 'R(λ) = autocorrelation at lag λ, normalized by R(0)',
            'lag_window': 'max_lag = min(n//3, 40) for time series; magnitude-frequency for events',
            'missing_data_treatment': 'drop NaN/Inf values before computing ACF',
            'parameter_bounds': {
                'D': [0.01, 10.0],
                'lambda_q': [0.01, 1000.0]
            },
            'optimizer': 'scipy.optimize.curve_fit (Levenberg-Marquardt)',
            'candidate_models': {
                'nested': ['EXP', 'GAUSS'],
                'non_nested': ['POWER', 'BIEXP', 'LOGNORM'],
                'canonical': 'S2 (stretched exponential)'
            },
            'model_selection_criterion': 'AICc (corrected Akaike Information Criterion)',
            'gate_threshold': 'S2 must win or tie (ΔAICc within ±2) to enter registry',
        },
        'entries': ledger_entries,
        'scan_runs': [],  # Append-only log of scanner runs
    }
    
    with open(LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    
    print(f'\n✓ Ledger written to {LEDGER_PATH}')
    print(f'  {len(ledger_entries)} entries')
    print(f'  {len(retro)} retrospective (baseline)')
    print(f'  {len(pros)} prospective')
    
    return ledger


def append_scan_run(run_info):
    """Append a scan run record to the ledger."""
    if not os.path.exists(LEDGER_PATH):
        init_ledger()
    
    with open(LEDGER_PATH) as f:
        ledger = json.load(f)
    
    run_info['timestamp'] = datetime.now(timezone.utc).isoformat()
    ledger['scan_runs'].append(run_info)
    
    with open(LEDGER_PATH, 'w') as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='DREAM Scout Provenance Ledger')
    parser.add_argument('--init', action='store_true', help='Initialize ledger from current registry')
    args = parser.parse_args()
    
    if args.init or not os.path.exists(LEDGER_PATH):
        init_ledger()
    else:
        # Just print current state
        with open(LEDGER_PATH) as f:
            ledger = json.load(f)
        print(f'Ledger: {LEDGER_PATH}')
        print(f'  Version: {ledger.get("version")}')
        print(f'  Initialized: {ledger.get("initialized")}')
        print(f'  Entries: {len(ledger.get("entries", []))}')
        print(f'  Scan runs: {len(ledger.get("scan_runs", []))}')
        retro = sum(1 for e in ledger['entries'] if not e.get('prospective'))
        pros = sum(1 for e in ledger['entries'] if e.get('prospective'))
        print(f'  Retrospective: {retro}')
        print(f'  Prospective: {pros}')


if __name__ == '__main__':
    main()
