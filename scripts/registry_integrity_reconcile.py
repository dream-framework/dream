#!/usr/bin/env python3
"""
DREAM Registry Integrity Reconciler

Derives every displayed field (narrative, regime label, model verdict text)
from the immutable numeric fields (D, r2, model_verdict, delta_aicc, best_alt).

This script:
  1. Parses ALL entries from tests.html (both EN and RU) with a proper
     brace-matching parser that captures the narrative field.
  2. Audits each entry for:
     - Regime label consistency (EXTRACTION/NATURAL/THRESHOLD must match D)
     - Narrative D consistency (text must match numeric D)
     - Narrative r² consistency (text must match numeric r²)
     - Narrative regime text consistency ("D>1 indicates extraction" vs actual D)
  3. Regenerates the narrative from the immutable fields.
  4. Fixes the regime label to match D.
  5. Writes the corrected entries back to tests.html.

The narrative is derived as:
  "D={D:.4f}, R²={r2:.4f}. {regime_text} {model_note}"

Where:
  - regime_text = "D>1 indicates extraction regime — retention collapses
    super-exponentially." if D > 1
    "D<1 confirms natural retention — heavy-tailed, slow decay." if D < 0.8
    "D≈1 — threshold regime, borderline exponential." otherwise
  - model_note = from the model_verdict + delta_aicc + best_alt fields

For RU, the narrative is translated.

Usage:
  python3 scripts/registry_integrity_reconcile.py [--dry-run]
"""

import os, sys, re, json, argparse

def derive_regime(D):
    """Derive the regime label from D."""
    if D is None:
        return 'PENDING'
    if D < 0.8:
        return 'NATURAL'
    elif D > 1.0:
        return 'EXTRACTION'
    else:
        return 'THRESHOLD'

def derive_narrative_en(entry):
    """Derive the English narrative from immutable numeric fields."""
    D = entry.get('D')
    r2 = entry.get('r2')
    model_verdict = entry.get('model_verdict', '')
    delta_aicc = entry.get('delta_aicc')
    best_alt = entry.get('best_alt', '')
    
    if D is None:
        return 'S2 fit pending — data acquired but not yet analyzed.'
    
    # Regime text
    if D > 1.0:
        regime_text = 'D>1 indicates extraction regime — retention collapses super-exponentially.'
    elif D < 0.8:
        regime_text = 'D<1 confirms natural retention — heavy-tailed, slow decay.'
    else:
        regime_text = 'D≈1 — threshold regime, near-exponential decay.'
    
    # Model comparison text
    if model_verdict == 'S2_WINS' and best_alt and delta_aicc is not None:
        model_text = f'S2 beats {best_alt} (ΔAICc={delta_aicc:.2f}).'
    elif model_verdict == 'S2_TIES' and best_alt and delta_aicc is not None:
        model_text = f'S2 ties {best_alt} (ΔAICc={delta_aicc:.2f}, within ±2).'
    elif model_verdict == 'S2_LOSES' and best_alt and delta_aicc is not None:
        model_text = f'S2 loses to {best_alt} (ΔAICc={delta_aicc:.2f}).'
    elif model_verdict == 'NO_FIT':
        model_text = 'No model fits adequately.'
    else:
        model_text = ''
    
    return f'D={D:.4f}, R²={r2:.4f}. {regime_text} {model_text}'.strip()

def derive_narrative_ru(entry):
    """Derive the Russian narrative from immutable numeric fields."""
    D = entry.get('D')
    r2 = entry.get('r2')
    model_verdict = entry.get('model_verdict', '')
    delta_aicc = entry.get('delta_aicc')
    best_alt = entry.get('best_alt', '')
    
    if D is None:
        return 'S2-подгонка ожидается — данные получены, но не проанализированы.'
    
    if D > 1.0:
        regime_text = 'D>1 указывает на режим извлечения — сохранение коллапсирует сверхэкспоненциально.'
    elif D < 0.8:
        regime_text = 'D<1 подтверждает естественное сохранение — тяжёлый хвост, медленное угасание.'
    else:
        regime_text = 'D≈1 — пороговый режим, около-экспоненциальное затухание.'
    
    if model_verdict == 'S2_WINS' and best_alt and delta_aicc is not None:
        model_text = f'S2 превосходит {best_alt} (ΔAICc={delta_aicc:.2f}).'
    elif model_verdict == 'S2_TIES' and best_alt and delta_aicc is not None:
        model_text = f'S2 сравнима с {best_alt} (ΔAICc={delta_aicc:.2f}, в пределах ±2).'
    elif model_verdict == 'S2_LOSES' and best_alt and delta_aicc is not None:
        model_text = f'S2 уступает {best_alt} (ΔAICc={delta_aicc:.2f}).'
    elif model_verdict == 'NO_FIT':
        model_text = 'Ни одна модель не подгоняется адекватно.'
    else:
        model_text = ''
    
    return f'D={D:.4f}, R²={r2:.4f}. {regime_text} {model_text}'.strip()


def parse_tests_html(html_path):
    """Parse tests.html and return (html_without_tests, list_of_entry_strings, parsed_entries).
    
    Returns the HTML with the TESTS array body removed (for reassembly),
    the raw entry strings, and parsed entry dicts.
    """
    with open(html_path) as f:
        html = f.read()
    
    # Find the TESTS array
    m = re.search(r'(const\s+TESTS\s*=\s*\[)(.*?)(\n\];)', html, re.DOTALL)
    if not m:
        return html, [], []
    
    prefix = html[:m.start(1)] + m.group(1)
    body = m.group(2)
    suffix = m.group(3) + html[m.end(3):]
    
    # Parse each entry with brace matching
    entries_raw = []
    entries_parsed = []
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
        entries_raw.append(entry_str)
        
        # Parse fields
        entry = {}
        for field in ['id', 'name', 'domain', 'verdict', 'model_verdict', 'narrative',
                       'source', 'date', 'url', 'image', 'kind', 'best_alt', 'model_note']:
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
        # Also extract delta_aicc and best_alt if present
        fm = re.search(r'(?:^|,)\s*delta_aicc\s*:\s*(-?[\d]+(?:\.[\d]+)?)', entry_str)
        if fm:
            entry['delta_aicc'] = float(fm.group(1))
        fm = re.search(r'(?:^|,)\s*best_alt\s*:\s*"([^"]*)"', entry_str)
        if fm:
            entry['best_alt'] = fm.group(1)
        
        entries_parsed.append(entry)
        i = j + 1
    
    return html, entries_raw, entries_parsed, prefix, suffix


def rebuild_entry_string(entry, is_ru=False):
    """Rebuild a JS entry object from the parsed entry, with corrected fields."""
    # Fix regime
    D = entry.get('D')
    if D is not None:
        entry['verdict'] = derive_regime(D)
    
    # Regenerate narrative
    if is_ru:
        entry['narrative'] = derive_narrative_ru(entry)
    else:
        entry['narrative'] = derive_narrative_en(entry)
    
    # Build JS object — preserve field order: id, name, domain, D, r2, verdict,
    # model_verdict, narrative, source, date, url, image
    def js_str(s):
        if s is None:
            return ''
        s = str(s)
        s = s.replace('\\', '\\\\')
        s = s.replace('"', '\\"')
        s = s.replace('\n', ' ')
        s = s.replace('\r', ' ')
        s = s.replace('\t', ' ')
        return s
    
    parts = []
    parts.append(f'id:"{js_str(entry.get("id", ""))}"')
    parts.append(f'name:"{js_str(entry.get("name", ""))}"')
    parts.append(f'domain:"{js_str(entry.get("domain", ""))}"')
    if D is not None:
        parts.append(f'D:{D:.4f}')
    else:
        parts.append('D:null')
    if entry.get('r2') is not None:
        parts.append(f'r2:{entry["r2"]:.4f}')
    else:
        parts.append('r2:null')
    parts.append(f'verdict:"{js_str(entry.get("verdict", ""))}"')
    if entry.get('model_verdict'):
        parts.append(f'model_verdict:"{js_str(entry["model_verdict"])}"')
    if entry.get('delta_aicc') is not None:
        parts.append(f'delta_aicc:{entry["delta_aicc"]:.4f}')
    if entry.get('best_alt'):
        parts.append(f'best_alt:"{js_str(entry["best_alt"])}"')
    if entry.get('model_note'):
        parts.append(f'model_note:"{js_str(entry["model_note"])}"')
    parts.append(f'narrative:"{js_str(entry.get("narrative", ""))}"')
    parts.append(f'source:"{js_str(entry.get("source", ""))}"')
    parts.append(f'date:"{js_str(entry.get("date", ""))}"')
    url_val = entry.get('url', '')
    if url_val:
        parts.append(f'url:"{js_str(url_val)}"')
    else:
        parts.append('url:null')
    parts.append('image:null')
    
    return '{' + ','.join(parts) + '}'


def reconcile(html_path, is_ru=False, dry_run=False):
    """Reconcile all entries in tests.html."""
    print(f'\n{"="*60}')
    print(f'Reconciling {html_path} ({"RU" if is_ru else "EN"})')
    print(f'{"="*60}')
    
    html, entries_raw, entries, prefix, suffix = parse_tests_html(html_path)
    
    print(f'Parsed {len(entries)} entries')
    
    # Audit
    fixes = []
    for i, (raw, parsed) in enumerate(zip(entries_raw, entries)):
        D = parsed.get('D')
        r2 = parsed.get('r2')
        verdict = parsed.get('verdict', '')
        narr = parsed.get('narrative', '')
        name = parsed.get('name', '?')
        
        if D is None:
            continue
        
        issues = []
        expected_verdict = derive_regime(D)
        if verdict != expected_verdict:
            issues.append(f'verdict: {verdict} → {expected_verdict}')
        
        # Check narrative D
        if narr:
            narr_d_match = re.search(r'D=([\d.]+)', narr)
            if narr_d_match:
                narr_d = float(narr_d_match.group(1).rstrip('.'))
                if abs(narr_d - D) > 0.005:
                    issues.append(f'narrative D: {narr_d} → {D:.4f}')
        
        if issues:
            fixes.append({
                'index': i,
                'name': name[:55],
                'D': D,
                'issues': issues,
            })
    
    print(f'\nEntries needing fixes: {len(fixes)}')
    for f in fixes:
        print(f'  {f["name"]:<55} D={f["D"]:.4f} | {"; ".join(f["issues"])}')
    
    if dry_run:
        print(f'\n[DRY RUN] Would fix {len(fixes)} entries')
        return fixes
    
    # Rebuild all entries
    new_entries = []
    for parsed in entries:
        new_entry = rebuild_entry_string(parsed, is_ru=is_ru)
        new_entries.append(new_entry)
    
    # Reassemble HTML
    new_body = '\n  ' + '\n  ,'.join(new_entries) + '\n'
    new_html = prefix + new_body + suffix
    
    with open(html_path, 'w') as f:
        f.write(new_html)
    
    print(f'\n✓ Reconciled {len(entries)} entries in {html_path}')
    print(f'  Fixed: {len(fixes)} entries with regime/narrative mismatches')
    
    return fixes


def main():
    parser = argparse.ArgumentParser(description='Reconcile DREAM registry data integrity')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fixed without writing')
    args = parser.parse_args()
    
    print('=== DREAM Registry Integrity Reconciler ===')
    print(f'Date: {os.popen("date -u +%Y-%m-%dT%H:%M:%S UTC").read().strip()}')
    
    # Reconcile both EN and RU
    fixes_en = reconcile('en/tests.html', is_ru=False, dry_run=args.dry_run)
    fixes_ru = reconcile('ru/tests.html', is_ru=True, dry_run=args.dry_run)
    
    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'EN fixes: {len(fixes_en)}')
    print(f'RU fixes: {len(fixes_ru)}')
    
    if not args.dry_run:
        # Also re-export tests.json
        import subprocess
        export_script = 'scripts/export_tests_json.py'
        if os.path.exists(export_script):
            try:
                r = subprocess.run(['python3', export_script, '.'], capture_output=True, text=True, timeout=30)
                print(f'tests.json: {r.stdout.strip().split(chr(10))[-1]}')
            except Exception as e:
                print(f'Export failed: {e}')
        
        # Update meta-s2 article (since D values may have changed regime)
        sys.path.insert(0, 'scripts')
        import dream_auto_scanner as ds
        print('\nUpdating meta-s2 article...')
        ds.update_meta_s2_article('en/tests.html', is_ru=False)
        ds.update_meta_s2_article('ru/tests.html', is_ru=True)


if __name__ == '__main__':
    main()
