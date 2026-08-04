#!/usr/bin/env python3
"""
Recover the 21 entries lost when 41e1bca was force-pushed over 8e842c4.

Background:
  - 11:00 UTC: CI auto-scan committed 8e842c4 with 126 entries (added 21 new)
  - 12:08 UTC: Local commit 41e1bca (rename REJECTED→DUST-DOMINATED) was pushed
    WITHOUT first pulling 8e842c4. The rebase was aborted. Force-push overwrote
    the CI's 21 new entries.
  - This script cherry-picks those 21 entries from 8e842c4 and re-appends them
    to the current tests.html (EN + RU), then re-exports tests.json.

The "DUST-DOMINATED" label is rendered client-side from model_verdict==="S2_LOSES",
so we just need to restore the data — no rename logic needed.
"""
import os, sys, json, subprocess, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI_SHA = '8e842c4'

def get_ci_tests():
    """Get the 21 entries that were in 8e842c4 but not in current tests.json."""
    out = subprocess.check_output(
        ['git', 'show', f'{CI_SHA}:en/tests.json'],
        cwd=REPO
    ).decode('utf-8')
    ci_tests = json.loads(out)['tests']

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        cur_tests = json.load(f)['tests']

    cur_names = set(t['name'] for t in cur_tests)
    lost = [t for t in ci_tests if t['name'] not in cur_names]
    print(f'CI commit {CI_SHA}: {len(ci_tests)} tests')
    print(f'Current: {len(cur_tests)} tests')
    print(f'Lost entries to recover: {len(lost)}')
    return lost

def entry_to_js(entry):
    """Convert a JSON entry dict to a JS object literal string matching tests.html format."""
    def js_str(s):
        if s is None: return ''
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
    D = entry.get('D')
    parts.append(f'D:{D:.4f}' if D is not None else 'D:null')
    r2 = entry.get('r2')
    parts.append(f'r2:{r2:.4f}' if r2 is not None else 'r2:null')
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

def append_to_tests_html(html_path, entries, is_ru=False):
    """Append entries to the TESTS array in tests.html."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    # Find the closing of the TESTS array
    m = re.search(r'(const\s+TESTS\s*=\s*\[.*?)(\n\];)', html, re.DOTALL)
    if not m:
        print(f'  ✗ TESTS array not found in {html_path}')
        return False

    prefix = html[:m.start(1)] + m.group(1)
    suffix = m.group(2) + html[m.end(2):]

    # Build new entries
    new_entry_strings = [entry_to_js(e) for e in entries]
    new_body = '\n  ,' + '\n  ,'.join(new_entry_strings) + '\n'

    new_html = prefix + new_body + suffix
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(new_html)
    print(f'  ✓ Appended {len(entries)} entries to {html_path}')
    return True

def main():
    print('=== RECOVERY: Restoring 21 entries lost in 41e1bca force-push ===\n')
    lost = get_ci_tests()
    if not lost:
        print('Nothing to recover.')
        return

    print('\nEntries to recover:')
    for e in lost:
        mv = e.get('model_verdict', '?')
        marker = '✓' if mv == 'S2_WINS' else ('~' if mv == 'S2_TIES' else '!')
        print(f'  {marker} {e["name"][:65]:<65} D={e.get("D", 0):.3f} {mv}')

    # Append to EN tests.html
    print('\n--- Appending to EN tests.html ---')
    append_to_tests_html(os.path.join(REPO, 'en/tests.html'), lost, is_ru=False)

    # Append to RU tests.html (same entries — names are kept in English for now)
    print('\n--- Appending to RU tests.html ---')
    append_to_tests_html(os.path.join(REPO, 'ru/tests.html'), lost, is_ru=True)

    # Re-export tests.json
    print('\n--- Re-exporting tests.json ---')
    r = subprocess.run(
        ['python3', 'scripts/export_tests_json.py', '.'],
        cwd=REPO, capture_output=True, text=True, timeout=30
    )
    print(r.stdout.strip())

    # Run reconciler to ensure narrative consistency
    print('\n--- Running registry integrity reconciler ---')
    r = subprocess.run(
        ['python3', 'scripts/registry_integrity_reconcile.py'],
        cwd=REPO, capture_output=True, text=True, timeout=60
    )
    # Print last 20 lines
    lines = r.stdout.strip().split('\n')
    for line in lines[-20:]:
        print(line)

    # Update meta-s2 article
    print('\n--- Updating meta-s2 article ---')
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import dream_auto_scanner as ds
    ds.update_meta_s2_article(os.path.join(REPO, 'en/tests.html'), is_ru=False)
    ds.update_meta_s2_article(os.path.join(REPO, 'ru/tests.html'), is_ru=True)

    # Final count
    print('\n=== FINAL STATE ===')
    with open(os.path.join(REPO, 'en/tests.json')) as f:
        d = json.load(f)
    print(f'Total tests: {d["total_tests"]}')

    from collections import Counter
    mv = Counter(t.get('model_verdict', '?') for t in d['tests'])
    print(f'By model_verdict: {dict(mv)}')

if __name__ == '__main__':
    main()
