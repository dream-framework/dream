#!/usr/bin/env python3
"""
Remove S2_NO_FIT entries that were recorded for the wrong reasons.

User policy: only record S2_NO_FIT when ALL of the following are true:
  1. Data was successfully fetched (n >= 50 valid observations)
  2. ACF was computed (not None — series has variance)
  3. S2 curve_fit optimizer failed to converge

Fetch failures (URL unreachable, unsupported format) and low-n datasets
should NOT be shown in the registry. They are internal retry candidates
for the next scout, not evidence of S2 failure.

This script removes any S2_NO_FIT entry with rejection_reason in:
  - fetch_failed
  - insufficient_rows / insufficient_values / insufficient_lags (n < 50)
  - no_numeric_column (parse failure, not S2 failure)
  - csv_parse_error
  - acf_failed with n=0 (effectively fetch failure)

Keep only:
  - no_fit (curve_fit didn't converge on real data)
  - acf_failed with n >= 50 (real data, but ACF computation failed)
"""
import os, sys, json, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reasons that should NOT be shown in the registry
HIDDEN_REASONS = {
    'fetch_failed',
    'insufficient_rows',
    'insufficient_values',
    'insufficient_lags',
    'no_numeric_column',
    'csv_parse_error',
}

def remove_hidden_rejected(html_path):
    """Remove S2_NO_FIT entries with hidden rejection reasons from tests.html."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    # Find all S2_NO_FIT entries
    pattern = re.compile(
        r'(\{[^{}]*?model_verdict:"S2_NO_FIT"[^{}]*?\})',
        re.DOTALL
    )

    removed = 0
    kept = 0
    def filter_one(m):
        nonlocal removed, kept
        entry = m.group(1)
        # Extract rejection_reason
        rm = re.search(r'rejection_reason:"((?:[^"\\]|\\.)*)"', entry)
        reason = rm.group(1).replace('\\"', '"').replace('\\\\', '\\') if rm else ''
        # Also check n if present
        nm = re.search(r'\bn:(\d+)', entry)
        n_val = int(nm.group(1)) if nm else 0

        # For acf_failed, check if n was actually large enough
        if reason == 'acf_failed' and n_val >= 50:
            kept += 1
            return entry  # keep — real data, ACF failed
        if reason == 'no_fit':
            kept += 1
            return entry  # keep — real data, optimizer failed
        if reason in HIDDEN_REASONS:
            removed += 1
            return ''  # remove
        if reason.startswith('insufficient'):
            removed += 1
            return ''
        if reason.startswith('no_fit'):
            kept += 1
            return entry  # keep — real optimizer failure
        # Unknown reason — keep for safety
        kept += 1
        return entry

    new_html = pattern.sub(filter_one, html)
    # Clean up any double commas left by removals
    new_html = re.sub(r',\s*,', ',', new_html)
    new_html = re.sub(r'\[\s*,', '[', new_html)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(new_html)
    return removed, kept

def main():
    print('=' * 60)
    print('REMOVING HIDDEN S2_NO_FIT ENTRIES')
    print('(fetch_failed, insufficient_n, parse_errors)')
    print('=' * 60)

    en_removed, en_kept = remove_hidden_rejected(os.path.join(REPO, 'en/tests.html'))
    print(f'\nEN: removed {en_removed} hidden entries, kept {en_kept} real S2 failures')
    ru_removed, ru_kept = remove_hidden_rejected(os.path.join(REPO, 'ru/tests.html'))
    print(f'RU: removed {ru_removed} hidden entries, kept {ru_kept} real S2 failures')

    # Re-export tests.json
    import subprocess
    r = subprocess.run(
        ['python3', 'scripts/export_tests_json.py', '.'],
        cwd=REPO, capture_output=True, text=True, timeout=30
    )
    print(r.stdout.strip())

    # Run reconciler
    r = subprocess.run(
        ['python3', 'scripts/registry_integrity_reconcile.py'],
        cwd=REPO, capture_output=True, text=True, timeout=60
    )
    for line in r.stdout.strip().split('\n')[-8:]:
        print(line)

    # Update meta-s2
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
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

if __name__ == '__main__':
    main()
