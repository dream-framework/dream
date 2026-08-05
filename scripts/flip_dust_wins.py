#!/usr/bin/env python3
"""
Flip S2_LOSES → S2_DUST_WINS ONLY for entries where:
  1. best_alt = S2_DUST (the fit already ran, S2_DUST is the best model)
  2. The original model_note contains D1=/D2=/R²= values (fit was validated)
  3. The dust decomposition passes physical validation:
     - D1 > 0 and D2 > 0 (no degenerate zeros)
     - R²_dust >= 0.5
     - 0.05 < D1, D2 < 9.0 (not at boundary)
     - |log(D1/D2)| > 0.5 (distinct scales)

Entries where best_alt=S2_DUST but the fit wasn't validated (no D1/D2 in
model_note) are LEFT as S2_LOSES — we can't confirm the dust decomposition
is real, not just 6-parameter overfitting.
"""
import os, sys, json, re, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def validate_dust_from_note(note):
    """Extract D1, D2, R² from model_note and validate. Returns (is_legit, issues)."""
    m = re.search(r'D1=([\d.]+),\s*D2=([\d.]+),\s*R²=([\d.]+)', note)
    if not m:
        return False, ['no D1/D2/R² in model_note (fit not validated)']
    d1, d2, r2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    issues = []
    if d1 <= 0 or d2 <= 0:
        issues.append(f'degenerate (D1={d1}, D2={d2})')
    if r2 < 0.5:
        issues.append(f'R²_dust={r2:.3f} < 0.5')
    if d1 > 0 and not (0.05 < d1 < 9.0):
        issues.append(f'D1={d1:.3f} at boundary')
    if d2 > 0 and not (0.05 < d2 < 9.0):
        issues.append(f'D2={d2:.3f} at boundary')
    if d1 > 0 and d2 > 0:
        ratio = abs(math.log(d1 / d2))
        if ratio < 0.5:
            issues.append(f'D1≈D2 (|log|={ratio:.2f}<0.5)')
    return (len(issues) == 0), issues

def flip_entries(html_path):
    """Flip S2_LOSES with best_alt=S2_DUST to S2_DUST_WINS in tests.html,
    but ONLY if the dust decomposition passes physical validation."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    flips = 0
    skipped = 0
    pattern = re.compile(
        r'(\{[^{}]*?model_verdict:"S2_LOSES"[^{}]*?best_alt:"S2_DUST"[^{}]*?\})',
        re.DOTALL
    )

    def flip_one(m):
        nonlocal flips, skipped
        entry = m.group(1)
        # Extract model_note to validate
        nm = re.search(r'model_note:"((?:[^"\\]|\\.)*)"', entry)
        if not nm:
            skipped += 1
            return entry  # don't flip
        note = nm.group(1).replace('\\"', '"').replace('\\\\', '\\')

        is_legit, issues = validate_dust_from_note(note)
        if not is_legit:
            skipped += 1
            return entry  # don't flip — keep as S2_LOSES

        # Flip model_verdict
        new_entry = entry.replace('model_verdict:"S2_LOSES"', 'model_verdict:"S2_DUST_WINS"')
        flips += 1
        return new_entry

    new_html = pattern.sub(flip_one, html)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(new_html)
    return flips, skipped

def main():
    print('=== Flipping S2_LOSES → S2_DUST_WINS where best_alt=S2_DUST ===\n')

    en_flips, en_skipped = flip_entries(os.path.join(REPO, 'en/tests.html'))
    print(f'EN: flipped {en_flips} entries, skipped {en_skipped} (failed validation)')
    ru_flips, ru_skipped = flip_entries(os.path.join(REPO, 'ru/tests.html'))
    print(f'RU: flipped {ru_flips} entries, skipped {ru_skipped} (failed validation)')

    # Re-export tests.json
    import subprocess
    r = subprocess.run(
        ['python3', 'scripts/export_tests_json.py', '.'],
        cwd=REPO, capture_output=True, text=True, timeout=30
    )
    print(r.stdout.strip())

    # Run reconciler to sync narratives
    r = subprocess.run(
        ['python3', 'scripts/registry_integrity_reconcile.py'],
        cwd=REPO, capture_output=True, text=True, timeout=60
    )
    for line in r.stdout.strip().split('\n')[-10:]:
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
