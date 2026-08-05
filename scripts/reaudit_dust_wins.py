#!/usr/bin/env python3
"""
Re-audit S2_DUST_WINS entries with strict physical validation criteria.

S2_DUST (6 params) almost always beats S2 (3 params) and BIEXP (4 params) on
AICc just from parameter flexibility — especially on noisy or structured data.
A legitimate dust decomposition must satisfy ALL of:

  1. D1 > 0 and D2 > 0 (no degenerate zeros / failed fits)
  2. R²_dust >= 0.5 (the dust model must actually fit the curve)
  3. 0.05 < D1 < 9.0 and 0.05 < D2 < 9.0 (not at optimizer boundary)
  4. |log(D1/D2)| > 0.5 (the two scales must be distinct — ratio > 1.65x)

Entries that fail any criterion are reverted to S2_LOSES with an honest
model_note explaining why the dust decomposition was rejected.
"""
import os, sys, json, re, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_dust_params(entry):
    """Extract D1, D2, R²_dust from model_note. Returns (d1, d2, r2) or (None, None, None)."""
    note = entry.get('model_note', '')
    m = re.search(r'D1=([\d.]+),\s*D2=([\d.]+),\s*R²=([\d.]+)', note)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None

def validate_dust(d1, d2, r2_dust):
    """Return list of validation failures (empty list = legit)."""
    issues = []
    if d1 is None or d2 is None or r2_dust is None:
        return ['no D1/D2/R² in model_note (fit not validated)']
    if d1 <= 0 or d2 <= 0:
        issues.append(f'degenerate (D1={d1}, D2={d2})')
    if r2_dust < 0.5:
        issues.append(f'R²_dust={r2_dust:.3f} < 0.5')
    if d1 > 0 and not (0.05 < d1 < 9.0):
        issues.append(f'D1={d1:.3f} at boundary')
    if d2 > 0 and not (0.05 < d2 < 9.0):
        issues.append(f'D2={d2:.3f} at boundary')
    if d1 > 0 and d2 > 0:
        ratio = abs(math.log(d1 / d2))
        if ratio < 0.5:
            issues.append(f'D1≈D2 (ratio={d1/d2:.2f}, |log|={ratio:.2f}<0.5)')
    return issues

def revert_entry(html_path, name, issues):
    """Revert a S2_DUST_WINS entry back to S2_LOSES in tests.html."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    pattern = re.compile(
        r'(\{[^{}]*?model_verdict:"S2_DUST_WINS"[^{}]*?\})',
        re.DOTALL
    )

    name_clean = name.replace('"', '').replace('\\', '')

    for m in pattern.finditer(html):
        entry = m.group(1)
        nm = re.search(r'name:"((?:[^"\\]|\\.)*)"', entry)
        if not nm:
            continue
        entry_name = nm.group(1).replace('\\"', '"').replace('\\\\', '\\')
        entry_name_clean = entry_name.replace('"', '').replace('\\', '')

        if entry_name_clean[:40] == name_clean[:40]:
            # Always revert — validation was already done in the audit step.
            new_entry = entry.replace('model_verdict:"S2_DUST_WINS"', 'model_verdict:"S2_LOSES"')
            delta_match = re.search(r'delta_aicc:([\d.]+)', new_entry)
            delta = float(delta_match.group(1)) if delta_match else 0.0
            reason = '; '.join(issues)
            new_note = (f'S2 loses to S2_DUST (ΔAICc={delta:.2f}). '
                        f'S2+dust decomposition REJECTED: {reason}. '
                        f'Likely overfitting (6-param model). Treated as S2_LOSES.')
            new_note_escaped = new_note.replace('\\', '\\\\').replace('"', '\\"')
            if 'model_note:"' in new_entry:
                new_entry = re.sub(
                    r'model_note:"(?:[^"\\]|\\.)*"',
                    f'model_note:"{new_note_escaped}"',
                    new_entry, count=1
                )
            html = html.replace(entry, new_entry, 1)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            return True
    return False

def main():
    print('=' * 60)
    print('RE-AUDIT OF S2_DUST_WINS ENTRIES')
    print('=' * 60)

    with open(os.path.join(REPO, 'en/tests.json')) as f:
        data = json.load(f)

    redeemed = [t for t in data['tests'] if t.get('model_verdict') == 'S2_DUST_WINS']
    print(f'\nS2_DUST_WINS entries to audit: {len(redeemed)}\n')

    legit = []
    revert = []

    for t in redeemed:
        d1, d2, r2_dust = extract_dust_params(t)
        issues = validate_dust(d1, d2, r2_dust)
        if issues:
            revert.append((t, issues))
            print(f'  ✗ REVERT {t["name"][:55]:<55} — {"; ".join(issues)}')
        else:
            legit.append(t)
            print(f'  ✓ KEEP   {t["name"][:55]:<55} D1={d1:.3f} D2={d2:.3f} R²={r2_dust:.3f}')

    print(f'\n{"=" * 60}')
    print(f'Legitimate dust decompositions: {len(legit)}')
    print(f'Reverting to S2_LOSES:          {len(revert)}')

    for t, issues in revert:
        revert_entry(os.path.join(REPO, 'en/tests.html'), t['name'], issues)
        revert_entry(os.path.join(REPO, 'ru/tests.html'), t['name'], issues)

    import subprocess
    r = subprocess.run(
        ['python3', 'scripts/export_tests_json.py', '.'],
        cwd=REPO, capture_output=True, text=True, timeout=30
    )
    print(f'\n{r.stdout.strip()}')

    r = subprocess.run(
        ['python3', 'scripts/registry_integrity_reconcile.py'],
        cwd=REPO, capture_output=True, text=True, timeout=60
    )
    for line in r.stdout.strip().split('\n')[-8:]:
        print(line)

    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    from dream_auto_scanner import update_meta_s2_article
    update_meta_s2_article(os.path.join(REPO, 'en/tests.html'), is_ru=False)
    update_meta_s2_article(os.path.join(REPO, 'ru/tests.html'), is_ru=True)

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
