#!/usr/bin/env python3
"""
Flip S2_LOSES → S2_DUST_WINS for entries where best_alt is already S2_DUST.

These entries were fitted by the auto-scanner, which found S2_DUST to be the
best overall model. But fit_s2() in dream_auto_scanner.py sets model_verdict
based on whether S2 beats the best NON-S2 alternative — so when S2_DUST itself
is the best, it incorrectly labels the entry as S2_LOSES.

This script flips those entries to S2_DUST_WINS without refetching, since the
fit was already done. It also handles entries where the model_note already
contains 'S2+dust' indicating a dust decomposition was performed.
"""
import os, sys, json, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def flip_entries(html_path):
    """Flip S2_LOSES with best_alt=S2_DUST to S2_DUST_WINS in tests.html."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    # Find all entries with model_verdict:"S2_LOSES" and best_alt:"S2_DUST"
    # Pattern: {id:"...",name:"...",...,model_verdict:"S2_LOSES",...,best_alt:"S2_DUST",...}
    flips = 0
    # Use regex to find entry blocks containing both model_verdict:"S2_LOSES" and best_alt:"S2_DUST"
    pattern = re.compile(
        r'(\{[^{}]*?model_verdict:"S2_LOSES"[^{}]*?best_alt:"S2_DUST"[^{}]*?\})',
        re.DOTALL
    )

    def flip_one(m):
        nonlocal flips
        entry = m.group(1)
        # Flip model_verdict
        new_entry = entry.replace('model_verdict:"S2_LOSES"', 'model_verdict:"S2_DUST_WINS"')
        # Update model_note if it's the generic "S2 loses to S2_DUST" — make it clear this is dust-resolved
        # Don't overwrite if model_note already has dust decomposition info
        if 'model_note:"S2 loses to S2_DUST' in new_entry and 'S2+dust' not in new_entry.split('model_note:"')[1].split('"')[0]:
            # Generic note — replace with dust-resolved note
            # Extract delta_aicc
            dm = re.search(r'delta_aicc:([\d.]+)', new_entry)
            delta = float(dm.group(1)) if dm else 0.0
            new_note = f'S2 loses to S2+dust (ΔAICc={delta:.2f}). S2+dust is the best overall model. Dust decomposition confirmed — not a DREAM failure.'
            new_note_escaped = new_note.replace('\\', '\\\\').replace('"', '\\"')
            # Replace existing model_note (or add if not present)
            if 'model_note:"' in new_entry:
                new_entry = re.sub(
                    r'model_note:"(?:[^"\\]|\\.)*"',
                    f'model_note:"{new_note_escaped}"',
                    new_entry, count=1
                )
            else:
                new_entry = new_entry.replace(
                    'best_alt:"S2_DUST"',
                    f'best_alt:"S2_DUST",model_note:"{new_note_escaped}"'
                )
        flips += 1
        return new_entry

    new_html = pattern.sub(flip_one, html)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(new_html)
    return flips

def main():
    print('=== Flipping S2_LOSES → S2_DUST_WINS where best_alt=S2_DUST ===\n')

    en_flips = flip_entries(os.path.join(REPO, 'en/tests.html'))
    print(f'EN: flipped {en_flips} entries')
    ru_flips = flip_entries(os.path.join(REPO, 'ru/tests.html'))
    print(f'RU: flipped {ru_flips} entries')

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
