#!/usr/bin/env python3
"""
Remove blacklisted entries from tests.html (EN + RU) and prevent re-addition.

Blacklist: arXiv papers that have no downloadable data — they sit as PENDING
forever and pollute the registry count without contributing to Meta-S2.
"""
import re, json, os, sys

BLACKLIST = [
    'arXiv: Stretched Exponential Decay in the Edwards-Wilkins',
    'arXiv: Exponential decay of correlations for random inter',
    'arXiv: Cost-Aware Logging: Measuring the Financial Impact',
    'arXiv: Large deviations for dynamical systems with stretc',
    'arXiv: On the Polynomial and Exponential Decay of Eigen-F',
]

# Also blacklist by URL pattern
BLACKLIST_PATTERNS = [
    r'arxiv\.org/abs/',
]

def is_blacklisted(entry_str):
    for bl in BLACKLIST:
        if bl in entry_str:
            return True
    for pat in BLACKLIST_PATTERNS:
        if re.search(pat, entry_str):
            return True
    return False

def remove_blacklisted(html_path):
    with open(html_path) as f:
        html = f.read()
    
    m = re.search(r'(const\s+TESTS\s*=\s*\[)(.*?)(\n\];)', html, re.DOTALL)
    if not m:
        print(f'  ! No TESTS array in {html_path}')
        return 0
    
    prefix = m.group(1)
    body = m.group(2)
    suffix = m.group(3)
    
    # Parse entries
    kept = []
    removed = 0
    i = 0
    while i < len(body):
        brace = body.find('{', i)
        if brace < 0: break
        depth = 0; j = brace; in_str = False; esc = False
        while j < len(body):
            c = body[j]
            if esc: esc = False
            elif c == '\\': esc = True
            elif c == '"': in_str = not in_str
            elif not in_str:
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0: break
            j += 1
        if depth != 0: break
        entry_str = body[brace:j+1]
        
        if is_blacklisted(entry_str):
            removed += 1
            nm = re.search(r'name:"([^"]*)"', entry_str)
            print(f'  ✗ Removed: {nm.group(1)[:60] if nm else "?"}')
        else:
            # Keep the leading comma if present
            leading = body[i:brace]
            kept.append(leading + entry_str)
        i = j + 1
    
    # Rebuild
    new_body = '\n  '.join(kept) if kept else ''
    new_html = html[:m.start()] + prefix + new_body + suffix + html[m.end():]
    
    with open(html_path, 'w') as f:
        f.write(new_html)
    
    return removed

# Write blacklist file for scanner
def write_blacklist_file():
    bl = {
        'blacklisted_names': BLACKLIST,
        'blacklisted_url_patterns': [p.replace('\\', '') for p in BLACKLIST_PATTERNS],
        'reason': 'arXiv papers with no downloadable data — PENDING forever, no D value, no Meta-S2 contribution',
    }
    with open('scan_blacklist.json', 'w') as f:
        json.dump(bl, f, indent=2)
    print(f'✓ Blacklist file written: scan_blacklist.json')

if __name__ == '__main__':
    print('=== Removing blacklisted entries ===')
    total_removed = 0
    for lang in ('en', 'ru'):
        print(f'\n--- {lang.upper()} ---')
        removed = remove_blacklisted(f'{lang}/tests.html')
        print(f'  Removed: {removed}')
        total_removed += removed
    write_blacklist_file()
    print(f'\nTotal removed: {total_removed}')
