#!/usr/bin/env python3
"""
Wrap all '10D' and '10-dimensional' mentions in body text with a tooltip
span explaining that 10D is a parsimonious modeling choice.

Also fixes meta descriptions to not claim '10→4' as empirical fact.

Rules:
  - Skip <meta> tags (handled separately)
  - Skip <title> tags
  - Skip already-wrapped instances (class="dim-tip")
  - Skip instances inside <script> or <style> blocks
  - Skip instances in already-explanatory context (e.g., 'parsimonious 10D')
  - Wrap bare '10D' and '10-dimensional' with tooltip span
"""
import os, re, glob

REPO = '/home/z/my-project/dream_repo'
TOOLTIP = '10D is a parsimonious modeling choice — the simplest presently adopted realization, not an empirically observed dimensionality'

# Pages to process
html_files = glob.glob(os.path.join(REPO, 'en', '*.html')) + \
             glob.glob(os.path.join(REPO, 'en', 'articles', '*.html')) + \
             glob.glob(os.path.join(REPO, 'ru', '*.html')) + \
             glob.glob(os.path.join(REPO, 'ru', 'articles', '*.html')) + \
             glob.glob(os.path.join(REPO, '*.html'))

def wrap_10d(text):
    """Wrap bare 10D mentions with tooltip span, skipping already-explained contexts."""
    # Don't wrap if already inside a dim-tip span
    # Don't wrap if preceded by 'parsimonious' or 'modeling' or 'choice' (already explained)
    # Don't wrap in meta tags, title, script, style

    # Pattern: match '10D' not preceded by dim-tip wrapper and not in explanatory context
    # We use negative lookbehind for common explanatory phrases
    # and negative lookahead for already-wrapped

    # Skip lines that contain 'parsimonious.*10D' or 'modeling.*10D' or 'choice.*10D'
    # or 'assumption.*10D' or 'working.*10D' — these already explain

    def replacer(m):
        full = m.group(0)
        # Check if this is inside an HTML tag
        if '<' in full or '>' in full:
            return full
        # Check if already wrapped
        if 'dim-tip' in m.group(0):
            return full
        # Get just the '10D' or '10-dimensional' part
        match_text = m.group(1)
        return f'<span class="dim-tip" title="{TOOLTIP}">{match_text}</span>'

    # Match '10D' or '10-dimensional' as standalone text (not inside tags)
    # Negative lookbehind: not preceded by dim-tip", not preceded by explanatory words within 30 chars
    # We'll do this more carefully: process line by line

    lines = text.split('\n')
    result = []
    for line in lines:
        # Skip meta tags, title, script, style lines
        stripped = line.strip()
        if stripped.startswith('<meta ') or stripped.startswith('<title>'):
            result.append(line)
            continue
        if '<script' in line or '</script>' in line or '<style' in line or '</style>' in line:
            result.append(line)
            continue

        # Check if this line already has explanatory context for 10D
        lower = line.lower()
        has_explanation = any(x in lower for x in [
            'parsimonious', 'modeling assumption', 'modeling choice',
            'not an observed', 'not an independently',
            'working dimensionality', 'not established',
            'dim-tip', 'could in principle', 'could be 14',
            'codimension-six', 'not a gratuitous',
        ])

        if has_explanation:
            # Don't wrap — context already explains
            result.append(line)
            continue

        # Wrap standalone '10D' (not inside a word like '10D→4D')
        # Match '10D' as a token, possibly followed by →4 or space or punctuation
        # But don't match inside HTML attribute values
        new_line = re.sub(
            r'(?<![a-zA-Z])(10D)(?![a-zA-Z"])',
            r'<span class="dim-tip" title="' + TOOLTIP + r'">\1</span>',
            line
        )
        # Also wrap '10-dimensional'
        new_line = re.sub(
            r'(?<![a-zA-Z])(10-dimensional)',
            r'<span class="dim-tip" title="' + TOOLTIP + r'">\1</span>',
            new_line
        )
        result.append(new_line)

    return '\n'.join(result)


def fix_meta_description(text):
    """Fix meta descriptions to not claim 10→4 as empirical fact."""
    old_desc = 'A 10→4 projection-first physics framework with the Retention Law as foundation.'
    new_desc = 'A projection-first physics framework modeling 4D observables as finite-resolution projections of a higher-dimensional source, with the Retention Law as foundation.'
    return text.replace(old_desc, new_desc)


def main():
    total_files = 0
    total_wraps = 0

    for filepath in sorted(html_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        modified = fix_meta_description(original)

        # Count existing dim-tip to know how many are already there
        before_count = modified.count('class="dim-tip"')

        modified = wrap_10d(modified)

        after_count = modified.count('class="dim-tip"')
        new_wraps = after_count - before_count

        if modified != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            total_files += 1
            total_wraps += new_wraps
            print(f'  ✓ {os.path.relpath(filepath, REPO)}: {new_wraps} wraps added')

    print(f'\nTotal: {total_files} files modified, {total_wraps} tooltips added')


if __name__ == '__main__':
    main()
