#!/usr/bin/env python3
"""
DREAM Pixel Boundary Tracker — Scout
=====================================

Daily scout of recent publications on optical probing limits.

The DREAM framework's "pixel is real" claim predicts that no amount of
engineering improvement can push the kernel's resolution λ_q lower — the
abyss is structural, not engineering-limited. To track this claim live,
we monitor the published state of the art for optical resolution, the
smallest scale at which structures or interference can be resolved.

Each daily run:
1. Query arXiv for recent papers in physics.optics and quant-ph
   with keywords like "optical resolution", "nanoscopy", "interferometry",
   "quantum metrology", "displacement sensitivity"
2. Extract the smallest length-scale mentioned in title/abstract
3. Append to a JSON timeline (pixel_tracker/records.json)
4. Commit to the repo (handled by the GitHub Action workflow)

The tracker page on the site plots the timeline of smallest-resolved-scale
over time. If DREAM is right, the curve should flatten (saturate at the
kernel's pixel). If standard physics is right, the curve keeps dropping
with engineering progress.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json, os, re, time, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
ARXIV_BASE = "http://export.arxiv.org/api/query"
RECORDS_FILE = Path(__file__).parent / "records.json"
BASELINE_FILE = Path(__file__).parent / "baseline.json"

# Search terms targeting optical-probing limits
QUERIES = [
    'all:"optical resolution" AND cat:physics.optics',
    'all:"nanoscopy" AND cat:physics.optics',
    'all:"super-resolution microscopy" AND cat:physics.optics',
    'all:"displacement sensitivity" AND cat:quant-ph',
    'all:"interferometry" AND all:"sensitivity" AND cat:physics.optics',
    'all:"quantum metrology" AND all:"length"',
    'all:"wavefront sensing" AND all:"resolution"',
]

# Scale regex — capture length-scale mentions in abstracts
# Examples: "10 nm", "1.5 Å", "10^-14 m", "femtometer", "attometer"
SCALE_PATTERNS = [
    # Numeric with explicit unit (m, nm, pm, fm, am, Å)
    (re.compile(r'(\d+\.?\d*)\s*(am|attometer[s]?)\b', re.I), lambda m: float(m.group(1)) * 1e-18),
    (re.compile(r'(\d+\.?\d*)\s*(fm|femtometer[s]?)\b', re.I), lambda m: float(m.group(1)) * 1e-15),
    (re.compile(r'(\d+\.?\d*)\s*(pm|picometer[s]?)\b', re.I), lambda m: float(m.group(1)) * 1e-12),
    (re.compile(r'(\d+\.?\d*)\s*(nm|nanometer[s]?)\b', re.I), lambda m: float(m.group(1)) * 1e-9),
    (re.compile(r'(\d+\.?\d*)\s*(μm|um|micron[s]?|micrometer[s]?)\b', re.I), lambda m: float(m.group(1)) * 1e-6),
    (re.compile(r'(\d+\.?\d*)\s*Å\b'), lambda m: float(m.group(1)) * 1e-10),
    # Scientific notation: 10^-14 m, 10e-20 m
    (re.compile(r'10\^(-?\d+)\s*m\b', re.I), lambda m: 10 ** int(m.group(1))),
    (re.compile(r'10\^(-?\d+)\s*met(er|re)s?\b', re.I), lambda m: 10 ** int(m.group(1))),
    (re.compile(r'10e(-?\d+)\s*m\b', re.I), lambda m: 10 ** int(m.group(1))),
]

def extract_scales(text):
    """Return list of length-scale values (in meters) mentioned in text."""
    scales = []
    for pat, conv in SCALE_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = conv(m)
                # Filter: scales of physical interest — between 10^-22 and 10^-3
                if 1e-22 <= v <= 1e-3:
                    scales.append(v)
            except Exception:
                pass
    return scales

def query_arxiv(query, max_results=20):
    """Query arXiv API for recent papers."""
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    }
    url = ARXIV_BASE + '?' + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'DREAM-pixel-scout/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"  query failed: {e}")
        return []
    # Parse Atom feed
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml)
    results = []
    for entry in root.findall('a:entry', ns):
        try:
            arxiv_id = entry.find('a:id', ns).text.split('/')[-1]
            title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('a:summary', ns).text.strip().replace('\n', ' ')
            published = entry.find('a:published', ns).text[:10]
            link = entry.find('a:id', ns).text
            results.append({
                'arxiv_id': arxiv_id,
                'title': title,
                'summary': summary,
                'published': published,
                'link': link,
            })
        except Exception as e:
            continue
    return results

def load_records():
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE) as f:
            return json.load(f)
    return {'records': [], 'last_run': None}

def save_records(data):
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORDS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def already_seen(records, arxiv_id):
    seen = {r.get('arxiv_id') for r in records.get('records', [])}
    return arxiv_id in seen

def main():
    print("=== DREAM Pixel Boundary Scout ===")
    print(f"Date: {datetime.date.today().isoformat()}")
    print(f"Time: {datetime.datetime.now().isoformat()}")

    data = load_records()
    if already_seen(data, '__placeholder__'):
        pass  # noop

    new_records = []
    seen_ids = {r.get('arxiv_id') for r in data.get('records', [])}

    for q in QUERIES:
        print(f"\nQuery: {q}")
        papers = query_arxiv(q, max_results=15)
        print(f"  got {len(papers)} results")
        for p in papers:
            if p['arxiv_id'] in seen_ids:
                continue
            text = (p['title'] + ' ' + p['summary']).lower()
            scales = extract_scales(text)
            if not scales:
                continue
            smallest = min(scales)
            # Skip obvious false positives (scales way outside optical probing regime)
            if smallest > 1e-3 or smallest < 1e-22:
                continue
            rec = {
                'arxiv_id': p['arxiv_id'],
                'title': p['title'],
                'published': p['published'],
                'link': p['link'],
                'smallest_scale_m': smallest,
                'smallest_scale_str': format_scale(smallest),
                'query': q,
                'scouted': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            new_records.append(rec)
            seen_ids.add(p['arxiv_id'])
            print(f"  + {p['arxiv_id']} ({p['published']}) {format_scale(smallest)} — {p['title'][:70]}")
        # Be polite to arXiv API
        time.sleep(3)

    # Append new records to the timeline
    data['records'].extend(new_records)
    data['records'].sort(key=lambda r: r.get('published', ''))
    data['last_run'] = datetime.datetime.now().isoformat(timespec='seconds')
    data['total_records'] = len(data['records'])

    save_records(data)
    print(f"\n=== Summary ===")
    print(f"New records: {len(new_records)}")
    print(f"Total records: {data['total_records']}")
    print(f"Last run: {data['last_run']}")

    if new_records:
        # Output a small JSON for the workflow to detect changes
        print("\nNEW_RECORDS_JSON:" + json.dumps({'count': len(new_records)}))

def format_scale(m):
    """Format a length in meters as a human-readable string."""
    if m >= 1e-9:
        return f"{m*1e9:.2g} nm"
    if m >= 1e-12:
        return f"{m*1e12:.2g} pm"
    if m >= 1e-15:
        return f"{m*1e15:.2g} fm"
    if m >= 1e-18:
        return f"{m*1e18:.2g} am"
    return f"{m:.2e} m"

if __name__ == '__main__':
    main()
