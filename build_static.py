#!/usr/bin/env python3
"""
build_static.py — convert the Flask DREAM templates into static HTML.

For each *_body.html under templates/i18n/{en,ru}/:
  * strip Jinja2 block tags ({% block content %}, {% endblock %})
  * inline partial includes ({% include 'partials/foo.html' %})
  * strip {{ _('ui.read_more') }} → "Read more" / "Читать далее"
  * rewrite url_for('static', filename='plotter.html') → /plotter.html
  * rewrite url_for('xxx') → relative link, e.g. retention.html
  * wrap the body in the dream_static base template (header / nav / footer /
    chat bubble / fractal canvas / MathJax) and write to en/{page}.html
    or ru/{page}.html.

Run:  python3 build_static.py
"""

import os
import re
import sys
from pathlib import Path

SRC_ROOT = Path("/home/z/my-project/dream_physics_clone")
TPL_ROOT = SRC_ROOT / "templates"
DST_ROOT = Path("/home/z/my-project/dream_static")

# ---------------------------------------------------------------------------
# nav (matches app.py:NAV, plus plotter & npa toy & intervention simulator)
# ---------------------------------------------------------------------------
NAV = [
    ("home",         "Home",              "Главная"),
    ("case",         "Case for D.R.E.A.M","Сила D.R.E.A.M"),
    ("retention",    "Retention Law",     "Принцип Сохранения"),
    ("axioms",       "Axioms",            "Аксиомы"),
    ("theorems",     "Theorems",          "Теоремы"),
    ("math",         "Math Frame",        "Матмодель"),
    ("kernel",       "Kernel",            "Проекция"),
    ("topology",     "Topology",          "Топология"),
    ("spectrum",     "Fractal Spectrum",  "Фрактальность"),
    ("predictions",  "Predictions",       "Предсказания"),
    ("falsification","Falsification",     "Фальсификация"),
    ("faq",          "FAQ Clouds",        "Вопрос-Ответ"),
    ("about",        "AI Analysis",       "Оценка ИИ"),
    ("time",         "Time",              "Время"),
    ("memory",       "Memory",            "Память"),
    ("articles",     "Articles",          "Статьи"),
]

TOY_LINKS = [
    ("plotter.html",              "Plotter Toy",   "График Удержания"),
    ("npa-calculator.html",       "NPA Toy",       "NPA Калькулятор"),
    ("intervention-simulator.html","Intervention Sim","Симулятор Вмешательства"),
]

# map page key → file slug (most match the key, a few differ)
PAGE_FILE = {
    "home":          "index",
    "case":          "case",
    "retention":     "retention",
    "axioms":        "axioms",
    "theorems":      "theorems",
    "math":          "math",
    "kernel":        "kernel",
    "topology":      "topology",
    "spectrum":      "spectrum",
    "predictions":   "predictions",
    "falsification": "falsification",
    "faq":           "faq",
    "about":         "about",
    "time":          "time",
    "memory":        "memory",
    "articles":      "articles",
}

# ---------------------------------------------------------------------------
# partial inlining
# ---------------------------------------------------------------------------

PARTIAL_CACHE = {}

def read_partial(name: str) -> str:
    """Read templates/<name>, stripping any leading Jinja comment block."""
    if name in PARTIAL_CACHE:
        return PARTIAL_CACHE[name]
    p = TPL_ROOT / name
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8")
    # strip leading {# ... #} comments
    txt = re.sub(r"\{#.*?#\}", "", txt, flags=re.DOTALL)
    PARTIAL_CACHE[name] = txt
    return txt

INCLUDE_RE = re.compile(r"\{%\s*include\s+['\"]([^'\"]+)['\"]\s*%\}")

def inline_partials(text: str) -> str:
    """Recursively replace {% include 'partials/x.html' %} with file content."""
    for _ in range(8):
        new = INCLUDE_RE.sub(lambda m: read_partial(m.group(1)), text)
        if new == text:
            return new
        text = new
    return text

# ---------------------------------------------------------------------------
# url_for + i18n stripping
# ---------------------------------------------------------------------------

URL_FOR_STATIC_RE = re.compile(
    r"\{\{\s*url_for\(\s*['\"]static['\"]\s*,\s*filename\s*=\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"
)
URL_FOR_RE = re.compile(
    r"\{\{\s*url_for\(\s*['\"]([a-zA-Z0-9_.]+)['\"](?:[^)]*)?\)\s*\}\}"
)
BLOCK_RE = re.compile(r"\{%\s*(?:block\s+\w+|endblock)\s*%\}")
TRANS_RE = re.compile(r"\{\{\s*_\(\s*['\"]([\w.]+)['\"]\s*\)\s*\}\}")
JINJA_COMMENT_RE = re.compile(r"\{#.*?#\}", flags=re.DOTALL)
LEFTOVER_JINJA_RE = re.compile(r"\{%[^%]*%\}")  # any remaining {% ... %}
LEFTOVER_EXPR_RE = re.compile(r"\{\{[^}]*\}\}")  # any remaining {{ ... }}
INLINE_HEAD_RE = re.compile(r"^\s*<head>.*?</head>\s*", flags=re.DOTALL | re.IGNORECASE)
# inline <script> blocks (in body content) that re-configure or re-load MathJax.
# The base template already provides a complete MathJax config + SVG loader;
# body content shouldn't try to override it (causes double-load / renderer flips).
INLINE_MATHJAX_CFG_RE = re.compile(
    r"\s*<script>\s*window\.MathJax\s*=\s*\{[^}]*\}\s*;?\s*</script>",
    flags=re.DOTALL
)
INLINE_MATHJAX_LOADER_RE = re.compile(
    r'\s*<script[^>]*src="[^"]*mathjax[^"]*"[^>]*>\s*</script>',
    flags=re.IGNORECASE
)

def strip_jinja(text: str, lang: str) -> str:
    # static asset urls → relative (the toys live at the site root, and pages
    # live under /en/ or /ru/, so use ../toys)
    text = URL_FOR_STATIC_RE.sub(lambda m: "../" + m.group(1), text)
    # url_for('weave.data') → leave empty (will be replaced by client weave.js)
    # url_for('xxx') → relative link, e.g. retention.html (and ru/retention.html)
    def url_repl(m):
        endpoint = m.group(1)
        if endpoint == "weave.data":
            return ""  # weave_embed has its own URL fetch path we'll override
        # endpoint names match page keys
        slug = PAGE_FILE.get(endpoint, endpoint)
        return f"{slug}.html"
    text = URL_FOR_RE.sub(url_repl, text)
    # {{ _('ui.read_more') }} → "Read more" / "Читать далее"
    read_more = "Читать далее" if lang == "ru" else "Read more"
    def trans_repl(m):
        key = m.group(1)
        if key == "ui.read_more":
            return read_more
        # any other i18n key → empty
        return ""
    text = TRANS_RE.sub(trans_repl, text)
    # block tags
    text = BLOCK_RE.sub("", text)
    # comments
    text = JINJA_COMMENT_RE.sub("", text)
    # any leftover {% %} or {{ }} → remove
    text = LEFTOVER_JINJA_RE.sub("", text)
    text = LEFTOVER_EXPR_RE.sub("", text)
    # strip any inline <head>...</head> that body content tries to inject
    # (only the FAQ body has one; the base template already provides <head>)
    text = INLINE_HEAD_RE.sub("", text)
    # strip inline MathJax config + loader duplicates from body content
    text = INLINE_MATHJAX_CFG_RE.sub("", text)
    text = INLINE_MATHJAX_LOADER_RE.sub("", text)
    # FAQ body hardcodes "/static/kb/parsed_faqs_<lang>.json" — rewrite to
    # a relative path so it works on GitHub Pages subpaths.
    text = re.sub(
        r'"/static/kb/(parsed_faqs_\w+\.json)"',
        r'"../kb/\1"',
        text
    )
    return text

# ---------------------------------------------------------------------------
# base template
# ---------------------------------------------------------------------------

def nav_html(lang: str, active_key: str | None, depth: int = 0) -> str:
    """
    Build nav HTML. `depth` is the number of `../` prefixes needed:
      depth=0  → page is at /<lang>/<slug>.html      (sibling nav links)
      depth=1  → page is at /<lang>/articles/<slug>.html  (need ../ prefix)
    """
    prefix = "../" * depth if depth else ""
    out = []
    for key, en, ru in NAV:
        label = ru if lang == "ru" else en
        slug = PAGE_FILE[key]
        # sibling-level links within the language folder
        href = f"{prefix}{slug}.html" if slug != "index" else f"{prefix}index.html"
        if key == active_key:
            cls = "nav-link active"
        else:
            cls = "nav-link"
        # add foundation marker for predictions (matches original app.py styling)
        if key == "predictions":
            cls += " foundation"
        out.append(f'      <a class="{cls}" href="{href}">{label}</a>')
    # toys — they live at the site root, so we need one extra ../ for depth
    toy_prefix = "../" * (depth + 1)
    for fname, en, ru in TOY_LINKS:
        label = ru if lang == "ru" else en
        href = f"{toy_prefix}{fname}"
        out.append(f'      <a class="nav-link" href="{href}">{label}</a>')
    return "\n".join(out)

def lang_toggle_html(lang: str, depth: int = 0) -> str:
    """Return language toggle for a page in en/ or ru/.
    `depth` matches nav_html's depth: 0 for /<lang>/<page>.html, 1 for /<lang>/articles/<page>.html.
    The other language's same-page lives at ../<lang>/<slug>.html (depth=0) or ../../<lang>/<slug>.html (depth=1).
    The rewrite JS at the bottom of each page rewrites these to point to the
    equivalent page in the other language."""
    ru_on = "on" if lang == "ru" else ""
    en_on = "on" if lang == "en" else ""
    return f'''        <div class="lang">
          <a class="lang-link {ru_on}" href="{'../' * (depth + 1)}ru/" title="Русский">
            <img src="https://flagcdn.com/24x18/ru.png"
                 srcset="https://flagcdn.com/48x36/ru.png 2x, https://flagcdn.com/72x54/ru.png 3x"
                 width="24" height="16" alt="Russia" class="flag-icon">
          </a>
          <a class="lang-link {en_on}" href="{'../' * (depth + 1)}en/" title="English">
            <img src="https://flagcdn.com/24x18/gb.png"
                 srcset="https://flagcdn.com/48x36/gb.png 2x, https://flagcdn.com/72x54/gb.png 3x"
                 width="24" height="16" alt="United Kingdom" class="flag-icon">
          </a>
        </div>'''

# `data-lang-switch` JS at the bottom of each page rewrites the lang toggle
# links so they point to the equivalent page in the other language.
LANG_SWITCH_JS = """
<script>
(function(){
  // Rewrite language toggle so it switches to the *same page* in the other language.
  // Handles /<lang>/<slug>.html  AND  /<lang>/articles/<slug>.html
  const here = location.pathname.replace(/\\/index\\.html$/i, '/');
  const m = here.match(/\\/(en|ru)\\/(.+?)\\.html$/i);
  let rest = '';
  if (m) rest = m[2];                                  // e.g. "retention" or "articles/introduction"
  document.querySelectorAll('.lang-link').forEach(a=>{
    const isRu = /flagcdn\\.com\\/.*?ru\\.png/.test(a.querySelector('img')?.src || '');
    const targetLang = isRu ? 'ru' : 'en';
    // Compute the prefix: walk up to the language root, then into the other language.
    // For /en/retention.html: depth=0 → prefix is "../"
    // For /en/articles/introduction.html: depth=1 → prefix is "../../"
    const parts = rest.split('/');
    const depth = parts.length - 1;                     // 0 for top-level, 1 for articles/
    const prefix = '../'.repeat(depth + 1);
    a.setAttribute('href', prefix + targetLang + '/' + rest + '.html');
  });
})();
</script>
"""

def base_template(lang: str, page_key: str, title: str, body_html: str, page_slug: str, depth: int = 0) -> str:
    is_article = page_key == "articles"
    html_lang = lang
    nav = nav_html(lang, page_key, depth=depth)
    lang_toggle = lang_toggle_html(lang, depth=depth)

    # toggles (Math + Speculative) — same as original
    math_label = "Математика" if lang == "ru" else "Math"
    spec_label = "Интерпретации" if lang == "ru" else "Speculative"
    math_title = "Показать/скрыть математическое содержание" if lang == "ru" else "Show/Hide mathematical content"
    spec_title = "Показывать интерпретационные разделы" if lang == "ru" else "Show speculative content"

    footer_text = (
        "прототип; ядро науки и интерпретации чётко разделены."
        if lang == "ru"
        else "prototype; core science and speculation are clearly separated."
    )

    # css/js paths: depth=0 → /<lang>/<page>.html, so prefix is "../"
    #              depth=1 → /<lang>/articles/<page>.html, so prefix is "../../"
    css_prefix = "../" * (depth + 1)
    css_path = f"{css_prefix}css/global.css"
    js_path  = lambda p: f"{css_prefix}js/{p}"

    # brand link — go up to language root
    brand_href = ("../" * depth) + "index.html"

    # MathJax config (same as original base.html)
    mathjax_cfg = """  window.MathJax = {
    tex: {
      inlineMath: [['\\\\(','\\\\)'], ['$', '$']],
      displayMath: [['\\\\[','\\\\]'], ['$$','$$']],
      processEscapes: true,
      packages: {'[+]': ['ams']}
    },
    chtml: { linebreaks:{automatic:true, width:'container'}, displayAlign:'center', displayIndent:'0' },
    svg:   { linebreaks:{automatic:true, width:'container'}, displayAlign:'center', displayIndent:'0' },
    options: { skipHtmlTags:['script','noscript','style','textarea','pre','code'] }
  };"""

    # The mode-toggle, spec-toggle, and math-typeset scripts that were inlined
    # into base.html's tail
    inline_tail_js = """
<script>
document.addEventListener('DOMContentLoaded', () => {
  const cb = document.getElementById('specToggle');
  const BODY_FLAG = 'show-spec';
  const KEY = 'spec:on';
  const saved = localStorage.getItem(KEY);
  if (saved !== null) {
    const on = saved === '1';
    document.body.classList.toggle(BODY_FLAG, on);
    if (cb) cb.checked = on;
  } else if (cb) {
    document.body.classList.toggle(BODY_FLAG, cb.checked);
  }
  cb?.addEventListener('change', e => {
    const on = e.target.checked;
    document.body.classList.toggle(BODY_FLAG, on);
    localStorage.setItem(KEY, on ? '1' : '0');
  });
});
</script>
<script>
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.math-block, .math-inline').forEach(el => {
    let t = el.textContent;
    if (/\\\\[/.test(t) && !/\\\\]/.test(t)) t = t.replace(/$/, '\\n\\\\]');
    if (/\\\\(/.test(t) && !/\\\\)/.test(t)) t = t.replace(/$/, '\\\\)');
    if (t !== el.textContent) el.textContent = t;
  });
  const typeset = nodes =>
    (window.MathJax && MathJax.typeset) && MathJax.typeset(nodes || undefined);
  if (document.body.classList.contains('show-math')) typeset();
  const mt = document.getElementById('mathToggle');
  mt && mt.addEventListener('change', () => setTimeout(() => typeset(), 0));
  document.querySelectorAll('details.more').forEach(d => {
    d.addEventListener('toggle', () => d.open && typeset([d]));
  });
});
</script>
"""

    return f"""<!doctype html>
<html lang="{html_lang}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <meta name="description" content="D.R.E.A.M — Dimensional Resonant Emergent Attractors in Manifold. A 10→4 projection-first physics framework with the Retention Law as foundation." />
  <link rel="stylesheet" href="{css_path}">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
  <script>
{mathjax_cfg}
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    /* per-page overrides for article pages */
    {"""
    body.is-article .site-header{ width:100vw; margin-left:calc(50% - 50vw); margin-right:calc(50% - 50vw); border-radius:0; }
    """ if is_article else ""}
  </style>
</head>
<body data-lang="{lang}" class="{'is-article' if is_article else ''}">

  <!-- ===== Fractal backdrop ===== -->
  <div id="fx" aria-hidden="true">
    <canvas id="mb"></canvas>
  </div>

  <!-- ===== Header / Nav ===== -->
  <header class="site-header">
    <div class="brand">
      <span class="dot" aria-hidden="true"></span>
      <a href="{brand_href}">D.R.E.A.M</a>
    </div>
    <nav class="nav" aria-label="Primary">
{nav}
    </nav>
    <div class="controls">
      <div class="toggles">
        <label class="switch" title="{math_title}">
          <input type="checkbox" id="mathToggle"><span class="slider" aria-hidden="true"></span>
          <span class="switch-label">{math_label}</span>
        </label>
        <label class="switch" title="{spec_title}">
          <input type="checkbox" id="specToggle"><span class="slider" aria-hidden="true"></span>
          <span class="switch-label">{spec_label}</span>
        </label>
      </div>
{lang_toggle}
    </div>
  </header>

  <!-- ===== Main ===== -->
  <main class="container">
{body_html}
  </main>

  <!-- ===== Footer ===== -->
  <footer class="site-footer">
    <div class="footer-content">
      <small>© 2024–2025 DREAM — {footer_text}</small>
    </div>
  </footer>

  <!-- ===== Chat bubble + dock (built by js/bot.js) ===== -->

  <!-- ===== Math toggle / speculative / typeset hooks ===== -->
{inline_tail_js}

  <!-- ===== Language switch rewrite (must run before bot.js so bot picks
        the right language pack from data-lang) ===== -->
{LANG_SWITCH_JS}

  <!-- ===== Fractal backdrop ===== -->
  <script defer src="{js_path('fractal.js')}"></script>
  <!-- ===== Kernel weave generator (only used on kernel page, but harmless elsewhere) ===== -->
  <script defer src="{js_path('weave.js')}"></script>
  <!-- ===== Bot (chat bubble) ===== -->
  <script defer src="{js_path('bot.js')}"></script>
  <!-- ===== Fullscreen toggle for toys ===== -->
  <script defer src="{js_path('fs-toggle.js')}"></script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Special-case rewrites for the weave_embed partial (no backend available)
# ---------------------------------------------------------------------------

WEAVE_FETCH_RE = re.compile(
    r"async function fetchData\(\)\s*\{[\s\S]*?\n  \}"
)
WEAVE_DATA_URL_LINE_RE = re.compile(r"\s*const DATA_URL\s*=.*?;\s*\n")

def rewrite_weave_embed(html: str, lang: str) -> str:
    """
    Replace the weave_embed's fetch-based data loader with a client-side
    call to window.DreamWeave.generate(...).
    """
    # 1) drop the kw-endpoint anchor (no longer needed)
    html = re.sub(
        r'\s*<a id="kw-endpoint"[^>]*></a>',
        "",
        html
    )
    # 2) replace `const DATA_URL = document.getElementById('kw-endpoint').href;`
    html = WEAVE_DATA_URL_LINE_RE.sub("", html)
    # 3) replace the async fetchData() function body
    new_fetch = (
        "async function fetchData(){\n"
        "    const params = { lam: parseFloat(elLam.value), lam_q: parseFloat(elLq.value), deff: parseFloat(elDeff.value), seed: parseInt(elSeed.value) || 42, n: 3500 };\n"
        "    return window.DreamWeave ? window.DreamWeave.generate(params) : { traces: { filament:{x:[],y:[],z:[],count:0}, facet:{x:[],y:[],z:[],count:0}, hub:{x:[],y:[],z:[],count:0}, dust:{x:[],y:[],z:[],count:0} } };\n"
        "  }"
    )
    html = WEAVE_FETCH_RE.sub(new_fetch, html, count=1)
    # 4) fix the catch error message that referenced the now-removed DATA_URL
    html = html.replace(
        "say(`Could not load data (${e.message}). Check ${DATA_URL}.`);",
        "say(`Could not generate weave data (${e.message}). Make sure js/weave.js is loaded.`);"
    )
    return html

# ---------------------------------------------------------------------------
# Articles page (manual list expansion — replaces the Jinja loops)
# ---------------------------------------------------------------------------

ARTICLES = {
    "en": [
        ("introduction.html", "Introduction to D.R.E.A.M"),
        ("kernel.html",       "The Projection Kernel: Constants-as-Settings, Retention, and the Resolution Cliff"),
        ("retention.html",    "The Retention Cliff: Measuring the Edge Between Quantum and Classical"),
    ],
    "ru": [
        ("introduction.html", "Введение в D.R.E.A.M"),
        ("kernel.html",       "Проекционное ядро"),
        ("retention.html",    "Скала удержания: как конечное разрешение формирует реальность"),
    ],
}

def render_articles_body(lang: str) -> str:
    items = ARTICLES[lang]
    if lang == "ru":
        title = "Статьи"
        sub = "Просмотрите статьи ниже."
        empty = "Статей пока нет."
    else:
        title = "Articles"
        sub = "Browse the articles below."
        empty = "No articles yet."
    cards = []
    for fname, t in items:
        cards.append(
            f'    <a class="card doc-card" href="articles/{fname}">\n'
            f'      <div class="card-body">\n'
            f'        <div class="doc-title">{t}</div>\n'
            f'      </div>\n'
            f'      <div class="card-tail">{fname}</div>\n'
            f'    </a>'
        )
    return f'''<div class="container">
  <section class="content-section">
    <div class="section-header">
      <h1 class="section-title">{title}</h1>
      <p class="muted">{sub}</p>
    </div>
    <div class="cards-grid">
{chr(10).join(cards)}
    </div>
  </section>
</div>

<style>
  .content-section {{ display:grid; gap:12px; }}
  .section-header {{ display:grid; gap:4px; }}
  .section-title {{ margin:0; font-size:22px; font-weight:800; }}
</style>'''

# ---------------------------------------------------------------------------
# Articles (the actual article pages under /articles/)
# ---------------------------------------------------------------------------

ARTICLE_TITLES = {
    ("en", "introduction.html"): "Introduction to D.R.E.A.M",
    ("en", "kernel.html"):       "The Projection Kernel",
    ("en", "retention.html"):    "The Retention Cliff",
    ("ru", "introduction.html"): "Введение в D.R.E.A.M",
    ("ru", "kernel.html"):       "Проекционное ядро",
    ("ru", "retention.html"):    "Скала удержания",
}

# articles include a header via templates/articles.html — but the body content
# is in templates/i18n/{lang}/articles/<slug>.html and we just need to inline it.

ARTICLES_WRAPPER_RE = re.compile(
    r"\{%\s*block\s+content\s*%\}(.*?)\{%\s*endblock\s*%\}",
    flags=re.DOTALL,
)

# templates/articles.html wrapper — extract just the article body include
def render_article_page(lang: str, slug: str) -> tuple[str, str] | None:
    """Return (title, body_html) for an article, or None if missing.

    Article body files are complete HTML documents (DOCTYPE, <html>, <head>,
    <body>...). We extract just the <body> innerHTML so it inlines cleanly
    into our base template. Any <style> blocks in the article's <head> are
    preserved (they get transplanted into the body, which browsers tolerate).
    """
    body_path = TPL_ROOT / "i18n" / lang / "articles" / slug
    if not body_path.exists():
        return None
    raw = body_path.read_text(encoding="utf-8")
    raw = inline_partials(raw)
    raw = strip_jinja(raw, lang)

    # Pull <style>...</style> blocks out of the article's <head> so we can
    # re-inject them next to the body content (preserves article-specific CSS).
    head_match = re.search(r"<head[^>]*>(.*?)</head>", raw, flags=re.DOTALL | re.IGNORECASE)
    styles = []
    if head_match:
        head = head_match.group(1)
        for s in re.findall(r"<style[^>]*>.*?</style>", head, flags=re.DOTALL | re.IGNORECASE):
            styles.append(s)

    # Extract just the <body>...</body> inner content.
    body_match = re.search(r"<body[^>]*>(.*?)</body>", raw, flags=re.DOTALL | re.IGNORECASE)
    if body_match:
        body = body_match.group(1).strip()
    else:
        # No <body> tag — use the raw content as-is.
        body = raw.strip()

    # Re-inject any preserved <style> blocks at the top of the body content.
    if styles:
        body = "\n".join(styles) + "\n" + body

    title = ARTICLE_TITLES.get((lang, slug), slug.replace("-", " ").title())
    return (title, body)

# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def page_title(lang: str, key: str) -> str:
    labels = {
        "home":         "DREAM — Dimensional Resonant Emergent Attractors in Manifold",
        "case":         "Case for D.R.E.A.M",
        "retention":    "Retention Law · D.R.E.A.M",
        "axioms":       "Axioms · D.R.E.A.M",
        "theorems":     "Theorems · D.R.E.A.M",
        "math":         "Mathematical Frame · D.R.E.A.M",
        "kernel":       "Projection Kernel · D.R.E.A.M",
        "topology":     "Topology · D.R.E.A.M",
        "spectrum":     "Fractal Spectrum · D.R.E.A.M",
        "predictions":  "Predictions · D.R.E.A.M",
        "falsification":"Falsification · D.R.E.A.M",
        "faq":          "FAQ · D.R.E.A.M",
        "about":        "AI Analysis · D.R.E.A.M",
        "time":         "Time · D.R.E.A.M",
        "memory":       "Memory · D.R.E.A.M",
        "articles":     "Articles · D.R.E.A.M",
    }
    ru_titles = {
        "home":         "DREAM — размерностные резонансные эмерджентные аттракторы в многообразии",
        "case":         "Сила D.R.E.A.M",
        "retention":    "Принцип Сохранения · D.R.E.A.M",
        "axioms":       "Аксиомы · D.R.E.A.M",
        "theorems":     "Теоремы · D.R.E.A.M",
        "math":         "Матмодель · D.R.E.A.M",
        "kernel":       "Проекция · D.R.E.A.M",
        "topology":     "Топология · D.R.E.A.M",
        "spectrum":     "Фрактальность · D.R.E.A.M",
        "predictions":  "Предсказания · D.R.E.A.M",
        "falsification":"Фальсификация · D.R.E.A.M",
        "faq":          "Вопрос-Ответ · D.R.E.A.M",
        "about":        "Оценка ИИ · D.R.E.A.M",
        "time":         "Время · D.R.E.A.M",
        "memory":       "Память · D.R.E.A.M",
        "articles":     "Статьи · D.R.E.A.M",
    }
    if lang == "ru":
        return ru_titles.get(key, labels.get(key, "D.R.E.A.M"))
    return labels.get(key, "D.R.E.A.M")

def build_page(lang: str, key: str) -> None:
    slug = PAGE_FILE[key]
    out_path = DST_ROOT / lang / f"{slug}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if key == "articles":
        body_html = render_articles_body(lang)
    else:
        body_path = TPL_ROOT / "i18n" / lang / f"{key}_body.html"
        if not body_path.exists():
            # fall back to EN body (matches the Flask app's
            # ['i18n/<lang>/X_body.html', 'i18n/en/X_body.html'] include list).
            body_path = TPL_ROOT / "i18n" / "en" / f"{key}_body.html"
            if not body_path.exists():
                print(f"  ! missing: {body_path}")
                return
            print(f"  (note: {key} has no RU body — using EN as fallback)")
        body = body_path.read_text(encoding="utf-8")
        body = inline_partials(body)
        # weave_embed needs special rewriting (no backend)
        if key == "kernel":
            body = rewrite_weave_embed(body, lang)
        body = strip_jinja(body, lang)
        body_html = body.strip()

    title = page_title(lang, key)
    html = base_template(lang, key, title, body_html, slug)
    out_path.write_text(html, encoding="utf-8")
    print(f"  ✓ {out_path.relative_to(DST_ROOT)}  ({len(html):,} bytes)")

def build_articles_subpages(lang: str) -> None:
    out_dir = DST_ROOT / lang / "articles"
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug in ["introduction.html", "kernel.html", "retention.html"]:
        result = render_article_page(lang, slug)
        if not result:
            print(f"  ! missing article: {lang}/articles/{slug}")
            continue
        title, body_html = result
        # article wrapper: same base template, but depth=1 (one ../ to language root)
        html = base_template(lang, "articles", f"{title} · D.R.E.A.M", body_html, slug, depth=1)
        out_path = out_dir / slug
        out_path.write_text(html, encoding="utf-8")
        print(f"  ✓ {out_path.relative_to(DST_ROOT)}  ({len(html):,} bytes)")

def main():
    if not TPL_ROOT.exists():
        print(f"Source templates not found at {TPL_ROOT}", file=sys.stderr)
        sys.exit(1)
    print("Building static DREAM site into", DST_ROOT)
    for lang in ("en", "ru"):
        print(f"\n=== {lang.upper()} pages ===")
        for key in PAGE_FILE:
            build_page(lang, key)
        print(f"\n=== {lang.upper()} article subpages ===")
        build_articles_subpages(lang)
    print("\nDone.")

if __name__ == "__main__":
    main()
