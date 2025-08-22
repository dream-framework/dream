import os, json, threading, webbrowser
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for
from jinja2 import TemplateNotFound

app = Flask(__name__, template_folder="templates", static_folder="static")

# ------------------------------------------------------------------
# i18n (JSON-based)
# ------------------------------------------------------------------
LANGS = ["en", "ru"]
I18N_DIR = Path("i18n")
TX = {l: {} for l in LANGS}
MT = {l: 0 for l in LANGS}

def _load(lang):
    p = I18N_DIR / f"{lang}.json"
    if p.exists():
        TX[lang] = json.loads(p.read_text(encoding="utf-8"))
        MT[lang] = p.stat().st_mtime
    else:
        TX[lang] = {}
        MT[lang] = 0

for l in LANGS:
    _load(l)

def _refresh():
    for l in LANGS:
        p = I18N_DIR / f"{l}.json"
        if p.exists() and p.stat().st_mtime != MT.get(l, 0):
            _load(l)

def get_lang():
    lang = request.cookies.get("lang", "en")
    return lang if lang in LANGS else "en"

def t(key):
    _refresh()
    def lookup(lang):
        val = TX.get(lang, {})
        for part in key.split("."):
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return None
        return val
    v = lookup(get_lang())
    return v if v is not None else (lookup("en") or key)

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------
NAV = [
    ("home", "nav.home", "/"),
    ("overview", "nav.overview", "/overview"),
    ("axioms", "nav.axioms", "/axioms"),
    ("math", "nav.math", "/math"),
    ("kernel", "nav.kernel", "/kernel"),
    ("topology", "nav.topology", "/topology"),
    ("spectrum", "nav.spectrum", "/spectrum"),
    ("predictions", "nav.predictions", "/predictions"),
    ("falsification", "nav.falsification", "/falsification"),
    ("faq", "nav.faq", "/faq"),
    ("about", "nav.about", "/about"),
]

@app.context_processor
def inject():
    return dict(_=t, NAV=NAV, LANGS=LANGS, lang=get_lang())

# ------------------------------------------------------------------
# Helper to render localized templates
# ------------------------------------------------------------------
def render_lang(template_name, **context):
    lang = get_lang()
    try:
        return render_template(f"{lang}/{template_name}", **context)
    except TemplateNotFound:
        return render_template(template_name, **context)  # fallback to default

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/setlang/<lang_code>")
def setlang(lang_code):
    resp = redirect(request.referrer or url_for("home"))
    if lang_code in LANGS:
        resp.set_cookie("lang", lang_code, max_age=60 * 60 * 24 * 365)
    return resp

@app.route("/")
def home():
    return render_lang("home.html", title=t("meta.title"))

@app.route("/overview")
def overview():
    return render_lang("overview.html", title=t("nav.overview"))

@app.route("/axioms")
def axioms():
    return render_lang("axioms.html", title=t("nav.axioms"))

@app.route("/math")
def math():
    return render_lang("math.html", title=t("nav.math"))

@app.route("/kernel")
def kernel():
    return render_lang("kernel.html", title=t("nav.kernel"))

@app.route("/topology")
def topology():
    return render_lang("topology.html", title=t("nav.topology"))

@app.route("/spectrum")
def spectrum():
    return render_lang("spectrum.html", title=t("nav.spectrum"))

@app.route("/predictions")
def predictions():
    return render_lang("predictions.html", title=t("nav.predictions"))

@app.route("/falsification")
def falsification():
    return render_lang("falsification.html", title=t("nav.falsification"))

@app.route("/faq")
def faq():
    return render_lang("faq.html", title=t("nav.faq"))

@app.route("/about")
def about():
    return render_lang("about.html", title=t("nav.about"))

# ------------------------------------------------------------------
if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, use_reloader=False)
