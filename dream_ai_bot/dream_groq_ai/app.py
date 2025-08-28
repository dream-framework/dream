# app.py
import os, re, json, threading, webbrowser, time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from flask_cors import CORS

from copy import deepcopy

# === FAQ API (serves /faq/search and /faq/item) ===
from faq_api import faq_bp

# === Parser helpers (bilingual build + JSON loader) ===
from faq_parser import build_bilingual_jsons, load_faqs

from flask import request, jsonify
from groq_bot import warm_index, groq_answer

# .env loading (tries .env first; falls back to en.env)
from dotenv import load_dotenv, find_dotenv
_loaded = load_dotenv(find_dotenv())  # looks for .env up the tree
if not _loaded:
    load_dotenv("en.env")  # fallback if you keep your file named en.env
load_dotenv() 

# Warm the index once at startup (points to your DREAM PDF)
warm_index("kb/dream_faqs.pdf")  # or "kb/dream.pdf" if that’s your path



app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Google Analytics (GA4) support ---
import os as _os
try:
    app.config["GA_MEASUREMENT_ID"] = _os.getenv("GA_MEASUREMENT_ID", "").strip()
except Exception:
    app.config["GA_MEASUREMENT_ID"] = ""

@app.context_processor
def _inject_ga():
    return {"GA_MEASUREMENT_ID": app.config.get("GA_MEASUREMENT_ID", "")}


CORS(app)  # allow JS from all pages to call this API

# ------------------------------------------------------------------
# Knowledge Base locations (parser-based, no FAISS/vectorstore)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
KB_DIR       = PROJECT_ROOT / "kb"
PDF_PATH     = str(KB_DIR / "dream_faqs.pdf")

JSON_EN_PATH = str(KB_DIR / "parsed_faqs_en.json")
JSON_RU_PATH = str(KB_DIR / "parsed_faqs_ru.json")

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def ensure_bilingual_cache():
    """
    Build/refresh EN+RU caches when:
      - either JSON is missing, OR
      - PDF is newer than either JSON.
    Uses the parser's build_bilingual_jsons (translation mode set in env/CLI).
    """
    os.makedirs(KB_DIR, exist_ok=True)

    pdf_m = _mtime(PDF_PATH)
    en_m  = _mtime(JSON_EN_PATH)
    ru_m  = _mtime(JSON_RU_PATH)

    needs_build = (not os.path.exists(JSON_EN_PATH)
                   or not os.path.exists(JSON_RU_PATH)
                   or (pdf_m and (pdf_m > en_m or pdf_m > ru_m)))

    if not os.path.exists(PDF_PATH):
        # Create empty caches so endpoints don't 404
        for p, lang in [(JSON_EN_PATH, "en"), (JSON_RU_PATH, "ru")]:
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"metadata": {"lang": lang, "total_faqs": 0}, "faqs": []}, f, indent=2, ensure_ascii=False)
        return

    if needs_build:
        # Honour translation mode via env var FAQ_TRANS_MODE (off|google|marian)
        # Example: os.environ["FAQ_TRANS_MODE"] = "google"
        try:
            build_bilingual_jsons(PDF_PATH, JSON_EN_PATH, JSON_RU_PATH)
        except Exception:
            # Fallback: write EN to both, so the app still works
            from faq_parser import parse_faq_pdf, save_faqs_to_json_lang
            faqs = parse_faq_pdf(PDF_PATH)
            save_faqs_to_json_lang(faqs, "en", JSON_EN_PATH)
            save_faqs_to_json_lang(faqs, "ru", JSON_RU_PATH)

# ------------------------------------------------------------------
# i18n
# ------------------------------------------------------------------
LANGS = ["en", "ru"]
I18N_DIR = PROJECT_ROOT / "i18n"
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

def _json_for_lang(lang: str) -> str:
    return JSON_RU_PATH if lang == "ru" else JSON_EN_PATH

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
# Routes
# ------------------------------------------------------------------
@app.get("/setlang/<lang_code>")
def setlang(lang_code):
    resp = redirect(request.referrer or url_for("home"))
    if lang_code in LANGS:
        resp.set_cookie("lang", lang_code, max_age=60*60*24*365)
    return resp

@app.route("/")
def home():
    return render_template("home.html", title=t("meta.title"))

@app.route("/overview")
def overview():
    return render_template("overview.html", title=t("nav.overview"))

@app.route("/axioms")
def axioms():
    return render_template("axioms.html", title=t("nav.axioms"))

@app.route("/math")
def math():
    return render_template("math.html", title=t("nav.math"))

@app.route("/kernel")
def kernel():
    return render_template("kernel.html", title=t("nav.kernel"))

@app.route("/topology")
def topology():
    return render_template("topology.html", title=t("nav.topology"))

@app.route("/spectrum")
def spectrum():
    return render_template("spectrum.html", title=t("nav.spectrum"))

@app.route("/predictions")
def predictions():
    return render_template("predictions.html", title=t("nav.predictions"))

@app.route("/falsification")
def falsification():
    return render_template("falsification.html", title=t("nav.falsification"))

@app.route("/faq")
def faq():
    return render_template("faq.html", title=t("nav.faq"))

@app.route("/about")
def about():
    return render_template("about.html", title=t("nav.about"))

# -------------------- Simple chat over parsed FAQs (bilingual) ----------------
@app.route("/chat", methods=["POST"])
def chat():
    user_message = (request.json.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    try:
        lang = get_lang()
        json_path = _json_for_lang(lang)
        faqs = load_faqs(json_path) if os.path.exists(json_path) else []
        qlower = user_message.lower()
        tokens = [tok for tok in re.findall(r"\w+", qlower) if len(tok) > 2]

        best = None
        best_score = 0
        for f in faqs:
            q = (f.get("question") or "").lower()
            a = (f.get("answer") or "").lower()
            score = sum(tok in q for tok in tokens) * 2 + sum(tok in a for tok in tokens)
            if score > best_score:
                best_score = score
                best = f

        if best and best_score > 0:
            reply = f"**{best['question']}**\n\n{best['answer']}"
            return jsonify({"reply": reply, "source": f"faq-{lang}"})
        else:
            msg = "I couldn't find that in the FAQs. Open the FAQ Bot and try searching the titles, or rephrase your question."
            return jsonify({"reply": msg})

    except Exception:
        return jsonify({"reply": "Sorry, I encountered an error while processing your request."})

# ----------------------- Status endpoint --------------------------------------
@app.get("/api/rag-status")
def rag_status():
    def count(p):
        try:
            return len(load_faqs(p)) if os.path.exists(p) else 0
        except Exception:
            return 0
    return jsonify({
        "pdf_exists": os.path.exists(PDF_PATH),
        "pdf_mtime": _mtime(PDF_PATH),
        "en_json_exists": os.path.exists(JSON_EN_PATH),
        "ru_json_exists": os.path.exists(JSON_RU_PATH),
        "en_count": count(JSON_EN_PATH),
        "ru_count": count(JSON_RU_PATH),
        "json_en_path": JSON_EN_PATH,
        "json_ru_path": JSON_RU_PATH,
    })

@app.post("/groq-chat")
def groq_chat():
    payload = request.get_json(force=True, silent=True) or {}
    q = (payload.get("message") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Empty message."}), 400
    try:
        answer = groq_answer(q, top_k=6)
        return jsonify({"ok": True, "reply": answer})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
@app.post("/groq/ask")
def groq_ask():
    data = request.get_json(force=True) or {}
    q = (data.get("message") or "").strip()
    history = data.get("history") or []
    # use UI language when present; groq_answer will still auto-detect from text if needed
    lang = get_lang()
    ans = groq_answer(q, history=history, lang=lang)
    return jsonify({"ok": True, "answer": ans})

# --- add after JSON_EN_PATH / JSON_RU_PATH ---
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "January@1")  # set to something strong in prod
SAVE_LOCK = threading.Lock()

def _check_admin():
    if not ADMIN_TOKEN:
        return  # open dev mode
    token = request.headers.get("X-Admin-Token") or request.args.get("token")
    if token != ADMIN_TOKEN:
        abort(401)

def _write_json_atomic(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_en_ru():
    """Return (faqs_en, faqs_ru). Each is a list of {number, question, answer}."""
    en = load_faqs(JSON_EN_PATH) if os.path.exists(JSON_EN_PATH) else []
    ru = load_faqs(JSON_RU_PATH) if os.path.exists(JSON_RU_PATH) else []
    return en, ru

def load_bilingual_flat():
    """Merge EN+RU by number -> list of {number, question_en, answer_en, question_ru, answer_ru}."""
    en, ru = _load_en_ru()
    by_id = {}
    for f in en:
        by_id[f["number"]] = {
            "number": f["number"],
            "question_en": f.get("question",""),
            "answer_en":   f.get("answer",""),
            "question_ru": "",
            "answer_ru":   ""
        }
    for f in ru:
        slot = by_id.setdefault(f["number"], {
            "number": f["number"],
            "question_en": "", "answer_en":"", "question_ru":"", "answer_ru":""
        })
        slot["question_ru"] = f.get("question","")
        slot["answer_ru"]   = f.get("answer","")
    return sorted(by_id.values(), key=lambda x: x["number"])

def save_bilingual_flat(items):
    """Write EN & RU JSON files from merged list."""
    with SAVE_LOCK:
        # Build EN payload
        faqs_en = []
        faqs_ru = []
        for it in items:
            n = int(it["number"])
            faqs_en.append({
                "number": n,
                "question": it.get("question_en","") or "",
                "answer":   it.get("answer_en","") or "",
                "source":   "admin_en"
            })
            faqs_ru.append({
                "number": n,
                "question": it.get("question_ru","") or "",
                "answer":   it.get("answer_ru","") or "",
                "source":   "admin_ru"
            })

        meta_en = {
            "metadata": {
                "source": "Admin-edited",
                "lang": "en",
                "total_faqs": len(faqs_en),
                "parsed_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "faqs": sorted(faqs_en, key=lambda x: x["number"])
        }
        meta_ru = deepcopy(meta_en)
        meta_ru["metadata"]["lang"] = "ru"
        meta_ru["faqs"] = sorted(faqs_ru, key=lambda x: x["number"])

        os.makedirs(KB_DIR, exist_ok=True)
        _write_json_atomic(JSON_EN_PATH, meta_en)
        _write_json_atomic(JSON_RU_PATH, meta_ru)

# ---------- Admin UI ----------
@app.get("/admin")
def admin_page():
    _check_admin()
    # simple page; UI below in templates/admin.html
    return render_template("admin.html", title="Admin · FAQs")

# ---------- Admin API (CRUD) ----------
@app.get("/admin/api/faqs")
def admin_list_faqs():
    _check_admin()
    return jsonify({"ok": True, "items": load_bilingual_flat()})

@app.get("/admin/api/faqs/<int:number>")
def admin_get_faq(number):
    _check_admin()
    items = load_bilingual_flat()
    for it in items:
        if it["number"] == number:
            return jsonify({"ok": True, "item": it})
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.post("/admin/api/faqs")
def admin_create_faq():
    _check_admin()
    data = request.get_json(force=True) or {}
    items = load_bilingual_flat()
    used = {it["number"] for it in items}
    # assign next number if missing
    number = int(data.get("number") or (max(used) + 1 if used else 1))
    if number in used:
        return jsonify({"ok": False, "error": "Number already exists"}), 400
    new_item = {
        "number": number,
        "question_en": data.get("question_en",""),
        "answer_en":   data.get("answer_en",""),
        "question_ru": data.get("question_ru",""),
        "answer_ru":   data.get("answer_ru","")
    }
    items.append(new_item)
    save_bilingual_flat(items)
    return jsonify({"ok": True, "item": new_item})

@app.put("/admin/api/faqs/<int:number>")
def admin_update_faq(number):
    _check_admin()
    data = request.get_json(force=True) or {}
    items = load_bilingual_flat()
    for it in items:
        if it["number"] == number:
            it["question_en"] = data.get("question_en", it["question_en"])
            it["answer_en"]   = data.get("answer_en",   it["answer_en"])
            it["question_ru"] = data.get("question_ru", it["question_ru"])
            it["answer_ru"]   = data.get("answer_ru",   it["answer_ru"])
            save_bilingual_flat(items)
            return jsonify({"ok": True, "item": it})
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.delete("/admin/api/faqs/<int:number>")
def admin_delete_faq(number):
    _check_admin()
    items = load_bilingual_flat()
    new_items = [it for it in items if it["number"] != number]
    if len(new_items) == len(items):
        return jsonify({"ok": False, "error": "Not found"}), 404
    save_bilingual_flat(new_items)
    return jsonify({"ok": True, "deleted": number})

# ----------------------- Mount the FAQ API blueprint --------------------------
app.register_blueprint(faq_bp, url_prefix="/faq")

# ------------------------------------------------------------------
if __name__ == "__main__":
    ensure_bilingual_cache()  # make sure both JSONs exist / are fresh
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, use_reloader=False)