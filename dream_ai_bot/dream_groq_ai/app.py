# app.py — Render-ready, pulls/pushes FAQs from GitHub kb-data, hot-reloads Groq RAG

import os, re, json, time, base64, urllib.request, urllib.error, threading, webbrowser
from pathlib import Path
from copy import deepcopy
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from flask_cors import CORS

# === .env ===
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv() or "")
load_dotenv()  # second call is harmless; keeps local dev flexible

# === KB / Groq ===
from faq_api import faq_bp
from faq_parser import build_bilingual_jsons, load_faqs
from groq_bot import warm_index, groq_answer, reload_index

# -----------------------------------------------------------------------------------
# App + GA
# -----------------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

app.config["GA_MEASUREMENT_ID"] = os.getenv("GA_MEASUREMENT_ID", "").strip()

@app.context_processor
def _inject_ga():
    return {"GA_MEASUREMENT_ID": app.config.get("GA_MEASUREMENT_ID", "")}

# -----------------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
KB_DIR       = PROJECT_ROOT / "kb"
PDF_PATH     = str(KB_DIR / "dream_faqs.pdf")
JSON_EN_PATH = str(KB_DIR / "parsed_faqs_en.json")
JSON_RU_PATH = str(KB_DIR / "parsed_faqs_ru.json")

def _mtime(p: str) -> float:
    try: return os.path.getmtime(p)
    except Exception: return 0.0

def _ensure_json_files_exist():
    os.makedirs(KB_DIR, exist_ok=True)
    for path, lang in [(JSON_EN_PATH, "en"), (JSON_RU_PATH, "ru")]:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"metadata": {"lang": lang, "total_faqs": 0}, "faqs": []},
                          f, ensure_ascii=False, indent=2)

# -----------------------------------------------------------------------------------
# Build from PDF (fallback only)
# -----------------------------------------------------------------------------------
def ensure_bilingual_cache():
    """
    Fallback builder: if JSONs are missing, or PDF is newer than either JSON.
    In the normal flow the KB is fed from GitHub; this only runs when needed.
    """
    _ensure_json_files_exist()
    pdf_m, en_m, ru_m = _mtime(PDF_PATH), _mtime(JSON_EN_PATH), _mtime(JSON_RU_PATH)
    needs = (not os.path.exists(JSON_EN_PATH) or
             not os.path.exists(JSON_RU_PATH) or
             (pdf_m and (pdf_m > en_m or pdf_m > ru_m)))
    if not os.path.exists(PDF_PATH):
        return
    if needs:
        try:
            build_bilingual_jsons(PDF_PATH, JSON_EN_PATH, JSON_RU_PATH)
        except Exception:
            # fallback: copy EN to RU so the UI remains usable
            from faq_parser import parse_faq_pdf, save_faqs_to_json_lang
            faqs = parse_faq_pdf(PDF_PATH)
            save_faqs_to_json_lang(faqs, "en", JSON_EN_PATH)
            save_faqs_to_json_lang(faqs, "ru", JSON_RU_PATH)

# -----------------------------------------------------------------------------------
# GitHub kb-data integration (pull/push with stdlib)
# -----------------------------------------------------------------------------------
GH_OWNER   = os.getenv("GH_OWNER",  "dream-framework")
GH_REPO    = os.getenv("GH_REPO",   "dream")
GH_BRANCH  = os.getenv("GH_BRANCH", "kb-data")
GH_JSON_EN = os.getenv("GH_JSON_EN","kb/parsed_faqs_en.json")
GH_JSON_RU = os.getenv("GH_JSON_RU","kb/parsed_faqs_ru.json")
GH_TOKEN   = os.getenv("GH_TOKEN",  "")  # fine-grained PAT, Contents: read & write

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        # REST v3 prefers 'token' scheme (not 'Bearer')
        h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _gh_get_contents(path_in_repo: str) -> bytes | None:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}?ref={GH_BRANCH}"
    req = urllib.request.Request(url, headers=_gh_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = json.loads(r.read().decode("utf-8"))
            b64 = payload.get("content")
            return base64.b64decode(b64) if b64 else None
    except urllib.error.HTTPError as e:
        print(f"[kb] GET {path_in_repo} -> {e.code}")
    except Exception as e:
        print(f"[kb] GET {path_in_repo} error:", e)
    return None

def pull_kb_from_github() -> bool:
    """
    Pull EN/RU JSONs from kb-data into local kb/, return True if any changed.
    """
    os.makedirs(KB_DIR, exist_ok=True)
    changed = False
    for remote, local in [(GH_JSON_EN, JSON_EN_PATH), (GH_JSON_RU, JSON_RU_PATH)]:
        blob = _gh_get_contents(remote)
        if not blob:
            continue
        old = open(local, "rb").read() if os.path.exists(local) else None
        if old != blob:
            with open(local, "wb") as f:
                f.write(blob)
            changed = True
            print(f"[kb] updated {local} from {GH_BRANCH}:{remote}")
    return changed

def _gh_current_sha(path_in_repo: str) -> str | None:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}?ref={GH_BRANCH}"
    req = urllib.request.Request(url, headers=_gh_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8")).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"[kb] sha {path_in_repo} -> {e.code}")
    except Exception as e:
        print(f"[kb] sha error:", e)
    return None

def push_kb_to_github(path_in_repo: str, raw_bytes: bytes, message: str) -> bool:
    if not GH_TOKEN:
        print("[kb] GH_TOKEN missing; skip push")
        return False
    body = {
        "message": message,
        "branch": GH_BRANCH,
        "content": base64.b64encode(raw_bytes).decode("ascii"),
    }
    sha = _gh_current_sha(path_in_repo)
    if sha: body["sha"] = sha
    put_url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}"
    req = urllib.request.Request(put_url, data=json.dumps(body).encode("utf-8"),
                                 method="PUT", headers=_gh_headers())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            _ = r.read()
            print(f"[kb] pushed {path_in_repo} to {GH_BRANCH}")
            return True
    except Exception as e:
        print(f"[kb] push error ({path_in_repo}):", e)
        return False

def start_kb_poll(interval_sec: int = 180):
    """Background poller: pulls from kb-data and reloads retriever when changed."""
    def _loop():
        while True:
            try:
                if pull_kb_from_github():
                    reload_index(force=True)
            except Exception as e:
                print("[kb] poll error:", e)
            time.sleep(max(30, interval_sec))
    threading.Thread(target=_loop, daemon=True).start()

# -----------------------------------------------------------------------------------
# i18n
# -----------------------------------------------------------------------------------
LANGS = ["en", "ru"]
I18N_DIR = PROJECT_ROOT / "i18n"
TX, MT = {l: {} for l in LANGS}, {l: 0 for l in LANGS}

def _load(lang: str):
    p = I18N_DIR / f"{lang}.json"
    if p.exists():
        TX[lang] = json.loads(p.read_text(encoding="utf-8"))
        MT[lang] = p.stat().st_mtime
    else:
        TX[lang], MT[lang] = {}, 0

for _l in LANGS: _load(_l)

def _refresh_i18n():
    for l in LANGS:
        p = I18N_DIR / f"{l}.json"
        if p.exists() and p.stat().st_mtime != MT.get(l, 0):
            _load(l)

def get_lang():
    lang = request.cookies.get("lang", "en")
    return lang if lang in LANGS else "en"

def t(key: str):
    _refresh_i18n()
    def lookup(lang):
        val = TX.get(lang, {})
        for part in key.split("."):
            if isinstance(val, dict) and part in val: val = val[part]
            else: return None
        return val
    v = lookup(get_lang())
    return v if v is not None else (lookup("en") or key)

def _json_for_lang(lang: str) -> str:
    return JSON_RU_PATH if lang == "ru" else JSON_EN_PATH

# -----------------------------------------------------------------------------------
# Hot reload of local KB files
# -----------------------------------------------------------------------------------
_KB_SIG = None
_KB_LOCK = threading.Lock()

def _sig_tuple() -> tuple:
    return (_mtime(PDF_PATH), _mtime(JSON_EN_PATH), _mtime(JSON_RU_PATH))

def _kb_refresh_if_stale():
    global _KB_SIG
    with _KB_LOCK:
        new_sig = _sig_tuple()
        if _KB_SIG is None:
            _KB_SIG = new_sig
            return
        pdf_m_new, en_m_new, ru_m_new = new_sig
        pdf_m_old, en_m_old, ru_m_old = _KB_SIG

        json_changed = (en_m_new != en_m_old) or (ru_m_new != ru_m_old)
        pdf_newer    = pdf_m_new and (pdf_m_new > en_m_new or pdf_m_new > ru_m_new)

        if pdf_newer:
            ensure_bilingual_cache()
            reload_index(force=True)
            _KB_SIG = _sig_tuple()
        elif json_changed:
            reload_index(force=True)
            _KB_SIG = new_sig

@app.before_request
def _before_request_hot_reload():
    _kb_refresh_if_stale()

# -----------------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------------
NAV = [
    ("home","nav.home","/"),
    ("overview","nav.overview","/overview"),
    ("axioms","nav.axioms","/axioms"),
    ("math","nav.math","/math"),
    ("kernel","nav.kernel","/kernel"),
    ("topology","nav.topology","/topology"),
    ("spectrum","nav.spectrum","/spectrum"),
    ("predictions","nav.predictions","/predictions"),
    ("falsification","nav.falsification","/falsification"),
    ("faq","nav.faq","/faq"),
    ("about","nav.about","/about"),
]

@app.context_processor
def inject():
    return dict(_=t, NAV=NAV, LANGS=LANGS, lang=get_lang())

# -----------------------------------------------------------------------------------
# Routes (pages)
# -----------------------------------------------------------------------------------
@app.get("/setlang/<lang_code>")
def setlang(lang_code):
    resp = redirect(request.referrer or url_for("home"))
    if lang_code in LANGS:
        resp.set_cookie("lang", lang_code, max_age=60*60*24*365)
    return resp

@app.route("/")
def home(): return render_template("home.html", title=t("meta.title"))

@app.route("/overview")
def overview(): return render_template("overview.html", title=t("nav.overview"))

@app.route("/axioms")
def axioms(): return render_template("axioms.html", title=t("nav.axioms"))

@app.route("/math")
def math(): return render_template("math.html", title=t("nav.math"))

@app.route("/kernel")
def kernel(): return render_template("kernel.html", title=t("nav.kernel"))

@app.route("/topology")
def topology(): return render_template("topology.html", title=t("nav.topology"))

@app.route("/spectrum")
def spectrum(): return render_template("spectrum.html", title=t("nav.spectrum"))

@app.route("/predictions")
def predictions(): return render_template("predictions.html", title=t("nav.predictions"))

@app.route("/falsification")
def falsification(): return render_template("falsification.html", title=t("nav.falsification"))

@app.route("/faq")
def faq(): return render_template("faq.html", title=t("nav.faq"))

@app.route("/about")
def about(): return render_template("about.html", title=t("nav.about"))

# -----------------------------------------------------------------------------------
# Simple FAQ search (bilingual JSONs)
# -----------------------------------------------------------------------------------
@app.post("/chat")
def chat():
    user_message = (request.json.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})
    try:
        lang = get_lang()
        faqs = load_faqs(_json_for_lang(lang)) if os.path.exists(_json_for_lang(lang)) else []
        qlower = user_message.lower()
        tokens = [t for t in re.findall(r"\w+", qlower) if len(t) > 2]
        best, score = None, 0
        for f in faqs:
            q = (f.get("question") or "").lower()
            a = (f.get("answer") or "").lower()
            s = sum(t in q for t in tokens)*2 + sum(t in a for t in tokens)
            if s > score:
                best, score = f, s
        if best and score > 0:
            return jsonify({"reply": f"**{best['question']}**\n\n{best['answer']}", "source": f"faq-{lang}"})
        return jsonify({"reply": "I couldn't find that in the FAQs. Try the FAQ Bot or rephrase your question."})
    except Exception:
        return jsonify({"reply": "Sorry, I encountered an error while processing your request."})

# -----------------------------------------------------------------------------------
# Status
# -----------------------------------------------------------------------------------
@app.get("/api/rag-status")
def rag_status():
    def count(p):
        try: return len(load_faqs(p)) if os.path.exists(p) else 0
        except Exception: return 0
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

# -----------------------------------------------------------------------------------
# Groq endpoints
# -----------------------------------------------------------------------------------
@app.post("/groq-chat")
def groq_chat():
    payload = request.get_json(force=True, silent=True) or {}
    q = (payload.get("message") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Empty message."}), 400
    try:
        ans = groq_answer(q, top_k=6)
        return jsonify({"ok": True, "reply": ans})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/groq/ask")
def groq_ask():
    data = request.get_json(force=True) or {}
    q = (data.get("message") or "").strip()
    history = data.get("history") or []
    ans = groq_answer(q, history=history, lang=get_lang())
    return jsonify({"ok": True, "answer": ans})

# -----------------------------------------------------------------------------------
# Admin (CRUD) — also pushes KB changes back to kb-data
# -----------------------------------------------------------------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "January@1")
SAVE_LOCK = threading.Lock()

def _check_admin():
    if not ADMIN_TOKEN: return
    token = request.headers.get("X-Admin-Token") or request.args.get("token")
    if token != ADMIN_TOKEN: abort(401)

def _write_json_atomic(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_en_ru():
    en = load_faqs(JSON_EN_PATH) if os.path.exists(JSON_EN_PATH) else []
    ru = load_faqs(JSON_RU_PATH) if os.path.exists(JSON_RU_PATH) else []
    return en, ru

def load_bilingual_flat():
    en, ru = _load_en_ru()
    by = {}
    for f in en:
        by[f["number"]] = {"number": f["number"], "question_en": f.get("question",""),
                           "answer_en": f.get("answer",""), "question_ru":"", "answer_ru":""}
    for f in ru:
        slot = by.setdefault(f["number"], {"number": f["number"], "question_en":"", "answer_en":"",
                                           "question_ru":"", "answer_ru":""})
        slot["question_ru"] = f.get("question",""); slot["answer_ru"] = f.get("answer","")
    return sorted(by.values(), key=lambda x: x["number"])

def save_bilingual_flat(items):
    """Write EN/RU JSONs locally, push to GitHub, and refresh retriever."""
    with SAVE_LOCK:
        faqs_en, faqs_ru = [], []
        for it in items:
            n = int(it["number"])
            faqs_en.append({"number": n, "question": it.get("question_en","") or "",
                            "answer": it.get("answer_en","") or "", "source": "admin_en"})
            faqs_ru.append({"number": n, "question": it.get("question_ru","") or "",
                            "answer": it.get("answer_ru","") or "", "source": "admin_ru"})
        meta_en = {"metadata": {"source": "Admin-edited", "lang": "en",
                                "total_faqs": len(faqs_en),
                                "parsed_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                   "faqs": sorted(faqs_en, key=lambda x: x["number"])}
        meta_ru = json.loads(json.dumps(meta_en, ensure_ascii=False))
        meta_ru["metadata"]["lang"] = "ru"
        meta_ru["faqs"] = sorted(faqs_ru, key=lambda x: x["number"])

        os.makedirs(KB_DIR, exist_ok=True)
        _write_json_atomic(JSON_EN_PATH, meta_en)
        _write_json_atomic(JSON_RU_PATH, meta_ru)

        # push to kb-data (best-effort)
        try:
            push_kb_to_github(GH_JSON_EN, json.dumps(meta_en, ensure_ascii=False, indent=2).encode("utf-8"),
                              "admin: update EN FAQs")
            push_kb_to_github(GH_JSON_RU, json.dumps(meta_ru, ensure_ascii=False, indent=2).encode("utf-8"),
                              "admin: update RU FAQs")
        except Exception as e:
            print("[kb] push skipped:", e)

    # retriever sees changes immediately
    global _KB_SIG
    _KB_SIG = _sig_tuple()
    reload_index(force=True)

@app.get("/admin")
def admin_page():
    _check_admin()
    return render_template("admin.html", title="Admin · FAQs")

@app.get("/admin/api/faqs")
def admin_list_faqs():
    _check_admin()
    return jsonify({"ok": True, "items": load_bilingual_flat()})

@app.get("/admin/api/faqs/<int:number>")
def admin_get_faq(number):
    _check_admin()
    for it in load_bilingual_flat():
        if it["number"] == number:
            return jsonify({"ok": True, "item": it})
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.post("/admin/api/faqs")
def admin_create_faq():
    _check_admin()
    data = request.get_json(force=True) or {}
    items = load_bilingual_flat()
    used = {it["number"] for it in items}
    number = int(data.get("number") or (max(used) + 1 if used else 1))
    if number in used:
        return jsonify({"ok": False, "error": "Number already exists"}), 400
    items.append({"number": number,
                  "question_en": data.get("question_en",""), "answer_en": data.get("answer_en",""),
                  "question_ru": data.get("question_ru",""), "answer_ru": data.get("answer_ru","")})
    save_bilingual_flat(items)
    return jsonify({"ok": True, "item": items[-1]})

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
    items = [it for it in load_bilingual_flat() if it["number"] != number]
    if len(items) == len(load_bilingual_flat()):
        return jsonify({"ok": False, "error": "Not found"}), 404
    save_bilingual_flat(items)
    return jsonify({"ok": True, "deleted": number})

# Mount FAQ API blueprint
app.register_blueprint(faq_bp, url_prefix="/faq")

# -----------------------------------------------------------------------------------
# Boot
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Pull from kb-data (preferred), else build from PDF
    pulled = pull_kb_from_github()
    if not pulled:
        ensure_bilingual_cache()

    # 2) Warm Groq retriever (PDF path is only for naming; groq_bot reads JSONs itself)
    warm_index("kb/dream_faqs.pdf")

    # 3) Init signature and start KB poller
    _KB_SIG = _sig_tuple()
    start_kb_poll(interval_sec=60)

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(debug=True, use_reloader=False)