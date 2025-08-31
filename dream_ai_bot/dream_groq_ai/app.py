# app.py — GitHub-only FAQs + Groq RAG
import os, re, json, time, base64, urllib.request as _urlreq, urllib.error as _urlerr
import threading, webbrowser
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# ===== env =====
load_dotenv(find_dotenv() or "")
load_dotenv()

# ===== Groq bot =====
from groq_bot import warm_index, groq_answer, reload_index
from faq_parser import load_faqs  # used only for /api/rag-status counts if you keep a local file

# =====================================================================================
# App + GA
# =====================================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["GA_MEASUREMENT_ID"] = os.getenv("GA_MEASUREMENT_ID", "").strip()

@app.context_processor
def _inject_ga():
    return {"GA_MEASUREMENT_ID": app.config.get("GA_MEASUREMENT_ID", "")}

# =====================================================================================
# Paths (PDF for RAG only; FAQs come from GitHub)
# =====================================================================================
PROJECT_ROOT   = Path(__file__).resolve().parent
KB_DIR         = PROJECT_ROOT / "kb"
PDF_PATH       = str(KB_DIR / "dream_faqs.pdf")   # optional for RAG
LOCAL_JSON_EN  = str(KB_DIR / "parsed_faqs_en.json")  # not used for reads; only for status if exists
LOCAL_JSON_RU  = str(KB_DIR / "parsed_faqs_ru.json")  # not used for reads; only for status if exists

def _mtime(p: str) -> float:
    try: return os.path.getmtime(p)
    except Exception: return 0.0

# =====================================================================================
# GitHub kb-data (SOURCE OF TRUTH — no local fallback)
# =====================================================================================
GH_OWNER   = os.getenv("GH_OWNER",  "dream-framework")
GH_REPO    = os.getenv("GH_REPO",   "dream")
GH_BRANCH  = os.getenv("GH_BRANCH", "kb-data")
# IMPORTANT: these must match the repo exactly (see your screenshot)
GH_JSON_EN = (os.getenv("GH_JSON_EN") or "kb/parsed_faqs_en.json").lstrip("/")
GH_JSON_RU = (os.getenv("GH_JSON_RU") or "kb/parsed_faqs_ru.json").lstrip("/")
GH_TOKEN   = os.getenv("GH_TOKEN",  "").strip()  # PAT with contents read/write

def _gh_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"token {GH_TOKEN}"  # PAT scheme for REST v3
    return h

def _gh_contents_url(path_in_repo: str, ref: Optional[str] = None) -> str:
    path_in_repo = path_in_repo.lstrip("/")
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}"
    if ref:
        url += f"?ref={ref}"
    return url

def _gh_get_file(path_in_repo: str) -> Dict[str, Any]:
    req = _urlreq.Request(_gh_contents_url(path_in_repo, GH_BRANCH), headers=_gh_headers())
    with _urlreq.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))

def _gh_get_file_bytes(path_in_repo: str) -> bytes:
    """
    Strict GitHub read. Raises on any problem.
    """
    try:
        payload = _gh_get_file(path_in_repo)
    except _urlerr.HTTPError as e:
        raise RuntimeError(f"GitHub GET {path_in_repo} -> {e.code} {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"GitHub GET {path_in_repo} error: {e}") from e

    if isinstance(payload, list):
        # you pointed to a directory instead of the file
        raise RuntimeError(f"GitHub path '{path_in_repo}' is a directory, expected file")

    content_b64 = payload.get("content")
    encoding = payload.get("encoding")
    if not content_b64 or encoding != "base64":
        raise RuntimeError(f"GitHub file '{path_in_repo}' has no base64 content")
    try:
        # base64 content may contain newlines
        return base64.b64decode(content_b64.encode("ascii"))
    except Exception as e:
        raise RuntimeError(f"Base64 decode failed for '{path_in_repo}': {e}") from e

def _normalize_faq_json(raw_bytes: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(raw_bytes.decode("utf-8", "ignore"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON in GH file: {e}") from e

    # Accept a few shapes: {"faqs":[...]}, {"items":[...]}, or top-level list
    faqs = []
    if isinstance(obj, dict):
        if "faqs" in obj and isinstance(obj["faqs"], list):
            faqs = obj["faqs"]
        elif "items" in obj and isinstance(obj["items"], list):
            faqs = obj["items"]
        else:
            raise RuntimeError("JSON object has no 'faqs' or 'items' array")
    elif isinstance(obj, list):
        faqs = obj
    else:
        raise RuntimeError("Unexpected JSON structure for FAQs")

    # Normalize each entry
    out: List[Dict[str, Any]] = []
    for f in faqs:
        if not isinstance(f, dict):
            continue
        n = int(f.get("number") or 0)
        q = f.get("question", "") or ""
        a = f.get("answer", "") or ""
        out.append({"number": n, "question": q, "answer": a})
    out.sort(key=lambda x: x["number"])
    return {"metadata": obj.get("metadata", {}), "faqs": out}

def gh_load_lang_json(lang: str) -> Dict[str, Any]:
    path = GH_JSON_RU if lang == "ru" else GH_JSON_EN
    raw = _gh_get_file_bytes(path)
    data = _normalize_faq_json(raw)
    # Hard guarantee: do not allow silent empty sets
    if not data["faqs"]:
        raise RuntimeError(f"GitHub file '{path}' contained 0 FAQ items")
    app.logger.info("[kb] loaded %s FAQs from %s (%d items)", lang, path, len(data["faqs"]))
    return data

def _gh_current_sha(path_in_repo: str) -> Optional[str]:
    try:
        payload = _gh_get_file(path_in_repo)
        if isinstance(payload, dict):
            return payload.get("sha")
    except Exception:
        pass
    return None

def push_kb_to_github(path_in_repo: str, payload: Dict[str, Any], message: str) -> bool:
    """
    Strict write back to the exact path in the kb-data branch.
    """
    if not GH_TOKEN:
        raise RuntimeError("GH_TOKEN is not set — cannot push to GitHub")

    body = {
        "message": message,
        "branch": GH_BRANCH,
        "content": base64.b64encode(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii"),
    }
    sha = _gh_current_sha(path_in_repo)
    if sha:
        body["sha"] = sha

    put_url = _gh_contents_url(path_in_repo)
    req = _urlreq.Request(put_url, data=json.dumps(body).encode("utf-8"),
                          method="PUT", headers=_gh_headers())
    try:
        with _urlreq.urlopen(req, timeout=25) as r:
            _ = r.read()
        app.logger.info("[kb] pushed %s (%d items)", path_in_repo, len((payload or {}).get("faqs", [])))
        return True
    except Exception as e:
        raise RuntimeError(f"GitHub push error for '{path_in_repo}': {e}") from e

# =====================================================================================
# i18n
# =====================================================================================
LANGS = ["en", "ru"]
I18N_DIR = PROJECT_ROOT / "i18n"
TX, MT = {l: {} for l in LANGS}, {l: 0 for l in LANGS}

def _load_i18n(lang: str):
    p = I18N_DIR / f"{lang}.json"
    if p.exists():
        TX[lang] = json.loads(p.read_text(encoding="utf-8"))
        MT[lang] = p.stat().st_mtime
    else:
        TX[lang], MT[lang] = {}, 0

for _l in LANGS: _load_i18n(_l)

def _refresh_i18n():
    for l in LANGS:
        p = I18N_DIR / f"{l}.json"
        if p.exists() and p.stat().st_mtime != MT.get(l, 0):
            _load_i18n(l)

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

# =====================================================================================
# Navigation + pages
# =====================================================================================
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

app.url_map.strict_slashes = False

# =====================================================================================
# Admin (GET UI only; CRUD uses GH as source of truth)
# =====================================================================================
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "January@1")

def _check_admin():
    if not ADMIN_TOKEN:
        return
    token = request.headers.get("X-Admin-Token") or request.args.get("token")
    if token != ADMIN_TOKEN:
        abort(401)

@app.route("/admin", methods=["GET"])
@app.route("/admin/", methods=["GET"])
def admin_page():
    _check_admin()
    return render_template("admin.html", title="Admin FAQs")

# =====================================================================================
# Simple FAQ search (GitHub JSON ONLY)
# =====================================================================================
@app.post("/chat")
def chat():
    user_message = (request.json.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})
    lang = get_lang()
    try:
        data = gh_load_lang_json(lang)
        faqs = data.get("faqs", [])
    except Exception as e:
        return jsonify({"reply": f"FAQ load error from GitHub: {e}"})

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

# =====================================================================================
# Status (only reports counts if local files happen to exist)
# =====================================================================================
@app.get("/api/rag-status")
def rag_status():
    def count(local_path):
        try:
            if os.path.exists(local_path):
                return len(load_faqs(local_path))
        except Exception:
            pass
        return 0
    return jsonify({
        "pdf_exists": os.path.exists(PDF_PATH),
        "pdf_mtime": _mtime(PDF_PATH),
        "en_cache_exists": os.path.exists(LOCAL_JSON_EN),
        "ru_cache_exists": os.path.exists(LOCAL_JSON_RU),
        "en_count": count(LOCAL_JSON_EN),
        "ru_count": count(LOCAL_JSON_RU),
        "json_en_repo": GH_JSON_EN,
        "json_ru_repo": GH_JSON_RU,
        "branch": GH_BRANCH,
        "repo": f"{GH_OWNER}/{GH_REPO}",
    })

# =====================================================================================
# Groq endpoints (unchanged)
# =====================================================================================
@app.post("/groq-chat")
def groq_chat():
    payload = request.get_json(force=True, silent=True) or {}
    q = (payload.get("message") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Empty message."})
    try:
        ans = groq_answer(q, top_k=6)
        return jsonify({"ok": True, "reply": ans})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/groq/ask")
def groq_ask():
    data = request.get_json(force=True) or {}
    q = (data.get("message") or "").strip()
    history = data.get("history") or []
    ans = groq_answer(q, history=history, lang=get_lang())
    return jsonify({"ok": True, "answer": ans})

# =====================================================================================
# /faq API — ALWAYS reads/writes GitHub
# =====================================================================================
def _lang_from_request() -> str:
    lang = request.args.get("lang", "").lower() or (request.json or {}).get("lang", "")
    return "ru" if lang == "ru" else "en"

@app.get("/faq/list")
def faq_list():
    lang = _lang_from_request()
    try:
        data = gh_load_lang_json(lang)
        items = [{"number": int(f["number"]), "question": f["question"], "answer": f["answer"]}
                 for f in data["faqs"]]
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "items": [], "error": f"GitHub fetch failed: {e}"})

@app.get("/faq/item/<int:number>")
def faq_item(number):
    lang = _lang_from_request()
    try:
        data = gh_load_lang_json(lang)
        for f in data["faqs"]:
            if int(f["number"]) == number:
                return jsonify({"ok": True, "item": f})
        return jsonify({"ok": False, "error": "Not found"}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": f"GitHub fetch failed: {e}"}), 502

@app.post("/faq/save")
def faq_save():
    _check_admin()
    body = request.get_json(force=True) or {}
    lang = "ru" if (body.get("lang") or "en").lower() == "ru" else "en"
    item = body.get("item") or {}
    n = int(item.get("number") or 0)
    if n <= 0:
        return jsonify({"ok": False, "error": "Invalid number"}), 400
    try:
        data = gh_load_lang_json(lang)
        faqs = data["faqs"]
        updated = False
        for f in faqs:
            if int(f["number"]) == n:
                f["question"] = item.get("question","")
                f["answer"]   = item.get("answer","")
                updated = True
                break
        if not updated:
            faqs.append({"number": n, "question": item.get("question",""), "answer": item.get("answer","")})
        faqs.sort(key=lambda x: int(x["number"]))
        meta = data.setdefault("metadata", {})
        meta["lang"] = lang
        meta["total_faqs"] = len(faqs)
        meta["parsed_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        path = GH_JSON_RU if lang == "ru" else GH_JSON_EN
        push_kb_to_github(path, data, f"admin: save {lang.upper()} FAQ #{n}")
        reload_index(force=True)  # if you later blend JSON into retrieval
        return jsonify({"ok": True, "number": n})
    except Exception as e:
        return jsonify({"ok": False, "error": f"GitHub save failed: {e}"}), 502

@app.delete("/faq/delete/<int:number>")
def faq_delete(number):
    _check_admin()
    lang = _lang_from_request()
    try:
        data = gh_load_lang_json(lang)
        before = list(data["faqs"])
        after = [f for f in before if int(f["number"]) != number]
        if len(after) == len(before):
            return jsonify({"ok": False, "error": "Not found"}), 404
        data["faqs"] = after
        meta = data.setdefault("metadata", {})
        meta["lang"] = lang
        meta["total_faqs"] = len(after)
        meta["parsed_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        path = GH_JSON_RU if lang == "ru" else GH_JSON_EN
        push_kb_to_github(path, data, f"admin: delete {lang.upper()} FAQ #{number}")
        reload_index(force=True)
        return jsonify({"ok": True, "deleted": number})
    except Exception as e:
        return jsonify({"ok": False, "error": f"GitHub delete failed: {e}"}), 502

# =====================================================================================
# Boot
# =====================================================================================
_BOOT_DONE = False
_BOOT_LOCK = threading.Lock()

def _boot_once():
    global _BOOT_DONE
    with _BOOT_LOCK:
        if _BOOT_DONE:
            return
        # Warm Groq retriever (if PDF present)
        try:
            if os.path.exists(PDF_PATH):
                warm_index(PDF_PATH)
            else:
                warm_index()
        except Exception as e:
            app.logger.warning("[rag] warm_index failed: %s", e)
        _BOOT_DONE = True
        app.logger.info("[boot] Ready (GitHub-only FAQs).")

_boot_once()

# For local dev convenience
if __name__ == "__main__":
    try:
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    except Exception:
        pass
    app.run(debug=True, use_reloader=False)