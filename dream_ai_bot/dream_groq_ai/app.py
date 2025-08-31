# app.py — GitHub-first FAQs + hot-reload Groq RAG
import os, re, json, time, base64, urllib.request as _urlreq, urllib.error as _urlerr
import threading, webbrowser
from pathlib import Path
from typing import Optional, Tuple
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# ===== env =====
load_dotenv(find_dotenv() or "")
load_dotenv()

# ===== Groq bot bits =====
from groq_bot import warm_index, groq_answer, reload_index
from faq_parser import build_bilingual_jsons, load_faqs  # for status / local cache introspection

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
# Paths (PDF only used by RAG; JSONs come from GitHub)
# =====================================================================================
PROJECT_ROOT   = Path(__file__).resolve().parent
KB_DIR         = PROJECT_ROOT / "kb"
PDF_PATH       = str(KB_DIR / "dream_faqs.pdf")
LOCAL_JSON_EN  = str(KB_DIR / "parsed_faqs_en.json")  # local cache (optional)
LOCAL_JSON_RU  = str(KB_DIR / "parsed_faqs_ru.json")  # local cache (optional)

def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except Exception:
        return 0.0

# =====================================================================================
# GitHub kb-data integration (SOURCE OF TRUTH)
# =====================================================================================
GH_OWNER   = os.getenv("GH_OWNER",  "dream-framework")
GH_REPO    = os.getenv("GH_REPO",   "dream")
GH_BRANCH  = os.getenv("GH_BRANCH", "kb-data")
GH_JSON_EN = os.getenv("GH_JSON_EN","kb/parsed_faqs_en.json")
GH_JSON_RU = os.getenv("GH_JSON_RU","kb/parsed_faqs_ru.json")
GH_TOKEN   = os.getenv("GH_TOKEN",  "").strip()  # PAT with contents read+write

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        h["Authorization"] = f"token {GH_TOKEN}"  # REST v3 PAT scheme
    return h

def _gh_contents_url(path_in_repo: str, ref: Optional[str] = None) -> str:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}"
    if ref:
        url += f"?ref={ref}"
    return url

def _gh_get_file_and_sha(path_in_repo: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Return (raw_bytes_or_None, sha_or_None) from GitHub contents API at GH_BRANCH."""
    req = _urlreq.Request(_gh_contents_url(path_in_repo, GH_BRANCH), headers=_gh_headers())
    try:
        with _urlreq.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read().decode("utf-8", "ignore"))
            content_b64 = payload.get("content")
            data = base64.b64decode(content_b64.encode("ascii")) if content_b64 else None
            return data, payload.get("sha")
    except _urlerr.HTTPError as e:
        if e.code != 404:
            app.logger.warning("[kb] GET %s -> %s", path_in_repo, e)
    except Exception as e:
        app.logger.warning("[kb] GET %s error: %s", path_in_repo, e)
    return None, None

def _gh_current_sha(path_in_repo: str) -> Optional[str]:
    try:
        _, sha = _gh_get_file_and_sha(path_in_repo)
        return sha
    except Exception:
        return None

def push_kb_to_github(path_in_repo: str, raw_bytes: bytes, message: str) -> bool:
    if not GH_TOKEN:
        app.logger.info("[kb] GH_TOKEN missing; skip push")
        return False
    body = {
        "message": message,
        "branch": GH_BRANCH,
        "content": base64.b64encode(raw_bytes).decode("ascii"),
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
            app.logger.info("[kb] pushed %s to %s", path_in_repo, GH_BRANCH)
            return True
    except Exception as e:
        app.logger.warning("[kb] push error (%s): %s", path_in_repo, e)
        return False

def _lang_to_repo_path(lang: str) -> str:
    return GH_JSON_RU if lang == "ru" else GH_JSON_EN

def _safe_json_parse(raw_bytes: Optional[bytes]) -> dict:
    if not raw_bytes:
        return {"metadata": {"lang":"en","total_faqs":0}, "faqs":[]}
    try:
        return json.loads(raw_bytes.decode("utf-8", "ignore"))
    except Exception:
        return {"metadata": {"lang":"en","total_faqs":0}, "faqs":[]}

def gh_load_lang_json(lang: str) -> dict:
    """ALWAYS hit GitHub for admin/FAQ list routes; also refresh local cache."""
    path = _lang_to_repo_path(lang)
    raw, _sha = _gh_get_file_and_sha(path)
    data = _safe_json_parse(raw)
    try:
        os.makedirs(KB_DIR, exist_ok=True)
        cache_path = LOCAL_JSON_RU if lang == "ru" else LOCAL_JSON_EN
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data

def gh_save_lang_json(lang: str, payload: dict, commit_msg: str) -> bool:
    path = _lang_to_repo_path(lang)
    ok = push_kb_to_github(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), commit_msg)
    try:
        cache_path = LOCAL_JSON_RU if lang == "ru" else LOCAL_JSON_EN
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return ok

# --- background cache puller (also used on boot)
def _pull_and_cache_from_github() -> bool:
    """Fetch EN/RU from GitHub into local cache; return True if changed."""
    changed = False
    try:
        os.makedirs(KB_DIR, exist_ok=True)
        for remote, local in [(GH_JSON_EN, LOCAL_JSON_EN), (GH_JSON_RU, LOCAL_JSON_RU)]:
            blob, _sha = _gh_get_file_and_sha(remote)
            if blob is None:
                continue
            current = open(local, "rb").read() if os.path.exists(local) else None
            if current != blob:
                with open(local, "wb") as f:
                    f.write(blob)
                changed = True
                app.logger.info("[kb] cached %s from %s:%s", local, GH_BRANCH, remote)
    except Exception as e:
        app.logger.warning("[kb] cache pull error: %s", e)
    return changed

# --- compat alias for any legacy calls
def pull_kb_from_github() -> bool:
    return _pull_and_cache_from_github()

def start_kb_poll(interval_sec: int = 60):
    def _loop():
        while True:
            try:
                if _pull_and_cache_from_github():
                    reload_index(force=True)
            except Exception as e:
                app.logger.warning("[kb] poll error: %s", e)
            time.sleep(max(30, interval_sec))
    threading.Thread(target=_loop, daemon=True).start()

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

for _l in LANGS:
    _load_i18n(_l)

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
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return None
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
# Admin UI (GET only; data APIs are /faq/* below)
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
    try:
        pull_kb_from_github()  # ensure latest from GH
    except Exception as _e:
        app.logger.warning("admin pull failed: %s", _e)
    return render_template("admin.html", title="Admin FAQs")

# =====================================================================================
# Simple FAQ search (uses GitHub JSON)
# =====================================================================================
@app.post("/chat")
def chat():
    user_message = (request.json.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Please enter a message."})
    try:
        lang = get_lang()
        data = gh_load_lang_json(lang)
        faqs = data.get("faqs", [])
        qlower = user_message.lower()
        tokens = [t for t in re.findall(r"\w+", qlower) if len(t) > 2]
        best, score = None, 0
        for f in faqs:
            q = (f.get("question") or "").lower()
            a = (f.get("answer") or "").lower()
            s = sum(t in q for t in tokens) * 2 + sum(t in a for t in tokens)
            if s > score:
                best, score = f, s
        if best and score > 0:
            return jsonify({"reply": f"**{best['question']}**\n\n{best['answer']}", "source": f"faq-{lang}"})
        return jsonify({"reply": "I couldn't find that in the FAQs. Try the FAQ Bot or rephrase your question."})
    except Exception as e:
        return jsonify({"reply": f"Error while processing your request: {str(e)}"})

# =====================================================================================
# RAG status (reads local cache counts if present)
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
# Groq endpoints
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
# /faq API — ALWAYS hits GitHub as source of truth
# =====================================================================================
@app.get("/faq/list")
def faq_list():
    lang = request.args.get("lang", "en").lower()
    lang = "ru" if lang == "ru" else "en"
    try:
        data = gh_load_lang_json(lang)
        faqs = data.get("faqs", [])
        items = [{"number": int(f.get("number") or 0),
                  "question": f.get("question", ""),
                  "answer": f.get("answer", "")} for f in faqs]
        items.sort(key=lambda x: x["number"])
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "items": [], "error": f"GH fetch failed: {str(e)}"})

@app.get("/faq/item/<int:number>")
def faq_item(number):
    lang = request.args.get("lang", "en").lower()
    lang = "ru" if lang == "ru" else "en"
    try:
        data = gh_load_lang_json(lang)
        for f in data.get("faqs", []):
            if int(f.get("number") or 0) == number:
                return jsonify({"ok": True, "item": {
                    "number": int(f.get("number") or 0),
                    "question": f.get("question", ""),
                    "answer": f.get("answer", ""),
                }})
        return jsonify({"ok": False, "error": "Not found"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"GH fetch failed: {str(e)}"})

@app.post("/faq/save")
def faq_save():
    _check_admin()
    body = request.get_json(force=True) or {}
    lang = (body.get("lang") or "en").lower()
    lang = "ru" if lang == "ru" else "en"
    item = body.get("item") or {}
    try:
        data = gh_load_lang_json(lang)
        faqs = data.get("faqs", [])
        n = int(item.get("number") or 0)
        q = item.get("question", "")
        a = item.get("answer", "")
        updated = False
        for f in faqs:
            if int(f.get("number") or 0) == n:
                f["question"] = q
                f["answer"] = a
                updated = True
                break
        if not updated:
            faqs.append({"number": n, "question": q, "answer": a, "source": f"admin_{lang}"})
        faqs.sort(key=lambda x: int(x.get("number") or 0))
        data["faqs"] = faqs
        meta = data.setdefault("metadata", {})
        meta["lang"] = lang
        meta["total_faqs"] = len(faqs)
        meta["parsed_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        ok = gh_save_lang_json(lang, data, f"admin: save {lang.upper()} FAQ #{n}")
        if ok:
            reload_index(force=True)
        return jsonify({"ok": True, "number": n})
    except Exception as e:
        return jsonify({"ok": False, "error": f"GH save failed: {str(e)}"})

@app.delete("/faq/delete/<int:number>")
def faq_delete(number):
    _check_admin()
    lang = request.args.get("lang", "en").lower()
    lang = "ru" if lang == "ru" else "en"
    try:
        data = gh_load_lang_json(lang)
        before = list(data.get("faqs", []))
        after = [f for f in before if int(f.get("number") or 0) != number]
        if len(after) == len(before):
            return jsonify({"ok": False, "error": "Not found"})
        data["faqs"] = sorted(after, key=lambda x: int(x.get("number") or 0))
        meta = data.setdefault("metadata", {})
        meta["lang"] = lang
        meta["total_faqs"] = len(data["faqs"])
        meta["parsed_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ok = gh_save_lang_json(lang, data, f"admin: delete {lang.upper()} FAQ #{number}")
        if ok:
            reload_index(force=True)
        return jsonify({"ok": True, "deleted": number})
    except Exception as e:
        return jsonify({"ok": False, "error": f"GH delete failed: {str(e)}"})

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
        # ensure kb cache exists locally (status page, dev offline)
        try:
            _pull_and_cache_from_github()
        except Exception as e:
            app.logger.warning("[kb] initial cache pull failed: %s", e)
        # warm Groq retriever (guard if PDF missing)
        try:
            if os.path.exists(PDF_PATH):
                warm_index(PDF_PATH)
            else:
                warm_index()
        except Exception as e:
            app.logger.warning("[rag] warm_index failed: %s", e)
        # poll GH for updates
        start_kb_poll(interval_sec=60)
        _BOOT_DONE = True
        app.logger.info("[boot] Ready (GitHub-first FAQs).")

_boot_once()

# For local dev convenience
if __name__ == "__main__":
    try:
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    except Exception:
        pass
    app.run(debug=True, use_reloader=False)