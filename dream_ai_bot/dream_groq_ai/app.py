# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import threading
import webbrowser
import urllib.request as _urlreq
import urllib.error as _urlerr
from pathlib import Path
from typing import Optional, Any, Dict, List

from flask import (
    Flask, render_template, request, redirect, url_for, jsonify,
    abort, session, flash, make_response
)
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from datetime import timedelta

import hmac, hashlib

# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------
load_dotenv(find_dotenv() or "")
load_dotenv()

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["GA_MEASUREMENT_ID"] = os.getenv("GA_MEASUREMENT_ID", "").strip()
app.config["JSON_AS_ASCII"] = False

app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev-change-me"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,     # set True only behind HTTPS
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_PATH="/",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
)

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "January@1").strip()

def validate_token(tok: str) -> bool:
    return bool(tok) and bool(ADMIN_TOKEN) and hmac.compare_digest(tok, ADMIN_TOKEN)

@app.context_processor
def _inject_ga():
    return {"GA_MEASUREMENT_ID": app.config.get("GA_MEASUREMENT_ID", "")}

# -------- Admin auth: signed cookie ----------------------------------
_ADMIN_COOKIE = "admin_auth"
_ADMIN_TTL = 12 * 60 * 60  # 12h

def _make_admin_cookie() -> str:
    ts = str(int(time.time()))
    msg = f"1|{ts}"
    sig = hmac.new(app.config["SECRET_KEY"].encode(), msg.encode(), hashlib.sha256).hexdigest()
    return f"{msg}|{sig}"

def _check_admin_cookie(val: Optional[str]) -> bool:
    if not val:
        return False
    try:
        part, ts, sig = val.split("|", 2)
        if part != "1":
            return False
        msg = f"{part}|{ts}"
        expect = hmac.new(app.config["SECRET_KEY"].encode(), msg.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expect):
            return False
        if time.time() - int(ts) > _ADMIN_TTL:
            return False
        return True
    except Exception:
        return False

def is_admin() -> bool:
    return _check_admin_cookie(request.cookies.get(_ADMIN_COOKIE))

@app.before_request
def _mirror_admin_into_session():
    if is_admin():
        session.permanent = True
        session["is_admin"] = True

# ---------------------------------------------------------------------
# KB / GH helpers (unchanged behavior)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
KB_DIR = PROJECT_ROOT / "kb"
PDF_PATH = str(KB_DIR / "dream_faqs.pdf")
LOCAL_JSON_EN = str(KB_DIR / "parsed_faqs_en.json")
LOCAL_JSON_RU = str(KB_DIR / "parsed_faqs_ru.json")

def _mtime(p: str) -> float:
    try: return os.path.getmtime(p)
    except Exception: return 0.0

GH_OWNER   = (os.getenv("GH_OWNER",  "dream-framework") or "").strip()
GH_REPO    = (os.getenv("GH_REPO",   "dream") or "").strip()
GH_BRANCH  = (os.getenv("GH_BRANCH", "kb-data") or "").strip()
GH_JSON_EN = (os.getenv("GH_JSON_EN","kb/parsed_faqs_en.json") or "").strip()
GH_JSON_RU = (os.getenv("GH_JSON_RU","kb/parsed_faqs_ru.json") or "").strip()
GH_TOKEN   = (os.getenv("GH_TOKEN",  "") or "").strip()

GITHUB_TIMEOUT_RAW   = int(os.getenv("GITHUB_TIMEOUT_RAW", "6"))
GITHUB_TIMEOUT_API   = int(os.getenv("GITHUB_TIMEOUT_API", "6"))
GITHUB_TOTAL_BUDGET  = int(os.getenv("GITHUB_TOTAL_BUDGET", "12"))

MUTATION_FRESH_TTL = int(os.getenv("FAQ_MUTATION_FRESH_TTL", "45"))
_API_FIRST_UNTIL: Dict[str, float] = {"en": 0.0, "ru": 0.0}
_LAST_PAYLOAD: Dict[str, Optional[Dict[str, Any]]] = {"en": None, "ru": None}

def _time_left(deadline: float) -> float:
    return max(0.5, deadline - time.monotonic())

def _gh_headers():
    h = {"Accept":"application/vnd.github+json", "User-Agent":"dream-app/1.0"}
    if GH_TOKEN: h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _gh_headers_raw():
    h = {"Accept":"application/vnd.github.raw", "User-Agent":"dream-app/1.0"}
    if GH_TOKEN: h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _gh_contents_url(path, ref=None):
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}"
    return url + (f"?ref={ref}" if ref else "")

def _gh_blob_url(sha): return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/git/blobs/{sha}"
def _raw_github_url(path): return f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{path}"

def _gh_get_file_bytes(path_in_repo: str, deadline=None, *, api_first=False) -> bytes:
    if deadline is None: deadline = time.monotonic() + GITHUB_TOTAL_BUDGET
    def _try_raw():
        try:
            req = _urlreq.Request(_raw_github_url(path_in_repo), headers=_gh_headers())
            with _urlreq.urlopen(req, timeout=min(GITHUB_TIMEOUT_RAW, _time_left(deadline))) as r:
                return r.read()
        except Exception: return None
    def _try_api():
        try:
            req = _urlreq.Request(_gh_contents_url(path_in_repo, GH_BRANCH), headers=_gh_headers())
            with _urlreq.urlopen(req, timeout=min(GITHUB_TIMEOUT_API, _time_left(deadline))) as r:
                payload = json.loads(r.read().decode("utf-8", "ignore"))
            if isinstance(payload, dict):
                if payload.get("content"):
                    import base64 as _b64
                    return _b64.b64decode(payload["content"].encode("ascii"))
                dl = payload.get("download_url")
                if dl:
                    with _urlreq.urlopen(_urlreq.Request(dl, headers=_gh_headers()),
                                         timeout=min(GITHUB_TIMEOUT_RAW, _time_left(deadline))) as rr:
                        return rr.read()
                sha = payload.get("sha")
                if sha:
                    req2 = _urlreq.Request(_gh_blob_url(sha), headers=_gh_headers_raw())
                    with _urlreq.urlopen(req2, timeout=min(GITHUB_TIMEOUT_API, _time_left(deadline))) as rr:
                        return rr.read()
        except Exception: return None
    data = _try_api() or _try_raw() if api_first else _try_raw() or _try_api()
    if data is None: raise RuntimeError(f"Cannot fetch '{path_in_repo}'")
    return data

_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_RE_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
_RE_BAD_U_ESCAPE   = re.compile(r'\\u(?![0-9a-fA-F]{4})')
_RE_TRAILING_BS    = re.compile(r'\\(?=$)')

def _safe_json_loads(txt: str) -> Any:
    try: return json.loads(txt)
    except json.JSONDecodeError:
        txt = _RE_INVALID_ESCAPE.sub(r"\\\\", txt)
        txt = _RE_BAD_U_ESCAPE.sub(r"\\\\u", txt)
        txt = _RE_TRAILING_BS.sub(r"\\\\", txt)
        return json.loads(txt)

def _decode_json_bytes(raw: bytes) -> Any:
    for enc in ("utf-8-sig","utf-8","utf-16","utf-16-le","utf-16-be","cp1251","latin-1"):
        try:
            txt = raw.decode(enc, errors="strict")
            txt = _CTRL_RE.sub("", txt)
            return _safe_json_loads(txt)
        except Exception: pass
    txt = raw.decode("utf-8","ignore")
    txt = _CTRL_RE.sub("", txt)
    return _safe_json_loads(txt)

def _normalize_faq_json_obj(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        faqs_src = obj.get("faqs")
        if not isinstance(faqs_src, list):
            for key in ("items","data","list"):
                if isinstance(obj.get(key), list):
                    faqs_src = obj[key]; break
            if not isinstance(faqs_src, list):
                for v in obj.values():
                    if isinstance(v, list) and (not v or all(isinstance(x, dict) for x in v)):
                        faqs_src = v; break
        meta = dict(obj.get("metadata") or {})
    elif isinstance(obj, list):
        faqs_src, meta = obj, {}
    else:
        faqs_src, meta = [], {}
    out: List[Dict[str, Any]] = []
    for f in (faqs_src or []):
        if not isinstance(f, dict): continue
        try: n = int(f.get("number", 0))
        except Exception: n = 0
        q = f.get("question") or f.get("title") or f.get("question_ru") or f.get("question_en") or ""
        a = f.get("answer")   or f.get("body")  or f.get("answer_ru")   or f.get("answer_en")   or ""
        out.append({"number": n, "question": q, "answer": a})
    out.sort(key=lambda x: x["number"])
    meta["total_faqs"] = len(out)
    return {"metadata": meta, "faqs": out}

def _dir_of(p: str) -> str:
    parts = (p or "").split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""

if not GH_JSON_RU or GH_JSON_RU.strip() in {"parsed_faqs_ru.json","ru/parsed_faqs_ru.json"}:
    GH_JSON_RU = f"{_dir_of(GH_JSON_EN) or 'kb'}/parsed_faqs_ru.json"

_GH_PATH_CACHE: Dict[str, List[str]] = {"en":[GH_JSON_EN], "ru":[GH_JSON_RU]}

def _read_local_cache(lang: str) -> Optional[Dict[str, Any]]:
    try:
        p = LOCAL_JSON_RU if lang=="ru" else LOCAL_JSON_EN
        if os.path.exists(p):
            with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: pass
    return None

def gh_load_lang_json(lang: str) -> Dict[str, Any]:
    now = time.monotonic()
    api_first = now < _API_FIRST_UNTIL.get(lang, 0.0)

    if api_first:
        mem = _LAST_PAYLOAD.get(lang)
        if mem and mem.get("faqs"): return mem

    attempts: List[str] = []
    cand_paths = list(dict.fromkeys((_GH_PATH_CACHE.get(lang) or []) + [GH_JSON_EN if lang=="en" else GH_JSON_RU]))
    last_err = None
    best = None; best_path=None
    deadline = time.monotonic() + GITHUB_TOTAL_BUDGET

    for path in cand_paths:
        if not path: continue
        attempts.append(path)
        try:
            raw = _gh_get_file_bytes(path, deadline=deadline, api_first=api_first)
            obj = _decode_json_bytes(raw)
            data = _normalize_faq_json_obj(obj)
            faqs = data.get("faqs", [])
            meta_total = int((data.get("metadata") or {}).get("total_faqs") or 0)
            if len(faqs)==0 and meta_total>0:
                _GH_PATH_CACHE[lang] = [p for p in _GH_PATH_CACHE.get(lang, []) if p != path]
                last_err = f"0 FAQs while metadata says {meta_total}"
                continue
            best, best_path = data, path
            break
        except Exception as e:
            last_err = str(e)
            _GH_PATH_CACHE[lang] = [p for p in _GH_PATH_CACHE.get(lang, []) if p != path]
            if _time_left(deadline) <= 0.6: break

    if best is None:
        cached = _read_local_cache(lang)
        if cached and cached.get("faqs"): return cached
        raise RuntimeError(last_err or "GH load failed")

    if best_path: _GH_PATH_CACHE[lang] = [best_path]

    try:
        os.makedirs(KB_DIR, exist_ok=True)
        with open(LOCAL_JSON_RU if lang=="ru" else LOCAL_JSON_EN, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
    except Exception: pass

    return best

def gh_save_lang_json(lang: str, payload: Dict[str, Any], commit_msg: str) -> bool:
    path = (_GH_PATH_CACHE.get(lang) or [GH_JSON_RU if lang=="ru" else GH_JSON_EN])[0]
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if not GH_TOKEN: return False
    body = {"message":commit_msg, "branch":GH_BRANCH, "content":base64.b64encode(raw).decode("ascii")}
    # optimistic PUT via contents API
    try:
        req = _urlreq.Request(f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}",
                              data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                              method="PUT", headers=_gh_headers())
        with _urlreq.urlopen(req, timeout=20): pass
        with open(LOCAL_JSON_RU if lang=="ru" else LOCAL_JSON_EN, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except Exception: return False

# ---------------------------------------------------------------------
# i18n + pages
# ---------------------------------------------------------------------
LANGS = ["en","ru"]
I18N_DIR = PROJECT_ROOT / "i18n"
TX, MT = {l:{} for l in LANGS}, {l:0 for l in LANGS}
def _load_i18n(lang):
    p = I18N_DIR / f"{lang}.json"
    if p.exists(): TX[lang]=json.loads(p.read_text(encoding="utf-8")); MT[lang]=p.stat().st_mtime
for _l in LANGS: _load_i18n(_l)
def _refresh_i18n():
    for l in LANGS:
        p = I18N_DIR / f"{l}.json"
        if p.exists() and p.stat().st_mtime != MT.get(l,0): _load_i18n(l)
def get_lang():
    lang = request.cookies.get("lang","en"); return lang if lang in LANGS else "en"
def t(key: str):
    _refresh_i18n()
    def look(lang):
        v=TX.get(lang,{}); 
        for part in key.split("."):
            if isinstance(v,dict) and part in v: v=v[part]
            else: return None
        return v
    return look(get_lang()) or look("en") or key

NAV = [
    ("home","nav.home","/"),
    ("case","nav.case","/case"),
    ("overview","nav.overview","/overview"),
    ("axioms","nav.axioms","/axioms"),
    ("theorems","nav.theorems","/theorems"),
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
    if lang_code in LANGS: resp.set_cookie("lang", lang_code, max_age=31536000)
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
@app.route("/case")
def case(): return render_template("case.html", title=t("nav.case"))
@app.route("/theorems")
def theorems(): return render_template("theorems.html", title=t("nav.theorems"))

app.url_map.strict_slashes = False

@app.after_request
def _no_store_for_faq(resp):
    try:
        if request.path.startswith("/faq/"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
    except Exception: pass
    return resp

# ---------------------------- Admin pages -----------------------------
def _require_admin():
    if not is_admin():
        abort(401)

@app.route("/admin", methods=["GET","POST"])
def admin():
    if request.method == "POST":
        token = (request.form.get("token") or "").strip()
        if validate_token(token):
            session["is_admin"] = True
            session.permanent = True
            resp = make_response(redirect(url_for("admin")))
            resp.set_cookie(_ADMIN_COOKIE, _make_admin_cookie(),
                            max_age=_ADMIN_TTL, httponly=True, samesite="Lax",
                            secure=bool(app.config.get("SESSION_COOKIE_SECURE", False)),
                            path="/")
            return resp
        flash("Invalid token", "error")
        return redirect(url_for("home"))

    # IMPORTANT: never redirect away here; just render and tell template the state
    return render_template("admin.html", is_admin=is_admin(), title="Admin")

@app.post("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    resp = jsonify({"ok": True})
    resp.set_cookie(_ADMIN_COOKIE, "", max_age=0, expires=0, path="/")
    return resp

# Simple probe for debugging in DevTools
@app.get("/api/admin/state")
def admin_state():
    return jsonify({
        "cookie_present": bool(request.cookies.get(_ADMIN_COOKIE)),
        "is_admin": is_admin(),
        "session_is_admin": bool(session.get("is_admin"))
    })

# ---------------------------- Bot / FAQ API (unchanged) --------------
try:
    from groq_bot import warm_index, groq_answer, reload_index  # type: ignore
except Exception:
    def groq_answer(*_, **__): return "RAG unavailable."
    def warm_index(*_, **__): return None
    def reload_index(*_, **__): return None

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
            s = sum(t in q for t in tokens)*2 + sum(t in a for t in tokens)
            if s > score: best, score = f, s
        if best and score > 0:
            return jsonify({"reply": f"**{best['question']}**\n\n{best['answer']}", "source": f"faq-{lang}"})
        return jsonify({"reply": "I couldn't find that in the FAQs. Try the FAQ Bot or rephrase your question."})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.get("/api/rag-status")
def rag_status():
    def safe_len(lang):
        try: return len(gh_load_lang_json(lang).get("faqs", []))
        except Exception: return -1
    return jsonify({
        "pdf_exists": os.path.exists(PDF_PATH),
        "pdf_mtime": _mtime(PDF_PATH),
        "en_count": safe_len("en"),
        "ru_count": safe_len("ru"),
        "json_en_repo": GH_JSON_EN,
        "json_ru_repo": GH_JSON_RU,
        "branch": GH_BRANCH,
        "repo": f"{GH_OWNER}/{GH_REPO}",
        "resolved_paths": dict(_GH_PATH_CACHE),
    })

@app.post("/groq-chat")
def groq_chat():
    payload = request.get_json(force=True, silent=True) or {}
    q = (payload.get("message") or "").strip()
    if not q: return jsonify({"ok": False, "error": "Empty message."})
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

# ---------------------------- Background poller ----------------------
def _pull_and_cache_from_github() -> bool:
    changed = False
    try:
        os.makedirs(KB_DIR, exist_ok=True)
        for lang, remote_list in (("en", _GH_PATH_CACHE.get("en") or [GH_JSON_EN]),
                                  ("ru", _GH_PATH_CACHE.get("ru") or [GH_JSON_RU])):
            for remote in (remote_list if isinstance(remote_list, list) else [remote_list]):
                try: blob = _gh_get_file_bytes(remote)
                except Exception: continue
                local = LOCAL_JSON_EN if lang=="en" else LOCAL_JSON_RU
                old = open(local,"rb").read() if os.path.exists(local) else None
                if old != blob:
                    with open(local,"wb") as f: f.write(blob)
                    changed = True
    except Exception: pass
    return changed

def start_kb_poll(interval_sec: int = 60):
    def _loop():
        while True:
            try:
                if _pull_and_cache_from_github():
                    try: reload_index(force=True)
                    except Exception: pass
            except Exception: pass
            time.sleep(max(30, interval_sec))
    threading.Thread(target=_loop, daemon=True).start()

_BOOT_DONE=False; _BOOT_LOCK=threading.Lock()
def _boot_once():
    global _BOOT_DONE
    with _BOOT_LOCK:
        if _BOOT_DONE: return
        try:
            if os.path.exists(PDF_PATH): warm_index(PDF_PATH)
            else: warm_index()
        except Exception: pass
        start_kb_poll(60)
        _BOOT_DONE=True
_boot_once()

if __name__ == "__main__":
    try: threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    except Exception: pass
    app.run(debug=True, use_reloader=False)