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

from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------
load_dotenv(find_dotenv() or "")
load_dotenv()

# Groq bot bits
from groq_bot import warm_index, groq_answer, reload_index

# ---------------------------------------------------------------------
# App + GA
# ---------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["GA_MEASUREMENT_ID"] = os.getenv("GA_MEASUREMENT_ID", "").strip()
app.config["JSON_AS_ASCII"] = False  # ensure UTF-8 JSON responses

@app.context_processor
def _inject_ga():
    return {"GA_MEASUREMENT_ID": app.config.get("GA_MEASUREMENT_ID", "")}

# ---------------------------------------------------------------------
# Paths (PDF only used by RAG; JSONs are loaded from GitHub)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
KB_DIR       = PROJECT_ROOT / "kb"
PDF_PATH     = str(KB_DIR / "dream_faqs.pdf")
LOCAL_JSON_EN = str(KB_DIR / "parsed_faqs_en.json")  # optional cache
LOCAL_JSON_RU = str(KB_DIR / "parsed_faqs_ru.json")  # optional cache

def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except Exception:
        return 0.0

# ---------------------------------------------------------------------
# GitHub kb-data integration (source of truth)
# ---------------------------------------------------------------------
GH_OWNER   = os.getenv("GH_OWNER",  "dream-framework")
GH_REPO    = os.getenv("GH_REPO",   "dream")
GH_BRANCH  = os.getenv("GH_BRANCH", "kb-data")
GH_JSON_EN = os.getenv("GH_JSON_EN","kb/parsed_faqs_en.json")
GH_JSON_RU = os.getenv("GH_JSON_RU","kb/parsed_faqs_ru.json")
GH_TOKEN   = os.getenv("GH_TOKEN",  "")  # fine-grained PAT (contents: read+write)

# Conservative timeouts + overall budget so requests never hang long enough to 502
GITHUB_TIMEOUT_RAW   = int(os.getenv("GITHUB_TIMEOUT_RAW", "6"))   # per raw/download hop
GITHUB_TIMEOUT_API   = int(os.getenv("GITHUB_TIMEOUT_API", "6"))   # per API hop
GITHUB_TOTAL_BUDGET  = int(os.getenv("GITHUB_TOTAL_BUDGET", "12")) # overall per fetch

def _time_left(deadline: float) -> float:
    return max(0.5, deadline - time.monotonic())

def _gh_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "dream-app/1.0"}
    if GH_TOKEN:
        h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _gh_headers_raw() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github.raw", "User-Agent": "dream-app/1.0"}
    if GH_TOKEN:
        h["Authorization"] = f"token {GH_TOKEN}"
    return h

def _gh_contents_url(path_in_repo: str, ref: Optional[str] = None) -> str:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}"
    if ref:
        url += f"?ref={ref}"
    return url

def _gh_blob_url(sha: str) -> str:
    return f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/git/blobs/{sha}"

def _raw_github_url(path_in_repo: str) -> str:
    return f"https://raw.githubusercontent.com/{GH_OWNER}/{GH_REPO}/{GH_BRANCH}/{path_in_repo}"

def _gh_get_file_bytes(path_in_repo: str, deadline: Optional[float] = None) -> bytes:
    """
    Return raw bytes for a file in repo/branch.
    Fast-fail strategy with per-hop timeouts and an overall budget:
      1) raw.githubusercontent.com
      2) Contents API (inline base64 or download_url)
      3) Git blob by sha
    Only treat 404 as path-tryable; other network errors don't cascade forever.
    """
    if deadline is None:
        deadline = time.monotonic() + GITHUB_TOTAL_BUDGET

    # 1) raw
    try:
        req = _urlreq.Request(_raw_github_url(path_in_repo), headers=_gh_headers())
        with _urlreq.urlopen(req, timeout=min(GITHUB_TIMEOUT_RAW, _time_left(deadline))) as r:
            return r.read()
    except _urlerr.HTTPError as e:
        # non-404: proceed to contents quickly but don't burn time
        if e.code != 404:
            pass
    except Exception:
        pass

    # 2) contents
    try:
        req = _urlreq.Request(_gh_contents_url(path_in_repo, GH_BRANCH), headers=_gh_headers())
        with _urlreq.urlopen(req, timeout=min(GITHUB_TIMEOUT_API, _time_left(deadline))) as r:
            payload = json.loads(r.read().decode("utf-8", "ignore"))
        if isinstance(payload, dict):
            if payload.get("content"):
                return base64.b64decode(payload["content"].encode("ascii"))
            dl = payload.get("download_url")
            if dl:
                with _urlreq.urlopen(_urlreq.Request(dl, headers=_gh_headers()),
                                     timeout=min(GITHUB_TIMEOUT_RAW, _time_left(deadline))) as rr:
                    return rr.read()
            sha = payload.get("sha")
            if sha:
                # 3) blob
                req2 = _urlreq.Request(_gh_blob_url(sha), headers=_gh_headers_raw())
                with _urlreq.urlopen(req2, timeout=min(GITHUB_TIMEOUT_API, _time_left(deadline))) as rr:
                    return rr.read()
    except _urlerr.HTTPError as e:
        if e.code != 404:
            app.logger.warning("GitHub contents fetch (%s) -> %s", path_in_repo, e)
    except Exception as e:
        app.logger.warning("GitHub contents error (%s): %s", path_in_repo, e)

    raise RuntimeError(f"Cannot fetch '{path_in_repo}' from GitHub (branch={GH_BRANCH}) within time budget")

def _gh_current_sha(path_in_repo: str) -> Optional[str]:
    try:
        req = _urlreq.Request(_gh_contents_url(path_in_repo, GH_BRANCH), headers=_gh_headers())
        with _urlreq.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read().decode("utf-8", "ignore"))
        return payload.get("sha")
    except Exception:
        return None

def push_kb_to_github(path_in_repo: str, raw_bytes: bytes, message: str) -> bool:
    if not GH_TOKEN:
        app.logger.warning("GH_TOKEN not set; skipping push to GitHub")
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
    req = _urlreq.Request(put_url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          method="PUT", headers=_gh_headers())
    try:
        with _urlreq.urlopen(req, timeout=25) as r:
            _ = r.read()
            app.logger.info("Pushed %s to %s", path_in_repo, GH_BRANCH)
            return True
    except Exception as e:
        app.logger.error("Push error (%s): %s", path_in_repo, e)
        return False

# ---------------------------------------------------------------------
# JSON decode/normalize (robust) + GitHub load/save
# ---------------------------------------------------------------------
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _decode_json_bytes(raw: bytes) -> Any:
    # Try several encodings (RU files are sometimes saved oddly)
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1251", "latin-1"):
        try:
            txt = raw.decode(enc)  # first pass
            # Remove stray control chars that break JSON
            txt = _CTRL_RE.sub("", txt)
            return json.loads(txt)
        except Exception:
            continue
    # Last-ditch: treat as UTF-8 ignoring errors
    txt = raw.decode("utf-8", "ignore")
    txt = _CTRL_RE.sub("", txt)
    return json.loads(txt)

def _normalize_faq_json_obj(obj: Any) -> Dict[str, Any]:
    # Find list of faqs
    if isinstance(obj, dict):
        faqs_src = obj.get("faqs")
        if not isinstance(faqs_src, list):
            # try common alternates
            for key in ("items", "data", "list"):
                if isinstance(obj.get(key), list):
                    faqs_src = obj[key]
                    break
            # or first list-of-dicts in values
            if not isinstance(faqs_src, list):
                for v in obj.values():
                    if isinstance(v, list) and (not v or all(isinstance(x, dict) for x in v)):
                        faqs_src = v
                        break
        meta = dict(obj.get("metadata") or {})
    elif isinstance(obj, list):
        faqs_src, meta = obj, {}
    else:
        faqs_src, meta = [], {}

    out: List[Dict[str, Any]] = []
    for f in (faqs_src or []):
        if not isinstance(f, dict):
            continue
        # normalize fields
        try:
            n = int(f.get("number", 0))
        except Exception:
            n = 0
        q = f.get("question") or f.get("title") or f.get("question_ru") or f.get("question_en") or ""
        a = f.get("answer")   or f.get("body")  or f.get("answer_ru")   or f.get("answer_en")   or ""
        out.append({"number": n, "question": q, "answer": a})

    out.sort(key=lambda x: x["number"])
    meta["total_faqs"] = len(out)  # recompute
    return {"metadata": meta, "faqs": out}

def _dir_of(p: str) -> str:
    parts = (p or "").split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""

# If RU path is generic, coerce it to live next to EN
if not GH_JSON_RU or GH_JSON_RU.strip() in {"parsed_faqs_ru.json", "ru/parsed_faqs_ru.json"}:
    GH_JSON_RU = f"{_dir_of(GH_JSON_EN) or 'kb'}/parsed_faqs_ru.json"

def _lang_to_repo_path_candidates(lang: str) -> List[str]:
    """
    Primary env-provided path first; sensible alternates next.
    Intentionally avoid repo-root 'parsed_faqs_ru.json' to prevent drift.
    """
    en_dir = _dir_of(GH_JSON_EN) or "kb"
    ru_dir = _dir_of(GH_JSON_RU) or en_dir

    if lang == "ru":
        alts = [
            GH_JSON_RU,
            f"{ru_dir}/parsed_faqs_ru.json",
            f"{en_dir}/parsed_faqs_ru.json",
            "kb/parsed_faqs_ru.json",
            "kb/ru/parsed_faqs_ru.json",
            "ru/parsed_faqs_ru.json",
        ]
    else:
        alts = [
            GH_JSON_EN,
            f"{en_dir}/parsed_faqs_en.json",
            "kb/parsed_faqs_en.json",
            "kb/en/parsed_faqs_en.json",
            "en/parsed_faqs_en.json",
        ]

    # Deduplicate while keeping order
    seen, ordered = set(), []
    for p in alts:
        if p and p not in seen:
            ordered.append(p); seen.add(p)
    return ordered

_GH_PATH_CACHE: Dict[str, str] = {}  # lang -> resolved path that worked
# Seed cache so we don't start at repo root by accident
_GH_PATH_CACHE.setdefault("en", GH_JSON_EN)
_GH_PATH_CACHE.setdefault("ru", GH_JSON_RU)

def _read_local_cache(lang: str) -> Optional[Dict[str, Any]]:
    try:
        p = LOCAL_JSON_RU if lang == "ru" else LOCAL_JSON_EN
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def gh_load_lang_json(lang: str) -> Dict[str, Any]:
    """
    Read from GitHub with a strict time budget; fall back to local cache if GitHub is slow/unavailable.
    If a candidate path yields 0 FAQs while metadata claims >0, try alternates; don't silently return 0.
    Evict a cached bad path immediately to avoid pinning to a wrong location.
    """
    attempts: List[str] = []

    # Use cached working path first (if any), then canonical candidates
    cand_paths = []
    cached = _GH_PATH_CACHE.get(lang)
    if cached:
        cand_paths.append(cached)
    cand_paths += [p for p in _lang_to_repo_path_candidates(lang) if p not in cand_paths]

    last_err: Optional[str] = None
    best_data: Optional[Dict[str, Any]] = None
    best_path: Optional[str] = None
    deadline = time.monotonic() + GITHUB_TOTAL_BUDGET

    for path in cand_paths:
        if not path:
            continue
        attempts.append(path)
        try:
            raw = _gh_get_file_bytes(path, deadline=deadline)
            obj = _decode_json_bytes(raw)
            data = _normalize_faq_json_obj(obj)

            faqs = data.get("faqs", [])
            meta_total = int((data.get("metadata") or {}).get("total_faqs") or 0)
            # If meta says there are FAQs but parse found 0, treat as mismatch (try next)
            if len(faqs) == 0 and meta_total > 0:
                last_err = f"Parsed 0 FAQs while metadata says {meta_total} at {path}"
                if path == cached:
                    _GH_PATH_CACHE.pop(lang, None)  # evict bad cached path
                continue

            best_data, best_path = data, path
            break
        except Exception as e:
            last_err = f"{e}"
            if path == cached:
                _GH_PATH_CACHE.pop(lang, None)      # evict failing cached path
            if _time_left(deadline) <= 0.6:
                break
            continue

    if best_data is None:
        # Serve local cache if possible (avoids 502s and keeps UI responsive)
        cached_local = _read_local_cache(lang)
        if cached_local and cached_local.get("faqs"):
            app.logger.warning("[kb] %s serving local cache; GH attempts=%s; last_err=%s",
                               lang.upper(), attempts, last_err)
            return cached_local
        app.logger.error("[kb] %s GH failed. attempts=%s last_err=%s", lang.upper(), attempts, last_err)
        raise RuntimeError(last_err or f"Unable to load {lang.upper()} FAQs from GitHub.")

    if len(best_data.get("faqs", [])) == 0:
        cached_local = _read_local_cache(lang)
        if cached_local and cached_local.get("faqs"):
            app.logger.warning("[kb] %s GitHub JSON contained 0 FAQs; serving cached copy.", lang.upper())
            return cached_local
        raise RuntimeError(f"{lang.upper()} JSON parsed successfully but contains 0 FAQs (path: {best_path}).")

    # Cache path that worked
    if best_path:
        _GH_PATH_CACHE[lang] = best_path

    # Write local cache (for status page / debugging)
    try:
        os.makedirs(KB_DIR, exist_ok=True)
        cache_path = LOCAL_JSON_RU if lang == "ru" else LOCAL_JSON_EN
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(best_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    app.logger.info("[kb] %s loaded %d FAQs from %s@%s",
                    lang.upper(), len(best_data["faqs"]), GH_BRANCH, best_path)
    return best_data

def gh_save_lang_json(lang: str, payload: Dict[str, Any], commit_msg: str) -> bool:
    # Save back to *the same* path that worked, or primary if none cached
    path = _GH_PATH_CACHE.get(lang) or (GH_JSON_RU if lang == "ru" else GH_JSON_EN)
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    ok = push_kb_to_github(path, raw, commit_msg)
    if ok:
        try:
            cache_path = LOCAL_JSON_RU if lang == "ru" else LOCAL_JSON_EN
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return ok

# ---------------------------------------------------------------------
# i18n helpers
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Navigation + pages
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Admin page (view)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Simple FAQ search (GitHub JSON as the source of truth)
# ---------------------------------------------------------------------
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
            if s > score:
                best, score = f, s
        if best and score > 0:
            return jsonify({"reply": f"**{best['question']}**\n\n{best['answer']}", "source": f"faq-{lang}"})
        return jsonify({"reply": "I couldn't find that in the FAQs. Try the FAQ Bot or rephrase your question."})
    except Exception as e:
        return jsonify({"reply": f"Error while processing your request: {str(e)}"})

# ---------------------------------------------------------------------
# Status (reads from GitHub directly and reports counts)
# ---------------------------------------------------------------------
@app.get("/api/rag-status")
def rag_status():
    def safe_len(lang_code: str) -> int:
        try:
            return len(gh_load_lang_json(lang_code).get("faqs", []))
        except Exception:
            return -1
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

# ---------------------------------------------------------------------
# Groq endpoints
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# /faq API — hits GitHub (with cache fallback via gh_load_lang_json)
# ---------------------------------------------------------------------
@app.get("/faq/list")
def faq_list():
    lang = request.args.get("lang", "en").lower()
    lang = "ru" if lang == "ru" else "en"
    try:
        data = gh_load_lang_json(lang)
        faqs = data.get("faqs", [])
        if not faqs:
            # if we reached here, it means GitHub said 0 items after all attempts — return 500-like JSON
            return jsonify({"ok": False, "items": [], "error": f"No FAQs for {lang.upper()} on GitHub."})
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

# ---------------------------------------------------------------------
# Debug helpers (optional but handy)
# ---------------------------------------------------------------------
@app.get("/api/debug/gh-ru-path")
def dbg_ru_path():
    return jsonify({
        "GH_JSON_EN": GH_JSON_EN,
        "GH_JSON_RU": GH_JSON_RU,
        "cached_en": _GH_PATH_CACHE.get("en"),
        "cached_ru": _GH_PATH_CACHE.get("ru"),
        "branch": GH_BRANCH,
        "repo": f"{GH_OWNER}/{GH_REPO}",
    })

@app.post("/api/debug/gh-evict-ru")
def dbg_evict_ru():
    _GH_PATH_CACHE.pop("ru", None)
    return jsonify({"ok": True, "message": "evicted ru path cache"})

# ---------------------------------------------------------------------
# Background poller: keep a local cache fresh for status/RAG (optional)
# ---------------------------------------------------------------------
def _pull_and_cache_from_github() -> bool:
    """Pull EN/RU JSONs from GitHub into local cache; returns True if changed."""
    changed = False
    try:
        os.makedirs(KB_DIR, exist_ok=True)
        for lang, remote in (("en", _GH_PATH_CACHE.get("en") or GH_JSON_EN),
                             ("ru", _GH_PATH_CACHE.get("ru") or GH_JSON_RU)):
            try:
                blob = _gh_get_file_bytes(remote)
            except Exception:
                continue
            local = LOCAL_JSON_EN if lang == "en" else LOCAL_JSON_RU
            old = open(local, "rb").read() if os.path.exists(local) else None
            if old != blob:
                with open(local, "wb") as f:
                    f.write(blob)
                changed = True
                app.logger.info("[kb] cached %s from %s:%s", local, GH_BRANCH, remote)
    except Exception as e:
        app.logger.warning("[kb] cache pull error: %s", e)
    return changed

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

# ---------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------
_BOOT_DONE = False
_BOOT_LOCK = threading.Lock()

def _boot_once():
    global _BOOT_DONE
    with _BOOT_LOCK:
        if _BOOT_DONE:
            return
        # Warm Groq retriever (if PDF exists)
        try:
            if os.path.exists(PDF_PATH):
                warm_index(PDF_PATH)
            else:
                warm_index()
        except Exception as e:
            app.logger.warning("[rag] warm_index failed: %s", e)

        # Start poller
        start_kb_poll(interval_sec=60)
        _BOOT_DONE = True
        app.logger.info("[boot] Ready (GitHub-first FAQs with robust RU loader).")

# Trigger on import and again on first request (idempotent)
_boot_once()

# For local dev convenience
if __name__ == "__main__":
    try:
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    except Exception:
        pass
    app.run(debug=True, use_reloader=False)