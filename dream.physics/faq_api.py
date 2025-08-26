# faq_api.py
import os, json, threading
from flask import Blueprint, request, jsonify, make_response, render_template

faq_bp = Blueprint("faq", __name__, template_folder="templates")

KB_DIR   = "kb"
EN_JSON  = os.path.join(KB_DIR, "parsed_faqs_en.json")
RU_JSON  = os.path.join(KB_DIR, "parsed_faqs_ru.json")
_LOCK    = threading.Lock()

def _ensure_files():
    os.makedirs(KB_DIR, exist_ok=True)
    for path, lang in [(EN_JSON, "en"), (RU_JSON, "ru")]:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"metadata":{"lang":lang,"total_faqs":0},"faqs":[]}, f, ensure_ascii=False, indent=2)

def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("faqs", data.get("items", []))

def _save(path, faqs, lang):
    payload = {
        "metadata": {"lang": lang, "total_faqs": len(faqs)},
        "faqs": faqs
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _path_for_lang(lang: str) -> str:
    return RU_JSON if (lang or "").lower() == "ru" else EN_JSON

def _lang_from_request() -> str:
    # Prefer explicit ?lang=, else cookie, else 'en'
    lang = request.args.get("lang")
    if not lang:
        lang = request.cookies.get("lang", "en")
    return "ru" if (lang or "").lower() == "ru" else "en"

@faq_bp.before_app_request
def _boot():
    _ensure_files()

# --- Admin page (HTML) ---
@faq_bp.get("/admin")
def admin_page():
    # Renders templates/admin.html
    resp = make_response(render_template("admin.html", title="FAQ Admin"))
    return resp

# --- List all FAQs in given lang ---
@faq_bp.get("/list")
def list_faqs():
    lang = _lang_from_request()
    path = _path_for_lang(lang)
    with _LOCK:
        faqs = _load(path)
    return jsonify({"ok": True, "items": faqs})

# --- Get single FAQ by number ---
@faq_bp.get("/item/<int:number>")
def get_item(number: int):
    lang = _lang_from_request()
    path = _path_for_lang(lang)
    with _LOCK:
        faqs = _load(path)
    for f in faqs:
        if int(f.get("number", -1)) == number:
            return jsonify({"ok": True, "item": f})
    return jsonify({"ok": False, "error": "Not found"}), 404

# --- Save (create/update) a FAQ ---
@faq_bp.post("/save")
def save_item():
    data = request.get_json(silent=True) or {}
    lang = (data.get("lang") or _lang_from_request())
    item = data.get("item") or {}
    q = (item.get("question") or "").strip()
    a = (item.get("answer") or "").strip()
    num = int(item.get("number") or 0)

    if not q:
        return jsonify({"ok": False, "error": "Question is required"}), 400

    path = _path_for_lang(lang)
    with _LOCK:
        faqs = _load(path)
        if num <= 0:
            num = (max([int(f.get("number", 0)) for f in faqs] + [0]) + 1)
        # upsert
        updated = False
        for i, f in enumerate(faqs):
            if int(f.get("number", -1)) == num:
                faqs[i] = {"number": num, "question": q, "answer": a}
                updated = True
                break
        if not updated:
            faqs.append({"number": num, "question": q, "answer": a})
        faqs.sort(key=lambda x: int(x.get("number", 0)))
        _save(path, faqs, lang)

    return jsonify({"ok": True, "number": num})

# --- Delete a FAQ by number ---
@faq_bp.delete("/delete/<int:number>")
def delete_item(number: int):
    lang = _lang_from_request()
    path = _path_for_lang(lang)
    with _LOCK:
        faqs = _load(path)
        new = [f for f in faqs if int(f.get("number", -1)) != number]
        if len(new) == len(faqs):
            return jsonify({"ok": False, "error": "Not found"}), 404
        _save(path, new, lang)
    return jsonify({"ok": True})

# --- Search titles for the FAQ dock (min 3 chars) ---
@faq_bp.get("/search")
def search():
    q = (request.args.get("q") or "").strip()
    lang = _lang_from_request()
    if len(q) < 3:
        return jsonify({"results": []})
    path = _path_for_lang(lang)
    with _LOCK:
        faqs = _load(path)

    ql = q.lower()
    results = []
    for f in faqs:
        title = f.get("question", "")
        if ql in title.lower():
            ans = f.get("answer", "") or ""
            prev = (ans[:100] + "...") if len(ans) > 100 else ans
            results.append({
                "id": int(f.get("number", 0)),
                "question": title,
                "preview": prev
            })
    # optional: sort by simple containment length
    return jsonify({"results": results})

# --- Optional: quick status ---
@faq_bp.get("/status")
def status():
    with _LOCK:
        en = len(_load(EN_JSON)) if os.path.exists(EN_JSON) else 0
        ru = len(_load(RU_JSON)) if os.path.exists(RU_JSON) else 0
    return jsonify({"ok": True, "counts": {"en": en, "ru": ru}})