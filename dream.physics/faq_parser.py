import os, re, json, datetime, logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

# ---------------- PDF extract ----------------
try:
    from pypdf import PdfReader as _PdfReader
except Exception:
    from PyPDF2 import PdfReader as _PdfReader

HEADER_HINTS = [
    "DREAM Framework — Comprehensive FAQ",
    "Dimensional Resonant Emergent Attractors",
    "Build timestamp",
]
SECTION_SENTINELS = {"REFERENCES", "APPENDIX", "CONCLUSION"}

def _read_pdf_text(pdf_path: str) -> str:
    log.info("Reading PDF: %s  (exists=%s)", pdf_path, os.path.exists(pdf_path))
    with open(pdf_path, "rb") as f:
        reader = _PdfReader(f)
        pages = []
        for idx, p in enumerate(reader.pages):
            txt = p.extract_text() or ""
            log.debug("Page %d chars: %d", idx + 1, len(txt))
            pages.append(txt)
        all_text = "\n\n".join(pages)
        log.info("Total extracted characters: %d", len(all_text))
        return all_text

def _basic_clean(text: str) -> str:
    lines = [ln for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if any(s.startswith(h) for h in HEADER_HINTS):
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", s):
            continue
        cleaned.append(s)
    text = "\n".join(cleaned)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # heal hyphenation
    text = re.sub(r"[ \t]+", " ", text)               # normalize spaces
    text = re.sub(r"\n{3,}", "\n\n", text)            # collapse blank lines
    return text

def _format_math(text: str) -> str:
    if text is None: return ""
    patterns = [
        (r'(\w+)\^(\d+)', r'\1^\2'),
        (r'(\d+)\s*\*\s*(\d+)', r'\1×\2'),
        (r'(\d+)\s*\+\s*(\d+)', r'\1 + \2'),
        (r'(\d+)\s*-\s*(\d+)', r'\1 - \2'),
        (r'(\d+)\s*/\s*(\d+)', r'\1/\2'),
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text

def parse_faq_pdf(pdf_path: str = "kb/dream_faqs.pdf") -> List[Dict[str, Any]]:
    log.info("parse_faq_pdf() starting for %s", os.path.abspath(pdf_path))
    raw = _read_pdf_text(pdf_path)
    if len(raw.strip()) < 50:
        raise RuntimeError("No extractable text found (OCR may be required).")
    text = _basic_clean(raw)
    lines = [ln.strip() for ln in text.splitlines()]

    faqs, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not m:
            i += 1; continue

        qnum = int(m.group(1))
        qbuf = [m.group(2)]
        i += 1
        while i < len(lines) and "?" not in "".join(qbuf):
            nxt = lines[i]
            if re.match(r"^\d+\.\s+", nxt):  # safety
                break
            qbuf.append(nxt); i += 1

        question = " ".join(qbuf).strip()

        abuf = []
        while i < len(lines):
            nxt = lines[i]
            if re.match(r"^\d+\.\s+", nxt): break
            if nxt.upper() in SECTION_SENTINELS: break
            abuf.append(nxt); i += 1

        answer = _format_math(re.sub(r"\s+", " ", " ".join(abuf).strip()))
        if question:
            faqs.append({"number": qnum, "question": question, "answer": answer, "source": "faq_pdf"})

    faqs.sort(key=lambda x: x["number"])
    log.info("Parsed FAQs: %d", len(faqs))
    return faqs

# ---------------- JSON helpers ----------------
def save_faqs_to_json_lang(faqs, lang: str, output_path: str):
    payload = {
        "metadata": {
            "source": "Parsed from structured FAQ PDF",
            "lang": lang,
            "total_faqs": len(faqs),
            "parsed_date": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "faqs": faqs,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("JSON written: %s (lang=%s, total_faqs=%d)", output_path, lang, len(faqs))
    return payload

def load_faqs(json_path="kb/parsed_faqs_en.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["faqs"]

# ---------------- Translation (pluggable; OFF by default) ----------------
# Choose mode via env/CLI: off | google | marian
TRANS_MODE = os.environ.get("FAQ_TRANS_MODE", "off").lower()  # default OFF

_MATH_RE = re.compile(r'(\$[^$]+\$|\\\([^)]*\\\)|\\\[[^]]*\\\])')
def _mask_math(text: str):
    spans = []
    def repl(m):
        idx = len(spans); spans.append(m.group(0)); return f"⟦M{idx}⟧"
    return _MATH_RE.sub(repl, text), spans

def _unmask_math(text: str, spans):
    def repl(m):
        idx = int(m.group(1))
        return spans[idx] if 0 <= idx < len(spans) else m.group(0)
    return re.sub(r'⟦M(\d+)⟧', repl, text)

def _translate_google(texts: List[str]) -> List[str]:
    try:
        from deep_translator import GoogleTranslator
        tr = GoogleTranslator(source="en", target="ru")
        out = []
        for t in texts:
            masked, spans = _mask_math(t or "")
            ru = tr.translate(masked)
            out.append(_unmask_math(ru, spans))
        return out
    except Exception as e:
        log.warning("GoogleTranslator failed: %s. Falling back to EN.", e)
        return texts

def _translate_marian(texts: List[str]) -> List[str]:
    # Import ONLY if requested, to avoid SentencePiece crashes
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        log.warning("transformers not available: %s. Falling back to EN.", e)
        return texts
    try:
        name = "Helsinki-NLP/opus-mt-en-ru"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
        out = []
        for t in texts:
            masked, spans = _mask_math(t or "")
            toks = tok(masked, return_tensors="pt", truncation=True)
            gen = mdl.generate(**toks, max_new_tokens=512)
            ru = tok.decode(gen[0], skip_special_tokens=True)
            out.append(_unmask_math(ru, spans))
        return out
    except Exception as e:
        log.warning("Marian load/translate failed: %s. Falling back to EN.", e)
        return texts

def translate_en_to_ru(texts: List[str]) -> List[str]:
    if not texts:
        return []
    mode = os.environ.get("FAQ_TRANS_MODE", "off").lower()
    log.info("translate_en_to_ru mode: %s", mode)
    if mode == "google":
        return _translate_google(texts)
    if mode == "marian":
        return _translate_marian(texts)
    # "off" or unknown -> mirror EN so the pipeline always works
    return texts

# ---------------- Bilingual builder ----------------
def build_bilingual_jsons(
    pdf_path="kb/dream_faqs.pdf",
    out_en="kb/parsed_faqs_en.json",
    out_ru="kb/parsed_faqs_ru.json",
) -> int:
    en_faqs = parse_faq_pdf(pdf_path)
    save_faqs_to_json_lang(en_faqs, "en", out_en)

    ru_q = translate_en_to_ru([f["question"] for f in en_faqs])
    ru_a = translate_en_to_ru([f["answer"]   for f in en_faqs])
    ru_faqs = [{
        "number": f["number"],
        "question": ru_q[i],
        "answer": ru_a[i],
        "source": "faq_pdf_ru",
    } for i, f in enumerate(en_faqs)]
    save_faqs_to_json_lang(ru_faqs, "ru", out_ru)
    return len(en_faqs)

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="kb/dream_faqs.pdf")
    ap.add_argument("--out-en", default="kb/parsed_faqs_en.json")
    ap.add_argument("--out-ru", default="kb/parsed_faqs_ru.json")
    ap.add_argument("--bilingual", action="store_true", help="Emit both EN and RU jsons")
    ap.add_argument("--trans-mode", choices=["off","google","marian"], default=os.environ.get("FAQ_TRANS_MODE","off"),
                    help="Translation backend (default: off).")
    args = ap.parse_args()

    # Apply CLI choice
    # Apply CLI choice
    os.environ["FAQ_TRANS_MODE"] = args.trans_mode
    # Also update the module variable if present
    try:
        TRANS_MODE = args.trans_mode  # harmless if not defined earlier
    except NameError:
        pass
    log.info("Translation mode: %s", args.trans_mode)

    if args.bilingual:
        n = build_bilingual_jsons(args.pdf, args.out_en, args.out_ru)
        print(f"Bilingual caches written. FAQs: {n}")
    else:
        faqs = parse_faq_pdf(args.pdf)
        save_faqs_to_json_lang(faqs, "en", args.out_en)
        print(f"EN cache written: {args.out_en} (FAQs: {len(faqs)})")