# groq_bot.py — drop-in, token-optimized prompts for richer EN/RU answers & formulas (free-tier friendly)
import os, re, html, json, threading
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# ----- PDF reader (pypdf preferred) -----
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    from PyPDF2 import PdfReader  # fallback

from rank_bm25 import BM25Okapi
from groq import Groq

# ================== Config ==================
DEFAULT_MODEL        = os.environ.get("GROQ_MODEL","llama-3.3-70b-versatile")
FALLBACK_MODELS      = ("llama-3.1-8b-instant",)
RU_GENERIC_FALLBACK  = os.environ.get("RU_GENERIC_FALLBACK", "1") == "1"
STOP_MARK            = "<END>"

# KB paths (override via env if needed)
FAQ_EN_PATH = os.environ.get("FAQ_EN_PATH", "kb/parsed_faqs_en.json")
FAQ_RU_PATH = os.environ.get("FAQ_RU_PATH", "kb/parsed_faqs_ru.json")

# ================== Lang detect =============
_CYR = re.compile(r"[\u0400-\u04FF]")
def detect_lang(text: str) -> str:
    """RU if any Cyrillic, else EN."""
    return "ru" if _CYR.search(text or "") else "en"

_MATH_RE = re.compile(r"(\\begin\{equation\}|\\\[|\\\(|\$\$|∑|∫|∂|λ|ε|≈|→|←|∇|∞|√|ℏ|±)")
def _has_math(text: str) -> bool:
    return bool(_MATH_RE.search(text or ""))

def _budget(lang: str, context_text: str, user_cap: int) -> int:
    """
    More headroom for formulas, still safe for free tier:
    EN base≈520; RU +30%; math +20%; hard cap 640.
    """
    base = 520
    if lang == "ru":
        base = int(base * 1.30)
    if _has_math(context_text):
        base = int(base * 1.20)
    return min(user_cap, base, 640)

# ================== Retrieval ===============
@dataclass
class Chunk:
    text: str
    start_page: int
    end_page: int
    source: str  # 'pdf' | 'faq-en' | 'faq-ru'

class DreamRetriever:
    """
    Unified retriever over:
      - PDF pages (chunked by words)
      - EN/RU FAQ JSONs (each FAQ becomes a chunk)
    """
    def __init__(
        self,
        pdf_path: str,
        faq_en_path: Optional[str] = None,
        faq_ru_path: Optional[str] = None,
        max_words_per_chunk: int = 220,
    ):
        self.pdf_path = pdf_path
        self.faq_en_path = faq_en_path or FAQ_EN_PATH
        self.faq_ru_path = faq_ru_path or FAQ_RU_PATH
        self.max_words_per_chunk = max_words_per_chunk

        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: List[List[str]] = []

    @staticmethod
    def _clean(s: str) -> str:
        s = s.replace("\r", "")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{2,}", "\n", s)
        return s.strip()

    # ---------- PDF ----------
    def _read_pdf(self) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            return pages
        with open(self.pdf_path, "rb") as f:
            pdf = PdfReader(f)
            for i, p in enumerate(getattr(pdf, "pages", [])):
                t = (p.extract_text() or "")
                pages.append((i + 1, self._clean(t)))
        return pages

    def _chunk_pages(self, pages: List[Tuple[int, str]]) -> List[Chunk]:
        buf: List[str] = []
        st_pg: Optional[int] = None
        chunks: List[Chunk] = []
        words_in_buf = 0

        def flush(end_pg: int) -> None:
            nonlocal buf, st_pg, words_in_buf
            if not buf:
                return
            text = DreamRetriever._clean(" ".join(buf))
            chunks.append(Chunk(text=text, start_page=int(st_pg or end_pg), end_page=end_pg, source="pdf"))
            buf, st_pg, words_in_buf = [], None, 0

        for pg, txt in pages:
            for para in [p for p in txt.split("\n") if p.strip()]:
                w = para.split()
                if st_pg is None:
                    st_pg = pg
                if words_in_buf and words_in_buf + len(w) > self.max_words_per_chunk:
                    flush(pg)
                    st_pg = pg
                buf.append(para)
                words_in_buf += len(w)

        if st_pg is not None:
            flush(pages[-1][0] if pages else 1)
        return chunks

    # ---------- FAQ JSONs ----------
    @staticmethod
    def _safe_load_json(path: str) -> Dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _faq_chunks_from_json(self, path: str, src_tag: str) -> List[Chunk]:
        payload = self._safe_load_json(path)
        faqs = payload.get("faqs", []) if isinstance(payload, dict) else []
        out: List[Chunk] = []
        for f in faqs:
            q = (f.get("question") or "").strip()
            a = (f.get("answer") or "").strip()
            if not (q or a):
                continue
            txt = self._clean(f"Q: {q}\nA: {a}")
            out.append(Chunk(text=txt, start_page=-1, end_page=-1, source=src_tag))
        return out

    # ---------- Build unified index ----------
    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[a-zA-Zа-яА-Я0-9]+", s.lower())

    def build(self) -> None:
        self.chunks = []
        # PDF
        pages = self._read_pdf()
        self.chunks.extend(self._chunk_pages(pages))
        # FAQs
        self.chunks.extend(self._faq_chunks_from_json(self.faq_en_path, "faq-en"))
        self.chunks.extend(self._faq_chunks_from_json(self.faq_ru_path, "faq-ru"))
        # BM25
        self._tokenized = [self._tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, k: int = 6, min_rel: float = 0.28) -> List[Chunk]:
        """BM25 with intro-page penalty for PDF; relevance threshold; NumPy-safe."""
        if not self._bm25:
            self.build()

        toks = self._tokenize(query)
        if not toks:
            return []

        scores_arr = self._bm25.get_scores(toks)
        scores = list(map(float, getattr(scores_arr, "tolist", lambda: scores_arr)()))
        if not scores:
            return []

        max_s = max(scores)
        if max_s <= 0:
            return []

        # Penalize early (intro) pages slightly — only for PDF chunks
        def penalized(i: int) -> float:
            if self.chunks[i].source != "pdf":
                return scores[i]
            mid_pg = (self.chunks[i].start_page + self.chunks[i].end_page) / 2.0
            return scores[i] * (0.85 if mid_pg <= 3 else 1.0)

        ranked = sorted(range(len(scores)), key=penalized, reverse=True)

        picks: List[Chunk] = []
        for i in ranked:
            if scores[i] < (min_rel * max_s):
                break
            picks.append(self.chunks[i])
            if len(picks) >= k:
                break
        return picks

# ================== Answerer =================
class GroqAnswerer:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_tokens: int = 800):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in environment.")
        self.client = Groq(api_key=api_key)
        self.model = model.strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    # ------ private: build system (Context embedded) ------
    def _system(self, lang: str, layman_header: str, context_text: str) -> str:
        """Short, directive, token-light system prompt (keeps all rules)."""
        if lang == "ru":
            rules_ru = (
                "Русский ответ. Используй ТОЛЬКО раздел Context ниже. Без прелюдий, извинений, ссылок/страниц. "
                "Формулы — LaTeX, кратко поясни символы. Не переводить имена переменных. "
                "ФОРМАТ: две секции — `### Научно` и `### {lay}`. "
                "Technical: 4–6 пунктов, ≤180 слов суммарно, до 2 коротких формульных блоков. "
                "{lay}: 3 пункта, ≤90 слов, минимум символов. Всегда выдай ОБЕ секции. "
                f"Заверши сообщением {STOP_MARK}."
            ).format(lay=layman_header)
            base = rules_ru
        else:
            base = (
                "English reply. Use ONLY the Context below. No preambles/apologies/citations/pages. "
                "Formulas in LaTeX; briefly define symbols. Do not translate variable names. "
                "FORMAT: two sections — `### Technical` and `### {lay}`. "
                "Technical: 4–6 bullets, ≤180 words total, up to 2 short equation blocks. "
                "{lay}: 3 bullets, ≤90 words, minimal symbols. Always output BOTH sections. "
                f"Finish with {STOP_MARK}."
            ).format(lay=layman_header)

        if context_text and context_text != "NO_CONTEXT":
            base += "\n\n---\n### Context\n" + context_text + "\n---"
        return base

    @staticmethod
    def _strip_ru_anchor(q: str) -> str:
        return re.sub(r'(\s*(?:в|во)\s*DREAM)(?=[\s\.\!\?…]*$)', '', (q or ''), flags=re.I).strip()

    @staticmethod
    def _compact(s: str, max_chars: int) -> str:
        s = s.replace("\r", "")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s[:max_chars]

    @staticmethod
    def _tokens(s: str) -> List[str]:
        return re.findall(r"[a-zA-Zа-яА-Я0-9]+", (s or "").lower())

    def _key_terms(self, question: str, lang: str) -> List[str]:
        EN_STOP = {"the","a","an","and","or","of","to","in","on","for","with","by","from","at","as",
                   "is","are","was","were","be","being","been","that","this","these","those","it","its",
                   "we","you","he","she","they","i","my","our","your","their","but","if","than","then",
                   "about","into","over","after","before","not","no","yes","do","does","did","can","could"}
        RU_STOP = {"и","или","а","но","к","в","во","на","над","под","о","об","от","до","по","при","для",
                   "это","тот","та","те","этот","эта","эти","что","как","так","же","не","да","нет","ни",
                   "есть","быть","был","была","были","будет","мы","вы","он","она","они","мой","наш","ваш","их"}
        stop = RU_STOP if lang == "ru" else EN_STOP
        toks = self._tokens(question)
        out, seen = [], set()
        for t in toks:
            if len(t) >= 3 and t not in stop and t not in seen:
                seen.add(t); out.append(t)
        return out[:8]

    def _generic_answer(self, question: str, layman_header: str) -> str:
        """RU-only generic fallback, token-light, still structured."""
        sys = (
            "Русский ответ. Без упоминаний DREAM/PDF. "
            "ФОРМАТ: `### Technical` (4–6 пунктов, ≤180 слов, до 2 коротких формул) и `### {lay}` (3 пункта, ≤90 слов). "
            f"Заверши {STOP_MARK}."
        ).format(lay=layman_header)
        usr = "Дай краткое общее объяснение по теме вопроса:\n" + question.strip() + "\n\nОтвет:"

        try:
            gen_tok = _budget("ru", "", self.max_tokens)
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=min(self.temperature, 0.25),
                max_tokens=min(gen_tok, 560),
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.0,
                stop=[STOP_MARK],
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return (
                "### Technical\n- Временная ошибка генерации.\n- Повторите позже.\n\n"
                f"### {layman_header}\n- Сейчас ответ недоступен.\n{STOP_MARK}"
            )

    # ------ public: make an answer ------
    def answer(
        self,
        question: str,
        context_chunks: List[Chunk],
        lang: Optional[str] = None,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:

        lang = (lang or detect_lang(question)).lower()
        lang = "ru" if lang == "ru" else "en"
        science_header = "Научно" if lang == "ru" else "Technical"
        layman_header = "Простыми словами" if lang == "ru" else "Simply Put"

        # Build compact context (top 3); shorter by default, longer when math present
        ctx_blocks: List[str] = []
        for c in (context_chunks or [])[:3]:
            limit = 1200 if _has_math(c.text) else 900
            ctx_blocks.append(self._compact(c.text or "", limit))
        context = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "NO_CONTEXT"

        # Context gate
        if context != "NO_CONTEXT":
            ctx_text_lc = context.lower()
            q_terms = self._key_terms(question, lang)
            need_hits = 1 if len(q_terms) <= 3 else 2
            hits_list = [t for t in q_terms if t in ctx_text_lc]
        else:
            hits_list, need_hits = [], 1

        def _insufficient_reply() -> str:
            if lang == "ru":
                missing = ", ".join([t for t in (self._key_terms(question, lang)) if t not in (context.lower() if context != "NO_CONTEXT" else "")][:5]) or "нужные термины"
                return (
                    "### Technical\n- Недостаточно контекста для точного ответа.\n"
                    f"- В контексте нет ключевых терминов: {missing}.\n\n"
                    f"### {layman_header}\n- Уточните разделы/формулы или переформулируйте вопрос.\n{STOP_MARK}"
                )
            else:
                missing = ", ".join([t for t in (self._key_terms(question, lang)) if t not in (context.lower() if context != "NO_CONTEXT" else "")][:5]) or "specific terms"
                return (
                    "### Technical\n- Context is insufficient for a precise answer.\n"
                    f"- Missing key terms in context: {missing}.\n\n"
                    f"### {layman_header}\n- Point to sections/formulas or rephrase the question.\n{STOP_MARK}"
                )

        if context == "NO_CONTEXT" or len(hits_list) < need_hits:
            if lang == "ru" and RU_GENERIC_FALLBACK:
                q_plain = self._strip_ru_anchor(question)
                return self._generic_answer(q_plain, layman_header)
            return _insufficient_reply()

        # Prompts
        system = self._system(lang, layman_header, context)

        user = (
            (f"Вопрос:\n{question.strip()}\n\n"
             "Отвечай, опираясь ТОЛЬКО на раздел «Context» системного сообщения. Следуй формату.\nОтвет:")
            if lang == "ru"
            else
            (f"Question:\n{question.strip()}\n\n"
             'Answer using ONLY the "Context" section in the system message. Follow the format.\nAnswer:')
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for role, content in (history or [])[-6:]:
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": self._compact(content or "", 1600)})
        messages.append({"role": "user", "content": user})

        def _call(model_name: str, max_tok: int):
            return self.client.chat.completions.create(
                model=model_name,
                temperature=self.temperature,
                max_tokens=max_tok,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.0,
                stop=[STOP_MARK],
                messages=messages,
            )

        try:
            max_tok = _budget(lang, context, self.max_tokens)
            resp = _call(self.model, max_tok)
        except Exception as e:
            err = str(e)
            if "rate limit" in err.lower() or "429" in err:
                for fb in FALLBACK_MODELS:
                    try:
                        resp = _call(fb, max_tok)
                        break
                    except Exception:
                        continue
                else:
                    return "Sorry, the model is busy. Please retry shortly."
            else:
                return f"Error: {html.escape(err)}"

        out = (resp.choices[0].message.content or "").strip()
        finish_reason = getattr(resp.choices[0], "finish_reason", None)

        # Ensure layman section exists
        if f"### {layman_header}".lower() not in out.lower():
            try:
                fix_tok = min(140, max(80, int(max_tok * 0.28)))
                fix = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=fix_tok,
                    frequency_penalty=0.35,
                    presence_penalty=0.0,
                    stop=[STOP_MARK],
                    messages=messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user", "content": f"Add ONLY the missing second section:\n\n### {layman_header}\n- 3 bullets, ≤90 words total.\n{STOP_MARK}"},
                    ],
                )
                out += "\n" + (fix.choices[0].message.content or "").strip()
            except Exception:
                pass

        # Brief continuation if cut
        if finish_reason == "length":
            try:
                cont_tok = min(160, max(90, int(max_tok * 0.30)))
                cont = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=cont_tok,
                    frequency_penalty=0.35,
                    presence_penalty=0.0,
                    stop=[STOP_MARK],
                    messages=messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user", "content": f"Continue briefly and finish the '{layman_header}' section within 3 bullets."},
                    ],
                )
                out += "\n" + (cont.choices[0].message.content or "").strip()
            except Exception:
                pass

        return out

# ================== Facade & hot-reload ===================
_retriever: Optional[DreamRetriever] = None
_answerer:  Optional[GroqAnswerer]  = None

# Signature of KB inputs to detect staleness
_KB_SIG: Optional[Tuple] = None
_RELOAD_LOCK = threading.Lock()

def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def _pick_default_pdf() -> str:
    envp = os.environ.get("DREAM_PDF_PATH")
    if envp and os.path.exists(envp):
        return envp
    for p in ("kb/dream_faqs.pdf", "kb/dream.pdf"):
        if os.path.exists(p):
            return p
    return "kb/dream.pdf"

def _kb_signature(custom_pdf: Optional[str] = None) -> Tuple:
    pdf = custom_pdf or _pick_default_pdf()
    return (
        pdf, _file_mtime(pdf),
        FAQ_EN_PATH, _file_mtime(FAQ_EN_PATH),
        FAQ_RU_PATH, _file_mtime(FAQ_RU_PATH),
    )

def reload_index(force: bool = False) -> bool:
    """
    Rebuild the unified retriever if inputs changed (or on demand).
    Returns True if a rebuild actually happened.
    """
    global _retriever, _KB_SIG
    with _RELOAD_LOCK:
        sig = _kb_signature()
        if force or _retriever is None or sig != _KB_SIG:
            pdf_path = sig[0]
            _retriever = DreamRetriever(pdf_path, FAQ_EN_PATH, FAQ_RU_PATH)
            _retriever.build()
            _KB_SIG = sig
            return True
        return False

def _refresh_if_stale() -> None:
    """Cheap call per request; rebuilds only if something changed."""
    reload_index(False)

def warm_index(pdf_path: Optional[str] = None) -> None:
    """Build the retriever index once (idempotent)."""
    global _retriever, _KB_SIG
    with _RELOAD_LOCK:
        if pdf_path:
            _retriever = DreamRetriever(pdf_path, FAQ_EN_PATH, FAQ_RU_PATH)
            _retriever.build()
            _KB_SIG = _kb_signature(pdf_path)
        else:
            reload_index(force=True)

# --- lightweight term extraction + anchored retrieval ---
_EN_STOP = {"the","a","an","and","or","of","to","in","on","for","with","by","from","at","as",
            "is","are","was","were","be","being","been","that","this","these","those","it","its",
            "we","you","he","she","they","i","my","our","your","their","but","if","than","then",
            "about","into","over","after","before","not","no","yes","do","does","did","can","could"}
_RU_STOP = {"и","или","а","но","к","в","во","на","над","под","о","об","от","до","по","при","для",
            "это","тот","та","те","этот","эта","эти","что","как","так","же","не","да","нет","ни",
            "есть","быть","был","была","были","будет","мы","вы","он","она","они","мой","наш","ваш","их"}

def _terms(text: str, lang: str) -> List[str]:
    toks = re.findall(r"[a-zA-Zа-яА-Я0-9]+", (text or "").lower())
    stop = _RU_STOP if lang == "ru" else _EN_STOP
    out, seen = [], set()
    for t in toks:
        if len(t) >= 3 and t not in stop and t not in seen:
            seen.add(t); out.append(t)
    return out

def _select_chunks(question: str, lang: str, top_k: int) -> List[Chunk]:
    global _retriever
    base = _retriever.search(question, k=top_k)
    if not base:
        anchored_q = question + (" в контексте DREAM" if lang == "ru" else " in the DREAM context")
        return _retriever.search(anchored_q, k=top_k)

    ctx_text = " ".join(c.text for c in base).lower()
    qts = _terms(question, lang)
    need = 1 if len(qts) <= 3 else 2
    hits = sum(1 for t in qts if t in ctx_text)

    if hits >= need:
        return base

    anchored_q = question + (" в фреймворке DREAM" if lang == "ru" else " in the DREAM framework")
    alt = _retriever.search(anchored_q, k=top_k)
    return alt or base

def _append_ru_anchor(text: str, lang: str) -> str:
    """Append 'в DREAM' to RU questions unless DREAM is already present."""
    if lang != "ru":
        return text
    t = (text or "").strip()
    if re.search(r"\bDREAM\b", t, re.IGNORECASE):
        return t
    m = re.search(r"([\.!\?…]+)$", t)
    if m:
        return t[:m.start()] + " в DREAM" + m.group(0)
    return t + " в DREAM"

def groq_answer(
    question: str,
    top_k: int = 6,
    lang: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Convenience wrapper used by app.py (hot-reload + unified KB)."""
    global _answerer, _retriever
    _refresh_if_stale()
    if _retriever is None:
        warm_index()
    if _answerer is None:
        _answerer = GroqAnswerer()

    resolved_lang = (lang or detect_lang(question)).lower()
    resolved_lang = "ru" if resolved_lang == "ru" else "en"

    # RU anchoring helps retrieval
    q_aug = _append_ru_anchor(question, resolved_lang)

    # Retrieval over PDF + FAQs
    chunks = _select_chunks(q_aug, resolved_lang, top_k)

    # Ask the model
    return _answerer.answer(q_aug, chunks, lang=resolved_lang, history=history)