# groq_bot.py
import os, re, html
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
FALLBACK_MODELS      = ("llama-3.1-8b-instant",)  # <-- fix: real tuple
RU_GENERIC_FALLBACK  = os.environ.get("RU_GENERIC_FALLBACK", "1") == "1"
STOP_MARK            = "<END>"

# ================== Lang detect =============
_CYR = re.compile(r"[\u0400-\u04FF]")
def detect_lang(text: str) -> str:
    """RU if any Cyrillic, else EN."""
    return "ru" if _CYR.search(text or "") else "en"

def _has_math(text: str) -> bool:
    return bool(re.search(r"(\\begin\{equation\}|\\\[|\\\(|\$\$|∑|∫|∂|λ|ε|≈|→|←)", text or ""))

def _budget(lang: str, context_text: str, user_cap: int) -> int:
    base = 420
    if lang == "ru":
        base = int(base * 1.35)  # ≈ 567
    if _has_math(context_text):
        base = int(base * 1.15)  # +15% for equations
    return min(user_cap, base)

# ================== Retrieval ===============
@dataclass
class Chunk:
    text: str
    start_page: int
    end_page: int

class DreamRetriever:
    def __init__(self, pdf_path: str, max_words_per_chunk: int = 220):
        self.pdf_path = pdf_path
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

    def _read_pdf(self) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
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
            chunks.append(Chunk(text=text, start_page=int(st_pg or end_pg), end_page=end_pg))
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

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[a-zA-Zа-яА-Я0-9]+", s.lower())

    def build(self) -> None:
        pages = self._read_pdf()
        self.chunks = self._chunk_pages(pages)
        self._tokenized = [self._tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, k: int = 6, min_rel: float = 0.28) -> List[Chunk]:
        """BM25 with intro-page penalty and relevance threshold; NumPy-safe."""
        if not self._bm25:
            self.build()

        toks = self._tokenize(query)
        if not toks:
            return []

        # get_scores may be a NumPy array; convert safely to Python list[float]
        scores_arr = self._bm25.get_scores(toks)
        scores = list(map(float, getattr(scores_arr, "tolist", lambda: scores_arr)()))
        if len(scores) == 0:
            return []

        max_s = max(scores)
        if max_s <= 0:
            return []

        # Penalize early (intro) pages slightly
        def penalized(i: int) -> float:
            mid_pg = (self.chunks[i].start_page + self.chunks[i].end_page) / 2.0
            penalty = 0.85 if mid_pg <= 3 else 1.0
            return scores[i] * penalty

        ranked = sorted(range(len(scores)), key=penalized, reverse=True)

        picks: List[Chunk] = []
        for i in ranked:
            if scores[i] < (min_rel * max_s):
                break  # remaining are weaker
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

    # ------ private: build system (PDF context embedded) ------
    def _system(self, lang: str, layman_header: str, context_text: str) -> str:
        system_lang = "Русский" if lang == "ru" else "English"

        ru_lex_rules = (
            "When replying in Russian, avoid English nouns entirely (no code-switching). "
            "Prefer Russian equivalents for technical terms; keep math symbols/LaTeX unmodified. "
            "Use: dimension→измерение; dimensions→измерения; manifold→многообразие; kernel (operator)→ядро; "
            "projection/map→проекция/отображение; density→плотность; metric→метрика; invariant→инвариант; "
            "topology→топология; spectrum→спектр; resolution→разрешение; scale→масштаб; coarse-graining→укрупнение масштаба; "
            "fiber (bundle)→слой/расслоение; Jacobian→Якобиан."
        )

        parts = [
            "You are highly educated D.R.E.A.M. Assistant.",
            f"- Always reply in {system_lang}.",
            "- Use the provided context to ground your answer, but DO NOT quote verbatim and DO NOT mention page numbers or citations.",
            "- When formulas help, present them in LaTeX and briefly define symbols.",
            "- Preserve all formulas and symbols exactly in LaTeX; do not translate variable names.",
        ]
        if lang == "ru":
            parts.append(f"- {ru_lex_rules}")

        # Strict brevity + headings + grounding (localized)
        parts += [
            "- Be concise. No preambles, no apologies, no repetition.",
            "- OUTPUT FORMAT: exactly two markdown sections with these headings:",
            "  • Use the heading `### Technical`.",
            f"  • Use the heading `### {layman_header}`.",
            "- Under **Technical**: 3–5 bullets, ≤150 words total, ≤1 equation block.",
            f"- Under **{layman_header}**: 2–3 bullets, ≤80 words total; minimal symbols.",
            "- Always include BOTH sections even if context is insufficient.",
            "- Do not restate the question.",
            "- Always include any equations that appear in the context; do not omit them.",
            f"- End your message with {STOP_MARK} (do not output anything after it).",
        ]

        if lang == "ru":
            parts += [
                "- Используй ТОЛЬКО сведения из раздела Context (PDF) ниже.",
                "НЕ начинай ответ словами «DREAM:» и не давай общий обзор фреймворка; начинай сразу с `### Technical`.",
            ]
        else:
            parts += [
                "- Use ONLY the information present in the Context (PDF) section below.",
                "Do NOT start the answer with 'DREAM:' or any framework boilerplate; start directly with `### Technical`.",
            ]

        base = "\n".join(parts)

        # Embed the PDF chunks inside the system message
        if context_text and context_text != "NO_CONTEXT":
            base += "\n\n---\n### Context (PDF)\n" + context_text + "\n---"

        return base

    @staticmethod
    def _strip_ru_anchor(q: str) -> str:
        # remove a trailing "в DREAM" (optionally preceded by "во") before punctuation/whitespace/EOI
        return re.sub(r'(\s*(?:в|во)\s*DREAM)(?=[\s\.\!\?…]*$)', '', (q or ''), flags=re.I).strip()

    # ------ private: compact helpers ------
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

    # ------ private: RU generic fallback ------
    def _generic_answer(self, question: str, layman_header: str) -> str:
        system = (
            "You are a concise subject-matter explainer.\n"
            "- Reply in Russian.\n"
            "- Start with EXACTLY one line: '⚠️ Общее объяснение (не из DREAM)'.\n"
            "- Then output EXACTLY two markdown sections with these headings:\n"
            "  • `### Technical` — 3–5 bullets, ≤150 words total, ≤1 equation block.\n"
            f"  • `### {layman_header}` — 2–3 bullets, ≤80 words total; minimal symbols.\n"
            "- Do not restate the question.\n"
            "- Do NOT mention DREAM, PDF, or any framework; if the question contains them, IGNORE them and explain the concept in general.\n"
            f"- End your message with {STOP_MARK} (nothing after it)."
        )
        user = (
            "Дай краткое общее объяснение темы из вопроса ниже. "
            "Не ссылайся на PDF/страницы и не упоминай фреймворки (включая DREAM).\n\n"
            f"Вопрос:\n{question.strip()}\n\n"
            "Ответ:"
        )

        try:
            gen_tok = _budget("ru", "", self.max_tokens)
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=min(self.temperature, 0.3),
                max_tokens=min(gen_tok, 420),
                top_p=0.9,
                frequency_penalty=0.35,
                presence_penalty=0.0,
                stop=[STOP_MARK],
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return (
                "⚠️ Общее объяснение (не из DREAM)\n\n"
                "### Technical\n"
                "- Краткий общий ответ недоступен.\n"
                "- Повторите попытку позднее.\n\n"
                f"### {layman_header}\n"
                "- Сейчас не удалось сформировать общий ответ.\n"
                f"{STOP_MARK}"
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
        layman_header = "Простыми словами" if lang == "ru" else "Simply Put"

        # Context: top-3 chunks, trimmed to save tokens
        ctx_blocks: List[str] = []
        for c in (context_chunks or [])[:3]:
            header = f"[pages {c.start_page}–{c.end_page}]"
            text = self._compact(c.text or "", 1400)
            ctx_blocks.append(f"{header}\n{text}")
        context = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "NO_CONTEXT"

        # ----- Context gate: avoid generic answers when overlap is poor -----
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
                    "### Technical\n"
                    "- Недостаточно контекста из PDF для точного ответа.\n"
                    f"- В контексте нет ключевых терминов запроса: {missing}.\n"
                    "- Уточните разделы/формулы или переформулируйте вопрос.\n\n"
                    f"### {layman_header}\n"
                    "- В показанных фрагментах нет нужных деталей.\n"
                    "- Скажите, какие именно разделы или формулы нужны.\n"
                    f"{STOP_MARK}"
                )
            else:
                missing = ", ".join([t for t in (self._key_terms(question, lang)) if t not in (context.lower() if context != "NO_CONTEXT" else "")][:5]) or "specific terms"
                return (
                    "### Technical\n"
                    "- Context from the PDF is insufficient for a precise answer.\n"
                    f"- The context lacks key query terms: {missing}.\n"
                    "- Please point to sections/formulas or rephrase the question.\n\n"
                    f"### {layman_header}\n"
                    "- The provided snippets don’t include the needed details.\n"
                    "- Tell me which sections or formulas you want.\n"
                    f"{STOP_MARK}"
                )

        # If overlap is weak, either generic RU fallback or explicit insufficient
        if context == "NO_CONTEXT" or len(hits_list) < need_hits:
            if lang == "ru" and RU_GENERIC_FALLBACK:
                q_plain = self._strip_ru_anchor(question)  # <-- use de-anchored question
                return self._generic_answer(q_plain, layman_header)
            return _insufficient_reply()

        # ----- Build prompts (localized user turn) -----
        system = self._system(lang, layman_header, context)

        if lang == "ru":
            user = (
                f"Вопрос:\n{question.strip()}\n\n"
                "Отвечай, используя ТОЛЬКО раздел «Context (PDF)» в системном сообщении выше.\n"
                "Строго следуй OUTPUT FORMAT.\n"
                "Ответ:"
            )
        else:
            user = (
                f"Question:\n{question.strip()}\n\n"
                "Answer using ONLY the \"Context (PDF)\" section in the system message above.\n"
                "Follow the OUTPUT FORMAT exactly.\n"
                "Answer:"
            )

        # Messages: keep last 3 turns; never carry old system prompts
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for role, content in (history or [])[-6:]:
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": self._compact(content or "", 2500)})
        messages.append({"role": "user", "content": user})

        # --- API call with concise knobs ---
        def _call(model_name: str, max_tok: int):
            return self.client.chat.completions.create(
                model=model_name,
                temperature=self.temperature,
                max_tokens=max_tok,
                top_p=0.9,
                frequency_penalty=0.35,
                presence_penalty=0.0,
                stop=[STOP_MARK],
                messages=messages,
            )

        try:
            max_tok = _budget(lang, context, self.max_tokens)  # <-- apply RU/math budget
            resp = _call(self.model, max_tok)
        except Exception as e:
            err = str(e)
            if "rate limit" in err.lower() or "429" in err:
                for fb in FALLBACK_MODELS:
                    try:
                        resp = _call(fb, max_tok)  # <-- keep same budget
                        break
                    except Exception:
                        continue
                else:
                    return "Sorry, the model is busy. Please retry shortly."
            else:
                return f"Error: {html.escape(err)}"

        out = (resp.choices[0].message.content or "").strip()
        finish_reason = getattr(resp.choices[0], "finish_reason", None)

        # Ensure the layman section exists; if missing, add it briefly
        if f"### {layman_header}".lower() not in out.lower():
            try:
                fix_tok = min(140, max(80, int(max_tok * 0.3)))
                fix = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=fix_tok,
                    frequency_penalty=0.4,
                    presence_penalty=0.0,
                    stop=[STOP_MARK],
                    messages=messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user",
                         "content": f"Add ONLY the missing second section with this heading:\n\n### {layman_header}\n- 2–3 bullets, ≤80 words total.\n{STOP_MARK}"},
                    ],
                )
                out += "\n" + (fix.choices[0].message.content or "").strip()
            except Exception:
                pass

        # Brief continuation to finish bullets/equation if cut
        if finish_reason == "length":
            try:
                cont_tok = min(180, max(100, int(max_tok * 0.35)))
                cont = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=cont_tok,
                    frequency_penalty=0.4,
                    presence_penalty=0.0,
                    stop=[STOP_MARK],
                    messages=messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user",
                         "content": f"Continue briefly and finish the '{layman_header}' section within 3 bullets."},
                    ],
                )
                out += "\n" + (cont.choices[0].message.content or "").strip()
            except Exception:
                pass

        return out

# ================== Facade & helpers ===================
_retriever: Optional[DreamRetriever] = None
_answerer:  Optional[GroqAnswerer]  = None

def _pick_default_pdf() -> str:
    envp = os.environ.get("DREAM_PDF_PATH")
    if envp and os.path.exists(envp):
        return envp
    for p in ("kb/dream_faqs.pdf", "kb/dream.pdf"):
        if os.path.exists(p):
            return p
    return "kb/dream.pdf"  # last resort

def warm_index(pdf_path: Optional[str] = None) -> None:
    """Build the retriever index once (idempotent)."""
    global _retriever
    pdf_path = pdf_path or _pick_default_pdf()
    _retriever = DreamRetriever(pdf_path)
    _retriever.build()

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
    """Append 'в DREAM' to RU questions unless DREAM is already present.
    Keeps trailing punctuation (., !, ?, …) at the very end."""
    if lang != "ru":
        return text
    t = (text or "").strip()
    if re.search(r"\bDREAM\b", t, re.IGNORECASE):
        return t  # already anchored
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
    """Convenience wrapper used by app.py."""
    global _answerer, _retriever
    if _retriever is None:
        warm_index()
    if _answerer is None:
        _answerer = GroqAnswerer()

    resolved_lang = (lang or detect_lang(question)).lower()
    resolved_lang = "ru" if resolved_lang == "ru" else "en"

    # Always anchor RU questions (helps retrieval)
    q_aug = _append_ru_anchor(question, resolved_lang)

    # Retrieval (with overlap logic inside _select_chunks)
    chunks = _select_chunks(q_aug, resolved_lang, top_k)

    # Ask the model with the same augmented question
    return _answerer.answer(q_aug, chunks, lang=resolved_lang, history=history)