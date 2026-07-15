// ============================================================================
// D.R.E.A.M — Groq chat bot (static-site version)
// Floating bubble bottom-right; opens a docked panel. Calls a Render-hosted
// Express proxy that holds GROQ_API_KEY as an env var.
//
// Configure BACKEND_URL below (or set window.DREAM_GROQ_BACKEND before this
// script loads). The chat bubble stays hidden until the backend is up.
// ============================================================================

(function () {
  // ─────────────────────────────────────────────────────────────────────────
  // SET THIS to your deployed Render backend URL after `render.yaml` is live.
  // Example: 'https://dream-groq.onrender.com'
  // Leave empty ('') to keep the chat hidden until the backend is deployed.
  // ─────────────────────────────────────────────────────────────────────────
  const BACKEND_URL = (typeof window !== 'undefined' && window.DREAM_GROQ_BACKEND) || 'https://dream-groq.onrender.com';

  const lang = (document.body.getAttribute('data-lang') || document.documentElement.lang || 'en').startsWith('ru') ? 'ru' : 'en';

  const T = {
    en: {
      toggle:    'Ask D.R.E.A.M',
      title:     'D.R.E.A.Mer the Bot',
      placeholder: 'Type your question… (Shift+Enter = newline)',
      sending:   'Thinking…',
      hello:     'Hello! Ask me about the DREAM framework.',
      error:     'Network error talking to the bot.',
      empty:     'Please type a question first.',
      footer:    'Answers are grounded in the D.R.E.A.M theory.',
    },
    ru: {
      toggle:    'Спросить D.R.E.A.M',
      title:     'D.R.E.A.M · ИИ Бот',
      placeholder: 'Введите вопрос… (Shift+Enter = новая строка)',
      sending:   'Думаю…',
      hello:     'Привет! Спроси меня о теории DREAM.',
      error:     'Сетевая ошибка при обращении к боту.',
      empty:     'Введите вопрос.',
      footer:    'Ответы основаны на теории D.R.E.A.M.',
    }
  }[lang];

  // Build the chat UI into the page (replaces the Flask base template's
  // #chat-container / .faq-dock markup with a single source of truth).
  const html = `
    <div id="chat-container">
      <div id="chat-bubble" aria-label="${T.toggle}" title="${T.toggle}" role="button" tabindex="0">
        <span class="chat-icon" aria-hidden="true">✦</span>
        <span class="chat-tooltip">${T.toggle}</span>
      </div>
    </div>
    <div id="faqDock" class="faq-dock" aria-hidden="true" role="dialog" aria-modal="false" aria-label="${T.title}">
      <div class="faq-header">
        <h3 id="faqTitle" class="faq-title">${T.title}</h3>
        <button class="faq-close" id="faqCloseBtn" aria-label="Close">×</button>
      </div>
      <div class="faq-body">
        <div class="faq-main">
          <div id="llmMessages" aria-live="polite">
            <div class="msg bot">${T.hello}</div>
          </div>
          <div class="llm-input-row">
            <textarea id="llmInput" placeholder="${T.placeholder}" rows="1"></textarea>
            <button id="llmSend">${lang === 'ru' ? 'Отправить' : 'Send'}</button>
          </div>
        </div>
      </div>
      <div class="faq-footer">
        <small class="muted">${T.footer}</small>
        <div></div>
      </div>
    </div>
  `;
  document.body.insertAdjacentHTML('beforeend', html);

  const $ = (id) => document.getElementById(id);
  const bubble   = $('chat-bubble');
  const dock     = $('faqDock');
  const closeBtn = $('faqCloseBtn');
  const messages = $('llmMessages');
  const input    = $('llmInput');
  const sendBtn  = $('llmSend');

  function ensureBottomPadding() {
    const bar = document.querySelector('.llm-input-row');
    const h = (bar?.offsetHeight || 56);
    messages.style.paddingBottom = (h + 16) + 'px';
  }
  function scrollBottom() { messages.scrollTop = messages.scrollHeight; }

  const bar = document.querySelector('.llm-input-row');
  if (window.ResizeObserver && bar) {
    const ro = new ResizeObserver(() => { ensureBottomPadding(); scrollBottom(); });
    ro.observe(bar);
  }
  window.addEventListener('resize', ensureBottomPadding);

  function openDock() {
    dock.classList.add('show');
    dock.setAttribute('aria-hidden', 'false');
    setTimeout(() => { ensureBottomPadding(); input.focus(); }, 0);
  }
  function closeDock() {
    dock.classList.remove('show');
    dock.setAttribute('aria-hidden', 'true');
  }
  bubble.addEventListener('click', openDock);
  bubble.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDock(); }
  });
  closeBtn.addEventListener('click', closeDock);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && dock.classList.contains('show')) closeDock();
  });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]
    ));
  }

  // Render **limited** markdown (### headings, ** bold **, $...$ math kept verbatim)
  function renderMarkdown(text) {
    let t = escapeHtml(text);
    // Headings
    t = t.replace(/^###\s+(.*)$/gm, '<strong style="display:block;margin:.4rem 0 .15rem;color:var(--accent);">$1</strong>');
    // Bold
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Inline code
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Newlines
    t = t.replace(/\n/g, '<br>');
    return t;
  }

  function appendMsg(text, who, asHtml = false) {
    const el = document.createElement('div');
    el.className = `msg ${who}`;
    el.innerHTML = asHtml ? text : escapeHtml(text).replace(/\n/g, '<br>');
    messages.appendChild(el);
    ensureBottomPadding();
    return el;
  }

  function typingOn() {
    const t = document.createElement('div');
    t.className = 'msg bot';
    t.dataset.typing = '1';
    t.innerHTML = '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
    messages.appendChild(t);
    ensureBottomPadding();
    scrollBottom();
  }
  function typingOff() {
    const t = messages.querySelector('.msg.bot[data-typing="1"]');
    if (t) t.remove();
    ensureBottomPadding();
    scrollBottom();
  }

  function scrollToBottomOf(targetEl) {
    const m = messages;
    if (!m || !targetEl) return;
    requestAnimationFrame(() => {
      const mRect = m.getBoundingClientRect();
      const tRect = targetEl.getBoundingClientRect();
      const desiredTop = m.scrollTop + (tRect.top - mRect.top);
      m.scrollTo({ top: desiredTop, behavior: 'auto' });
    });
  }

  async function send() {
    const q = (input.value || '').trim();
    if (!q) { input.focus(); return; }

    appendMsg(q, 'user');
    input.value = '';
    ensureBottomPadding();
    typingOn();
    sendBtn.disabled = true;

    try {
      if (!BACKEND_URL) {
        // No backend deployed — give a friendly local reply instead of an error.
        typingOff();
        const fallback = lang === 'ru'
          ? 'Бот ещё не подключён к Groq. Админ: задайте BACKEND_URL в js/bot.js и разверните Render-прокси из папки server/.'
          : 'The bot is not yet wired to Groq. Admin: set BACKEND_URL in js/bot.js and deploy the Render proxy from the server/ folder.';
        const el = appendMsg(fallback, 'bot');
        scrollToBottomOf(el);
        return;
      }

      const r = await fetch(BACKEND_URL + '/groq-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q, lang }),
      });
      const j = await r.json().catch(() => ({ ok: false, reply: T.error }));
      typingOff();

      if (j && (j.ok === true || j.reply)) {
        const el = appendMsg(j.reply || T.error, 'bot', true);
        scrollToBottomOf(el);
        if (window.MathJax && MathJax.typesetPromise) {
          MathJax.typesetPromise([el]).then(() => scrollToBottomOf(el)).catch(() => scrollToBottomOf(el));
        }
      } else {
        const el = appendMsg((j && j.error) || T.error, 'bot');
        scrollToBottomOf(el);
      }
    } catch (e) {
      typingOff();
      const el = appendMsg(T.error, 'bot');
      scrollToBottomOf(el);
      console.error(e);
    } finally {
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
    setTimeout(() => { ensureBottomPadding(); scrollBottom(); }, 0);
  });
  input.addEventListener('input', ensureBottomPadding);

  // If backend is configured, health-check before showing the bubble.
  // Otherwise hide the bubble entirely.
  if (BACKEND_URL) {
    fetch(BACKEND_URL + '/health')
      .then((r) => r.json())
      .then((j) => {
        if (j && j.ok && j.groq_configured) {
          document.getElementById('chat-container').style.display = '';
        }
      })
      .catch(() => { /* keep visible anyway — chat still gives fallback message */ });
  }
})();
