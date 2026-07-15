// ============================================================================
// D.R.E.A.M — Groq chat proxy (Render-hosted)
// Holds GROQ_API_KEY as an env var (set on Render, never in the repo).
// The static GitHub-Pages site calls /groq-chat; we proxy to api.groq.com.
// ============================================================================

const express = require('express');

const app = express();
const PORT = process.env.PORT || 3000;
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const DEFAULT_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';

// ── Per-IP rate limit (in-memory, resets on redeploy) ─────────────────────
// Groq free tier = 30 req/min global, 14k/day. 5/min/IP is generous + safe.
const RATE_LIMIT = { windowMs: 60_000, max: 5 };
const ipHits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const hits = (ipHits.get(ip) || []).filter(h => now - h.ts < RATE_LIMIT.windowMs);
  if (hits.length >= RATE_LIMIT.max) return true;
  hits.push({ ts: now });
  ipHits.set(ip, hits);
  return false;
}

// ── Middleware ────────────────────────────────────────────────────────────
app.use(express.json({ limit: '1mb' }));

// CORS for ALL routes — set headers on every response, including errors.
app.use((req, res, next) => {
  res.set('Access-Control-Allow-Origin', '*');
  res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(204).end();
  next();
});

// ── Health check ──────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    ok: true,
    service: 'dream-groq-proxy',
    groq_configured: !!process.env.GROQ_API_KEY,
  });
});

// ── System prompts (mirror of groq_bot.py _system) ────────────────────────
const SYS_EN = `You are D.R.E.A.Mer, the guide bot for the DREAM physics framework (Dimensional Resonant Emergent Attractors in Manifold).
Answer concisely about: the projection kernel (10D→4D), the Retention Law R(λ)=exp[-(λ/λ_q)^D_eff], kernel invariants (λ_q, D_eff, locality, regularity), the coherence cliff, axioms A1–A6, theorems T1–T6, falsification gates, and cross-domain validation (Sweden FTT, China gaming ban, Montréal Protocol, Volcker Rule).
FORMAT: two sections — ### Technical (4–6 bullets, ≤180 words, up to 2 short LaTeX equation blocks) and ### Simply Put (3 bullets, ≤90 words, minimal symbols).
Formulas in LaTeX. Do not translate variable names. Cite formulas when relevant. If a question is off-topic, redirect politely to DREAM themes: projection, retention, kernel invariants, falsifiability.
Finish with <END>.`;

const SYS_RU = `Ты — D.R.E.A.Mer, бот-гид по физической теории DREAM (Dimensional Resonant Emergent Attractors in Manifold).
Отвечай кратко о: проекционном ядре (10D→4D), законе удержания R(λ)=exp[-(λ/λ_q)^D_eff], инвариантах ядра (λ_q, D_eff, локальность, регулярность), когерентном обрыве, аксиомах A1–A6, теоремах T1–T6, фальсификации, кросс-доменной валидации (Швеция FTT, Китай игровой запрет, Монреальский протокол, правило Волкера).
ФОРМАТ: две секции — ### Научно (4–6 пунктов, ≤180 слов, до 2 коротких LaTeX-блоков) и ### Простыми словами (3 пункта, ≤90 слов, минимум символов).
Формулы в LaTeX. Не переводить имена переменных. Если вопрос не по теме, вежливо вернись к темам DREAM.
Заверши сообщением <END>.`;

// ── Main endpoint ─────────────────────────────────────────────────────────
app.post('/groq-chat', async (req, res) => {
  const ip = req.headers['x-forwarded-for']?.split(',')[0] || req.ip || 'unknown';
  if (rateLimited(ip)) {
    return res.status(429).json({ error: 'Rate limit: 5 requests per minute. Try again shortly.' });
  }

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({ error: 'GROQ_API_KEY not set on server. Set it in Render → Environment.' });
  }

  try {
    const { message, lang, model } = req.body || {};
    const userMsg = (message || '').trim();
    if (!userMsg) {
      return res.status(400).json({ error: 'Provide "message" field.' });
    }

    const isRu = lang === 'ru' || /[\u0400-\u04FF]/.test(userMsg);
    const systemPrompt = isRu ? SYS_RU : SYS_EN;

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMsg },
    ];

    const groqResp = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: model || DEFAULT_MODEL,
        messages,
        temperature: 0.3,
        max_tokens: 640,
        top_p: 0.9,
        frequency_penalty: 0.3,
        stop: ['<END>'],
      }),
    });

    if (!groqResp.ok) {
      const errText = await groqResp.text();
      let msg = `Groq API HTTP ${groqResp.status}`;
      try {
        const j = JSON.parse(errText);
        msg += `: ${j.error?.message || errText.slice(0, 200)}`;
      } catch {
        msg += `: ${errText.slice(0, 200)}`;
      }
      return res.status(502).json({ error: msg });
    }

    const groqJson = await groqResp.json();
    let reply = groqJson.choices?.[0]?.message?.content?.trim() || '(no reply)';
    reply = reply.replace(/<END>/g, '').trim();
    res.json({ ok: true, reply });
  } catch (err) {
    console.error('[/groq-chat] error:', err);
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`D.R.E.A.M Groq proxy on :${PORT}`);
  console.log(`  Groq: ${process.env.GROQ_API_KEY ? 'configured' : 'NOT configured (set GROQ_API_KEY on Render)'}`);
});
