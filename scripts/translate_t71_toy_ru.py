#!/usr/bin/env python3
"""Translate visible text in t71-toy_ru.html to Russian (clean re-do)."""
import re

with open('t71-toy_ru.html') as f:
    html = f.read()

# Apply replacements in a sequence
replacements = [
    ('<html lang="en">', '<html lang="ru">'),
    ('<title>D.R.E.A.M. — T7.1 Multi-Regime S2 Toy (live data, multi-model)</title>',
     '<title>D.R.E.A.M. — T7.1 Мультирежимный S2 (живые данные, мульти-модель)</title>'),

    # Header
    ('<h1>T7.1 Multi-Regime S2 Toy</h1>', '<h1>T7.1 Мультирежимный S2 — Toy</h1>'),
    ('<span class="kicker">Live datasets · 6 competing models · piecewise vs single-S2 head-to-head</span>',
     '<span class="kicker">Живые датасеты · 6 моделей · кусочный vs одиночный S2 — прямое сравнение</span>'),
    ('<span class="badge t71">ΔAICc &amp; binomial p vs single-S2 null</span>',
     '<span class="badge t71">ΔAICc &amp; биномиальное p vs нулевой одиночного S2</span>'),
    ('<a class="home-link" href="en/retention.html#t71">← back to T7.1</a>',
     '<a class="home-link" href="ru/retention.html#t71">← назад к T7.1</a>'),

    # Controls
    ('<label for="dataset">Live dataset:</label>', '<label for="dataset">Живой датасет:</label>'),
    ('<optgroup label="Crypto (Binance daily)">', '<optgroup label="Крипто (Binance, дневные)">'),
    ('<optgroup label="Crypto (CoinGecko daily)">', '<optgroup label="Крипто (CoinGecko, дневные)">'),
    ('<optgroup label="Weather (Open-Meteo ACF)">', '<optgroup label="Погода (Open-Meteo, ACF)">'),
    ('<optgroup label="Economy (World Bank)">', '<optgroup label="Экономика (World Bank)">'),
    ('<optgroup label="Geophysical (USGS)">', '<optgroup label="Геофизика (USGS)">'),

    # Option labels
    ('>Binance — BTC daily (1yr)<', '>Binance — BTC дневной (1год)<'),
    ('>Binance — ETH daily (1yr)<', '>Binance — ETH дневной (1год)<'),
    ('>Binance — SOL daily (1yr)<', '>Binance — SOL дневной (1год)<'),
    ('>Binance — ADA daily (1yr)<', '>Binance — ADA дневной (1год)<'),
    ('>Binance — LINK daily (1yr)<', '>Binance — LINK дневной (1год)<'),
    ('>Binance — XRP daily (1yr)<', '>Binance — XRP дневной (1год)<'),
    ('>CoinGecko — Bitcoin (1yr)<', '>CoinGecko — Bitcoin (1год)<'),
    ('>CoinGecko — Ethereum (1yr)<', '>CoinGecko — Ethereum (1год)<'),
    ('>CoinGecko — Solana (1yr)<', '>CoinGecko — Solana (1год)<'),
    ('>CoinGecko — Polkadot (1yr)<', '>CoinGecko — Polkadot (1год)<'),
    ('>Open-Meteo — Berlin temperature (1yr)<', '>Open-Meteo — Температура Берлин (1год)<'),
    ('>Open-Meteo — Tokyo temperature (1yr)<', '>Open-Meteo — Температура Токио (1год)<'),
    ('>Open-Meteo — NYC temperature (1yr)<', '>Open-Meteo — Температура Нью-Йорк (1год)<'),
    ('>Open-Meteo — London temperature (1yr)<', '>Open-Meteo — Температура Лондон (1год)<'),
    ('>Open-Meteo — Sydney temperature (1yr)<', '>Open-Meteo — Температура Сидней (1год)<'),
    ('>World Bank — USA GDP (annual, 1960–)<', '>World Bank — ВВП США (ежегодно, 1960–)<'),
    ('>World Bank — USA CPI (annual, 1960–)<', '>World Bank — ИПЦ США (ежегодно, 1960–)<'),
    ('>USGS — Earthquakes (30d, hourly ACF)<', '>USGS — Землетрясения (30д, часовой ACF)<'),

    # Button
    ('<button id="load" class="primary">Fetch &amp; Compare</button>',
     '<button id="load" class="primary">Загрузить &amp; Сравнить</button>'),

    # Narrative card
    ('<h2 style="margin:0 0 8px 0; font-size:18px;">The story in one sentence</h2>',
     '<h2 style="margin:0 0 8px 0; font-size:18px;">Суть в одном предложении</h2>'),
    ('<p><span class="t71">T7.1 says:</span> S2 is the <em>local</em> law of retention, but the global curve is a <em>piecewise composition</em> of S2 regimes — the kernel enters different regimes at reproducible fractional positions.</p>',
     '<p><span class="t71">T7.1 говорит:</span> S2 — <em>локальный</em> закон сохранения, но глобальная кривая — <em>кусочная композиция</em> режимов S2 — ядро входит в различные режимы на воспроизводимых дробных позициях.</p>'),
    ('<p style="margin-top:.5rem"><span class="punch">This toy fetches real live data, fits 6 competing models head-to-head, and shows you exactly when piecewise S2 wins (and when it doesn\'t).</span></p>',
     '<p style="margin-top:.5rem"><span class="punch">Этот toy загружает реальные живые данные, фитирует 6 моделей head-to-head и показывает точно, когда кусочный S2 выигрывает (а когда — нет).</span></p>'),
    ('<div class="label">Layman analogy</div>', '<div class="label">Аналогия для не-технарей</div>'),

    # Chart card
    ('<h2 style="margin:0 0 6px 0; font-size:16px;">Live retention curve + 6 model fits</h2>',
     '<h2 style="margin:0 0 6px 0; font-size:16px;">Живая кривая сохранения + 6 фитов моделей</h2>'),
    ('<p class="small" style="margin:0 0 10px 0;">Black dots = empirical retention R(λ) computed from the autocorrelation function of the live data. Lines = best-fit of each model.</p>',
     '<p class="small" style="margin:0 0 10px 0;">Чёрные точки = эмпирическое R(λ) — автокорреляционная функция живых данных. Линии = лучший фит каждой модели.</p>'),

    # Model comparison
    ('<h2 style="margin:0 0 6px 0; font-size:16px;">Model comparison (ΔAICc, lower = better)</h2>',
     '<h2 style="margin:0 0 6px 0; font-size:16px;">Сравнение моделей (ΔAICc, меньше = лучше)</h2>'),
    ('<p class="small" style="margin:0 0 10px 0;">For each model: fit on the full retention curve, compute AICc, then ΔAICc = AICc_model − AICc_best. ΔAICc ≤ 2 = strong support; 2–4 = moderate; &gt;10 = essentially no support.</p>',
     '<p class="small" style="margin:0 0 10px 0;">Для каждой модели: фит на полной кривой, вычислить AICc, затем ΔAICc = AICc_модели − AICc_лучшей. ΔAICc ≤ 2 = сильная поддержка; 2–4 = умеренная; &gt;10 = поддержки нет.</p>'),
    ('<th>Model</th>', '<th>Модель</th>'),
    ('<th>k (params)</th>', '<th>k (параметры)</th>'),
    ('<th>Verdict</th>', '<th>Вердикт</th>'),
    ('<tr><td colspan="7" style="color:var(--muted); text-align:center;">Fetch a dataset to begin.</td></tr>',
     '<tr><td colspan="7" style="color:var(--muted); text-align:center;">Загрузите датасет, чтобы начать.</td></tr>'),

    # T7.1 params
    ('<h3 style="font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin:0 0 6px 0;">T7.1 regime parameters (if piecewise won)</h3>',
     '<h3 style="font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin:0 0 6px 0;">Параметры режимов T7.1 (если кусочный выиграл)</h3>'),
    ('<h3 style="font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin:0 0 6px 0;">Headline</h3>',
     '<h3 style="font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin:0 0 6px 0;">Заголовок</h3>'),

    # Math card
    ('<h2 style="margin:0 0 8px 0; font-size:16px;">What the 6 models say (math)</h2>',
     '<h2 style="margin:0 0 8px 0; font-size:16px;">Что говорят 6 моделей (математика)</h2>'),
    ('<summary><strong>How the fit works (technical)</strong></summary>',
     '<summary><strong>Как работает фит (технически)</strong></summary>'),

    # Interpretation card
    ('<h2 style="margin:0 0 8px 0; font-size:16px;">How to read the result</h2>',
     '<h2 style="margin:0 0 8px 0; font-size:16px;">Как читать результат</h2>'),
]

# Apply each
applied = 0
failed = []
for old, new in replacements:
    if old in html:
        html = html.replace(old, new, 1)
        applied += 1
    else:
        failed.append(old[:60])

print(f'Applied: {applied}/{len(replacements)}')
if failed:
    print('Failed:')
    for f in failed:
        print(f'  {f}')

with open('t71-toy_ru.html', 'w') as f:
    f.write(html)
print(f'Saved: t71-toy_ru.html ({len(html)} chars)')
