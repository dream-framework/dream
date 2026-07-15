# D.R.E.A.M — static GitHub Pages build

A static conversion of the Flask app at
[`dream_physics_clone/`](https://github.com/dream-framework/dream-physics)
deployed to GitHub Pages. The original site lives at
<https://dream-physics.onrender.com/>.

This is a **pure static site** — no Flask, no Jinja2, no server-side
rendering. The only backend is a tiny Render-hosted Express proxy that holds
the `GROQ_API_KEY` for the chat bubble.

## What's here

```
dream_static/
├── index.html                     # root redirect (auto-picks EN or RU)
├── en/                            # English pages (17 + 3 article subpages)
│   ├── index.html
│   ├── retention.html
│   ├── axioms.html
│   ├── theorems.html
│   ├── math.html
│   ├── kernel.html
│   ├── topology.html
│   ├── spectrum.html
│   ├── predictions.html
│   ├── falsification.html
│   ├── faq.html
│   ├── about.html
│   ├── case.html
│   ├── time.html
│   ├── memory.html
│   ├── articles.html
│   └── articles/{introduction,kernel,retention}.html
├── ru/                            # Russian pages (same set)
│   └── …
├── css/
│   └── global.css                 # new sleeker glassmorphism stylesheet
├── js/
│   ├── fractal.js                 # WebGL Mandelbrot backdrop (CPU fallback)
│   ├── weave.js                   # client-side kernel-weave generator
│   │                                (replaces the /weave/data backend)
│   ├── bot.js                     # Groq chat bubble (calls Render proxy)
│   └── fs-toggle.js               # fullscreen toggle for toys
├── plotter.html                   # retention curve plotter toy (EN, original)
├── plotter_ru.html                # retention curve plotter toy (RU, original)
├── npa-calculator.html            # NPA calculator toy (from tech-communism)
├── intervention-simulator.html    # interactive D-order-parameter simulator
├── server/                        # Groq proxy backend (Render)
│   ├── index.js
│   ├── package.json
│   └── render.yaml
├── .github/workflows/deploy.yml   # GitHub Pages deploy workflow
├── build_static.py                # build script (Flask → static HTML)
└── README.md                      # this file
```

## How the static build works

`build_static.py` reads each `templates/i18n/{en,ru}/*_body.html`, strips
Jinja2 syntax (`{% block %}`, `{{ _('ui.read_more') }}`,
`{{ url_for('static', filename='…') }}`), inlines any partial includes,
wraps the result in a base template (header / nav / footer / chat bubble /
fractal canvas / MathJax), and writes the result to `en/<page>.html` or
`ru/<page>.html`.

Re-run after editing source templates:

```bash
cd dream_static
python3 build_static.py
```

## Key adaptations from the Flask original

| Original (Flask)                             | Static site                                |
| -------------------------------------------- | ------------------------------------------ |
| `templates/base.html` (818-line Jinja2)      | `build_static.py::base_template()`         |
| `app.py` `NAV` list                          | `build_static.py::NAV` (same pages + toys) |
| `i18n/{en,ru}.json` translations             | baked directly into per-language pages     |
| `/weave/data` backend (`weave_blueprint.py`) | `js/weave.js` (pure-JS orthonormal proj + PCA) |
| `/groq-chat` backend (`groq_bot.py`)         | `server/index.js` (Express → Groq API)     |
| `static/plotter.html`                        | `plotter.html` (unchanged)                 |
| `static/plotter_ru.html`                     | `plotter_ru.html` (unchanged)              |

The **kernel weave toy** on the Kernel page originally fetched from a Flask
`/weave/data` endpoint. The static build replaces that with a pure-JS
implementation (`js/weave.js`) that generates the same JSON shape:
seeded RNG → 10-D weave points → orthonormal projection via modified
Gram-Schmidt → PCA via power iteration with deflation. Plotly renders the
3-D scatter exactly as before.

## Styling

The new `css/global.css` uses a **soft tech-communism palette**:

| token     | value     | use                              |
| --------- | --------- | -------------------------------- |
| `--bg`    | `#1e293b` | page background (top of gradient)|
| `--bg2`   | `#172033` | page background (bottom)         |
| `--card`  | `#3b4a5f` | card base (with glassmorphism)   |
| `--accent`| `#8ec5e8` | sky accent                       |
| `--accent-2` | `#f0c878` | warm gold                    |
| `--good`  | `#7dd3a8` | success / confirmed              |
| `--bad`   | `#f0a0a0` | danger / extraction              |
| `--warn`  | `#f0c878` | warning                          |

Cards use `backdrop-filter: blur(8px) saturate(140%)` for glassmorphism
over the WebGL Mandelbrot fractal backdrop. Smooth transitions on hover,
focus, and toggle. Fully mobile-responsive (sticky header collapses to a
grid below 900 px; nav scrolls horizontally).

## Deployment

### Frontend — GitHub Pages

The workflow in `.github/workflows/deploy.yml` runs on every push to
`main` / `master`, copies everything except `server/` and `build_static.py`
into `_site/`, and uploads it as a Pages artifact.

After the first deploy:

1. Go to **Settings → Pages** in the GitHub repo.
2. Under **Build and deployment → Source**, pick **GitHub Actions**.
3. The site will be live at
   `https://<user>.github.io/<repo>/` — typically
   `https://dream-framework.github.io/dream-physics/`.

If you serve under a sub-path (e.g. `/dream-physics/`), all links already
use **relative paths** (`../css/global.css`, `../plotter.html`,
`retention.html`) so no rewrites are needed.

### Backend — Groq proxy on Render

1. Push `server/` to a Render-connected repo (or use the same repo with
   `rootDir: server`).
2. Render reads `server/render.yaml` and creates a `dream-groq` web service
   on the free plan.
3. In the Render dashboard, go to **Environment** and set
   `GROQ_API_KEY` to your Groq API key.
4. Once deployed, edit `js/bot.js` and set:
   ```js
   const BACKEND_URL = 'https://dream-groq.onrender.com';
   ```
5. Re-deploy the static site. The chat bubble appears bottom-right after
   the bot's `/health` endpoint confirms `groq_configured: true`.

If `BACKEND_URL` is left empty, the bubble still appears but the bot gives
a friendly "not wired yet" message instead of calling Groq.

## Local preview

The site is pure static HTML — any HTTP server works:

```bash
cd dream_static
python3 -m http.server 3000
# or:  npx serve .
```

Open <http://localhost:3000/> in your browser.

## Toys

| Toy                          | What it does                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `plotter.html`               | Original retention-curve plotter. Pulls live data from SWPC GOES, USGS, World Bank, CoinGecko, iNaturalist. Fits S(W) ≈ exp(−(W/λ)^β). |
| `plotter_ru.html`            | Same, Russian UI.                                              |
| `npa-calculator.html`        | Net Present Attention calculator (ported from tech-communism).|
| `intervention-simulator.html`| Interactive: drag a slider, watch the order parameter D respond. Mirrors the four natural experiments (Sweden FTT, China gaming ban, Montréal Protocol, Volcker Rule). |

## Notes

- **MathJax v3** loads from CDN with the same config as the Flask site
  (SVG renderer, `$…$` and `\\(…\\)` inline math, `$$…$$` and `\\[…\\]`
  display math).
- **Plotly 2.35.2** loads from CDN on the topology, kernel, and time
  pages (for the kernel-weave 3-D scatter and the time-studio surface).
- **`prefers-reduced-motion`** disables the fractal backdrop and all
  transitions.
- **Print mode** hides the fractal, chat, header, and footer.
- All 17 pages × 2 languages are preserved with their full original text —
  no content was dropped. The Russian `/time/` page falls back to the
  English body (matching the Flask app's
  `['i18n/ru/time_body.html', 'i18n/en/time_body.html']` include list).
