// ============================================================================
// Universal chart fullscreen maximize/restore toggle
// Works on any element with class "fullscreenable" or by passing an element.
// Adds a floating ⛶ button to the top-right corner of the target.
// Click to expand to full viewport, click again (or press Esc) to restore.
// ============================================================================

(function () {
  const STYLE = `
    .fs-btn {
      position: absolute;
      top: 8px;
      right: 8px;
      z-index: 50;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      border: 1px solid var(--border, #54637a);
      background: rgba(40, 53, 72, 0.85);
      backdrop-filter: blur(6px);
      color: var(--fg, #f1f5f9);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      line-height: 1;
      transition: all 0.15s ease;
      opacity: 0.7;
    }
    .fs-btn:hover {
      opacity: 1;
      background: rgba(142, 197, 232, 0.15);
      border-color: rgba(142, 197, 232, 0.5);
      transform: scale(1.08);
    }
    .fs-btn:active { transform: scale(0.95); }

    /* Fullscreen state */
    .fs-target {
      transition: all 0.25s ease;
    }
    .fs-target.fs-active {
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      z-index: 9999 !important;
      border-radius: 0 !important;
      margin: 0 !important;
      background: var(--bg, #283548) !important;
    }
    /* When target is fullscreen, its children should fill it */
    .fs-target.fs-active .canvasWrap,
    .fs-target.fs-active canvas,
    .fs-target.fs-active .npv-plot,
    .fs-target.fs-active .npv-bars {
      width: 100% !important;
      height: 100% !important;
      max-height: 100% !important;
    }
    .fs-target.fs-active .chartArea {
      height: calc(100% - 40px) !important;
    }
    .fs-target.fs-active .legendOverlay {
      z-index: 10;
    }
    /* When the whole .main grid goes fullscreen, let it scroll and fill */
    .fs-target.fs-active.main {
      overflow: auto !important;
      padding: 12px !important;
      grid-template-columns: 1fr 380px !important;
    }
    .fs-target.fs-active.main .chart {
      min-height: 500px;
    }
    .fs-target.fs-active.main .chartArea {
      min-height: 350px;
      height: auto !important;
      flex: 1;
    }
    .fs-target.fs-active.main .panel {
      max-height: none !important;
    }
    .fs-target.fs-active.main .tabcontent {
      overflow: visible !important;
    }
    /* Dark backdrop behind fullscreen element (optional — element covers it anyway) */
    .fs-backdrop {
      display: none;
      position: fixed;
      top: 0; left: 0;
      width: 100vw; height: 100vh;
      background: rgba(0,0,0,0.6);
      z-index: 9998;
    }
    .fs-backdrop.fs-active { display: block; }
  `;

  // Inject styles once
  if (!document.getElementById('fs-toggle-style')) {
    const s = document.createElement('style');
    s.id = 'fs-toggle-style';
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  // Icon SVGs
  const ICON_EXPAND = '⤢';   // or use SVG below
  const ICON_RESTORE = '✕';

  function makeBtn() {
    const btn = document.createElement('button');
    btn.className = 'fs-btn';
    btn.type = 'button';
    btn.innerHTML = ICON_EXPAND;
    btn.setAttribute('aria-label', 'Maximize chart');
    return btn;
  }

  function makeBackdrop() {
    const bd = document.createElement('div');
    bd.className = 'fs-backdrop';
    return bd;
  }

  // State per target
  const states = new WeakMap();

  function toggle(target, backdrop) {
    const st = states.get(target) || { active: false };
    st.active = !st.active;
    states.set(target, st);

    if (st.active) {
      // Save original styles for restore
      st.orig = {
        position: target.style.position,
        top: target.style.top,
        left: target.style.left,
        width: target.style.width,
        height: target.style.height,
        zIndex: target.style.zIndex,
        borderRadius: target.style.borderRadius,
        margin: target.style.margin,
      };
      target.classList.add('fs-active');
      backdrop.classList.add('fs-active');
      target.querySelector('.fs-btn').innerHTML = ICON_RESTORE;
      target.querySelector('.fs-btn').setAttribute('aria-label', 'Restore chart');
    } else {
      target.classList.remove('fs-active');
      backdrop.classList.remove('fs-active');
      target.querySelector('.fs-btn').innerHTML = ICON_EXPAND;
      target.querySelector('.fs-btn').setAttribute('aria-label', 'Maximize chart');
    }

    // Notify charting libraries to resize — multiple passes for reliability
    [300, 600, 1000].forEach(delay => {
      setTimeout(() => {
        // Plotly charts: find the actual chart div inside the fullscreen target
        if (window.Plotly) {
          const plotDivs = target.querySelectorAll('#npvplot, #npv-bars, .js-plotly-plot');
          plotDivs.forEach(d => {
            try { Plotly.Plots.resize(d); } catch(e) {}
          });
        }
        // For canvas charts, dispatch a resize event so the app redraws
        if (target.querySelector('canvas')) {
          window.dispatchEvent(new Event('resize'));
        }
      }, delay);
    });
  }

  function init(target) {
    if (!target || target.querySelector('.fs-btn')) return;

    // Make target positionable so the button can sit inside it
    if (getComputedStyle(target).position === 'static') {
      target.style.position = 'relative';
    }
    target.classList.add('fs-target');

    const backdrop = makeBackdrop();
    document.body.appendChild(backdrop);

    const btn = makeBtn();
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggle(target, backdrop);
    });
    target.appendChild(btn);

    // Click backdrop to restore
    backdrop.addEventListener('click', () => {
      const st = states.get(target);
      if (st && st.active) toggle(target, backdrop);
    });

    // Esc to restore
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        const st = states.get(target);
        if (st && st.active) toggle(target, backdrop);
      }
    });
  }

  // Auto-init elements with [data-fullscreen] or .fullscreenable
  function autoInit() {
    document.querySelectorAll('[data-fullscreen], .fullscreenable').forEach(init);
  }

  // Expose
  window.ChartFullscreen = { init, autoInit, toggle };

  // Auto-init on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
})();
