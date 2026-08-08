// DREAM Pull-to-Refresh — lightweight, no dependencies
// Triggers a page reload when user pulls down from the top of the page on touch devices.
// Respects nested scroll containers (tab bars, test lists) — only activates when
// the body itself is scrolled to the top.

(function() {
  'use strict';

  // Skip on non-touch devices
  if (!('ontouchstart' in window) && navigator.maxTouchPoints === 0) return;

  const THRESHOLD = 70;        // px to trigger refresh
  const RESISTANCE = 0.5;      // rubber-band factor
  const START_DELAY = 5;       // px of grace before PTY engages

  let startY = null;
  let pulling = false;
  let indicator = null;

  function createIndicator() {
    if (indicator) return indicator;
    indicator = document.createElement('div');
    indicator.id = 'ptr-indicator';
    indicator.innerHTML = `
      <div class="ptr-spinner">
        <svg viewBox="0 0 24 24" width="22" height="22">
          <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 14.97 20 13.54 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 9.03 4 10.46 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z" fill="currentColor"/>
        </svg>
      </div>
      <div class="ptr-text">Pull to refresh</div>
    `;
    document.body.appendChild(indicator);
    return indicator;
  }

  function getIndicator() {
    return indicator || createIndicator();
  }

  // Inject styles once
  if (!document.getElementById('ptr-styles')) {
    const style = document.createElement('style');
    style.id = 'ptr-styles';
    style.textContent = `
      #ptr-indicator{
        position:fixed; top:0; left:0; right:0;
        height:0; overflow:hidden;
        display:flex; flex-direction:column; align-items:center; justify-content:flex-end;
        z-index:99998; pointer-events:none;
        transition:height .2s ease;
        background:linear-gradient(180deg, rgba(142,197,232,.08) 0%, transparent 100%);
      }
      #ptr-indicator.ptr-active{ transition:none; }
      #ptr-indicator.ptr-refreshing{
        height:54px !important; transition:height .2s ease;
      }
      .ptr-spinner{
        width:30px; height:30px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        color:var(--accent, #8ec5e8);
        transition:transform .2s ease, opacity .2s ease;
        transform:scale(0); opacity:0;
      }
      #ptr-indicator.ptr-active .ptr-spinner{ opacity:1; }
      .ptr-text{
        font-size:10px; color:var(--muted-2, #6b7280);
        margin-top:2px; margin-bottom:8px;
        font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing:.02em;
      }
      #ptr-indicator.ptr-refreshing .ptr-spinner{
        animation:ptr-spin .8s linear infinite;
      }
      #ptr-indicator.ptr-refreshing .ptr-text::before{ content:"Refreshing…"; }
      #ptr-indicator.ptr-refreshing .ptr-text{ font-size:0; }
      #ptr-indicator.ptr-refreshing .ptr-text::after{ content:""; }
      @keyframes ptr-spin{ from{transform:rotate(0)} to{transform:rotate(360deg)} }
      body.ptr-pulling{ overflow-y:hidden; touch-action:none; }
    `;
    document.head.appendChild(style);
  }

  // Check if a touch target is inside a scrollable container that isn't at its top
  function isInScrollableContainer(el) {
    let node = el;
    while (node && node !== document.body) {
      const style = getComputedStyle(node);
      const overflowY = style.overflowY;
      if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {
        // If the scrollable container is not at its top, don't engage PTY
        if (node.scrollTop > 0) return true;
      }
      node = node.parentElement;
    }
    return false;
  }

  function onTouchStart(e) {
    // Only single-finger
    if (e.touches.length !== 1) return;
    // Must be at top of document
    if (window.scrollY > 0 && document.documentElement.scrollTop > 0) return;
    // Don't engage if inside a scrollable container that's scrolled
    if (isInScrollableContainer(e.target)) return;
    // Skip if modal/dropdown open
    const openModal = document.querySelector('.nav-mobile-panel.active, .modal.open, [aria-expanded="true"].dropdown');
    if (openModal) return;

    startY = e.touches[0].clientY;
    pulling = false;
  }

  function onTouchMove(e) {
    if (startY === null) return;
    const dy = e.touches[0].clientY - startY;

    // If scrolling up, disengage
    if (dy <= 0) {
      if (pulling) {
        pulling = false;
        const ind = getIndicator();
        ind.classList.remove('ptr-active');
        ind.style.height = '0px';
        document.body.classList.remove('ptr-pulling');
      }
      return;
    }

    // Grace period — only engage after START_DELAY px
    if (dy < START_DELAY) return;

    // Re-check scroll position (in case page scrolled during touch)
    if (window.scrollY > 0 || document.documentElement.scrollTop > 0) {
      startY = null;
      return;
    }

    pulling = true;
    document.body.classList.add('ptr-pulling');
    const ind = getIndicator();
    ind.classList.add('ptr-active');

    // Rubber-band resistance
    const resisted = Math.max(0, (dy - START_DELAY) * RESISTANCE);
    const height = Math.min(resisted, THRESHOLD * 1.5);
    ind.style.height = height + 'px';

    // Scale spinner based on progress
    const progress = Math.min(1, height / THRESHOLD);
    const spinner = ind.querySelector('.ptr-spinner');
    if (spinner) {
      spinner.style.transform = `scale(${progress}) rotate(${progress * 180}deg)`;
    }

    // Update text
    const text = ind.querySelector('.ptr-text');
    if (text) {
      text.textContent = height >= THRESHOLD ? 'Release to refresh' : 'Pull to refresh';
    }

    // Prevent default scroll bounce
    if (e.cancelable) e.preventDefault();
  }

  function onTouchEnd(e) {
    if (!pulling) {
      startY = null;
      return;
    }
    const ind = getIndicator();
    pulling = false;
    document.body.classList.remove('ptr-pulling');

    // Check if threshold reached
    const currentHeight = parseFloat(ind.style.height) || 0;
    if (currentHeight >= THRESHOLD) {
      // Trigger refresh
      ind.classList.remove('ptr-active');
      ind.classList.add('ptr-refreshing');
      ind.style.height = '54px';

      // Reload after small delay for visual feedback
      setTimeout(() => {
        // Force reload from network, bypassing cache
        window.location.reload();
      }, 400);
    } else {
      // Snap back
      ind.classList.remove('ptr-active');
      ind.style.height = '0px';
    }
    startY = null;
  }

  // Attach listeners (passive:false so we can preventDefault)
  document.addEventListener('touchstart', onTouchStart, { passive: true });
  document.addEventListener('touchmove', onTouchMove, { passive: false });
  document.addEventListener('touchend', onTouchEnd, { passive: true });
  document.addEventListener('touchcancel', onTouchEnd, { passive: true });

  // Expose for debugging
  window.DREAM_PTR = { THRESHOLD, RESISTANCE };
})();
