/**
 * nav-mobile.js — Injects a hamburger menu into the site header for mobile.
 * Works alongside global.css. No page HTML changes required — auto-injects
 * the hamburger button + dropdown panel into every .site-header.
 *
 * Behavior:
 *   - Desktop (>768px): no-op, existing horizontal nav works.
 *   - Mobile (<=768px): hides .nav, shows hamburger button.
 *     Tap -> expands a full-width dropdown panel with all nav links
 *     in a vertical grid. Tap any link -> navigates + closes panel.
 *     Tap outside or X button -> closes panel.
 *
 * Accessibility:
 *   - Hamburger is a real <button> with aria-expanded/aria-controls.
 *   - Panel has role="menu", links have role="menuitem".
 *   - Esc key closes panel. Focus returns to hamburger.
 *   - Panel traps focus while open.
 */

(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn);
    } else {
      fn();
    }
  }

  function injectMobileNav() {
    const header = document.querySelector('.site-header');
    if (!header) return;
    if (header.querySelector('.nav-mobile-btn')) return; // already injected

    const nav = header.querySelector('.nav');
    if (!nav) return;

    const links = Array.from(nav.querySelectorAll('a.nav-link')).map(a => ({
      href: a.getAttribute('href') || '#',
      text: a.textContent.trim(),
      active: a.classList.contains('active'),
      cls: a.className,
    }));

    // Hamburger button
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-mobile-btn';
    btn.setAttribute('aria-label', 'Open menu');
    btn.setAttribute('aria-expanded', 'false');
    btn.setAttribute('aria-controls', 'nav-mobile-panel');
    btn.innerHTML =
      '<span class="nav-mobile-icon" aria-hidden="true">' +
        '<span></span><span></span><span></span>' +
      '</span>' +
      '<span class="nav-mobile-label">Menu</span>';

    // Dropdown panel
    const panel = document.createElement('div');
    panel.id = 'nav-mobile-panel';
    panel.className = 'nav-mobile-panel';
    panel.setAttribute('role', 'menu');
    panel.setAttribute('aria-label', 'Site navigation');

    const panelInner = document.createElement('div');
    panelInner.className = 'nav-mobile-panel-inner';

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'nav-mobile-close';
    closeBtn.setAttribute('aria-label', 'Close menu');
    closeBtn.innerHTML = '&times;';

    const linksWrap = document.createElement('nav');
    linksWrap.className = 'nav-mobile-links';
    linksWrap.setAttribute('aria-label', 'Pages');

    links.forEach(l => {
      const a = document.createElement('a');
      a.href = l.href;
      a.className = 'nav-mobile-link' + (l.active ? ' active' : '');
      a.setAttribute('role', 'menuitem');
      a.textContent = l.text;
      if (l.cls.indexOf('tools') >= 0) a.classList.add('tools');
      if (l.cls.indexOf('foundation') >= 0) a.classList.add('foundation');
      a.addEventListener('click', () => closePanel());
      linksWrap.appendChild(a);
    });

    panelInner.appendChild(closeBtn);
    panelInner.appendChild(linksWrap);
    panel.appendChild(panelInner);

    const backdrop = document.createElement('div');
    backdrop.className = 'nav-mobile-backdrop';
    backdrop.setAttribute('aria-hidden', 'true');

    header.appendChild(btn);
    document.body.appendChild(backdrop);
    document.body.appendChild(panel);

    let isOpen = false;

    function openPanel() {
      isOpen = true;
      panel.classList.add('open');
      backdrop.classList.add('open');
      btn.classList.add('open');
      btn.setAttribute('aria-expanded', 'true');
      document.body.classList.add('nav-mobile-open');
      const firstLink = panel.querySelector('.nav-mobile-link');
      if (firstLink) setTimeout(() => firstLink.focus(), 50);
    }

    function closePanel() {
      isOpen = false;
      panel.classList.remove('open');
      backdrop.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-expanded', 'false');
      document.body.classList.remove('nav-mobile-open');
      btn.focus();
    }

    function togglePanel() {
      if (isOpen) closePanel(); else openPanel();
    }

    btn.addEventListener('click', togglePanel);
    closeBtn.addEventListener('click', closePanel);
    backdrop.addEventListener('click', closePanel);

    document.addEventListener('keydown', e => {
      if (!isOpen) return;
      if (e.key === 'Escape') { e.preventDefault(); closePanel(); }
      if (e.key === 'Tab') {
        const focusable = panel.querySelectorAll('a.nav-mobile-link, button.nav-mobile-close');
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault(); last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault(); first.focus();
        }
      }
    });

    window.addEventListener('resize', () => {
      if (window.innerWidth > 768 && isOpen) closePanel();
    });
  }

  ready(injectMobileNav);
  document.addEventListener('dream:header-replaced', injectMobileNav);
})();
