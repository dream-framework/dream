/**
 * nav-dropdown.js — handles the "Tools" dropdown in the site-header nav.
 * Clicking the toggle opens/closes the dropdown. Clicking outside closes it.
 * Esc closes it. The dropdown contains Time, Memory, NPA Calculator,
 * Intervention Sim.
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

  function initDropdown() {
    const toggles = document.querySelectorAll('.nav-more-toggle');
    toggles.forEach(toggle => {
      if (toggle.dataset.bound) return;
      toggle.dataset.bound = '1';
      const parent = toggle.closest('.nav-more');
      if (!parent) return;

      toggle.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Close all other dropdowns
        document.querySelectorAll('.nav-more.open').forEach(d => {
          if (d !== parent) d.classList.remove('open');
        });
        parent.classList.toggle('open');
      });
    });

    // Close on outside click
    if (!window.__navDropdownOutsideBound) {
      window.__navDropdownOutsideBound = true;
      document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-more')) {
          document.querySelectorAll('.nav-more.open').forEach(d => d.classList.remove('open'));
        }
      });
      // Esc closes
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          document.querySelectorAll('.nav-more.open').forEach(d => d.classList.remove('open'));
        }
      });
    }
  }

  ready(initDropdown);
  // Re-init if header is dynamically rebuilt (npa-calculator does this on lang switch)
  document.addEventListener('dream:nav-rebuilt', initDropdown);
})();
