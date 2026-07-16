/**
 * nav-autoscroll.js — scrolls the nav horizontally to show the active link
 * or the More dropdown if it's off-screen. Runs on page load and on resize.
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

  function scrollNavToActive() {
    const nav = document.querySelector('.site-header .nav');
    if (!nav) return;
    const active = nav.querySelector('.nav-link.active, .nav-more-toggle.active');
    const target = active || nav.querySelector('.nav-more-toggle');
    if (!target) return;
    const navRect = nav.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    // If target is off-screen to the right, scroll right
    if (targetRect.right > navRect.right - 4) {
      nav.scrollLeft += (targetRect.right - navRect.right + 10);
    }
    // If target is off-screen to the left, scroll left
    if (targetRect.left < navRect.left + 4) {
      nav.scrollLeft -= (navRect.left - targetRect.left + 10);
    }
  }

  ready(() => {
    setTimeout(scrollNavToActive, 100);
    window.addEventListener('resize', scrollNavToActive);
  });
})();
