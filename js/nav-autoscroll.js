/**
 * nav-autoscroll.js — scrolls the nav horizontally to ensure the active link
 * AND the More dropdown are visible. Runs on page load and on resize.
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

  function scrollNav() {
    const nav = document.querySelector('.site-header .nav');
    if (!nav) return;
    
    // First, try to scroll active link into view
    const active = nav.querySelector('.nav-link.active, .nav-more-toggle.active');
    if (active) {
      const navRect = nav.getBoundingClientRect();
      const targetRect = active.getBoundingClientRect();
      if (targetRect.right > navRect.right - 4) {
        nav.scrollLeft += (targetRect.right - navRect.right + 10);
      }
      if (targetRect.left < navRect.left + 4) {
        nav.scrollLeft -= (navRect.left - targetRect.left + 10);
      }
    }
    
    // Then, ensure the More dropdown toggle is also visible
    const moreToggle = nav.querySelector('.nav-more-toggle');
    if (moreToggle) {
      const navRect = nav.getBoundingClientRect();
      const moreRect = moreToggle.getBoundingClientRect();
      if (moreRect.right > navRect.right - 4) {
        nav.scrollLeft += (moreRect.right - navRect.right + 10);
      }
    }
  }

  ready(() => {
    setTimeout(scrollNav, 100);
    setTimeout(scrollNav, 500); // run again after fonts load
    window.addEventListener('resize', scrollNav);
  });
})();
