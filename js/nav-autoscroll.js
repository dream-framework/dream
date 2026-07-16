/**
 * nav-autoscroll.js — scrolls the .nav-links-scroll div to bring the active
 * link into view. The .nav-more dropdown is outside the scroll div so it's
 * never clipped.
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
    const scroll = document.querySelector('.nav-links-scroll');
    if (!scroll) return;
    
    const active = scroll.querySelector('.nav-link.active');
    if (!active) return;
    
    const scrollRect = scroll.getBoundingClientRect();
    const activeRect = active.getBoundingClientRect();
    
    // If active link is off-screen to the right, scroll right
    if (activeRect.right > scrollRect.right - 4) {
      scroll.scrollLeft += (activeRect.right - scrollRect.right + 10);
    }
    // If off-screen to the left, scroll left
    if (activeRect.left < scrollRect.left + 4) {
      scroll.scrollLeft -= (scrollRect.left - activeRect.left + 10);
    }
  }

  ready(() => {
    setTimeout(scrollNav, 100);
    setTimeout(scrollNav, 500);
    window.addEventListener('resize', scrollNav);
  });
})();
