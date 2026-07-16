/**
 * nav-autoscroll.js — shifts the nav content horizontally to show the active link
 * and the More dropdown. Uses transform:translateX since overflow:clip prevents
 * native scrolling.
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

  function shiftNav() {
    const nav = document.querySelector('.site-header .nav');
    if (!nav) return;
    
    // Reset transform first
    nav.style.transform = '';
    
    // Find the active link or More toggle
    const active = nav.querySelector('.nav-link.active, .nav-more-toggle.active');
    const target = active || nav.querySelector('.nav-more-toggle');
    if (!target) return;
    
    const navRect = nav.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    
    // Calculate how much we need to shift
    let shift = 0;
    
    // If target is off-screen to the right, shift left
    if (targetRect.right > navRect.right - 4) {
      shift = navRect.right - targetRect.right - 10;
    }
    // If target is off-screen to the left, shift right
    if (targetRect.left < navRect.left + 4) {
      shift = navRect.left - targetRect.left + 10;
    }
    
    if (shift !== 0) {
      // Apply transform to the nav's children wrapper
      // We need to shift all children. Use transform on the nav itself.
      const currentTransform = nav.style.transform || '';
      const currentShift = parseInt(currentTransform.match(/-?\d+/)?.[0] || '0');
      nav.style.transform = `translateX(${currentShift + shift}px)`;
    }
    
    // Also check if the More toggle is visible after shifting
    const moreToggle = nav.querySelector('.nav-more-toggle');
    if (moreToggle) {
      const moreRect = moreToggle.getBoundingClientRect();
      const navRect2 = nav.getBoundingClientRect();
      if (moreRect.right > navRect2.right - 4) {
        const currentTransform = nav.style.transform || '';
        const currentShift = parseInt(currentTransform.match(/-?\d+/)?.[0] || '0');
        const additionalShift = navRect2.right - moreRect.right - 10;
        nav.style.transform = `translateX(${currentShift + additionalShift}px)`;
      }
    }
  }

  ready(() => {
    setTimeout(shiftNav, 100);
    setTimeout(shiftNav, 500);
    window.addEventListener('resize', shiftNav);
  });
})();
