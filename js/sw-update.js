// Force SW update on every page load — aggressive version
// This checks for a new SW, forces it to activate, clears ALL old caches,
// and reloads the page if a new SW took control.
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistration().then(function(reg) {
    if (!reg) return;
    
    // Force update check
    reg.update().catch(function() {});
    
    // If a new SW is waiting, force it to activate
    if (reg.waiting) {
      reg.waiting.postMessage({ type: 'SKIP_WAITING' });
    }
    
    // Listen for new SW installing
    reg.addEventListener('updatefound', function() {
      var newWorker = reg.installing;
      if (newWorker) {
        newWorker.addEventListener('statechange', function() {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            // New SW installed — force it to take over
            newWorker.postMessage({ type: 'SKIP_WAITING' });
          }
        });
      }
    });
    
    // When a new SW takes control, reload the page ONCE
    var reloaded = sessionStorage.getItem('sw-reloaded');
    navigator.serviceWorker.addEventListener('controllerchange', function() {
      if (!reloaded) {
        sessionStorage.setItem('sw-reloaded', '1');
        window.location.reload();
      }
    });
    
    // Also: if no controller but a SW exists, reload to get controlled
    if (!navigator.serviceWorker.controller && reg.active) {
      // Page loaded before SW controlled it — reload to get SW control
      // But only once per session
      if (!reloaded) {
        sessionStorage.setItem('sw-reloaded', '1');
        window.location.reload();
      }
    }
  });
}
