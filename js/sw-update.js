// Force SW update on every page load
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistration().then(function(reg) {
    if (reg) {
      reg.update().then(function() {
        // Check if there's a waiting SW
        if (reg.waiting) {
          reg.waiting.postMessage({ type: 'SKIP_WAITING' });
        }
        // Listen for new SW taking control
        navigator.serviceWorker.addEventListener('controllerchange', function() {
          // Reload once when the new SW takes control
          if (!window.__swReloaded) {
            window.__swReloaded = true;
            window.location.reload();
          }
        });
      });
    }
  });
}
