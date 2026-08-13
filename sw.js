// DREAM Service Worker — PWA offline support
// Caches the app shell and serves stale-while-revalidate for pages

const CACHE_VERSION = 'dream-v14';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const RUNTIME_CACHE = `${CACHE_VERSION}-runtime`;

// Files to pre-cache on install (the app shell)
const PRECACHE_URLS = [
  './',
  './en/index.html',
  './ru/index.html',
  './css/global.css',
  './img/icon-192.png',
  './img/icon-512.png',
  './img/favicon.png',
  './img/apple-touch-icon.png',
  './manifest.json',
  './js/fractal.js',
  './js/nav-mobile.js',
  './js/nav-dropdown.js',
  './js/nav-autoscroll.js',
  './js/pull-to-refresh.js',
  './js/scan-notifications.js',
];

// Install — pre-cache the app shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
      .catch((err) => console.log('SW pre-cache error:', err))
  );
});

// Activate — clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys
          .filter((key) => key.startsWith('dream-') && key !== STATIC_CACHE && key !== RUNTIME_CACHE)
          .map((key) => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

// Fetch — stale-while-revalidate strategy
self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // Skip cross-origin requests (CDN fonts, MathJax, etc.) — let them pass through
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return;
  }

  // For navigation requests (HTML pages), try network first, fall back to cache
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Cache the new page
          const clone = response.clone();
          caches.open(RUNTIME_CACHE).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => {
          // Network failed — try cache, or fall back to index
          return caches.match(request).then((cached) => cached || caches.match('./en/index.html'));
        })
    );
    return;
  }

  // For static assets (CSS, JS, images), use stale-while-revalidate
  event.respondWith(
    caches.match(request).then((cached) => {
      const fetchPromise = fetch(request)
        .then((response) => {
          // Only cache successful responses
          if (response && response.status === 200) {
            const clone = response.clone();
            caches.open(RUNTIME_CACHE).then((cache) => cache.put(request, clone));
          }
          return response;
        })
        .catch(() => cached); // Network failed — return cached version
      return cached || fetchPromise;
    })
  );
});

// Message handler — for "skip waiting" from the page
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

// ═══ PERIODIC SYNC — for scan notifications (Android Chrome/Edge) ═══
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'dream-scan-check') {
    event.waitUntil(checkForScanUpdates());
  }
});

// Check for tests.json changes and notify
async function checkForScanUpdates() {
  try {
    const resp = await fetch('/dream/en/tests.json?t=' + Date.now(), { cache: 'no-store' });
    if (!resp.ok) return;
    const data = await resp.json();
    const tests = data.tests || [];
    const counts = { S2_WINS: 0, S2_TIES: 0, S2_LOSES: 0, S2_DUST_WINS: 0, S2_NO_FIT: 0 };
    for (const t of tests) {
      const mv = t.model_verdict || 'unknown';
      counts[mv] = (counts[mv] || 0) + 1;
    }
    const curr = {
      total: data.total_tests || tests.length,
      exported_at: data.exported_at,
      counts,
      hash: `${data.total_tests || tests.length}-${data.exported_at || ''}`,
    };

    // Read previous state from cache (since SW can't access localStorage)
    const cache = await caches.open('dream-scan-state-v2');
    const stateResp = await cache.match('/dream-scan-state');
    let prev = null;
    if (stateResp) {
      try { prev = await stateResp.json(); } catch {}
    }

    if (!prev) {
      // First check — just store
      await cache.put('/dream-scan-state', new Response(JSON.stringify(curr)));
      return;
    }

    if (prev.hash === curr.hash) return;  // no change

    // Compute diff
    const sign = n => n > 0 ? `+${n}` : `${n}`;
    const dTotal = curr.total - prev.total;
    const dWins = (curr.counts.S2_WINS || 0) - (prev.counts.S2_WINS || 0);
    const dTies = (curr.counts.S2_TIES || 0) - (prev.counts.S2_TIES || 0);
    const dLoses = (curr.counts.S2_LOSES || 0) - (prev.counts.S2_LOSES || 0);
    const dDust = (curr.counts.S2_DUST_WINS || 0) - (prev.counts.S2_DUST_WINS || 0);

    const lines = [];
    if (dTotal !== 0) lines.push(`Total: ${sign(dTotal)}`);
    if (dWins) lines.push(`S2 wins: ${sign(dWins)}`);
    if (dDust) lines.push(`Dust-resolved: ${sign(dDust)}`);
    if (dLoses) lines.push(`S2 loses: ${sign(dLoses)}`);

    const title = `DREAM scan: ${curr.total} tests`;
    const body = lines.length > 0 ? lines.join(', ') : 'Registry updated';

    await self.registration.showNotification(title, {
      body,
      icon: '/dream/img/icon-192.png',
      badge: '/dream/img/icon-192.png',
      tag: 'dream-scan-update',
      renotify: true,
      data: { url: '/dream/en/tests.html' },
      vibrate: [80, 40, 80],
    });

    // Update stored state
    await cache.put('/dream-scan-state', new Response(JSON.stringify(curr)));
  } catch (e) {
    console.warn('DREAM SW: scan check failed', e);
  }
}

// ═══ NOTIFICATION CLICK — open tests page ═══
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || '/dream/en/tests.html';
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      // Focus existing window if open
      for (const client of clientList) {
        if (client.url.includes('/dream/') && 'focus' in client) {
          return client.navigate(url).then(c => c.focus());
        }
      }
      // Open new window
      if (clients.openWindow) return clients.openWindow(url);
    })
  );
});
