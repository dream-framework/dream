// DREAM Service Worker — PWA offline support
// Caches the app shell and serves stale-while-revalidate for pages

const CACHE_VERSION = 'dream-v2';
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
