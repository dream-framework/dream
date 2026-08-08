// DREAM Scan Notifications
// Notifies user when the tests registry changes (new entries, verdict flips, etc.)
//
// HOW IT WORKS:
//   1. User clicks "Subscribe" button → request Notification permission
//   2. Register Periodic Background Sync (Android Chrome, Edge) — checks every 6h
//   3. Fallback for iOS Safari / no periodicSync: poll when app is foreground
//   4. On check: fetch tests.json, compare to last-seen hash in localStorage
//   5. If changed: show notification with diff stats
//
// iOS NOTE: Notifications work on iOS 16.4+ ONLY when PWA is installed to
// home screen. If user opens in Safari, they get in-app banner only.
//
// ANDROID NOTE: Works in any browser (Chrome, Firefox, Samsung Internet).

(function() {
  'use strict';

  const POLL_INTERVAL_MS = 6 * 60 * 60 * 1000;  // 6 hours (matches CI cadence)
  const PERIODIC_SYNC_TAG = 'dream-scan-check';
  const STORAGE_KEY = 'dream_last_scan_state';
  const SUBSCRIBED_KEY = 'dream_scan_subscribed';

  // ── State helpers ────────────────────────────────────────────────────

  function getLastState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function setLastState(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {}
  }

  function isSubscribed() {
    return localStorage.getItem(SUBSCRIBED_KEY) === '1' &&
           Notification.permission === 'granted';
  }

  function setSubscribed(val) {
    localStorage.setItem(SUBSCRIBED_KEY, val ? '1' : '0');
  }

  // ── Fetch current state from tests.json ───────────────────────────────

  async function fetchCurrentState() {
    try {
      // Cache-bust to always get fresh data
      const url = '/dream/en/tests.json?t=' + Date.now();
      const resp = await fetch(url, { cache: 'no-store' });
      if (!resp.ok) return null;
      const data = await resp.json();
      const tests = data.tests || [];
      const counts = { S2_WINS: 0, S2_TIES: 0, S2_LOSES: 0, S2_DUST_WINS: 0, S2_NO_FIT: 0, unknown: 0 };
      for (const t of tests) {
        const mv = t.model_verdict || 'unknown';
        counts[mv] = (counts[mv] || 0) + 1;
      }
      return {
        total: data.total_tests || tests.length,
        exported_at: data.exported_at,
        counts,
        hash: `${data.total_tests || tests.length}-${data.exported_at || ''}`,
      };
    } catch (e) {
      console.warn('DREAM scan-notifications: fetch failed', e);
      return null;
    }
  }

  // ── Compute diff between states ─────────────────────────────────────

  function computeDiff(prev, curr) {
    if (!prev) return null;
    const deltaTotal = curr.total - prev.total;
    if (deltaTotal === 0 && prev.hash === curr.hash) return null;  // no change

    const dWins = (curr.counts.S2_WINS || 0) - (prev.counts.S2_WINS || 0);
    const dTies = (curr.counts.S2_TIES || 0) - (prev.counts.S2_TIES || 0);
    const dLoses = (curr.counts.S2_LOSES || 0) - (prev.counts.S2_LOSES || 0);
    const dDust = (curr.counts.S2_DUST_WINS || 0) - (prev.counts.S2_DUST_WINS || 0);
    const dNoFit = (curr.counts.S2_NO_FIT || 0) - (prev.counts.S2_NO_FIT || 0);

    return { deltaTotal, dWins, dTies, dLoses, dDust, dNoFit };
  }

  // ── Build notification body ─────────────────────────────────────────

  function buildNotification(diff, curr, isRu) {
    const sign = n => n > 0 ? `+${n}` : `${n}`;
    if (isRu) {
      const title = `DREAM скан: ${curr.total} тестов`;
      const lines = [];
      if (diff.deltaTotal !== 0) lines.push(`Всего: ${sign(diff.deltaTotal)}`);
      if (diff.dWins) lines.push(`S2 побед: ${sign(diff.dWins)}`);
      if (diff.dDust) lines.push(`Пыль разрешена: ${sign(diff.dDust)}`);
      if (diff.dLoses) lines.push(`S2 уступает: ${sign(diff.dLoses)}`);
      if (diff.dNoFit) lines.push(`Нет подгонки: ${sign(diff.dNoFit)}`);
      const body = lines.length > 0 ? lines.join(', ') : 'Реестр обновлён';
      return { title, body };
    } else {
      const title = `DREAM scan: ${curr.total} tests`;
      const lines = [];
      if (diff.deltaTotal !== 0) lines.push(`Total: ${sign(diff.deltaTotal)}`);
      if (diff.dWins) lines.push(`S2 wins: ${sign(diff.dWins)}`);
      if (diff.dDust) lines.push(`Dust-resolved: ${sign(diff.dDust)}`);
      if (diff.dLoses) lines.push(`S2 loses: ${sign(diff.dLoses)}`);
      if (diff.dNoFit) lines.push(`No fit: ${sign(diff.dNoFit)}`);
      const body = lines.length > 0 ? lines.join(', ') : 'Registry updated';
      return { title, body };
    }
  }

  // ── Show notification ───────────────────────────────────────────────

  async function showNotification(title, body) {
    if (Notification.permission !== 'granted') return;
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      if (reg && reg.showNotification) {
        reg.showNotification(title, {
          body,
          icon: '/dream/img/icon-192.png',
          badge: '/dream/img/icon-192.png',
          tag: 'dream-scan-update',
          renotify: true,
          data: { url: '/dream/en/tests.html' },
          vibrate: [80, 40, 80],
        });
      } else {
        // Fallback: in-app Notification (no service worker)
        new Notification(title, { body, icon: '/dream/img/icon-192.png' });
      }
    } catch (e) {
      console.warn('DREAM scan-notifications: showNotification failed', e);
    }
  }

  // ── Check for changes ───────────────────────────────────────────────

  async function checkForChanges(isRu) {
    if (Notification.permission !== 'granted') return;
    const curr = await fetchCurrentState();
    if (!curr) return;
    const prev = getLastState();
    if (!prev) {
      // First check — just store state, don't notify
      setLastState(curr);
      return;
    }
    const diff = computeDiff(prev, curr);
    if (!diff) return;  // no change
    const { title, body } = buildNotification(diff, curr, isRu);
    await showNotification(title, body);
    setLastState(curr);
  }

  // ── Subscribe / unsubscribe ─────────────────────────────────────────

  async function subscribe(isRu) {
    if (!('Notification' in window)) {
      alert(isRu ? 'Ваше устройство не поддерживает push-уведомления.'
                 : 'Your device does not support push notifications.');
      return false;
    }
    if (Notification.permission === 'granted') {
      setSubscribed(true);
      // Initial state capture
      const curr = await fetchCurrentState();
      if (curr) setLastState(curr);
      // Try periodic sync registration
      await registerPeriodicSync();
      // Start foreground polling fallback
      startForegroundPolling(isRu);
      return true;
    }
    const perm = await Notification.requestPermission();
    if (perm === 'granted') {
      setSubscribed(true);
      const curr = await fetchCurrentState();
      if (curr) setLastState(curr);
      await registerPeriodicSync();
      startForegroundPolling(isRu);
      return true;
    }
    return false;
  }

  function unsubscribe() {
    setSubscribed(false);
    unregisterPeriodicSync();
    stopForegroundPolling();
  }

  async function registerPeriodicSync() {
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      if (!reg || !reg.periodicSync) return false;
      // Check existing permission for periodic sync
      const status = await navigator.permissions.query({ name: 'periodic-background-sync' });
      if (status.state !== 'granted') return false;
      await reg.periodicSync.register(PERIODIC_SYNC_TAG, {
        minInterval: POLL_INTERVAL_MS,
      });
      console.log('DREAM: periodic sync registered');
      return true;
    } catch (e) {
      console.warn('DREAM: periodic sync not available', e);
      return false;
    }
  }

  async function unregisterPeriodicSync() {
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      if (reg && reg.periodicSync) {
        await reg.periodicSync.unregister(PERIODIC_SYNC_TAG);
      }
    } catch {}
  }

  // ── Foreground polling fallback (iOS Safari, browsers without periodicSync) ──

  let pollTimer = null;

  function startForegroundPolling(isRu) {
    stopForegroundPolling();
    // Check immediately
    checkForChanges(isRu);
    // Then every 30 min while app is open
    pollTimer = setInterval(() => checkForChanges(isRu), 30 * 60 * 1000);
    // Also check when page becomes visible again
    document.addEventListener('visibilitychange', onVisibilityChange);
  }

  function stopForegroundPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    document.removeEventListener('visibilitychange', onVisibilityChange);
  }

  function onVisibilityChange() {
    if (document.visibilityState === 'visible') {
      const isRu = window.location.pathname.includes('/ru/');
      checkForChanges(isRu);
    }
  }

  // ── UI: Subscribe button ────────────────────────────────────────────

  function updateButtonState(btn, isRu) {
    if (isRu) {
      if (isSubscribed()) {
        btn.textContent = '✓ Оповещения включены';
        btn.classList.add('subscribed');
        btn.setAttribute('aria-pressed', 'true');
      } else {
        btn.textContent = '🔔 Оповещения сканирования';
        btn.classList.remove('subscribed');
        btn.setAttribute('aria-pressed', 'false');
      }
    } else {
      if (isSubscribed()) {
        btn.textContent = '✓ Scan alerts on';
        btn.classList.add('subscribed');
        btn.setAttribute('aria-pressed', 'true');
      } else {
        btn.textContent = '🔔 Scan alerts';
        btn.classList.remove('subscribed');
        btn.setAttribute('aria-pressed', 'false');
      }
    }
  }

  function initSubscribeButton() {
    const btn = document.getElementById('scan-alert-btn');
    if (!btn) return;
    const isRu = window.location.pathname.includes('/ru/');
    updateButtonState(btn, isRu);
    btn.addEventListener('click', async () => {
      if (isSubscribed()) {
        unsubscribe();
        updateButtonState(btn, isRu);
      } else {
        const ok = await subscribe(isRu);
        updateButtonState(btn, isRu);
        if (ok) {
          // Show a confirmation notification
          const title = isRu ? 'DREAM: оповещения включены' : 'DREAM: alerts enabled';
          const body = isRu
            ? 'Вы будете получать уведомления при каждом обновлении реестра (каждые ~6 часов).'
            : 'You will be notified each time the registry updates (~every 6 hours).';
          await showNotification(title, body);
        }
      }
    });

    // If already subscribed, restart foreground polling on page load
    if (isSubscribed()) {
      startForegroundPolling(isRu);
    }
  }

  // Init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSubscribeButton);
  } else {
    initSubscribeButton();
  }

  // Expose for service worker
  window.DREAM_SCAN_NOTIFICATIONS = {
    checkForChanges,
    fetchCurrentState,
    computeDiff,
    buildNotification,
  };
})();
