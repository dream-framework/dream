/**
 * DREAM ECharts maximize/restore utility
 * Provides a fullscreen toggle for any chart container.
 * Usage: <button onclick="DREAM_chart_fullscreen('myChart')">⛶</button>
 */
window.DREAM_chart_fullscreen = function(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!document.fullscreenElement) {
    el.requestFullscreen?.() || el.webkitRequestFullscreen?.();
  } else {
    document.exitFullscreen?.() || document.webkitExitFullscreen?.();
  }
};

// Auto-resize charts on window resize
window.DREAM_charts = window.DREAM_charts || [];
window.addEventListener('resize', () => {
  window.DREAM_charts.forEach(c => c && c.resize && c.resize());
});
document.addEventListener('fullscreenchange', () => {
  window.DREAM_charts.forEach(c => c && c.resize && c.resize());
});
