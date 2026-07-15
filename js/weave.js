// ============================================================================
// D.R.E.A.M — client-side kernel-weave generator (simplified, reliable)
// Generates structured 2D point clouds representing the 10D→4D projection.
// Exposes window.DreamWeave.generate({lam, lam_q, deff, n, seed})
// ============================================================================

(function () {
  function makeRng(seed) {
    let a = ((seed || 42) | 0) >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function generate({ lam = 0.30, lam_q = 1.0, deff = 2.2, n = 3500, seed = 42 } = {}) {
    const rng = makeRng(seed);
    const R = Math.exp(-Math.pow(lam / lam_q, deff));
    const noise = (1 - R) * 0.5; // noise grows as retention drops

    const traces = { filament: { x: [], y: [], z: [], count: 0 }, facet: { x: [], y: [], z: [], count: 0 }, hub: { x: [], y: [], z: [], count: 0 }, dust: { x: [], y: [], z: [], count: 0 } };

    // Filaments: elongated structures (lines with scatter)
    var nFil = Math.floor(n * 0.45);
    var nStrands = 6;
    for (var s = 0; s < nStrands; s++) {
      var cx = gauss(rng) * 0.8;
      var cy = gauss(rng) * 0.8;
      var angle = rng() * Math.PI * 2;
      var length = 1.5 + rng() * 1.5;
      var per = Math.floor(nFil / nStrands);
      for (var i = 0; i < per; i++) {
        var t = (rng() - 0.5) * length;
        traces.filament.x.push(cx + t * Math.cos(angle) + gauss(rng) * 0.05 * (1 + noise));
        traces.filament.y.push(cy + t * Math.sin(angle) + gauss(rng) * 0.05 * (1 + noise));
        traces.filament.z.push(0);
      }
    }
    traces.filament.count = traces.filament.x.length;

    // Facets: 2D sheet-like structures (ellipses)
    var nFac = Math.floor(n * 0.30);
    var nSheets = 3;
    for (var s = 0; s < nSheets; s++) {
      var cx = gauss(rng) * 0.6;
      var cy = gauss(rng) * 0.6;
      var rx = 0.4 + rng() * 0.4;
      var ry = 0.3 + rng() * 0.3;
      var rot = rng() * Math.PI;
      var per = Math.floor(nFac / nSheets);
      for (var i = 0; i < per; i++) {
        var a = rng() * Math.PI * 2;
        var r = Math.sqrt(rng());
        var lx = r * rx * Math.cos(a);
        var ly = r * ry * Math.sin(a);
        traces.facet.x.push(cx + lx * Math.cos(rot) - ly * Math.sin(rot) + gauss(rng) * 0.03 * (1 + noise));
        traces.facet.y.push(cy + lx * Math.sin(rot) + ly * Math.cos(rot) + gauss(rng) * 0.03 * (1 + noise));
        traces.facet.z.push(0);
      }
    }
    traces.facet.count = traces.facet.x.length;

    // Hubs: tight clusters (high-density nodes)
    var nHub = Math.floor(n * 0.15);
    var nClusters = 4;
    for (var s = 0; s < nClusters; s++) {
      var cx = gauss(rng) * 0.4;
      var cy = gauss(rng) * 0.4;
      var per = Math.floor(nHub / nClusters);
      for (var i = 0; i < per; i++) {
        traces.hub.x.push(cx + gauss(rng) * 0.06 * (1 + noise));
        traces.hub.y.push(cy + gauss(rng) * 0.06 * (1 + noise));
        traces.hub.z.push(0);
      }
    }
    traces.hub.count = traces.hub.x.length;

    // Dust: isotropic scatter (fills space)
    var nDust = n - (traces.filament.count + traces.facet.count + traces.hub.count);
    for (var i = 0; i < nDust; i++) {
      traces.dust.x.push(gauss(rng) * 1.2 * (1 + noise * 0.5));
      traces.dust.y.push(gauss(rng) * 1.2 * (1 + noise * 0.5));
      traces.dust.z.push(0);
    }
    traces.dust.count = traces.dust.x.length;

    return { r: R, lambda: lam, lambda_q: lam_q, deff: deff, traces: traces };
  }

  window.DreamWeave = { generate: generate };
})();
