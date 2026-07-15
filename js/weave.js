// ============================================================================
// D.R.E.A.M — client-side kernel-weave generator (3D, reliable)
// Generates structured 3D point clouds representing the 10D→4D projection.
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
    const noise = (1 - R) * 0.5;

    const traces = {
      filament: { x: [], y: [], z: [], count: 0 },
      facet: { x: [], y: [], z: [], count: 0 },
      hub: { x: [], y: [], z: [], count: 0 },
      dust: { x: [], y: [], z: [], count: 0 }
    };

    // Filaments: 3D line-like structures (random 3D direction vectors)
    var nFil = Math.floor(n * 0.45);
    var nStrands = 6;
    for (var s = 0; s < nStrands; s++) {
      // Random center in 3D
      var cx = gauss(rng) * 0.8;
      var cy = gauss(rng) * 0.8;
      var cz = gauss(rng) * 0.8;
      // Random 3D direction
      var dx = gauss(rng), dy = gauss(rng), dz = gauss(rng);
      var dlen = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
      dx /= dlen; dy /= dlen; dz /= dlen;
      var length = 1.5 + rng() * 1.5;
      var per = Math.floor(nFil / nStrands);
      for (var i = 0; i < per; i++) {
        var t = (rng() - 0.5) * length;
        traces.filament.x.push(cx + t * dx + gauss(rng) * 0.05 * (1 + noise));
        traces.filament.y.push(cy + t * dy + gauss(rng) * 0.05 * (1 + noise));
        traces.filament.z.push(cz + t * dz + gauss(rng) * 0.05 * (1 + noise));
      }
    }
    traces.filament.count = traces.filament.x.length;

    // Facets: 3D sheet-like structures (2D planes in 3D space)
    var nFac = Math.floor(n * 0.30);
    var nSheets = 3;
    for (var s = 0; s < nSheets; s++) {
      var cx = gauss(rng) * 0.6;
      var cy = gauss(rng) * 0.6;
      var cz = gauss(rng) * 0.6;
      // Two random orthogonal vectors defining the plane
      var u1 = gauss(rng), u2 = gauss(rng), u3 = gauss(rng);
      var ulen = Math.sqrt(u1*u1+u2*u2+u3*u3) || 1; u1/=ulen; u2/=ulen; u3/=ulen;
      // v = random vector, orthogonalize against u
      var v1 = gauss(rng), v2 = gauss(rng), v3 = gauss(rng);
      var dot = v1*u1 + v2*u2 + v3*u3;
      v1 -= dot*u1; v2 -= dot*u2; v3 -= dot*u3;
      var vlen = Math.sqrt(v1*v1+v2*v2+v3*v3) || 1; v1/=vlen; v2/=vlen; v3/=vlen;
      var rx = 0.4 + rng() * 0.4;
      var ry = 0.3 + rng() * 0.3;
      var per = Math.floor(nFac / nSheets);
      for (var i = 0; i < per; i++) {
        var a = rng() * Math.PI * 2;
        var r = Math.sqrt(rng());
        var s1 = r * rx * Math.cos(a);
        var s2 = r * ry * Math.sin(a);
        traces.facet.x.push(cx + s1*u1 + s2*v1 + gauss(rng) * 0.03 * (1 + noise));
        traces.facet.y.push(cy + s1*u2 + s2*v2 + gauss(rng) * 0.03 * (1 + noise));
        traces.facet.z.push(cz + s1*u3 + s2*v3 + gauss(rng) * 0.03 * (1 + noise));
      }
    }
    traces.facet.count = traces.facet.x.length;

    // Hubs: tight 3D clusters
    var nHub = Math.floor(n * 0.15);
    var nClusters = 4;
    for (var s = 0; s < nClusters; s++) {
      var cx = gauss(rng) * 0.5;
      var cy = gauss(rng) * 0.5;
      var cz = gauss(rng) * 0.5;
      var per = Math.floor(nHub / nClusters);
      for (var i = 0; i < per; i++) {
        traces.hub.x.push(cx + gauss(rng) * 0.08 * (1 + noise));
        traces.hub.y.push(cy + gauss(rng) * 0.08 * (1 + noise));
        traces.hub.z.push(cz + gauss(rng) * 0.08 * (1 + noise));
      }
    }
    traces.hub.count = traces.hub.x.length;

    // Dust: full 3D isotropic scatter
    var nDust = n - (traces.filament.count + traces.facet.count + traces.hub.count);
    for (var i = 0; i < nDust; i++) {
      traces.dust.x.push(gauss(rng) * 1.2 * (1 + noise * 0.5));
      traces.dust.y.push(gauss(rng) * 1.2 * (1 + noise * 0.5));
      traces.dust.z.push(gauss(rng) * 1.2 * (1 + noise * 0.5));
    }
    traces.dust.count = traces.dust.x.length;

    return { r: R, lambda: lam, lambda_q: lam_q, deff: deff, traces: traces };
  }

  window.DreamWeave = { generate: generate };
})();
