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

    // Filaments: 3D curves with thickness — random walk in 3D
    var nFil = Math.floor(n * 0.45);
    var nStrands = 6;
    for (var s = 0; s < nStrands; s++) {
      var cx = gauss(rng) * 0.6;
      var cy = gauss(rng) * 0.6;
      var cz = gauss(rng) * 0.6;
      var per = Math.floor(nFil / nStrands);
      for (var i = 0; i < per; i++) {
        var t = (rng() - 0.5) * 2.5;
        // 3D direction with equal spread
        var dx = gauss(rng), dy = gauss(rng), dz = gauss(rng);
        var dlen = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
        // Add perpendicular spread so it's a tube, not a line
        var px = gauss(rng) * 0.15, py = gauss(rng) * 0.15, pz = gauss(rng) * 0.15;
        traces.filament.x.push(cx + t * dx/dlen + px * (1 + noise));
        traces.filament.y.push(cy + t * dy/dlen + py * (1 + noise));
        traces.filament.z.push(cz + t * dz/dlen + pz * (1 + noise));
      }
    }
    traces.filament.count = traces.filament.x.length;

    // Facets: 3D thick sheets — 2D plane with z-thickness comparable to x/y
    var nFac = Math.floor(n * 0.30);
    var nSheets = 3;
    for (var s = 0; s < nSheets; s++) {
      var cx = gauss(rng) * 0.5;
      var cy = gauss(rng) * 0.5;
      var cz = gauss(rng) * 0.5;
      // Plane basis vectors
      var u = [gauss(rng), gauss(rng), gauss(rng)];
      var ulen = Math.sqrt(u[0]**2+u[1]**2+u[2]**2)||1; u[0]/=ulen; u[1]/=ulen; u[2]/=ulen;
      var v = [gauss(rng), gauss(rng), gauss(rng)];
      var dot = v[0]*u[0]+v[1]*u[1]+v[2]*u[2];
      v[0]-=dot*u[0]; v[1]-=dot*u[1]; v[2]-=dot*u[2];
      var vlen = Math.sqrt(v[0]**2+v[1]**2+v[2]**2)||1; v[0]/=vlen; v[1]/=vlen; v[2]/=vlen;
      // Normal vector (thickness direction)
      var nw = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
      var per = Math.floor(nFac / nSheets);
      for (var i = 0; i < per; i++) {
        var a = rng() * Math.PI * 2;
        var r = Math.sqrt(rng());
        var s1 = r * (0.4 + rng() * 0.4) * Math.cos(a);
        var s2 = r * (0.3 + rng() * 0.3) * Math.sin(a);
        // thickness along normal — comparable to sheet size
        var thick = gauss(rng) * 0.25 * (1 + noise);
        traces.facet.x.push(cx + s1*u[0] + s2*v[0] + thick*nw[0]);
        traces.facet.y.push(cy + s1*u[1] + s2*v[1] + thick*nw[1]);
        traces.facet.z.push(cz + s1*u[2] + s2*v[2] + thick*nw[2]);
      }
    }
    traces.facet.count = traces.facet.x.length;

    // Hubs: 3D blobs with equal spread in all dimensions
    var nHub = Math.floor(n * 0.15);
    var nClusters = 4;
    for (var s = 0; s < nClusters; s++) {
      var cx = gauss(rng) * 0.5;
      var cy = gauss(rng) * 0.5;
      var cz = gauss(rng) * 0.5;
      var spread = 0.15 + rng() * 0.1;
      var per = Math.floor(nHub / nClusters);
      for (var i = 0; i < per; i++) {
        traces.hub.x.push(cx + gauss(rng) * spread * (1 + noise));
        traces.hub.y.push(cy + gauss(rng) * spread * (1 + noise));
        traces.hub.z.push(cz + gauss(rng) * spread * (1 + noise));
      }
    }
    traces.hub.count = traces.hub.x.length;

    // Dust: full 3D isotropic
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
