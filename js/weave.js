// ============================================================================
// D.R.E.A.M — client-side kernel-weave generator
// Replaces the original /weave/data backend (weave_blueprint.py) with a
// pure-JS implementation: seeded RNG, orthonormal projection via QR (modified
// Gram-Schmidt), and PCA via SVD-free power iteration. Same JSON shape as the
// backend, so the existing partial's Plotly code keeps working.
//
// Exposes window.DreamWeave.generate({lam, lam_q, deff, n, seed})
// ============================================================================

(function () {
  // ---------- seeded RNG (mulberry32) ----------
  function makeRng(seed) {
    let a = (seed | 0) >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function normalArray(rng, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = gauss(rng);
    return out;
  }
  function normalMatrix(rng, rows, cols) {
    const M = [];
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      for (let j = 0; j < cols; j++) row[j] = gauss(rng);
      M.push(row);
    }
    return M;
  }

  // ---------- matrix ops ----------
  function matMul(A, B) {
    // A: m×k, B: k×n  -> m×n
    const m = A.length, k = B.length, n = B[0].length;
    const out = new Array(m);
    for (let i = 0; i < m; i++) {
      const row = new Array(n).fill(0);
      const Ai = A[i];
      for (let l = 0; l < k; l++) {
        const ail = Ai[l];
        if (ail === 0) continue;
        const Bl = B[l];
        for (let j = 0; j < n; j++) row[j] += ail * Bl[j];
      }
      out[i] = row;
    }
    return out;
  }
  function transpose(A) {
    const m = A.length, n = A[0].length;
    const out = new Array(n);
    for (let j = 0; j < n; j++) {
      const col = new Array(m);
      for (let i = 0; i < m; i++) col[i] = A[i][j];
      out[j] = col;
    }
    return out;
  }
  // Orthonormalize columns of A (m×k) via modified Gram-Schmidt.
  function orthonormalize(A) {
    const m = A.length, k = A[0].length;
    const Q = A.map((row) => row.slice());
    for (let j = 0; j < k; j++) {
      // normalize column j
      let norm = 0;
      for (let i = 0; i < m; i++) norm += Q[i][j] * Q[i][j];
      norm = Math.sqrt(norm) || 1e-12;
      for (let i = 0; i < m; i++) Q[i][j] /= norm;
      // subtract projection onto column j from later columns
      for (let l = j + 1; l < k; l++) {
        let dot = 0;
        for (let i = 0; i < m; i++) dot += Q[i][j] * Q[i][l];
        for (let i = 0; i < m; i++) Q[i][l] -= dot * Q[i][j];
      }
    }
    return Q;
  }
  // PCA via power iteration with deflation (find top-3 directions of Y, m×4)
  // Returns m×3 matrix of projected coordinates.
  function pca3(Y) {
    const m = Y.length, d = Y[0].length;
    // center
    const mean = new Array(d).fill(0);
    for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) mean[j] += Y[i][j];
    for (let j = 0; j < d; j++) mean[j] /= m;
    const Yc = Y.map((row) => row.map((v, j) => v - mean[j]));
    const Yt = transpose(Yc); // d×m
    // covariance d×d = (1/m) Yt * Y
    const cov = matMul(Yt, Yc).map((row) => row.map((v) => v / m));

    const dirs = [];
    const cDef = cov.map((row) => row.slice());
    for (let k = 0; k < 3; k++) {
      // random init
      let v = normalArray(makeRng(123 + k), d);
      let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      v = v.map((x) => x / norm);
      for (let iter = 0; iter < 60; iter++) {
        const w = cDef.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
        const wn = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1e-12;
        v = w.map((x) => x / wn);
      }
      // eigenvalue
      const lambda = v.reduce((s, x, j) => s + x * cov[j].reduce((t, y, k) => t + y * v[k], 0), 0);
      dirs.push(v);
      // deflate
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cDef[i][j] -= lambda * v[i] * v[j];
    }
    // project: Y (m×d) * dirsT (d×3)
    const dirsT = transpose(dirs); // 3×d, each row is a direction
    const out = new Array(m);
    for (let i = 0; i < m; i++) {
      const yi = Yc[i];
      out[i] = [
        yi.reduce((s, x, j) => s + x * dirsT[0][j], 0),
        yi.reduce((s, x, j) => s + x * dirsT[1][j], 0),
        yi.reduce((s, x, j) => s + x * dirsT[2][j], 0),
      ];
    }
    return out;
  }

  // ---------- weave generation (matches weave_blueprint.py generate_weave) ----------
  function generateWeave(nTotal, rng, nFil = 6, nFac = 3, nHubs = 4) {
    const parts = [], labels = [], focus = [];
    const Nf = Math.max(1, Math.floor(0.45 * nTotal));
    const Naf = Math.max(1, Math.floor(0.30 * nTotal));
    const Nh = Math.max(1, Math.floor(0.15 * nTotal));
    const Nd = Math.max(1, nTotal - (Nf + Naf + Nh));

    const perFil = Math.max(20, Math.floor(Nf / nFil));
    for (let s = 0; s < nFil; s++) {
      const c = normalArray(rng, 10).map((x) => x * 0.4);
      let v = normalArray(rng, 10);
      const vn = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      v = v.map((x) => x / vn);
      const pts = [];
      for (let i = 0; i < perFil; i++) {
        const t = gauss(rng) * 0.8;
        const E = normalArray(rng, 10);
        // E -= (E . v) * v
        const Ev = E.reduce((s, x, j) => s + x * v[j], 0);
        for (let j = 0; j < 10; j++) {
          pts.push(c[j] + t * v[j] + 0.03 * (E[j] - Ev * v[j]));
        }
        labels.push('filament');
        focus.push(0.25);
      }
      parts.push(pts);
    }

    const perFac = Math.max(20, Math.floor(Naf / nFac));
    for (let s = 0; s < nFac; s++) {
      const c = normalArray(rng, 10).map((x) => x * 0.6);
      const Uraw = normalMatrix(rng, 10, 2);
      const U = orthonormalize(Uraw); // 10×2
      const pts = [];
      for (let i = 0; i < perFac; i++) {
        const s_ = gauss(rng) * 0.6;
        const t = gauss(rng) * 0.35;
        for (let j = 0; j < 10; j++) {
          pts.push(c[j] + s_ * U[j][0] + t * U[j][1] + gauss(rng) * 0.015);
        }
        labels.push('facet');
        focus.push(0.45);
      }
      parts.push(pts);
    }

    const perHub = Math.max(10, Math.floor(Nh / nHubs));
    for (let s = 0; s < nHubs; s++) {
      const c = normalArray(rng, 10).map((x) => x * 0.3);
      const pts = [];
      for (let i = 0; i < perHub; i++) {
        for (let j = 0; j < 10; j++) {
          pts.push(c[j] + gauss(rng) * 0.04);
        }
        labels.push('hub');
        focus.push(0.35);
      }
      parts.push(pts);
    }

    const dPts = [];
    for (let i = 0; i < Nd; i++) {
      for (let j = 0; j < 10; j++) dPts.push(gauss(rng) * 0.9);
      labels.push('dust');
      focus.push(1.0);
    }
    parts.push(dPts);

    // flatten to N×10
    const X10 = [];
    for (const part of parts) {
      for (let i = 0; i < part.length; i += 10) X10.push(part.slice(i, i + 10));
    }
    return { X10, labels, focus };
  }

  function projectTo4D(X10, labels, focus, lam, lamQ, deff, rng, P) {
    if (!P) {
      const A = normalMatrix(rng, 10, 4);
      P = orthonormalize(A); // 10×4
    }
    const Y = matMul(X10, P); // N×4
    const R = Math.exp(-Math.pow(lam / lamQ, deff));
    const sigma0 = 0.06;
    const out = new Array(Y.length);
    for (let i = 0; i < Y.length; i++) {
      const sigma = sigma0 * (1.0 - R) * focus[i];
      const noise = normalArray(rng, 4).map((x) => x * sigma);
      out[i] = Y[i].map((v, j) => v + noise[j]);
    }
    return { Y4: out, R };
  }

  function generate({ lam = 0.30, lam_q = 1.0, deff = 2.2, n = 3500, seed = 42 } = {}) {
    const rng = makeRng(seed);
    const { X10, labels, focus } = generateWeave(n, rng);
    const rng2 = makeRng(seed + 1);
    const P = orthonormalize(normalMatrix(rng2, 10, 4));
    const { Y4, R } = projectTo4D(X10, labels, focus, lam, lam_q, deff, rng2, P);
    const PCs = pca3(Y4);

    const out = {};
    for (const lbl of ['filament', 'facet', 'hub', 'dust']) {
      const xs = [], ys = [], zs = [];
      for (let i = 0; i < labels.length; i++) {
        if (labels[i] === lbl) {
          xs.push(PCs[i][0]);
          ys.push(PCs[i][1]);
          zs.push(PCs[i][2]);
        }
      }
      out[lbl] = { x: xs, y: ys, z: zs, count: xs.length };
    }
    return { r: R, lambda: lam, lambda_q: lam_q, deff, traces: out };
  }

  window.DreamWeave = { generate };
})();
