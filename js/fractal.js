// ============================================================================
// D.R.E.A.M — fractal canvas backdrop
// WebGL Mandelbrot with eased pan between named centers; CPU fallback.
// Ported 1:1 from the original Flask base.html (lines 480-758).
// ============================================================================

(function () {
  const canvas = document.getElementById('mb');
  if (!canvas) return;

  let gl = canvas.getContext('webgl', { antialias: false, alpha: true, powerPreference: 'high-performance' });

  /* ---------- Hardcoded params ---------- */
  const SPEED_MAX       = 0.06;
  const USER_SPEED      = SPEED_MAX * 0.04;
  const BASE_SPEED      = 0.02;
  const DRIFT           = 0.00035;
  const SCALE_BASE      = 1.35;
  const MAX_ITER        = 180;
  const CHANGE_INTERVAL = 60000;
  const TRANSITION_DUR  = 12000;

  // soft palette — accent hue matches the new CSS
  const HUE = 210;

  const patterns = [
    { name: 'Seahorse Valley',   x: -0.743643887037151, y:  0.13182590420533 },
    { name: 'Elephant Valley',   x: -0.743000000000000, y:  0.13100000000000 },
    { name: 'Triple Spiral',     x: -0.390540000000000, y: -0.58679000000000 },
    { name: 'Filaments',         x: -0.101100000000000, y:  0.95630000000000 },
    { name: 'Seahorse Tail',     x: -0.745300000000000, y:  0.11270000000000 },
    { name: 'Elephant Trunk',    x: -0.748000000000000, y:  0.09950000000000 },
    { name: 'Mini Mandelbrot',   x: -1.250660000000000, y:  0.02012000000000 },
    { name: 'Spiral Island',     x: -0.562500000000000, y:  0.64250000000000 },
    { name: 'Valley Ridge',      x: -0.660000000000000, y:  0.44800000000000 },
    { name: 'Misiurewicz Point', x: -1.401155000000000, y:  0.00000000000000 },
  ];

  let patIdx = 1;
  let current = { x: patterns[patIdx].x, y: patterns[patIdx].y };
  let target  = { x: patterns[patIdx].x, y: patterns[patIdx].y };
  let lastChange = performance.now();

  function nextPattern() {
    patIdx = (patIdx + 1) % patterns.length;
    target = { x: patterns[patIdx].x, y: patterns[patIdx].y };
    lastChange = performance.now();
  }

  const params = { zoomExp: 0 };

  // ===== WebGL path =====
  if (gl) {
    const vsSrc = `
      attribute vec2 aPos;
      varying vec2 vUv;
      void main(){
        vUv = (aPos + 1.0) * 0.5;
        gl_Position = vec4(aPos, 0.0, 1.0);
      }`;
    const fsSrc = `
      precision highp float;
      varying vec2 vUv;
      uniform vec2  uResolution;
      uniform float uAspect;
      uniform float uScale;
      uniform vec2  uCenter;
      uniform float uHue;
      uniform int   uMaxIter;

      vec3 hsv2rgb(vec3 c){
        vec3 p = abs(fract(c.x + vec3(0., 2./6., 4./6.)) * 6. - 3.);
        return c.z * mix(vec3(1.), clamp(p - 1., 0., 1.), c.y);
      }

      void main(){
        vec2 uv = (vUv - 0.5) * vec2(uAspect, 1.0) * 2.0;
        vec2 c = vec2(uv.x * uScale + uCenter.x, uv.y * uScale + uCenter.y);

        vec2 z = vec2(0.0);
        float i = 0.0;
        const int ITR = 1024;
        for(int k = 0; k < ITR; k++){
          if(k >= uMaxIter) break;
          float x = (z.x*z.x - z.y*z.y) + c.x;
          float y = (2.0*z.x*z.y) + c.y;
          z = vec2(x, y);
          if(dot(z, z) > 4.0){ i = float(k); break; }
          i = float(k);
        }

        float mu = i;
        float r2 = dot(z, z);
        if(r2 > 1.0){
          mu = i - log2(log(sqrt(r2)));
        }

        float t = clamp(mu / 40.0, 0.0, 1.0);
        float v = 0.18 + pow(t, 0.75) * 0.35;
        float h = mod(uHue / 360.0, 1.0);
        vec3 col = hsv2rgb(vec3(h, 0.35, v));
        gl_FragColor = vec4(col, 1.0);
      }`;

    function compile(type, src) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(sh));
        return null;
      }
      return sh;
    }
    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(prog));
      gl = null;
    }
    if (gl) {
      gl.useProgram(prog);
      const quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
      const aPos = gl.getAttribLocation(prog, 'aPos');
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      const uResolution = gl.getUniformLocation(prog, 'uResolution');
      const uAspect     = gl.getUniformLocation(prog, 'uAspect');
      const uScale      = gl.getUniformLocation(prog, 'uScale');
      const uCenter     = gl.getUniformLocation(prog, 'uCenter');
      const uHue        = gl.getUniformLocation(prog, 'uHue');
      const uMaxIter    = gl.getUniformLocation(prog, 'uMaxIter');

      function fit() {
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        const w = Math.floor(window.innerWidth * dpr);
        const h = Math.floor(window.innerHeight * dpr);
        canvas.width = w; canvas.height = h;
        canvas.style.width = '100%'; canvas.style.height = '100%';
        gl.viewport(0, 0, w, h);
        gl.uniform2f(uResolution, w, h);
        gl.uniform1f(uAspect, w / Math.max(1, h));
      }
      window.addEventListener('resize', fit);
      fit();

      let last = performance.now();
      function render() {
        const now = performance.now();
        const dt = Math.max(0, now - last);
        last = now;

        const tSince = now - lastChange;
        if (tSince >= CHANGE_INTERVAL) nextPattern();
        if (tSince < TRANSITION_DUR) {
          const u = tSince / TRANSITION_DUR;
          current.x = current.x * (1 - u) + target.x * u;
          current.y = current.y * (1 - u) + target.y * u;
        } else {
          current.x = target.x;
          current.y = target.y;
        }

        const speedMultiplier = 0.5 + USER_SPEED * 10.0;
        params.zoomExp += BASE_SPEED * speedMultiplier * (dt / 1000);
        const scale = Math.exp(-params.zoomExp);

        const driftX = Math.sin(now * 0.00020) * DRIFT;
        const driftY = Math.cos(now * 0.00017) * DRIFT;

        gl.uniform1f(uScale, SCALE_BASE * scale);
        gl.uniform2f(uCenter, current.x + driftX, current.y + driftY);
        gl.uniform1f(uHue, HUE);
        gl.uniform1i(uMaxIter, MAX_ITER);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
      }
      requestAnimationFrame(render);
      return;
    }
  }

  // ===== CPU fallback (light) =====
  const ctx = canvas.getContext('2d', { alpha: true });
  const buf = document.createElement('canvas');
  const bctx = buf.getContext('2d', { alpha: true, willReadFrequently: true });

  function fitCPU() {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width  = Math.floor(window.innerWidth  * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width  = '100%';
    canvas.style.height = '100%';
    const targetW = Math.min(960, canvas.width / 2);
    const aspect  = canvas.width / Math.max(1, canvas.height);
    const targetH = Math.max(60, Math.round(targetW / Math.max(0.5, aspect)));
    buf.width  = Math.max(120, targetW | 0);
    buf.height = targetH | 0;
  }
  window.addEventListener('resize', fitCPU);
  fitCPU();

  let last = performance.now();
  function renderCPU() {
    const now = performance.now();
    const dt = Math.max(0, now - last);
    last = now;

    const tSince = now - lastChange;
    if (tSince >= CHANGE_INTERVAL) nextPattern();
    if (tSince < TRANSITION_DUR) {
      const u = tSince / TRANSITION_DUR;
      current.x = current.x * (1 - u) + target.x * u;
      current.y = current.y * (1 - u) + target.y * u;
    } else {
      current.x = target.x;
      current.y = target.y;
    }

    const speedMultiplier = 0.5 + USER_SPEED * 10.0;
    params.zoomExp += BASE_SPEED * speedMultiplier * (dt / 1000);
    const scale = Math.exp(-params.zoomExp);

    const driftX = Math.sin(now * 0.00020) * DRIFT;
    const driftY = Math.cos(now * 0.00017) * DRIFT;
    const cx = current.x + driftX;
    const cy = current.y + driftY;

    const W = buf.width, H = buf.height;
    const img = bctx.createImageData(W, H);
    const data = img.data;
    const aspect = W / H;
    const halfH = SCALE_BASE * scale, halfW = halfH * aspect;

    const hi = Math.floor(HUE / 60) % 6;
    const xFactor = 1 - Math.abs(((HUE / 60) % 2) - 1);

    let p = 0;
    for (let j = 0; j < H; j++) {
      const y0 = cy + (j / H - .5) * 2 * halfH;
      for (let i = 0; i < W; i++) {
        const x0 = cx + (i / W - .5) * 2 * halfW;
        let x = 0, y = 0, xx = 0, yy = 0, iter = 0;
        while (iter < MAX_ITER && (xx + yy) <= 4.0) {
          y = 2 * x * y + y0;
          x = xx - yy + x0;
          xx = x * x;
          yy = y * y;
          iter++;
        }

        let k = 0.0;
        if (iter < MAX_ITER) {
          const mu = iter - Math.log(Math.log(Math.sqrt(xx + yy))) / Math.log(2.0);
          const t = Math.min(1.0, Math.max(0.0, mu / 40.0));
          k = 0.18 + Math.pow(t, 0.75) * 0.35;
        }
        const c = k, X = c * xFactor * 0.35;
        let r = 0, g = 0, b = 0;
        switch (hi) {
          case 0: r = c; g = X; b = 0; break;
          case 1: r = X; g = c; b = 0; break;
          case 2: r = 0; g = c; b = X; break;
          case 3: r = 0; g = X; b = c; break;
          case 4: r = X; g = 0; b = c; break;
          default: r = c; g = 0; b = X; break;
        }
        data[p++] = (r * 255) | 0;
        data[p++] = (g * 255) | 0;
        data[p++] = (b * 255) | 0;
        data[p++] = 255;
      }
    }
    bctx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(buf, 0, 0, canvas.width, canvas.height);
    requestAnimationFrame(renderCPU);
  }
  requestAnimationFrame(renderCPU);
})();
