// Interactive figure: rotate about z from 0° to 360° and plot one component of each representation
(function () {
  const root = document.getElementById('continuity-plot');
  if (!root) return;
  const BLUE = '#4c8bf5', RED = '#c2255c';
  const W = 380, H = 180, pad = { l: 40, r: 12, t: 12, b: 24 };
  const NS = 'http://www.w3.org/2000/svg';
  const rad = d => d * Math.PI / 180;
  const reps = [
    { t: 'Euler angles: yaw ψ (degrees)', lo: -180, hi: 180, fn: d => d <= 180 ? d : d - 360 },
    { t: 'Rotation vector: r_z (degrees)', lo: -180, hi: 180, fn: d => d <= 180 ? d : d - 360 },
    { t: 'Quaternion (w ≥ 0): z component', lo: -1, hi: 1, fn: d => { const w = Math.cos(rad(d) / 2), z = Math.sin(rad(d) / 2); return w < -1e-9 ? -z : z; } },
    { t: '6D: y component of column 1 (sin θ)', lo: -1, hi: 1, fn: d => Math.sin(rad(d)) },
  ];

  root.innerHTML =
    `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px 20px"></div>` +
    `<div style="display:flex;align-items:center;gap:12px;max-width:460px;margin:10px auto 0">` +
    `<label for="cont-th" style="color:var(--secondary);font-size:0.85em">θ (about z)</label>` +
    `<input type="range" id="cont-th" min="0" max="360" value="150" style="flex:1">` +
    `<output style="font-size:0.85em;min-width:3.5em;text-align:right;font-variant-numeric:tabular-nums"></output></div>`;
  const grid = root.firstChild, sl = root.querySelector('input'), out = root.querySelector('output');

  const X = x => pad.l + x / 360 * (W - pad.l - pad.r);
  const charts = reps.map(rp => {
    const wrap = document.createElement('div');
    wrap.style.minWidth = '0';
    wrap.innerHTML = `<p style="margin:0 0 4px;font-size:0.75em;color:var(--primary)">${rp.t}</p>`;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', rp.t);
    svg.setAttribute('style', 'width:100%;height:auto;display:block;font-family:inherit;touch-action:none;cursor:crosshair');
    wrap.appendChild(svg);
    grid.appendChild(wrap);
    const el = (tag, attrs, style, text) => {
      const e = document.createElementNS(NS, tag);
      for (const a in attrs) e.setAttribute(a, attrs[a]);
      if (style) e.setAttribute('style', style);
      if (text !== undefined) e.textContent = text;
      svg.appendChild(e);
      return e;
    };
    const Y = y => pad.t + (rp.hi - y) / (rp.hi - rp.lo) * (H - pad.t - pad.b);
    [rp.lo, 0, rp.hi].forEach(y => {
      el('line', { x1: pad.l, y1: Y(y), x2: W - pad.r, y2: Y(y) }, `stroke:var(--border);stroke-width:1${y === 0 ? '' : ';stroke-dasharray:2 4'}`);
      el('text', { x: pad.l - 6, y: Y(y) + 4, 'text-anchor': 'end' }, 'fill:var(--secondary);font-size:11px', String(y).replace('-', '−'));
    });
    [0, 180, 360].forEach(x => el('text', { x: X(x), y: H - 6, 'text-anchor': x === 0 ? 'start' : x === 360 ? 'end' : 'middle' }, 'fill:var(--secondary);font-size:11px', x + '°'));
    // the curve, broken where the value jumps
    let d = '', prev = null;
    for (let k = 0; k <= 720; k++) {
      const x = k / 2, y = rp.fn(x);
      const jump = prev !== null && Math.abs(y - prev) > (rp.hi - rp.lo) * 0.3;
      d += (k === 0 || jump ? 'M' : 'L') + X(x).toFixed(1) + ',' + Y(y).toFixed(1);
      prev = y;
    }
    el('path', { d }, `fill:none;stroke:${BLUE};stroke-width:2;stroke-linejoin:round`);
    const cursor = el('line', { y1: pad.t, y2: H - pad.b }, 'stroke:var(--secondary);stroke-width:1;stroke-dasharray:3 3');
    const pt = el('circle', { r: 4.5 }, `fill:${RED}`);
    const lbl = el('text', {}, 'fill:var(--primary);font-size:12px;font-variant-numeric:tabular-nums');
    const pick = e => {
      const r = svg.getBoundingClientRect();
      const f = ((e.clientX - r.left) / r.width * W - pad.l) / (W - pad.l - pad.r);
      sl.value = Math.round(Math.max(0, Math.min(1, f)) * 360);
      draw();
    };
    svg.addEventListener('pointermove', pick);
    svg.addEventListener('pointerdown', pick);
    return cur => {
      const v = rp.fn(cur), x = X(cur), y = Y(v);
      cursor.setAttribute('x1', x); cursor.setAttribute('x2', x);
      pt.setAttribute('cx', x); pt.setAttribute('cy', y);
      const right = cur > 270;
      lbl.setAttribute('x', x + (right ? -8 : 8));
      lbl.setAttribute('y', Math.max(pad.t + 10, y - 8));
      lbl.setAttribute('text-anchor', right ? 'end' : 'start');
      lbl.textContent = Math.abs(rp.hi) > 2 ? Math.round(v) + '°' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
    };
  });

  function draw() {
    const cur = +sl.value;
    out.textContent = cur + '°';
    charts.forEach(f => f(cur));
  }
  sl.addEventListener('input', draw);
  draw();
})();
