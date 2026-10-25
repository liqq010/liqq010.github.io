// Interactive figure: multiplying z by cos θ + i sin θ = cos θ · z + sin θ · (i z)
(function () {
  const root = document.getElementById('complex-rot');
  if (!root) return;
  const BLUE = '#4c8bf5', ORANGE = '#e8590c', RED = '#c2255c';
  const W = 360, H = 320, ox = 180, oy = 170, s = 120;
  const z = [0.9, 0.35];
  const P = (x, y) => [ox + x * s, oy - y * s];
  const NS = 'http://www.w3.org/2000/svg';

  root.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Rotating z by multiplying with cos θ + i sin θ" style="width:100%;max-width:420px;height:auto;display:block;margin:0 auto;font-family:inherit"></svg>` +
    `<div style="display:flex;align-items:center;gap:12px;max-width:420px;margin:8px auto 0">` +
    `<label for="complex-rot-th" style="color:var(--secondary)">θ</label>` +
    `<input type="range" id="complex-rot-th" min="0" max="360" value="60" style="flex:1">` +
    `<output id="complex-rot-o" style="min-width:3.5em;text-align:right;font-variant-numeric:tabular-nums">60°</output></div>` +
    `<p id="complex-rot-eq" style="text-align:center;font-size:0.9em;color:var(--secondary);margin:6px 0 0;font-variant-numeric:tabular-nums"></p>`;
  const svg = root.querySelector('svg'), sl = root.querySelector('input'), out = root.querySelector('output'), eq = root.querySelector('#complex-rot-eq');

  const el = (tag, attrs, style) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (style) e.setAttribute('style', style);
    svg.appendChild(e);
    return e;
  };
  const defs = el('defs', {});
  [['b', BLUE], ['o', ORANGE], ['r', RED]].forEach(([n, c]) => {
    defs.innerHTML += `<marker id="cr-${n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:${c}"/></marker>`;
  });
  const line = (a, b, style, mk) => {
    const e = el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, style);
    if (mk) e.setAttribute('marker-end', `url(#${mk})`);
    return e;
  };
  const text = (p, t, col, anchor, dx, dy, italic) => {
    const e = el('text', { x: p[0] + dx, y: p[1] + dy, 'text-anchor': anchor }, `fill:${col};font-size:14px;${italic ? 'font-style:italic' : ''}`);
    e.textContent = t;
    return e;
  };

  line(P(-1.4, 0), P(1.4, 0), 'stroke:var(--border);stroke-width:1');
  line(P(0, -1.2), P(0, 1.3), 'stroke:var(--border);stroke-width:1');
  text(P(1.4, 0), 'Re', 'var(--secondary)', 'end', 0, 16);
  text(P(0, 1.3), 'Im', 'var(--secondary)', 'start', 6, 10);
  el('circle', { cx: ox, cy: oy, r: Math.hypot(z[0], z[1]) * s }, 'fill:none;stroke:var(--border);stroke-width:1;stroke-dasharray:3 4');

  const iz = [-z[1], z[0]];
  line(P(0, 0), P(...z), `stroke:${BLUE};stroke-width:2.2`, 'cr-b');
  text(P(...z), 'z', BLUE, 'start', 6, 4, true);
  line(P(0, 0), P(...iz), `stroke:${ORANGE};stroke-width:1.8;stroke-dasharray:5 4`, 'cr-o');
  text(P(...iz), 'i z', ORANGE, 'end', -6, 0, true);

  const cosSeg = line(P(0, 0), P(0, 0), `stroke:${BLUE};stroke-width:6;opacity:.3`);
  const sinSeg = line(P(0, 0), P(0, 0), `stroke:${ORANGE};stroke-width:2.2`, 'cr-o');
  const res = line(P(0, 0), P(0, 0), `stroke:${RED};stroke-width:2.6`, 'cr-r');
  const resLbl = text(P(0, 0), 'z′', RED, 'start', 6, -6, true);

  const f = x => (x < 0 ? '−' : '') + Math.abs(x).toFixed(2);
  function draw() {
    const t = sl.value * Math.PI / 180, c = Math.cos(t), sn = Math.sin(t);
    const a = [z[0] * c, z[1] * c], b = [a[0] + iz[0] * sn, a[1] + iz[1] * sn];
    const set = (e, p, q) => { e.setAttribute('x1', P(...p)[0]); e.setAttribute('y1', P(...p)[1]); e.setAttribute('x2', P(...q)[0]); e.setAttribute('y2', P(...q)[1]); };
    set(cosSeg, [0, 0], a); set(sinSeg, a, b); set(res, [0, 0], b);
    resLbl.setAttribute('x', P(...b)[0] + 6); resLbl.setAttribute('y', P(...b)[1] - 6);
    out.textContent = sl.value + '°';
    const pm = x => (x < 0 ? ' − ' : ' + ') + Math.abs(x).toFixed(2);
    eq.textContent = `z′ = (cos θ + i sin θ) z = ${f(c)}·z${pm(sn)}·(i z) = ${f(b[0])}${pm(b[1])} i`;
  }
  sl.addEventListener('input', draw);
  draw();
})();
