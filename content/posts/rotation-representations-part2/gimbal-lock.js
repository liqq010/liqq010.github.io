// Interactive figure: the yaw, pitch and roll axes of R = Rz(ψ) Ry(β) Rx(φ); at pitch = ±90° the roll axis lines up with the yaw axis
(function () {
  const root = document.getElementById('gimbal-lock');
  if (!root) return;
  const BLUE = '#4c8bf5', ORANGE = '#e8590c', RED = '#c2255c';
  const W = 420, H = 320, ox = 200, oy = 175, s = 115;
  // x toward the viewer (down-left), y to the right, z up
  const P = ([x, y, z]) => [ox + s * (0.95 * y - 0.55 * x), oy - s * (z - 0.32 * x - 0.12 * y)];
  const NS = 'http://www.w3.org/2000/svg';

  const slider = (id, label, min, max, val) =>
    `<div style="display:flex;align-items:center;gap:12px;max-width:420px;margin:6px auto 0">` +
    `<label for="${id}" style="color:var(--secondary);min-width:6.5em">${label}</label>` +
    `<input type="range" id="${id}" min="${min}" max="${max}" value="${val}" style="flex:1">` +
    `<output style="min-width:3.5em;text-align:right;font-variant-numeric:tabular-nums"></output></div>`;
  root.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Yaw, pitch and roll axes; at pitch 90 degrees the roll axis lines up with the yaw axis" style="width:100%;max-width:420px;height:auto;display:block;margin:0 auto;font-family:inherit"></svg>` +
    slider('gl-yaw', 'yaw ψ', -180, 180, -30) + slider('gl-pitch', 'pitch β', -90, 90, 40) +
    `<p id="gl-eq" style="text-align:center;font-size:0.9em;color:var(--secondary);margin:6px 0 0;font-variant-numeric:tabular-nums;min-height:2.6em"></p>`;
  const svg = root.querySelector('svg');
  const sy = root.querySelector('#gl-yaw'), sp = root.querySelector('#gl-pitch');
  const oyw = sy.parentNode.querySelector('output'), op = sp.parentNode.querySelector('output');
  const eq = root.querySelector('#gl-eq');

  const el = (tag, attrs, style) => {
    const e = document.createElementNS(NS, tag);
    for (const a in attrs) e.setAttribute(a, attrs[a]);
    if (style) e.setAttribute('style', style);
    svg.appendChild(e);
    return e;
  };
  const defs = el('defs', {});
  [['b', BLUE], ['o', ORANGE], ['r', RED], ['g', 'var(--border)']].forEach(([n, c]) => {
    defs.innerHTML += `<marker id="gl-${n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:${c}"/></marker>`;
  });
  const O = P([0, 0, 0]);
  // fixed world axes
  [[[1.25, 0, 0], 'x'], [[0, 1.3, 0], 'y'], [[0, 0, 1.3], 'z']].forEach(([v, n]) => {
    const p = P(v);
    el('line', { x1: O[0], y1: O[1], x2: p[0], y2: p[1], 'marker-end': 'url(#gl-g)' }, 'stroke:var(--border);stroke-width:1.2');
    const t = el('text', { x: p[0] + (n === 'x' ? -12 : 8), y: p[1] + (n === 'z' ? 4 : 14) }, 'fill:var(--secondary);font-size:13px;font-style:italic');
    t.textContent = n;
  });
  const mk = (col, id) => {
    const l = el('line', { x1: O[0], y1: O[1], 'marker-end': `url(#gl-${id})` }, `stroke:${col};stroke-width:2.6`);
    const t = el('text', {}, `fill:${col};font-size:13px`);
    return [l, t];
  };
  const yawA = mk(BLUE, 'b'), pitchA = mk(ORANGE, 'o'), rollA = mk(RED, 'r');
  const place = ([l, t], v, txt, dx, dy) => {
    const p = P(v);
    l.setAttribute('x2', p[0]); l.setAttribute('y2', p[1]);
    t.setAttribute('x', p[0] + dx); t.setAttribute('y', p[1] + dy);
    t.setAttribute('text-anchor', p[0] < O[0] - 10 ? 'end' : 'start');
    t.textContent = txt;
  };

  function draw() {
    const psi = +sy.value * Math.PI / 180, th = +sp.value * Math.PI / 180;
    oyw.textContent = sy.value + '°'; op.textContent = sp.value + '°';
    const c = Math.cos, sn = Math.sin;
    const yaw = [0, 0, 1];
    const pitch = [-sn(psi), c(psi), 0];                         // Rz(ψ) y
    const roll = [c(psi) * c(th), sn(psi) * c(th), -sn(th)];     // Rz(ψ) Ry(β) x
    const L = 1.05;
    place(yawA, yaw.map(x => x * L), 'yaw axis', 8, -2);
    place(pitchA, pitch.map(x => x * L), 'pitch axis', 6, -6);
    place(rollA, roll.map(x => x * L), 'roll axis', 6, 16);
    const ang = Math.acos(Math.max(-1, Math.min(1, roll[2]))) * 180 / Math.PI;
    const par = Math.min(ang, 180 - ang);
    eq.textContent = `angle between the yaw axis and the roll axis line: ${par.toFixed(0)}°` +
      (par < 0.5 ? ' — same line: one degree of freedom is lost' : '');
  }
  sy.addEventListener('input', draw);
  sp.addEventListener('input', draw);
  draw();
})();
