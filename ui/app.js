const $ = (s) => document.querySelector(s);
const token = new URLSearchParams(location.search).get('t') ?? '';
const COLORS = { claude: '#ff9d4d', codex: '#3df5c8', lead: '#e8ecff', bad: '#ff4d5e', none: '#7a8199', ok: '#3dff8a', warn: '#ffd23d' };

const S = {
  settings: { dir: '', mode: 'auto', slots: 2 },
  commands: [],
  accounts: new Map(),
  runs: new Map(),
  current: null,
  tab: 'feed',
  focus: null,
};

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const stick = (box, fn) => { const near = box.scrollHeight - box.scrollTop - box.clientHeight < 60; fn(); if (near) box.scrollTop = box.scrollHeight; };

// ---------- terminal ----------
const termOut = $('#termOut');
function say(text, cls = '') { stick(termOut, () => { const d = el('div', cls); d.textContent = text; termOut.append(d); }); }

const hist = []; let hi = 0;
$('#termForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = $('#termIn');
  const line = input.value.trim();
  if (!line) return;
  input.value = '';
  hist.push(line); hi = hist.length;
  if (line === 'clear') { termOut.innerHTML = ''; return; }
  await fetch('/api/cmd', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ line }) });
});
$('#termIn').addEventListener('keydown', (e) => {
  const input = e.target;
  if (e.key === 'ArrowUp') { e.preventDefault(); if (hi > 0) input.value = hist[--hi]; }
  else if (e.key === 'ArrowDown') { e.preventDefault(); input.value = hi < hist.length - 1 ? hist[++hi] : (hi = hist.length, ''); }
  else if (e.key === 'Tab') {
    e.preventDefault();
    const m = S.commands.filter((c) => c.startsWith(input.value));
    if (m.length === 1) input.value = `${m[0]} `;
    else if (m.length > 1) say(m.join('  '), 'dim');
  }
});
document.addEventListener('keydown', (e) => { if (!e.ctrlKey && !e.metaKey && !e.altKey && e.target === document.body) $('#termIn').focus(); });

// ---------- tabs ----------
document.querySelectorAll('nav button').forEach((b) => b.addEventListener('click', () => {
  S.tab = b.dataset.tab;
  document.querySelectorAll('nav button').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('on', p.id === S.tab));
}));

// ---------- state ----------
const run = (id) => {
  if (!S.runs.has(id)) S.runs.set(id, { id, task: '', status: 'running', agents: new Map(), claims: new Map(), report: '', feed: [] });
  return S.runs.get(id);
};
const cur = () => S.runs.get(S.current);

function harnessOf(a) { return a.role === 'lead' ? 'lead' : (a.harness ?? 'none'); }

// ---------- event reducer ----------
function apply(ev) {
  switch (ev.kind) {
    case 'hello': S.settings = ev.settings; S.commands = ev.commands; renderHeader(); break;
    case 'settings': S.settings = ev.settings; renderHeader(); break;
    case 'term': say(ev.text, ev.cls); break;
    case 'account': S.accounts.set(ev.account.id, ev.account); renderAccounts(); break;
    case 'account.remove': S.accounts.delete(ev.id); renderAccounts(); break;
    case 'run.start': { const r = run(ev.runId); r.task = ev.task; S.current = ev.runId; S.focus = null; resetPanes(); renderRunbar(); break; }
    case 'agent.start': { const r = run(ev.runId); r.agents.set(ev.id, { id: ev.id, role: ev.role, title: ev.title, parent: ev.parent, harness: null, account: null, status: 'queued', born: ev.t }); renderRunbar(); break; }
    case 'agent.assign': { const a = run(ev.runId).agents.get(ev.agentId); if (a) Object.assign(a, { harness: ev.harness, account: ev.account, status: 'running', pulse: ev.t }); break; }
    case 'agent.failover': { const a = run(ev.runId).agents.get(ev.agentId); if (a) { a.status = 'queued'; a.flash = ev.t; } break; }
    case 'agent.end': {
      const a = run(ev.runId).agents.get(ev.agentId);
      if (a) { a.status = ev.ok ? 'done' : 'failed'; a.error = ev.error; a.endedAt = ev.t; }
      if (ev.ok) clickGear();
      break;
    }
    case 'agent.log': {
      const r = run(ev.runId); const a = r.agents.get(ev.agentId);
      r.feed.push({ id: ev.agentId, kind: ev.logKind, text: ev.text, harness: a ? harnessOf(a) : 'none' });
      if (r.feed.length > 1500) r.feed.shift();
      if (ev.runId === S.current && (!S.focus || S.focus === ev.agentId)) addFeed(r.feed.at(-1));
      break;
    }
    case 'cand.result': {
      const a = run(ev.runId).agents.get(ev.agentId);
      if (a) Object.assign(a, { score: ev.score, ok: ev.ok, champion: ev.champion || a.champion, idea: ev.idea, family: ev.family });
      if (ev.champion) say(`★ ${ev.agentId} new champion ${Number(ev.score).toFixed(4)} · ${ev.idea}`, 'ok');
      break;
    }
    case 'claims': {
      const r = run(ev.runId);
      ev.claims.forEach((c) => r.claims.set(c.id, c));
      if (ev.runId === S.current) renderClaims();
      break;
    }
    case 'run.report': { const r = run(ev.runId); r.report = ev.markdown; if (ev.runId === S.current) renderReport(); break; }
    case 'run.end': { const r = run(ev.runId); r.status = ev.status; if (ev.note) say(`run ${ev.runId} ${ev.status}: ${ev.note}`, 'err'); else say(`run ${ev.runId} ${ev.status}`, ev.status === 'done' ? 'ok' : 'warn'); renderRunbar(); break; }
  }
  renderStats();
}

function resetPanes() { $('#feed').innerHTML = ''; renderClaims(); renderReport(); $('#focus').hidden = true; }

// ---------- renderers ----------
function renderHeader() {
  $('#ws').textContent = S.settings.dir;
  const m = $('#modeChip'); m.textContent = S.settings.mode; m.classList.toggle('yolo', S.settings.mode === 'yolo');
}

function renderStats() {
  const r = cur();
  const run_ = r ? [...r.agents.values()].filter((a) => a.status === 'running').length : 0;
  const cl = r ? [...r.claims.values()] : [];
  const n = (v) => cl.filter((c) => c.verdict === v).length;
  $('#stats').innerHTML = [['AGENTS', r ? r.agents.size : 0], ['ACTIVE', run_], ['CLAIMS', cl.length], ['CONFIRMED', n('confirmed')], ['REFUTED', n('refuted')]]
    .map(([k, v]) => `<div class="stat"><b>${v}</b><small>${k}</small></div>`).join('');
  $('#claimN').textContent = cl.length;
  $('#graphEmpty').classList.toggle('hide', !!r);
}

function renderRunbar() {
  const bar = $('#runbar'); bar.innerHTML = '';
  for (const r of S.runs.values()) {
    const b = el('button', `${r.id === S.current ? 'on' : ''} ${r.status === 'running' ? 'running' : ''}`);
    b.textContent = `${r.id} · ${r.task}`;
    b.onclick = () => { S.current = r.id; S.focus = null; resetPanes(); r.feed.slice(-400).forEach(addFeed); renderRunbar(); renderStats(); };
    bar.append(b);
  }
}

function renderAccounts() {
  const list = $('#accList');
  const accs = [...S.accounts.values()].sort((a, b) => a.id.localeCompare(b.id));
  const ready = accs.filter((a) => a.status !== 'limited').length;
  $('#poolSum').textContent = `${ready}/${accs.length} ready`;
  list.innerHTML = '';
  for (const a of accs) {
    const pips = Array.from({ length: S.settings.slots ?? 2 }, (_, i) => `<i class="${i < a.load ? 'on' : ''}"></i>`).join('');
    const d = el('div', `acc ${a.harness} ${a.status}`, `
      <div class="top"><span class="id">${esc(a.id)}</span><span class="led"></span></div>
      <div class="task">${esc(a.task ?? (a.status === 'limited' ? 'rate limited' : 'idle'))}</div>
      <div class="row"><span class="pips">${pips}</span><span>${a.done}✓ ${a.failed ? a.failed + '✗' : ''}</span>
        <span class="${a.status === 'limited' ? 'cd' : a.authed === false ? 'auth-no' : ''}" data-until="${a.limitedUntil || 0}">${a.authed === false ? 'not logged in' : ''}</span></div>`);
    list.append(d);
  }
  tickCountdowns();
}

function tickCountdowns() {
  document.querySelectorAll('[data-until]').forEach((n) => {
    const u = Number(n.dataset.until);
    if (!u) return;
    const s = Math.max(0, Math.round((u - Date.now()) / 1000));
    n.textContent = s ? `back ${s >= 3600 ? Math.floor(s / 3600) + 'h' : ''}${s >= 60 ? Math.floor((s % 3600) / 60) + 'm' : ''}${s % 60}s` : 'ready';
  });
}

function addFeed(f) {
  const box = $('#feed');
  stick(box, () => {
    const row = el('div', `ln ${f.kind}`);
    const who = el('span', `who ${f.harness}`); who.textContent = f.id;
    who.onclick = () => focusAgent(f.id);
    const tx = el('span', 'tx'); tx.textContent = f.text;
    row.append(who, tx); box.append(row);
    while (box.childElementCount > 500) box.firstChild.remove();
  });
}

function focusAgent(id) {
  const r = cur(); if (!r) return;
  S.focus = S.focus === id ? null : id;
  $('#feed').innerHTML = '';
  r.feed.filter((f) => !S.focus || f.id === S.focus).slice(-400).forEach(addFeed);
  const f = $('#focus'); f.hidden = !S.focus; f.textContent = S.focus ? `filtering: ${S.focus} · click to clear` : '';
  f.onclick = () => focusAgent(S.focus);
}

function renderClaims() {
  const box = $('#claims'); box.innerHTML = '';
  const r = cur();
  if (!r || !r.claims.size) { box.append(el('div', 'muted', 'claims appear as researchers report them')); return; }
  const order = { confirmed: 0, unclear: 1, pending: 2, refuted: 3 };
  [...r.claims.values()].sort((a, b) => order[a.verdict] - order[b.verdict] || a.id.localeCompare(b.id)).forEach((c) => {
    box.append(el('div', `claim ${c.verdict}`, `
      <div class="h"><span>${esc(c.id)} · ${esc(c.by ?? '?')}${c.checkedBy ? ` → ${esc(c.checkedBy)}` : ''}</span><span class="badge">${c.verdict.toUpperCase()}</span></div>
      <div class="t">${esc(c.text)}</div>
      ${c.source ? `<div class="src">src: ${esc(c.source)}</div>` : ''}${c.note ? `<div class="note">${esc(c.note)}</div>` : ''}`));
  });
}

function md(src) {
  const out = []; let code = false; let list = false;
  const inline = (t) => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  for (const line of src.split('\n')) {
    if (line.startsWith('```')) { out.push(code ? '</pre>' : '<pre>'); code = !code; continue; }
    if (code) { out.push(esc(line) + '\n'); continue; }
    const li = /^\s*[-*]\s+(.*)/.exec(line);
    if (li) { if (!list) { out.push('<ul>'); list = true; } out.push(`<li>${inline(li[1])}</li>`); continue; }
    if (list) { out.push('</ul>'); list = false; }
    const h = /^(#{1,3})\s+(.*)/.exec(line);
    out.push(h ? `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>` : line.trim() ? `<p>${inline(line)}</p>` : '');
  }
  if (list) out.push('</ul>'); if (code) out.push('</pre>');
  return out.join('');
}
function renderReport() {
  const r = cur();
  $('#report').innerHTML = r?.report ? md(r.report) : '<div class="muted">report appears when the swarm finishes</div>';
}

// ---------- gear ----------
let gearDeg = 0;
function clickGear() { gearDeg += 30; $('#gearRot').style.transform = `rotate(${gearDeg}deg)`; }

// ---------- swarm graph ----------
const cv = $('#graph'); const ctx = cv.getContext('2d');
let W = 0; let H = 0; const nodes = new Map(); let hover = null;
function resize() { const r = cv.getBoundingClientRect(); const d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0); W = r.width; H = r.height; }
new ResizeObserver(resize).observe(cv); resize();

function targets(r) {
  const cx = W / 2; const cy = H / 2 + 6; const m = Math.min(W, H);
  const out = new Map(); out.set('lead', { x: cx, y: cy, ang: 0 });
  const kids = new Map();
  for (const a of r.agents.values()) { if (a.id === 'lead') continue; const p = a.parent ?? 'lead'; if (!kids.has(p)) kids.set(p, []); kids.get(p).push(a); }
  const place = (pid, base, R, spread) => {
    const ks = (kids.get(pid) ?? []).filter((k) => k.id !== 'synth');
    ks.forEach((k, i) => {
      const ang = pid === 'lead' ? (i / ks.length) * Math.PI * 2 - Math.PI / 2 : base + (ks.length === 1 ? 0 : (i / (ks.length - 1) - 0.5) * spread);
      const pt = out.get(pid);
      const rr = pid === 'lead' ? m * 0.27 : R;
      out.set(k.id, { x: pt.x + Math.cos(ang) * rr, y: pt.y + Math.sin(ang) * rr * (pid === 'lead' ? 0.92 : 1), ang });
      place(k.id, ang, m * 0.13, 1.1);
    });
  };
  place('lead', 0, 0, 0);
  if (r.agents.has('synth')) out.set('synth', { x: cx, y: cy + m * 0.1, ang: Math.PI / 2 });
  return out;
}

const tmp = new Map();
function frame(t) {
  requestAnimationFrame(frame);
  ctx.clearRect(0, 0, W, H);
  const r = cur();
  // radar rings + sweep
  const cx = W / 2; const cy = H / 2 + 6; const m = Math.min(W, H);
  ctx.strokeStyle = 'rgba(61,245,200,.06)'; ctx.lineWidth = 1;
  [0.13, 0.27, 0.4].forEach((k) => { ctx.beginPath(); ctx.arc(cx, cy, m * k, 0, 7); ctx.stroke(); });
  const sw = (t / 4000) % (Math.PI * 2);
  const g = ctx.createConicGradient ? ctx.createConicGradient(sw, cx, cy) : null;
  if (g) { g.addColorStop(0, 'rgba(61,245,200,.10)'); g.addColorStop(0.08, 'rgba(61,245,200,0)'); g.addColorStop(1, 'rgba(61,245,200,0)'); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, m * 0.42, 0, 7); ctx.fill(); }
  if (!r) return;

  const tg = targets(r);
  for (const a of r.agents.values()) {
    const tp = tg.get(a.id); if (!tp) continue;
    let n = nodes.get(`${r.id}/${a.id}`);
    if (!n) { const pp = nodes.get(`${r.id}/${a.parent ?? 'lead'}`); n = { x: pp?.x ?? cx, y: pp?.y ?? cy, vx: 0, vy: 0, seed: Math.random() * 9 }; nodes.set(`${r.id}/${a.id}`, n); }
    n.vx = (n.vx + (tp.x + Math.sin(t / 900 + n.seed) * 4 - n.x) * 0.05) * 0.82;
    n.vy = (n.vy + (tp.y + Math.cos(t / 1100 + n.seed) * 4 - n.y) * 0.05) * 0.82;
    n.x += n.vx; n.y += n.vy; n.a = a;
  }
  const N = (id) => nodes.get(`${r.id}/${id}`);

  // edges + particles
  for (const a of r.agents.values()) {
    if (a.id === 'lead') continue;
    const n = N(a.id); const p = N(a.parent ?? 'lead'); if (!n || !p) continue;
    const col = COLORS[harnessOf(a)] ?? COLORS.none;
    const mx = (n.x + p.x) / 2 + (n.y - p.y) * 0.12; const my = (n.y + p.y) / 2 - (n.x - p.x) * 0.12;
    ctx.strokeStyle = a.status === 'failed' ? 'rgba(255,77,94,.35)' : a.status === 'done' ? col + '55' : col + '33';
    ctx.lineWidth = a.status === 'running' ? 1.6 : 1;
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.quadraticCurveTo(mx, my, n.x, n.y); ctx.stroke();
    if (a.status === 'running') {
      for (let i = 0; i < 3; i++) {
        const u = ((t / 1100 + i / 3 + n.seed) % 1);
        const x = (1 - u) * (1 - u) * p.x + 2 * (1 - u) * u * mx + u * u * n.x;
        const y = (1 - u) * (1 - u) * p.y + 2 * (1 - u) * u * my + u * u * n.y;
        ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 10; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
      }
    }
  }

  // nodes
  hover = null;
  for (const a of r.agents.values()) {
    const n = N(a.id); if (!n) continue;
    const col = a.status === 'failed' ? COLORS.bad : COLORS[harnessOf(a)] ?? COLORS.none;
    const base = a.role === 'lead' ? 18 : a.role === 'verify' ? 9 : a.role === 'sub' || a.role === 'cand' ? 8 : 12;
    const age = t - (tmp.get(`${r.id}/${a.id}`) ?? (tmp.set(`${r.id}/${a.id}`, t), t));
    const pop = Math.min(1, age / 400); const rad = base * (0.4 + 0.6 * pop);
    if (a.status === 'running') {
      const k = (t / 900 + n.seed) % 1;
      ctx.strokeStyle = col; ctx.globalAlpha = 1 - k; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(n.x, n.y, rad + k * 22, 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
    }
    if (a.status === 'queued') { ctx.setLineDash([3, 4]); }
    ctx.shadowColor = col; ctx.shadowBlur = a.status === 'running' ? 22 : a.status === 'done' ? 10 : 4;
    ctx.fillStyle = a.status === 'done' ? col : '#0a0c14'; ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(n.x, n.y, rad, 0, 7); ctx.fill(); ctx.stroke(); ctx.setLineDash([]); ctx.shadowBlur = 0;
    if (a.status === 'done') { ctx.strokeStyle = '#05060a'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(n.x - rad * 0.4, n.y); ctx.lineTo(n.x - rad * 0.1, n.y + rad * 0.35); ctx.lineTo(n.x + rad * 0.45, n.y - rad * 0.3); ctx.stroke(); }
    if (a.status === 'failed') { ctx.strokeStyle = COLORS.bad; ctx.beginPath(); ctx.moveTo(n.x - 4, n.y - 4); ctx.lineTo(n.x + 4, n.y + 4); ctx.moveTo(n.x + 4, n.y - 4); ctx.lineTo(n.x - 4, n.y + 4); ctx.stroke(); }
    if (a.flash && t - a.flash < 600) { ctx.strokeStyle = COLORS.warn; ctx.globalAlpha = 1 - (t - a.flash) / 600; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(n.x, n.y, rad + 8 + (t - a.flash) / 40, 0, 7); ctx.stroke(); ctx.globalAlpha = 1; }
    ctx.fillStyle = S.focus === a.id ? '#fff' : '#9aa3c4'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText(a.id === 'lead' ? 'LEAD' : a.id, n.x, n.y + rad + 13);
    if (a.score != null && Number.isFinite(a.score)) { ctx.fillStyle = a.champion ? COLORS.warn : '#7e87a8'; ctx.font = '9px ui-monospace, monospace'; ctx.fillText(Number(a.score).toFixed(3), n.x, n.y + rad + 35); }
    if (a.champion) { ctx.strokeStyle = COLORS.warn; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(n.x, n.y, rad + 5, 0, 7); ctx.stroke(); }
    if (a.account) { ctx.fillStyle = col + 'aa'; ctx.font = '9px ui-monospace, monospace'; ctx.fillText(a.account, n.x, n.y + rad + 24); }
    if (Math.hypot(mouse.x - n.x, mouse.y - n.y) < rad + 6) hover = a;
  }
  if (hover) {
    const label = `${hover.title ?? hover.id}${hover.account ? '  ·  ' + hover.account : ''}  ·  ${hover.status}`;
    ctx.font = '11px ui-monospace, monospace'; const w = ctx.measureText(label).width + 16;
    const x = Math.min(Math.max(mouse.x - w / 2, 6), W - w - 6); const y = mouse.y - 34;
    ctx.fillStyle = 'rgba(5,6,10,.92)'; ctx.strokeStyle = '#2a3050'; ctx.beginPath(); ctx.roundRect(x, y, w, 22, 5); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.fillText(label, x + 8, y + 15);
  }
  cv.style.cursor = hover ? 'pointer' : 'crosshair';
}
const mouse = { x: -99, y: -99 };
cv.addEventListener('mousemove', (e) => { const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
cv.addEventListener('mouseleave', () => { mouse.x = mouse.y = -99; });
cv.addEventListener('click', () => { if (hover) focusAgent(hover.id); });
requestAnimationFrame(frame);

// ---------- clock + connection ----------
setInterval(() => { $('#clock').textContent = new Date().toLocaleTimeString([], { hour12: false }); tickCountdowns(); }, 1000);
$('#clock').textContent = new Date().toLocaleTimeString([], { hour12: false });

function connect() {
  const es = new EventSource(`/api/events?t=${encodeURIComponent(token)}`);
  const conn = $('#conn');
  es.onopen = () => { conn.textContent = 'live'; conn.classList.add('on'); };
  es.onmessage = (m) => apply(JSON.parse(m.data));
  es.onerror = () => { conn.textContent = 'offline'; conn.classList.remove('on'); };
}
// replayed history rebuilds state from scratch on reconnect
connect();
