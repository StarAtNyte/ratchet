const $ = (s) => document.querySelector(s);
const token = new URLSearchParams(location.search).get('t') ?? '';
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLORS = {};
const loadColors = () => Object.assign(COLORS, { claude: css('--harness-claude'), codex: css('--harness-codex'), antigravity: css('--harness-antigravity'), lead: css('--color-ink'), bad: css('--color-bad'), none: css('--harness-other'), ok: css('--color-ok'), warn: css('--color-accent'), accent: css('--color-accent'), paper: css('--color-paper-3'), panel: css('--color-paper-2'), rule: css('--color-rule'), muted: css('--color-muted'), ink2: css('--color-ink-2') });
loadColors();
const tint = (c, pct) => `color-mix(in oklab, ${c} ${pct}%, transparent)`;

const S = {
  settings: { dir: '', mode: 'auto', slots: 2 },
  commands: [],
  accounts: new Map(),
  logins: {},
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

// command catalogue drives completion and the suggestion list
const CATALOG = [
  ['run', '<task>', 'launch a research swarm (or just type the task)'],
  ['opt', 'init | start [goal] | apply', 'evolutionary optimise loop'],
  ['resume', '<run>', 're-run an interrupted research run'],
  ['stop', '[run|all]', 'stop a run'],
  ['runs', '', 'list runs'],
  ['report', '[run]', 'print a run report'],
  ['accounts', '', 'list the account pool'],
  ['account', 'add | login | probe | rm', 'manage accounts'],
  ['dir', '<path> | recent | mk <path>', 'change the workspace'],
  ['mode', 'yolo | auto', 'agent permission level'],
  ['set', 'slots | workers | timeout | rounds <n>', 'tune the swarm'],
  ['clear', '', 'clear the console'],
  ['help', '', 'all commands'],
];
const hist = JSON.parse(localStorage.getItem('ratchet.hist') || '[]'); let hi = hist.length;
const sug = { items: [], sel: -1 };
const termIn = $('#termIn');

function argChoices(cmd, sub) {
  const runs = [...S.runs.keys()].reverse();
  const accts = [...S.accounts.keys()];
  if (['stop', 'report'].includes(cmd)) return ['all', ...runs];
  if (cmd === 'resume') return [...S.runs.values()].filter((r) => r.status !== 'running' && r.status !== 'done').map((r) => r.id).reverse();
  if (cmd === 'account') return sub === undefined ? ['add', 'login', 'probe', 'rm'] : sub === 'add' ? (S.harnesses ?? []) : accts;
  if (cmd === 'mode') return ['yolo', 'auto'];
  if (cmd === 'set') return sub === undefined ? ['slots', 'workers', 'timeout', 'rounds'] : [];
  if (cmd === 'opt') return sub === undefined ? ['init', 'start', 'apply'] : [];
  if (cmd === 'dir') return sub === undefined ? ['recent', 'mk', ...(S.settings.recent ?? [])] : [];
  return [];
}
async function computeSuggestions() {
  const v = termIn.value;
  const parts = v.split(/\s+/);
  if (!v.trim()) return [];
  if (parts.length === 1) return CATALOG.filter(([c]) => c.startsWith(parts[0]) && c !== parts[0]).map(([c, a, d]) => ({ value: `${c} `, label: c, hint: a, desc: d }));
  const [cmd, ...rest] = parts;
  if (!CATALOG.some(([c]) => c === cmd)) return [];
  const last = rest[rest.length - 1] ?? '';
  const head = v.slice(0, v.length - last.length);
  if (cmd === 'dir' && rest.length >= 1 && (last.startsWith('/') || last.startsWith('~') || last.startsWith('.'))) {
    const base = last.endsWith('/') ? last : last.slice(0, last.lastIndexOf('/') + 1) || './';
    const prefix = last.slice(base.length);
    try {
      const r = await api(`/api/ls?path=${encodeURIComponent(base)}`);
      return (r.dirs ?? []).filter((d) => d.startsWith(prefix)).slice(0, 12).map((d) => ({ value: `${head}${base}${d}/`, label: `${base}${d}/`, hint: '', desc: 'folder' }));
    } catch { return []; }
  }
  const choices = argChoices(cmd, rest.length > 1 ? rest[0] : undefined);
  return choices.filter((c) => String(c).startsWith(last) && c !== last).slice(0, 12).map((c) => ({ value: `${head}${c} `, label: String(c), hint: '', desc: S.runs.get(c)?.task ?? S.accounts.get(c)?.harness ?? '' }));
}
function renderSuggest() {
  const box = $('#suggest');
  if (!sug.items.length) { box.hidden = true; return; }
  box.hidden = false; box.innerHTML = '';
  sug.items.forEach((it, i) => {
    const row = el('div', `sg${i === sug.sel ? ' on' : ''}`, `<b>${esc(it.label)}</b>${it.hint ? ` <span class="h">${esc(it.hint)}</span>` : ''}<span class="d">${esc(it.desc ?? '')}</span>`);
    row.setAttribute('role', 'option');
    row.onmousedown = (e) => { e.preventDefault(); termIn.value = it.value; refreshSuggest(); termIn.focus(); };
    box.append(row);
  });
}
let sugTimer = 0;
function refreshSuggest() { clearTimeout(sugTimer); sugTimer = setTimeout(async () => { sug.items = await computeSuggestions(); sug.sel = -1; renderSuggest(); }, 60); }
termIn.addEventListener('input', refreshSuggest);
termIn.addEventListener('blur', () => setTimeout(() => { $('#suggest').hidden = true; }, 120));

async function submitLine(line) {
  if (!line) return;
  if (hist[hist.length - 1] !== line) hist.push(line);
  while (hist.length > 300) hist.shift();
  localStorage.setItem('ratchet.hist', JSON.stringify(hist)); hi = hist.length;
  if (line === 'clear') { termOut.innerHTML = ''; return; }
  await fetch('/api/cmd', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ line }) });
}
$('#termForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const line = termIn.value.trim();
  termIn.value = ''; sug.items = []; renderSuggest();
  await submitLine(line);
});
termIn.addEventListener('keydown', (e) => {
  const open = sug.items.length > 0 && !$('#suggest').hidden;
  if (e.key === 'Tab') {
    e.preventDefault();
    if (!sug.items.length) return;
    const it = sug.items[sug.sel >= 0 ? sug.sel : 0];
    termIn.value = it.value; refreshSuggest();
  } else if (open && (e.key === 'ArrowDown' || e.key === 'ArrowUp') && sug.sel >= 0) {
    e.preventDefault();
    sug.sel = (sug.sel + (e.key === 'ArrowDown' ? 1 : -1) + sug.items.length) % sug.items.length; renderSuggest();
  } else if (open && e.key === 'ArrowDown') { e.preventDefault(); sug.sel = 0; renderSuggest(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); if (hi > 0) termIn.value = hist[--hi]; sug.items = []; renderSuggest(); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); termIn.value = hi < hist.length - 1 ? hist[++hi] : (hi = hist.length, ''); }
  else if (e.key === 'Escape') { sug.items = []; renderSuggest(); }
  else if (e.key === 'l' && e.ctrlKey) { e.preventDefault(); termOut.innerHTML = ''; }
  else if (e.key === 'u' && e.ctrlKey) { e.preventDefault(); termIn.value = ''; refreshSuggest(); }
  else if (e.key === 'Enter' && open && sug.sel >= 0) { e.preventDefault(); termIn.value = sug.items[sug.sel].value; refreshSuggest(); }
});
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target !== document.body) return;
  if (TT.tab === 'shell' && xt.term) xt.term.focus(); else termIn.focus();
});

// quick chips
[['help', 'help'], ['accounts', 'accounts'], ['runs', 'runs'], ['stop all', 'stop all'], ['mode', 'mode'], ['dir recent', 'dir recent']].forEach(([label, line]) => {
  const b = el('button', 'chipBtn'); b.type = 'button'; b.textContent = label; b.onclick = () => submitLine(line); $('#quick').append(b);
});

// ---------- terminal tabs, shell, split, expand ----------
const TT = { tab: 'console' };
const xt = { term: null, fit: null, id: sessionStorage.getItem('ratchet.shell') || null, es: null, out: '', timer: 0, dead: false };
const toRgb = (c) => { const cv2 = document.createElement('canvas'); cv2.width = cv2.height = 1; const x = cv2.getContext('2d'); x.fillStyle = c; x.fillRect(0, 0, 1, 1); const [r, g, b] = x.getImageData(0, 0, 1, 1).data; return `#${[r, g, b].map((n) => n.toString(16).padStart(2, '0')).join('')}`; };
const shellPost = (op, body) => fetch(`/api/shell/${op}`, { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify(body) });
async function shellConnect(fresh) {
  if (!xt.term) {
    if (!window.Terminal) { $('#xterm').textContent = 'The shell needs /vendor/xterm.js. Reinstall Ratchet.'; return; }
    xt.term = new window.Terminal({
      fontFamily: css('--font-outlier') || 'monospace', fontSize: 13, lineHeight: 1.2, cursorBlink: true, allowProposedApi: false, scrollback: 5000,
      theme: { background: toRgb(css('--color-paper-3')), foreground: toRgb(css('--color-ink')), cursor: toRgb(css('--color-accent')), selectionBackground: toRgb(css('--color-accent-soft')), black: toRgb(css('--color-paper-2')), brightBlack: toRgb(css('--color-muted')), red: toRgb(css('--color-bad')), green: toRgb(css('--color-ok')), yellow: toRgb(css('--color-accent')), blue: toRgb(css('--color-cool')), cyan: toRgb(css('--harness-codex')), magenta: toRgb(css('--harness-antigravity')), white: toRgb(css('--color-ink-2')), brightWhite: toRgb(css('--color-ink')) },
    });
    xt.fit = new window.FitAddon.FitAddon();
    xt.term.loadAddon(xt.fit);
    xt.term.open($('#xterm'));
    xt.term.onData((d) => {
      if (xt.dead) { if (d === '\r') shellConnect(true); return; }
      xt.out += d; clearTimeout(xt.timer); xt.timer = setTimeout(() => { const data = xt.out; xt.out = ''; shellPost('input', { id: xt.id, data }); }, 6);
    });
    let rt = 0;
    new ResizeObserver(() => { if (TT.tab !== 'shell') return; clearTimeout(rt); rt = setTimeout(() => { try { xt.fit.fit(); } catch { /* hidden */ } if (xt.id && !xt.dead) shellPost('resize', { id: xt.id, cols: xt.term.cols, rows: xt.term.rows }); }, 80); }).observe($('#shellView'));
  }
  try { xt.fit.fit(); } catch { /* hidden */ }
  if (fresh) { xt.id = null; xt.term.reset(); }
  const r = await shellPost('start', { id: xt.id, cols: xt.term.cols, rows: xt.term.rows });
  if (!r.ok) { xt.term.write('\r\ncould not start a shell\r\n'); return; }
  const j = await r.json();
  if (j.id !== xt.id) xt.term.reset();
  xt.id = j.id; xt.dead = false; sessionStorage.setItem('ratchet.shell', j.id);
  xt.es?.close();
  xt.es = new EventSource(`/api/shell/stream?id=${encodeURIComponent(j.id)}&t=${encodeURIComponent(token)}`);
  xt.es.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.d) xt.term.write(Uint8Array.from(atob(msg.d), (ch) => ch.charCodeAt(0)));
    if (msg.exit) { xt.dead = true; xt.es.close(); xt.term.write(`\r\n\x1b[2m${msg.error ?? 'shell exited'}. Press Enter for a new shell.\x1b[0m\r\n`); }
  };
  xt.term.focus();
}
function setTermTab(tab) {
  TT.tab = tab;
  document.querySelectorAll('.ttabs button').forEach((b) => b.classList.toggle('on', b.dataset.tt === tab));
  $('#consoleView').classList.toggle('on', tab === 'console');
  $('#shellView').classList.toggle('on', tab === 'shell');
  $('#tNewShell').hidden = tab !== 'shell';
  if (tab === 'shell') shellConnect(false); else termIn.focus();
}
document.querySelectorAll('.ttabs button').forEach((b) => { b.onclick = () => setTermTab(b.dataset.tt); });
$('#tNewShell').onclick = () => { if (xt.id) shellPost('kill', { id: xt.id }); shellConnect(true); };
$('#tClear').onclick = () => { if (TT.tab === 'shell') xt.term?.clear(); else termOut.innerHTML = ''; };
$('#tMax').onclick = () => { const m = $('main'); m.classList.toggle('termMax'); $('#tMax').textContent = m.classList.contains('termMax') ? 'restore' : 'expand'; setTimeout(() => { try { xt.fit?.fit(); } catch { /* hidden */ } }, 50); };
(() => {
  const split = $('#split'); const mainEl = $('main');
  const saved = Number(localStorage.getItem('ratchet.split')); if (saved > 0.15 && saved < 0.85) mainEl.style.setProperty('--gfrac', saved);
  const setFrac = (f) => { f = Math.min(0.85, Math.max(0.15, f)); mainEl.style.setProperty('--gfrac', f); localStorage.setItem('ratchet.split', f); };
  split.addEventListener('pointerdown', (e) => {
    split.setPointerCapture(e.pointerId);
    const move = (ev) => { const r = mainEl.getBoundingClientRect(); setFrac((ev.clientY - r.top) / r.height); };
    const up = () => { split.removeEventListener('pointermove', move); split.removeEventListener('pointerup', up); try { xt.fit?.fit(); } catch { /* hidden */ } };
    split.addEventListener('pointermove', move); split.addEventListener('pointerup', up);
  });
  split.addEventListener('keydown', (e) => { const cur = Number(getComputedStyle(mainEl).getPropertyValue('--gfrac')) || 0.6; if (e.key === 'ArrowUp') setFrac(cur - 0.05); if (e.key === 'ArrowDown') setFrac(cur + 0.05); });
})();

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
    case 'hello': S.settings = ev.settings; S.commands = ev.commands; S.harnesses = ev.harnesses ?? []; renderHeader(); break;
    case 'settings': S.settings = ev.settings; renderHeader(); break;
    case 'term': say(ev.text, ev.cls); break;
    case 'account': S.accounts.set(ev.account.id, ev.account); renderAccounts(); break;
    case 'login': {
      const L = (S.logins[ev.id] = ev.reset ? { text: '', url: null, done: false } : (S.logins[ev.id] ?? { text: '', url: null, done: false }));
      if (!ev.reset || ev.text) L.text += `${ev.text}\n`;
      if (ev.url) L.url = ev.url;
      if (ev.done) { L.done = true; L.ok = ev.ok; }
      break;
    }
    case 'account.remove': S.accounts.delete(ev.id); renderAccounts(); break;
    case 'run.start': { const r = run(ev.runId); r.task = ev.task; r.startedAt = ev.t; S.current = ev.runId; S.focus = null; resetPanes(); renderRunbar(); break; }
    case 'agent.start': { const r = run(ev.runId); r.agents.set(ev.id, { id: ev.id, role: ev.role, title: ev.title, parent: ev.parent, harness: null, account: null, status: 'queued', born: ev.t, prompt: ev.prompt, failovers: 0, attempts: [] }); renderRunbar(); break; }
    case 'agent.assign': { const a = run(ev.runId).agents.get(ev.agentId); if (a) { Object.assign(a, { harness: ev.harness, account: ev.account, status: 'running', pulse: ev.t, startedAt: a.startedAt ?? ev.t }); (a.attempts ??= []).push(ev.account); } break; }
    case 'agent.failover': { const a = run(ev.runId).agents.get(ev.agentId); if (a) { a.status = 'queued'; a.flash = ev.t; a.failovers = (a.failovers ?? 0) + 1; } break; }
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
    case 'agent.output': { const a = run(ev.runId).agents.get(ev.agentId); if (a) a.output = ev.text; break; }
    case 'claims': {
      const r = run(ev.runId);
      ev.claims.forEach((c) => r.claims.set(c.id, c));
      if (ev.runId === S.current) renderClaims();
      break;
    }
    case 'run.report': { const r = run(ev.runId); r.report = ev.markdown; if (ev.runId === S.current) renderReport(); break; }
    case 'run.end': { const r = run(ev.runId); r.status = ev.status; r.endedAt = ev.t; if (ev.note) say(`run ${ev.runId} ${ev.status}: ${ev.note}`, 'err'); else say(`run ${ev.runId} ${ev.status}`, ev.status === 'done' ? 'ok' : 'warn'); renderRunbar(); break; }
  }
  renderStats(); scheduleLists(); if (ev.runId === S.current || ev.kind === 'account') renderOverview();
}

function resetPanes() { $('#feed').innerHTML = ''; renderClaims(); renderReport(); $('#focus').hidden = true; }

// ---------- renderers ----------
function renderHeader() {
  $('#ws').textContent = S.settings.dir;
  $('#tcwd').textContent = S.settings.dir;
  $('#mockChip').hidden = !S.settings.mock;
  const m = $('#modeChip'); m.textContent = S.settings.mode; m.classList.toggle('yolo', S.settings.mode === 'yolo');
}

function renderStats() {
  const r = cur();
  const run_ = r ? [...r.agents.values()].filter((a) => a.status === 'running').length : 0;
  const cl = r ? [...r.claims.values()] : [];
  const n = (v) => cl.filter((c) => c.verdict === v).length;
  $('#stats').innerHTML = [['agents', r ? r.agents.size : 0], ['active', run_], ['claims', cl.length], ['confirmed', n('confirmed')], ['refuted', n('refuted')]]
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
    if (r.status === 'running' && r.id === S.current) { const x = el('button', 'stopRun'); x.type = 'button'; x.textContent = 'stop run'; x.onclick = () => sendCmd(`stop ${r.id}`); bar.append(x); }
    if (r.status === 'interrupted' && r.id === S.current) { const x = el('button', 'stopRun'); x.type = 'button'; x.textContent = 'resume'; x.onclick = () => sendCmd(`resume ${r.id}`); bar.append(x); }
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
    d.onclick = () => openAccount(a.id);
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
    who.onclick = () => openAgent(f.id);
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

// ---------- swarm graph: world-space layout, zoom, pan, fit ----------
const cv = $('#graph'); const ctx = cv.getContext('2d');
let W = 0; let H = 0; const nodes = new Map(); let hover = null;
const view = { k: 1, x: 0, y: 0, auto: true };
const G = { layout: localStorage.getItem('ratchet.layout') || 'radial' };
function resize() { const r = cv.getBoundingClientRect(); const d = devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; W = r.width; H = r.height; }
new ResizeObserver(resize).observe(cv); resize();

function kidsOf(r) {
  const kids = new Map();
  for (const a of r.agents.values()) { if (a.id === 'lead') continue; const p = r.agents.has(a.parent) ? a.parent : 'lead'; if (!kids.has(p)) kids.set(p, []); kids.get(p).push(a); }
  return kids;
}
function targets(r) {
  const out = new Map(); out.set('lead', { x: 0, y: 0 });
  const kids = kidsOf(r);
  if (G.layout === 'tree') {
    // layered left to right: depth is the column, leaves get consecutive rows
    let row = 0;
    const walk = (id, depth) => {
      const ks = (kids.get(id) ?? []);
      if (!ks.length) { out.set(id, { x: depth * 230, y: row++ * 58 }); return; }
      ks.forEach((k) => walk(k.id, depth + 1));
      const ys = ks.map((k) => out.get(k.id).y);
      out.set(id, { x: depth * 230, y: (Math.min(...ys) + Math.max(...ys)) / 2 });
    };
    walk('lead', 0);
    return out;
  }
  const place = (pid, base, R, spread) => {
    const ks = (kids.get(pid) ?? []).filter((k) => k.id !== 'synth');
    ks.forEach((k, i) => {
      const ang = pid === 'lead' ? (i / ks.length) * Math.PI * 2 - Math.PI / 2 : base + (ks.length === 1 ? 0 : (i / (ks.length - 1) - 0.5) * spread);
      const pt = out.get(pid);
      const rr = pid === 'lead' ? Math.max(220, ks.length * 34) : R;
      out.set(k.id, { x: pt.x + Math.cos(ang) * rr, y: pt.y + Math.sin(ang) * rr });
      place(k.id, ang, 110, Math.min(1.4, 0.35 * ((kids.get(k.id) ?? []).length + 1)));
    });
  };
  place('lead', 0, 0, 0);
  if (r.agents.has('synth')) out.set('synth', { x: 0, y: 92 });
  return out;
}
function bbox(tg) {
  let x0 = Infinity; let y0 = Infinity; let x1 = -Infinity; let y1 = -Infinity;
  for (const p of tg.values()) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const padR = G.layout === 'tree' ? 200 : 70;
  return { x0: x0 - 70, y0: y0 - 60, x1: x1 + padR, y1: y1 + 70 };
}
function fitTo(b, snap) {
  const top = 70; const bottom = 40; // room for the run strip and legend
  const k = Math.min(1.6, Math.max(0.15, Math.min(W / (b.x1 - b.x0), (H - top - bottom) / (b.y1 - b.y0))));
  const x = W / 2 - ((b.x0 + b.x1) / 2) * k; const y = top + (H - top - bottom) / 2 - ((b.y0 + b.y1) / 2) * k;
  if (snap) Object.assign(view, { k, x, y }); else { view.k += (k - view.k) * 0.12; view.x += (x - view.x) * 0.12; view.y += (y - view.y) * 0.12; }
}
const toWorld = (sx, sy) => ({ x: (sx - view.x) / view.k, y: (sy - view.y) / view.k });

const tmp = new Map();
let lastTg = null;
function frame(t) {
  requestAnimationFrame(frame);
  const d = devicePixelRatio || 1;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const r = cur();
  const tg = r ? targets(r) : new Map([['lead', { x: 0, y: 0 }]]);
  lastTg = tg;
  if (view.auto) fitTo(bbox(tg), false);
  ctx.setTransform(d * view.k, 0, 0, d * view.k, d * view.x, d * view.y);

  // instrument-dial backdrop around the lead, in world space
  const lead = tg.get('lead');
  ctx.lineWidth = 1 / view.k; ctx.strokeStyle = tint(COLORS.muted, 12);
  if (G.layout === 'radial') {
    [110, 220, 330].forEach((rr) => { ctx.beginPath(); ctx.arc(lead.x, lead.y, rr, 0, 7); ctx.stroke(); });
    for (let i = 0; i < 72; i++) { const an = (i / 72) * Math.PI * 2; const r0 = 330; const r1 = r0 + (i % 6 === 0 ? 10 : 4); ctx.beginPath(); ctx.moveTo(lead.x + Math.cos(an) * r0, lead.y + Math.sin(an) * r0); ctx.lineTo(lead.x + Math.cos(an) * r1, lead.y + Math.sin(an) * r1); ctx.stroke(); }
  }
  ctx.strokeStyle = tint(COLORS.accent, 22); ctx.beginPath();
  const teeth = 28; const rot = t / 60000;
  for (let i = 0; i < teeth; i++) {
    const a0 = rot + (i / teeth) * Math.PI * 2; const a1 = a0 + (0.8 / teeth) * Math.PI * 2; const a2 = a0 + (1 / teeth) * Math.PI * 2;
    const ri = 34; const ro = 44;
    if (i === 0) ctx.moveTo(lead.x + Math.cos(a0) * ri, lead.y + Math.sin(a0) * ri);
    ctx.lineTo(lead.x + Math.cos(a1) * ro, lead.y + Math.sin(a1) * ro); ctx.lineTo(lead.x + Math.cos(a1) * ri, lead.y + Math.sin(a1) * ri); ctx.lineTo(lead.x + Math.cos(a2) * ri, lead.y + Math.sin(a2) * ri);
  }
  ctx.closePath(); ctx.stroke();
  if (!r) return;

  for (const a of r.agents.values()) {
    const tp = tg.get(a.id); if (!tp) continue;
    let n = nodes.get(`${r.id}/${a.id}`);
    if (!n) { const pp = nodes.get(`${r.id}/${a.parent ?? 'lead'}`); n = { x: pp?.x ?? tp.x, y: pp?.y ?? tp.y, vx: 0, vy: 0, seed: Math.random() * 9 }; nodes.set(`${r.id}/${a.id}`, n); }
    n.vx = (n.vx + (tp.x - n.x) * 0.08) * 0.78;
    n.vy = (n.vy + (tp.y - n.y) * 0.08) * 0.78;
    n.x += n.vx; n.y += n.vy; n.a = a;
  }
  const N = (id) => nodes.get(`${r.id}/${id}`);

  // edges
  for (const a of r.agents.values()) {
    if (a.id === 'lead') continue;
    const n = N(a.id); const p = N(r.agents.has(a.parent) ? a.parent : 'lead'); if (!n || !p) continue;
    const col = COLORS[harnessOf(a)] ?? COLORS.none;
    ctx.strokeStyle = a.status === 'failed' ? tint(COLORS.bad, 40) : a.status === 'done' ? tint(col, 40) : tint(col, 22);
    ctx.lineWidth = (a.status === 'running' ? 1.6 : 1) / Math.sqrt(view.k);
    ctx.beginPath(); ctx.moveTo(p.x, p.y);
    if (G.layout === 'tree') { const mx = (p.x + n.x) / 2; ctx.bezierCurveTo(mx, p.y, mx, n.y, n.x, n.y); }
    else { const mx = (n.x + p.x) / 2 + (n.y - p.y) * 0.1; const my = (n.y + p.y) / 2 - (n.x - p.x) * 0.1; ctx.quadraticCurveTo(mx, my, n.x, n.y); }
    ctx.stroke();
    if (a.status === 'running') { ctx.setLineDash([4, 6]); ctx.lineDashOffset = -t / 40; ctx.strokeStyle = col; ctx.stroke(); ctx.setLineDash([]); }
  }

  // nodes
  hover = null;
  const mw = toWorld(mouse.x, mouse.y);
  for (const a of r.agents.values()) {
    const n = N(a.id); if (!n) continue;
    const col = a.status === 'failed' ? COLORS.bad : COLORS[harnessOf(a)] ?? COLORS.none;
    const base = a.role === 'lead' ? 16 : a.role === 'verify' ? 8 : a.role === 'sub' || a.role === 'cand' ? 8 : 11;
    const age = t - (tmp.get(`${r.id}/${a.id}`) ?? (tmp.set(`${r.id}/${a.id}`, t), t));
    const zs = 1 / Math.sqrt(Math.max(view.k, 0.2)); // keep nodes legible when zoomed out
    const rad = base * zs * (0.4 + 0.6 * Math.min(1, age / 400));
    if (a.status === 'running') {
      const k = (t / 900 + n.seed) % 1;
      ctx.strokeStyle = col; ctx.globalAlpha = (1 - k) * 0.5; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(n.x, n.y, rad + k * 12, 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
    }
    if (a.status === 'queued') ctx.setLineDash([3, 4]);
    ctx.fillStyle = a.status === 'done' ? col : COLORS.paper; ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(n.x, n.y, rad, 0, 7); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    if (a.status === 'done') { ctx.strokeStyle = COLORS.paper; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(n.x - rad * 0.4, n.y); ctx.lineTo(n.x - rad * 0.1, n.y + rad * 0.35); ctx.lineTo(n.x + rad * 0.45, n.y - rad * 0.3); ctx.stroke(); }
    if (a.status === 'failed') { ctx.strokeStyle = COLORS.bad; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(n.x - 4, n.y - 4); ctx.lineTo(n.x + 4, n.y + 4); ctx.moveTo(n.x + 4, n.y - 4); ctx.lineTo(n.x - 4, n.y + 4); ctx.stroke(); }
    const fl = Date.now() - (a.flash ?? 0); if (a.flash && fl < 600) { ctx.strokeStyle = COLORS.warn; ctx.globalAlpha = 1 - fl / 600; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(n.x, n.y, rad + 8 + fl / 40, 0, 7); ctx.stroke(); ctx.globalAlpha = 1; }
    if (a.champion) { ctx.strokeStyle = COLORS.accent; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(n.x, n.y, rad + 5, 0, 7); ctx.stroke(); }
    // labels: below in radial, to the right in tree; hidden when zoomed far out
    if (view.k > 0.3 || a.role === 'lead') {
      const name = a.id === 'lead' ? (r.task.startsWith('optimize') ? 'baseline' : 'lead') : a.id;
      const score = a.score != null && Number.isFinite(a.score) ? Number(a.score).toFixed(3) : '';
      const u = 1 / view.k; // one screen pixel in world units: text stays a fixed screen size
      ctx.font = `${12 * u}px system-ui, sans-serif`;
      if (G.layout === 'tree') {
        ctx.textAlign = 'left'; ctx.fillStyle = S.focus === a.id ? COLORS.lead : COLORS.ink2;
        ctx.fillText(a.id === 'lead' ? name : `${name}  ${(a.title ?? '').slice(0, 26)}`, n.x + rad + 8 * u, n.y - 2 * u);
        ctx.font = `${10 * u}px system-ui, sans-serif`; ctx.fillStyle = tint(col, 75);
        ctx.fillText([a.account ?? (a.id === 'lead' ? '' : 'waiting'), score].filter(Boolean).join('  '), n.x + rad + 8 * u, n.y + 11 * u);
      } else {
        ctx.textAlign = 'center'; ctx.fillStyle = S.focus === a.id ? COLORS.lead : COLORS.ink2;
        ctx.fillText(name, n.x, n.y + rad + 14 * u);
        ctx.font = `${10 * u}px system-ui, sans-serif`; ctx.fillStyle = tint(col, 75);
        if (a.account) ctx.fillText(a.account, n.x, n.y + rad + 26 * u);
        if (score) { ctx.fillStyle = a.champion ? COLORS.accent : COLORS.muted; ctx.fillText(score, n.x, n.y + rad + 38 * u); }
      }
    }
    if (Math.hypot(mw.x - n.x, mw.y - n.y) < rad + 6 / view.k) hover = a;
  }

  // screen-space overlays: score curve and tooltip
  ctx.setTransform(d, 0, 0, d, 0, 0);
  const sc = [...r.agents.values()].filter((a) => a.score != null && Number.isFinite(a.score) && a.ok !== false).sort((x, y) => (parseInt(x.id.slice(1)) || 0) - (parseInt(y.id.slice(1)) || 0));
  if (sc.length > 1) {
    const vals = sc.map((a) => a.score); const asc = vals[vals.length - 1] >= vals[0];
    let best = vals[0]; const curve = vals.map((v) => (best = asc ? Math.max(best, v) : Math.min(best, v)));
    const lo = Math.min(...vals); const hi = Math.max(...vals); const bx = W - 200; const by = H - 112; const bw = 180; const bh = 48;
    ctx.fillStyle = COLORS.panel; ctx.strokeStyle = COLORS.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.rect(bx - 10, by - 24, bw + 20, bh + 36); ctx.fill(); ctx.stroke();
    ctx.fillStyle = COLORS.muted; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.fillText(`best ${curve[curve.length - 1].toFixed(3)} after ${sc.length} tries`, bx, by - 9);
    const X = (i) => bx + (i / (curve.length - 1)) * bw; const Y = (v) => by + bh - ((v - lo) / (hi - lo || 1)) * bh;
    ctx.strokeStyle = tint(COLORS.muted, 40); ctx.beginPath(); vals.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
    ctx.strokeStyle = COLORS.accent; ctx.lineWidth = 1.8; ctx.beginPath(); curve.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
  }
  if (hover && !drag.on) {
    const label = `${hover.id}  ${hover.title ?? ''}${hover.account ? `  on ${hover.account}` : ''}  (${hover.status})`;
    ctx.font = '12px system-ui, sans-serif'; const w = ctx.measureText(label).width + 16;
    const x = Math.min(Math.max(mouse.x - w / 2, 6), W - w - 6); const y = Math.max(6, mouse.y - 36);
    ctx.fillStyle = COLORS.panel; ctx.strokeStyle = COLORS.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.rect(x, y, w, 22); ctx.fill(); ctx.stroke();
    ctx.fillStyle = COLORS.lead; ctx.textAlign = 'left'; ctx.fillText(label, x + 8, y + 15);
  }
  cv.style.cursor = drag.on ? 'grabbing' : hover ? 'pointer' : 'grab';
}
const mouse = { x: -99, y: -99 };
const drag = { on: false, moved: false, sx: 0, sy: 0, vx: 0, vy: 0 };
const zoomAt = (sx, sy, f) => { const w = toWorld(sx, sy); view.k = Math.min(3, Math.max(0.1, view.k * f)); view.x = sx - w.x * view.k; view.y = sy - w.y * view.k; view.auto = false; syncTools(); };
cv.addEventListener('mousemove', (e) => {
  const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top;
  if (drag.on) { const dx = e.clientX - drag.sx; const dy = e.clientY - drag.sy; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; view.x = drag.vx + dx; view.y = drag.vy + dy; view.auto = false; syncTools(); }
});
cv.addEventListener('mousedown', (e) => { if (e.button !== 0) return; Object.assign(drag, { on: true, moved: false, sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y }); });
window.addEventListener('mouseup', () => { drag.on = false; });
cv.addEventListener('mouseleave', () => { mouse.x = mouse.y = -99; });
cv.addEventListener('click', () => { if (hover && !drag.moved) openAgent(hover.id); });
cv.addEventListener('dblclick', () => { view.auto = true; syncTools(); });
cv.addEventListener('wheel', (e) => { e.preventDefault(); zoomAt(mouse.x, mouse.y, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
function syncTools() {
  $('#gFit').classList.toggle('on', view.auto);
  $('#gLayout').textContent = G.layout === 'tree' ? 'radial' : 'tree';
}
$('#gFit').onclick = () => { view.auto = true; syncTools(); };
$('#gIn').onclick = () => zoomAt(W / 2, H / 2, 1.25);
$('#gOut').onclick = () => zoomAt(W / 2, H / 2, 0.8);
$('#gLayout').onclick = () => { G.layout = G.layout === 'tree' ? 'radial' : 'tree'; localStorage.setItem('ratchet.layout', G.layout); view.auto = true; syncTools(); };
cv.setAttribute('tabindex', '0');
cv.addEventListener('keydown', (e) => { if (e.key === '+' || e.key === '=') zoomAt(W / 2, H / 2, 1.2); if (e.key === '-') zoomAt(W / 2, H / 2, 0.83); if (e.key === '0' || e.key === 'f') { view.auto = true; syncTools(); } });
syncTools();
requestAnimationFrame(frame);

// ---------- run overview strip ----------
const fmtAgo = (ms) => { const s = Math.floor(ms / 1000); return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`; };
function renderOverview() {
  const box = $('#overview'); const r = cur();
  if (!r) { box.hidden = true; return; }
  box.hidden = false;
  const ag = [...r.agents.values()];
  const c = (st) => ag.filter((a) => a.status === st).length;
  const cl = [...r.claims.values()];
  const v = (x) => cl.filter((q) => q.verdict === x).length;
  const elapsed = (r.endedAt ?? Date.now()) - (r.startedAt ?? Date.now());
  const busy = [...S.accounts.values()].filter((a) => a.load > 0).map((a) => a.id);
  box.innerHTML = `<div class="ovTask"><span class="pill ${r.status === 'running' ? 'running' : r.status === 'done' ? 'done' : r.status === 'failed' || r.status === 'interrupted' ? 'failed' : ''}">${esc(r.status)}</span><b title="${esc(r.task)}">${esc(r.task)}</b></div>
    <dl class="ovStats"><div><dt>elapsed</dt><dd>${fmtAgo(elapsed)}</dd></div><div><dt>running</dt><dd>${c('running')}</dd></div><div><dt>waiting</dt><dd>${c('queued')}</dd></div><div><dt>done</dt><dd>${c('done')}</dd></div><div><dt>failed</dt><dd>${c('failed')}</dd></div>${cl.length ? `<div><dt>claims</dt><dd><span class="okc">${v('confirmed')}</span> / <span class="unc">${v('unclear') + v('pending')}</span> / <span class="badc">${v('refuted')}</span></dd></div>` : ''}<div><dt>busy accounts</dt><dd>${busy.length ? esc(busy.join(', ')) : 'none'}</dd></div></dl>`;
}
setInterval(renderOverview, 1000);


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

// ---------- workspace picker ----------
const pk = { path: '', home: '' };
const api = (u) => fetch(u + (u.includes('?') ? '&' : '?') + 't=' + encodeURIComponent(token)).then((r) => r.json());
const sendCmd = (line) => fetch('/api/cmd', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ line }) });

async function browse(path) {
  const r = await api(`/api/ls?path=${encodeURIComponent(path ?? '')}${$('#pickHidden').checked ? '&hidden=1' : ''}`);
  const box = $('#pickDirs'); box.innerHTML = '';
  if (r.error) { box.append(el('div', 'err', esc(r.error))); return; }
  pk.path = r.path; pk.home = r.home;
  $('#pickPath').value = r.path;
  const up = el('div', 'up', '..'); up.onclick = () => browse(r.parent); box.append(up);
  r.dirs.forEach((d) => { const row = el('div', '', esc(d)); row.onclick = () => browse(`${r.path.replace(/\/$/, '')}/${d}`); box.append(row); });
  if (!r.dirs.length) box.append(el('div', 'err', 'no subfolders'));
}
function openPicker() {
  $('#picker').hidden = false;
  const rec = $('#pickRecent'); rec.innerHTML = '';
  (S.settings.recent ?? []).forEach((d) => { const b = el('button', ''); b.type = 'button'; b.textContent = d; b.title = d; b.onclick = () => browse(d); rec.append(b); });
  browse(S.settings.dir);
  $('#pickPath').focus();
}
const closePicker = () => { $('#picker').hidden = true; $('#termIn').focus(); };
$('#ws').onclick = openPicker;
$('#pickCancel').onclick = closePicker;
$('#picker').addEventListener('mousedown', (e) => { if (e.target.id === 'picker') closePicker(); });
$('#pickForm').addEventListener('submit', (e) => { e.preventDefault(); browse($('#pickPath').value); });
$('#pickHidden').onchange = () => browse(pk.path);
$('#pickUse').onclick = async () => { await sendCmd(`dir ${pk.path}`); closePicker(); };
$('#pickNew').onclick = async () => {
  const name = prompt('New folder name (created inside the current folder):');
  if (!name) return;
  await sendCmd(`dir mk ${pk.path.replace(/\/$/, '')}/${name}`);
  closePicker();
};
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('#picker').hidden) closePicker(); });

// ---------- agents tab, drawers, add account ----------
const fmtDur = (ms) => (ms < 1000 ? '0s' : ms < 60000 ? `${Math.round(ms / 1000)}s` : `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`);
let listsQueued = false;
function scheduleLists() { if (listsQueued) return; listsQueued = true; requestAnimationFrame(() => { listsQueued = false; renderAgents(); refreshDrawer(); renderWizard(); }); }

function renderAgents() {
  const box = $('#agents'); const r = cur();
  $('#agentN').textContent = r ? r.agents.size : 0;
  if (S.tab !== 'agents') return;
  box.innerHTML = '';
  const add = el('button', 'dact primary addAgent', '+ Add agent'); add.type = 'button'; add.onclick = openAgentModal; box.append(add);
  if (!r || !r.agents.size) { box.append(el('div', 'muted', 'no agents yet')); return; }
  [...r.agents.values()].forEach((a) => {
    const col = a.status === 'failed' ? COLORS.bad : COLORS[harnessOf(a)] ?? COLORS.none;
    const row = el('div', 'arow', `<span class="dot" style="background:${col}"></span><span class="m">${esc(a.id)}</span><span class="t">${esc(a.title ?? a.role)}<span class="m"> · ${esc(a.account ?? 'waiting')}</span></span><span class="pill ${a.status}">${a.score != null && Number.isFinite(a.score) ? Number(a.score).toFixed(3) : a.status}</span>`);
    row.onclick = () => openAgent(a.id);
    box.append(row);
  });
}
document.querySelectorAll('nav button').forEach((b) => b.addEventListener('click', renderAgents));

let drawer = null;
function openAccount(id) { drawer = { type: 'account', id }; $('#drawer').hidden = false; refreshDrawer(); }
function openAgent(id) { drawer = { type: 'agent', id }; $('#drawer').hidden = false; refreshDrawer(); }
$('#dclose').onclick = () => { drawer = null; $('#drawer').hidden = true; };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && drawer && $('#picker').hidden) $('#dclose').click(); });

const api2 = (u, b) => fetch(u, { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify(b) });
const beginLogin = (id) => { S.logins[id] = { text: '', url: null, done: false }; api2('/api/login', { id }); };
function signInHtml(a) {
  const L = S.logins[a.id];
  const canAuto = a.harness === 'claude' || a.harness === 'codex' || (a.harness === 'antigravity' && !!a.configDir);
  const status = a.authed === true ? '<span class="pill done">signed in</span>' : a.authed === false ? '<span class="pill failed">not signed in</span>' : '<span class="pill">unknown</span>';
  let h = `<div class="dsec">Sign in ${status}</div>`;
  if (a.harness === 'antigravity' && !a.configDir) return h + `<div class="muted2">This account has no private folder. It uses whichever Google account is signed in to the Antigravity app on this machine. Add another Antigravity account for a separate Google login.</div><div class="dacts"><button class="dact primary" data-act="probe">check sign-in</button></div>${a.authed == null ? '<div class="muted2" style="margin-top:8px">Checking with a one-line test prompt…</div>' : ''}`;
  if (!canAuto) return h + `<div class="muted2">Automatic sign-in is not available for ${esc(a.harness)}. Run this in a terminal, then use check sign-in:</div><div class="dpre">${esc(a.hint || 'uses your existing CLI login')}</div><div class="dacts"><button class="dact" data-act="copy">copy command</button><button class="dact" data-act="probe">check sign-in</button></div>`;
  h += `<div class="dacts"><button class="dact primary" data-act="signin">${L && !L.done ? 'restart sign-in' : 'sign in'}</button>${L?.url ? '<button class="dact primary" data-act="open">open sign-in page</button>' : ''}</div>`;
  if (a.harness === 'claude' && L && !L.done) h += `<div class="muted2" style="margin-top:8px">After approving in the browser, paste the code it shows:</div><div class="codeRow"><input id="loginCode" spellcheck="false" placeholder="paste code"><button class="dact" data-act="send">submit</button></div>`;
  if (a.harness === 'antigravity' && L && !L.done) h += `<div class="muted2" style="margin-top:8px">Open the page and finish the Google sign-in within 60 seconds. If the page shows a code, paste it here:</div><div class="codeRow"><input id="loginCode" spellcheck="false" placeholder="paste code if shown"><button class="dact" data-act="send">submit</button></div>`;
  if (a.harness === 'codex' && L && !L.done) h += `<div class="muted2" style="margin-top:8px">Open the page, then enter the code shown below.</div>`;
  if (L) h += `<div class="dpre" style="margin-top:8px">${esc(L.text.trim() || 'waiting…')}</div>${L.done ? `<div class="${L.ok ? 'ok' : 'err'}" style="margin-top:4px">${L.ok ? 'Signed in. This account is ready.' : 'Sign-in did not finish. Try again.'}</div>` : ''}`;
  return h;
}
const kv = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
function refreshDrawer() {
  if (!drawer) return;
  const body = $('#dbody'); const keep = body.scrollTop; const typed = body.querySelector('#loginCode')?.value; const hadFocus = document.activeElement?.id === 'loginCode';
  if (drawer.type === 'account') {
    const a = S.accounts.get(drawer.id);
    if (!a) { $('#dclose').click(); return; }
    $('#dtitle').textContent = a.id;
    const jobs = [];
    for (const r of S.runs.values()) for (const g of r.agents.values()) if (g.account === a.id || g.attempts?.includes(a.id)) jobs.push({ r, g });
    const left = a.limitedUntil && a.limitedUntil > Date.now() ? `${fmtDur(a.limitedUntil - Date.now())} (until ${new Date(a.limitedUntil).toLocaleTimeString()})` : '-';
    body.innerHTML = kv([
      ['harness', esc(a.harness)], ['status', `<span class="pill ${a.status === 'working' ? 'running' : a.status === 'limited' ? 'failed' : 'done'}">${esc(a.status)}</span>`],
      ['logged in', a.authed === true ? 'yes' : a.authed === false ? 'no' : 'unknown (run probe)'],
      ['slots in use', `${a.load} / ${S.settings.slots}`], ['jobs done', a.done], ['jobs failed', a.failed],
      ['limit resets in', esc(left)], ['current task', esc(a.task ?? '-')], ['config folder', esc(a.configDir ?? '(default login)')],
    ]) + signInHtml(a)
      + `<div class="dacts"><button class="dact" data-act="probe">check login</button><button class="dact" data-act="rm">remove account</button></div>`
      + `<div class="dsec">Recent jobs (${jobs.length})</div><div class="dlist">${jobs.slice(-12).reverse().map(({ r, g }, i) => `<div data-job="${r.id}/${g.id}"><span>${esc(r.id)} · ${esc(g.id)} · ${esc(g.title ?? '')}</span><span class="pill ${g.status}">${g.status}</span></div>`).join('') || '<div class="muted2">none yet</div>'}</div>`;
    body.querySelectorAll('[data-act]').forEach((b) => { b.onclick = () => {
      const act = b.dataset.act;
      if (act === 'copy') navigator.clipboard?.writeText(a.hint);
      else if (act === 'signin') beginLogin(a.id);
      else if (act === 'open') window.open(S.logins[a.id]?.url, '_blank', 'noopener');
      else if (act === 'send') { const v = body.querySelector('#loginCode')?.value; if (v) { api2('/api/login/input', { id: a.id, text: v }); } }
      else if (act === 'rm') { if (confirm(`Remove ${a.id}?`)) { sendCmd(`account rm ${a.id}`); $('#dclose').click(); } }
      else sendCmd(`account ${act} ${a.id}`);
    }; });
    body.querySelectorAll('[data-job]').forEach((d) => { d.onclick = () => { const [rid, gid] = d.dataset.job.split('/'); S.current = rid; renderRunbar(); openAgent(gid); }; });
  } else {
    const r = cur(); const a = r?.agents.get(drawer.id);
    if (!a) return;
    $('#dtitle').textContent = `${a.id} · ${a.title ?? a.role}`;
    const logs = r.feed.filter((f) => f.id === a.id);
    const claims = [...r.claims.values()].filter((c) => c.agent === a.id);
    const dur = a.startedAt ? fmtDur((a.endedAt ?? Date.now()) - a.startedAt) : '-';
    body.innerHTML = kv([
      ['role', esc(a.role)], ['status', `<span class="pill ${a.status}">${esc(a.status)}</span>`],
      ['parent', a.parent ? `<a href="#" data-ag="${esc(a.parent)}">${esc(a.parent)}</a>` : '-'],
      ['harness', esc(a.harness ?? 'not assigned')], ['account', a.account ? `<a href="#" data-acc="${esc(a.account)}">${esc(a.account)}</a>` : 'waiting for a free account'],
      ['accounts tried', esc((a.attempts ?? []).join(' > ') || '-')], ['failovers', a.failovers ?? 0], ['duration', dur],
      ...(a.score != null ? [['score', Number.isFinite(a.score) ? Number(a.score).toFixed(4) : 'failed'], ['family', esc(a.family ?? '')], ['idea', esc(a.idea ?? '')], ['champion', a.champion ? 'yes' : 'no']] : []),
      ...(a.error ? [['error', esc(a.error)]] : []),
    ])
      + (claims.length ? `<div class="dsec">Claims (${claims.length})</div>${claims.map((c) => `<div class="claim ${c.verdict}"><div class="h"><span>${esc(c.id)}</span><span class="badge">${c.verdict.toUpperCase()}</span></div><div class="t">${esc(c.text)}</div>${c.note ? `<div class="note">${esc(c.note)}</div>` : ''}</div>`).join('')}` : '')
      + `<div class="dsec">Activity (${logs.length})</div><div class="dpre">${logs.map((f) => `${f.kind === 'tool' ? '> ' : f.kind === 'err' ? '! ' : ''}${esc(f.text)}`).join('\n') || 'no output yet'}</div>`
      + (a.output ? `<div class="dsec">Result</div><div class="dpre">${esc(a.output)}</div>` : '')
      + (a.prompt ? `<div class="dsec">Prompt</div><div class="dpre">${esc(a.prompt)}</div>` : '');
    body.querySelectorAll('[data-ag]').forEach((x) => { x.onclick = (e) => { e.preventDefault(); openAgent(x.dataset.ag); }; });
    body.querySelectorAll('[data-acc]').forEach((x) => { x.onclick = (e) => { e.preventDefault(); openAccount(x.dataset.acc); }; });
  }
  body.scrollTop = keep;
  const box = body.querySelector('#loginCode'); if (box && typed) { box.value = typed; if (hadFocus) box.focus(); }
}

// ---------- add-account wizard: pick a provider, then sign in ----------
const PROVIDERS = {
  claude: { name: 'Claude Code', how: 'Claude subscription or Console. Sign in in your browser, then paste the code it shows.', auto: true },
  codex: { name: 'Codex', how: 'ChatGPT plan. Sign in in your browser with a one-time device code.', auto: true },
  antigravity: { name: 'Antigravity', how: 'Google account. Sign in in your browser within 60 seconds of starting.', auto: true },
  gemini: { name: 'Gemini CLI', how: 'Sign in once from a terminal.' },
  opencode: { name: 'OpenCode', how: 'Sign in once from a terminal.' },
};
const WZ = { step: 'pick', harness: null, id: null };
const wizOpen = () => !$('#addModal').hidden;
function renderWizard() {
  if (!wizOpen()) return;
  const body = $('#addBody'); const typed = body.querySelector('#loginCode')?.value; const hadFocus = document.activeElement?.id === 'loginCode';
  $('#addBack').hidden = WZ.step === 'pick'; $('#addDone').hidden = true;
  if (WZ.step === 'pick') {
    $('#addTitle').textContent = 'Add account: choose a provider';
    body.innerHTML = '<div class="providers"></div>';
    const list = body.firstChild;
    (S.harnesses?.length ? S.harnesses : Object.keys(PROVIDERS)).forEach((h) => {
      const p = PROVIDERS[h] ?? { name: h, how: 'Custom harness from ~/.ratchet/harnesses.json. Sign in from a terminal.' };
      const have = [...S.accounts.values()].filter((a) => a.harness === h).length;
      const full = p.single && have > 0;
      const b = el('button', `provider ${h}`, `<b>${esc(p.name)}</b><span>${esc(p.how)}</span><small>${have ? `${have} connected` : 'none connected'}${full ? ', limit reached' : ''}</small>`);
      b.type = 'button'; b.disabled = full;
      b.onclick = async () => {
        const before = new Set(S.accounts.keys());
        await sendCmd(`account add ${h}`);
        setTimeout(() => {
          const id = [...S.accounts.keys()].find((k) => !before.has(k));
          if (!id) return;
          Object.assign(WZ, { step: 'signin', harness: h, id });
          if (PROVIDERS[h]?.auto) beginLogin(id);
          else if (h === 'antigravity') sendCmd(`account probe ${id}`);
          renderWizard();
        }, 300);
      };
      list.append(b);
    });
    return;
  }
  const a = S.accounts.get(WZ.id);
  if (!a) { WZ.step = 'pick'; return renderWizard(); }
  const p = PROVIDERS[a.harness] ?? { name: a.harness };
  $('#addTitle').textContent = `Sign in: ${p.name} (${a.id})`;
  const L = S.logins[a.id];
  const done = a.authed === true || (L?.done && L.ok);
  body.innerHTML = signInHtml(a) + (done ? '<div class="ok big">Connected. Ratchet will now spread work onto this account.</div>' : '');
  $('#addDone').hidden = !done;
  bindSignIn(body, a);
  const box = body.querySelector('#loginCode'); if (box && typed) { box.value = typed; if (hadFocus) box.focus(); }
}
function bindSignIn(root, a) {
  root.querySelectorAll('[data-act]').forEach((b) => { b.onclick = () => {
    const act = b.dataset.act;
    if (act === 'copy') navigator.clipboard?.writeText(a.hint);
    else if (act === 'signin') beginLogin(a.id);
    else if (act === 'open') window.open(S.logins[a.id]?.url, '_blank', 'noopener');
    else if (act === 'send') { const v = root.querySelector('#loginCode')?.value; if (v) api2('/api/login/input', { id: a.id, text: v }); }
    else if (act === 'rm') { if (confirm(`Remove ${a.id}?`)) { sendCmd(`account rm ${a.id}`); $('#dclose').click(); } }
    else sendCmd(`account ${act} ${a.id}`);
  }; });
}
const closeWizard = (keep) => {
  const a = WZ.id && S.accounts.get(WZ.id);
  // an account that never signed in is removed on cancel, so the pool holds only working accounts
  if (!keep && WZ.step === 'signin' && a && a.authed !== true && !(S.logins[a.id]?.ok)) sendCmd(`account rm ${a.id}`);
  Object.assign(WZ, { step: 'pick', harness: null, id: null });
  $('#addModal').hidden = true;
};
$('#addAcc').onclick = () => { Object.assign(WZ, { step: 'pick', harness: null, id: null }); $('#addModal').hidden = false; renderWizard(); };
$('#addCancel').onclick = () => closeWizard(false);
$('#addDone').onclick = () => { const id = WZ.id; closeWizard(true); if (id) openAccount(id); };
$('#addBack').onclick = () => { closeWizard(false); $('#addModal').hidden = false; renderWizard(); };
$('#addModal').addEventListener('mousedown', (e) => { if (e.target.id === 'addModal') closeWizard(false); });
setInterval(() => { if (drawer && drawer.type === 'agent') refreshDrawer(); }, 2000);

// ---------- add agent ----------
const ag = { harness: 'any' };
function fillAccounts() {
  const sel = $('#agAccount'); sel.innerHTML = '<option value="">any available</option>';
  [...S.accounts.values()].filter((a) => ag.harness === 'any' || a.harness === ag.harness).sort((x, y) => x.id.localeCompare(y.id))
    .forEach((a) => { const o = document.createElement('option'); o.value = a.id; o.textContent = `${a.id}${a.status === 'limited' ? ' (limited)' : ''}`; sel.append(o); });
}
function openAgentModal() {
  const seg = $('#agHarness'); seg.innerHTML = '';
  const kinds = ['any', ...new Set([...S.accounts.values()].map((a) => a.harness))];
  kinds.forEach((h) => { const b = el('button', h === ag.harness ? 'on' : ''); b.type = 'button'; b.textContent = h === 'any' ? 'auto' : h; b.onclick = () => { ag.harness = h; seg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b)); fillAccounts(); }; seg.append(b); });
  fillAccounts(); $('#agErr').textContent = ''; $('#agentModal').hidden = false; $('#agPrompt').focus();
}
$('#agCancel').onclick = () => { $('#agentModal').hidden = true; };
$('#agentModal').addEventListener('mousedown', (e) => { if (e.target.id === 'agentModal') $('#agentModal').hidden = true; });
$('#agRun').onclick = async () => {
  const prompt = $('#agPrompt').value.trim();
  if (!prompt) { $('#agErr').textContent = 'write what the agent should do'; return; }
  const r = await fetch('/api/agent', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ harness: ag.harness, account: $('#agAccount').value || null, title: $('#agTitle').value.trim(), prompt }) });
  if (!r.ok) { $('#agErr').textContent = r.status === 409 ? 'no accounts yet: add one first' : 'could not start the agent'; return; }
  const j = await r.json();
  $('#agentModal').hidden = true; $('#agPrompt').value = ''; $('#agTitle').value = '';
  setTimeout(() => openAgent(j.agentId), 200);
};
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('#agentModal').hidden) $('#agentModal').hidden = true; });
