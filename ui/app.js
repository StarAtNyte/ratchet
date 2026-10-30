const $ = (s) => document.querySelector(s);
const token = new URLSearchParams(location.search).get('t') ?? '';
const terminalClientId = sessionStorage.getItem('ratchet.terminal.client') || crypto.randomUUID();
sessionStorage.setItem('ratchet.terminal.client', terminalClientId);
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
  tab: 'agents',
  focus: null,
  feedTools: false,
  lastSeq: 0,
  clearSeq: Number(sessionStorage.getItem(`ratchet.clear.${token}`)) || 0,
};

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const stick = (box, fn) => { const near = box.scrollHeight - box.scrollTop - box.clientHeight < 60; fn(); if (near) box.scrollTop = box.scrollHeight; };

// ---------- terminal ----------
const termOut = $('#termOut');
function say(text, cls = '') { stick(termOut, () => { const d = el('div', cls); d.textContent = text; termOut.append(d); }); }

// command catalogue drives completion and the suggestion list
const CATALOG = [
  ['run', '<task>', 'launch a research swarm'],
  ['follow', '<instruction>', 'queue instructions for the active swarm'],
  ['opt', 'init | start [goal] | apply', 'evolutionary optimise loop'],
  ['resume', '<run>', 're-run an interrupted research run'],
  ['stop', '[run|all]', 'stop a run'],
  ['runs', '', 'list runs'],
  ['report', '[run]', 'print a run report'],
  ['accounts', '', 'list the account pool'],
  ['account', 'add | login | probe | enable | disable | rm', 'manage accounts'],
  ['cd', '<path>', 'change the workspace directory'],
  ['dir', '<path> | recent | mk <path>', 'change the workspace'],
  ['mode', 'yolo | auto', 'agent permission level'],
  ['set', 'slots | workers | timeout | rounds <n>', 'tune the swarm'],
  ['clear', '', 'clear the dashboard view'],
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
  if (cmd === 'account') return sub === undefined ? ['add', 'login', 'probe', 'enable', 'disable', 'rm'] : sub === 'add' ? (S.harnesses ?? []) : accts;
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
  if (line === 'clear') { clearOutput(); return; }
  try {
    const response = await fetch('/api/cmd', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ line, clientId: terminalClientId }) });
    if (response.status === 409) say('A command is already running. Press Ctrl+C to stop it.', 'warn');
    else if (!response.ok) say('Could not run the command.', 'err');
  } catch { say('Could not reach the server.', 'err'); }
}
$('#termForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const line = termIn.value.trim();
  termIn.value = ''; sug.items = []; renderSuggest();
  await submitLine(line);
});
termIn.addEventListener('keydown', (e) => {
  const open = sug.items.length > 0 && !$('#suggest').hidden;
  if (e.key.toLowerCase() === 'c' && e.ctrlKey && !e.metaKey && !e.altKey) {
    if (termIn.selectionStart !== termIn.selectionEnd || window.getSelection()?.toString()) return;
    e.preventDefault(); termIn.value = ''; sug.items = []; renderSuggest();
    fetch('/api/cmd/interrupt', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ clientId: terminalClientId }) }).then((r) => { if (r.status === 204) say('^C · command interrupted', 'warn'); else if (r.status !== 404) say('Could not interrupt the command.', 'err'); }).catch(() => say('Could not reach the server.', 'err'));
  } else if (e.key === 'Tab') {
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
  else if (e.key === 'l' && e.ctrlKey) { e.preventDefault(); clearOutput(); }
  else if (e.key === 'u' && e.ctrlKey) { e.preventDefault(); termIn.value = ''; refreshSuggest(); }
  else if (e.key === 'Enter' && open && sug.sel >= 0) { e.preventDefault(); termIn.value = sug.items[sug.sel].value; refreshSuggest(); }
});
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target !== document.body) return;
  if (TT.tab === 'console') termIn.focus();
});

// quick chips
[['help', 'help'], ['accounts', 'accounts'], ['runs', 'runs'], ['stop all', 'stop all'], ['mode', 'mode'], ['dir recent', 'dir recent']].forEach(([label, line]) => {
  const b = el('button', 'chipBtn'); b.type = 'button'; b.textContent = label; b.onclick = () => submitLine(line); $('#quick').append(b);
});

// ---------- terminal tabs, shell, split, expand ----------
const TT = { tab: 'console' };
function setTermTab(tab) {
  TT.tab = tab;
  document.querySelectorAll('.ttabs button').forEach((b) => b.classList.toggle('on', b.dataset.tt === tab));
  $('#consoleView').classList.toggle('on', tab === 'console');
  $('#activityView').classList.toggle('on', tab === 'activity');
  document.querySelectorAll('.ttabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tt === tab))); 
  if (tab === 'console') termIn.focus();
}
document.querySelectorAll('.ttabs button').forEach((b) => { b.onclick = () => setTermTab(b.dataset.tt); });
function clearOutput() {
  S.clearSeq = S.lastSeq; sessionStorage.setItem(`ratchet.clear.${token}`, String(S.clearSeq));
  termOut.replaceChildren();
  for (const r of S.runs.values()) { r.feed = []; r.cleared = true; }
  S.current = null; S.focus = null; $('#focus').hidden = true;
  if (drawer) $('#dclose').click();
  resetPanes(); renderRunbar(); renderStats(); renderAgents(); renderOverview();
}
$('#tClear').onclick = clearOutput;
$('#tMax').onclick = () => { const m = $('main'); m.classList.toggle('termMax'); $('#tMax').textContent = m.classList.contains('termMax') ? 'restore' : 'expand';  };
(() => {
  const split = $('#split'); const mainEl = $('main');
  const saved = Number(localStorage.getItem('ratchet.split')); if (saved > 0.15 && saved < 0.85) mainEl.style.setProperty('--gfrac', saved);
  const setFrac = (f) => { f = Math.min(0.85, Math.max(0.15, f)); mainEl.style.setProperty('--gfrac', f); localStorage.setItem('ratchet.split', f); };
  split.addEventListener('pointerdown', (e) => {
    split.setPointerCapture(e.pointerId);
    const move = (ev) => { const r = mainEl.getBoundingClientRect(); setFrac((ev.clientY - r.top) / r.height); };
    const up = () => { split.removeEventListener('pointermove', move); split.removeEventListener('pointerup', up);  };
    split.addEventListener('pointermove', move); split.addEventListener('pointerup', up);
  });
  split.addEventListener('keydown', (e) => { const cur = Number(getComputedStyle(mainEl).getPropertyValue('--gfrac')) || 0.6; if (e.key === 'ArrowUp') setFrac(cur - 0.05); if (e.key === 'ArrowDown') setFrac(cur + 0.05); });
})();

// ---------- tabs ----------
document.querySelectorAll('#side nav button').forEach((b) => b.addEventListener('click', () => {
  S.tab = b.dataset.tab;
  document.querySelectorAll('#side nav button').forEach((x) => x.classList.toggle('on', x === b));
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
  // SSE reconnects replay server history; do not apply the same event twice.
  if (ev.seq && ev.seq <= S.lastSeq) return;
  S.lastSeq = Math.max(S.lastSeq, ev.seq || 0);
  const showOutput = !ev.seq || ev.seq > S.clearSeq;
  switch (ev.kind) {
    case 'hello': S.settings = ev.settings; S.commands = ev.commands; S.harnesses = ev.harnesses ?? []; renderHeader(); break;
    case 'settings': S.settings = ev.settings; renderHeader(); break;
    case 'term': if (showOutput) say(ev.text, ev.cls); break;
    case 'account': S.accounts.set(ev.account.id, ev.account); renderAccounts(); break;
    case 'login': {
      const L = (S.logins[ev.id] = ev.reset ? { text: '', url: null, done: false } : (S.logins[ev.id] ?? { text: '', url: null, done: false }));
      if (!ev.reset || ev.text) L.text += `${ev.text}\n`;
      if (ev.url) L.url = ev.url;
      if (ev.done) { L.done = true; L.ok = ev.ok; }
      break;
    }
    case 'account.remove': S.accounts.delete(ev.id); renderAccounts(); break;
    case 'run.start': { const r = run(ev.runId); r.task = ev.task; r.startedAt = ev.t; r.status = 'running'; r.cleared = !showOutput; if (showOutput && (!ev.restored || !cur() || cur().status !== 'running')) { S.current = ev.runId; S.focus = null; resetPanes(); } renderRunbar(); break; }
    case 'agent.start': { const r = run(ev.runId); r.agents.set(ev.id, { id: ev.id, role: ev.role, title: ev.title, parent: ev.parent, harness: null, account: null, status: 'queued', born: ev.t, prompt: ev.prompt, failovers: 0, attempts: [] }); renderRunbar(); break; }
    case 'agent.assign': { if (showOutput) say(`[${ev.runId}] ${ev.agentId} started on ${ev.account} (${ev.harness})${ev.attempt ? ` · retry ${ev.attempt}` : ''}`, 'dim'); const a = run(ev.runId).agents.get(ev.agentId); if (a) { Object.assign(a, { harness: ev.harness, account: ev.account, status: 'running', pulse: ev.t, startedAt: a.startedAt ?? ev.t }); (a.attempts ??= []).push(ev.account); } break; }
    case 'agent.failover': { if (showOutput) say(`[${ev.runId}] ${ev.agentId} is switching accounts after ${ev.from}`, 'warn'); const a = run(ev.runId).agents.get(ev.agentId); if (a) { a.status = 'queued'; a.flash = ev.t; a.failovers = (a.failovers ?? 0) + 1; } break; }
    case 'agent.end': {
      const a = run(ev.runId).agents.get(ev.agentId);
      if (a) { a.status = ev.ok ? 'done' : ev.error === 'stopped' ? 'stopped' : 'failed'; a.error = ev.error; a.endedAt = ev.t; }
      if (showOutput) say(`[${ev.runId}] ${ev.agentId} ${ev.ok ? 'finished' : ev.error === 'stopped' ? 'stopped' : `failed: ${ev.error || 'unknown error'}`}`, ev.ok ? 'ok' : 'err');
      if (ev.ok) clickGear();
      break;
    }
    case 'agent.log': {
      const r = run(ev.runId); const a = r.agents.get(ev.agentId);
      if (!showOutput) break;
      r.feed.push({ id: ev.agentId, kind: ev.logKind, text: ev.text, t: ev.t, harness: a ? harnessOf(a) : 'none' });
      if (r.feed.length > 1500) r.feed.shift();
      if (ev.runId === S.current && (!S.focus || S.focus === ev.agentId) && (S.feedTools || ev.logKind !== 'tool')) addFeed(r.feed.at(-1));
      break;
    }
    case 'cand.result': {
      const a = run(ev.runId).agents.get(ev.agentId);
      if (a) Object.assign(a, { score: ev.score, ok: ev.ok, champion: ev.champion || a.champion, idea: ev.idea, family: ev.family });
      if (showOutput && ev.champion) say(`★ ${ev.agentId} new champion ${Number(ev.score).toFixed(4)} · ${ev.idea}`, 'ok');
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
    case 'run.end': { const r = run(ev.runId); r.status = ev.status; r.endedAt = ev.t; if (showOutput && ev.note) say(`run ${ev.runId} ${ev.status}: ${ev.note}`, 'err'); else if (showOutput) say(`run ${ev.runId} ${ev.status}`, ev.status === 'done' ? 'ok' : 'warn'); renderRunbar(); break; }
  }
  renderStats(); scheduleLists(); if (ev.runId === S.current || ev.kind === 'account') renderOverview();
}

function resetPanes() { renderFeed(); renderClaims(); renderReport(); $('#focus').hidden = true; }

// ---------- renderers ----------
function renderHeader() {
  $('#ws').textContent = S.settings.dir;
  $('#tcwd').textContent = S.settings.dir;
  $('#mockChip').hidden = !S.settings.mock;
  const m = $('#modeChip'); m.textContent = S.settings.mode; m.classList.toggle('yolo', S.settings.mode === 'yolo');
}

function renderStats() {
  const r = cur();
  const cl = r ? [...r.claims.values()] : [];
  $('#claimN').textContent = cl.length;
  $('#graphEmpty').classList.toggle('hide', !!r);
}

function renderRunbar() {
  const bar = $('#runbar'); bar.innerHTML = '';
  for (const r of S.runs.values()) {
    if (r.cleared) continue;
    const b = el('button', `${r.id === S.current ? 'on' : ''} ${r.status === 'running' ? 'running' : ''}`);
    b.textContent = `${r.id} · ${r.task}`;
    b.onclick = () => { S.current = r.id; S.focus = null; resetPanes(); renderFeed(); renderRunbar(); renderStats(); };
    bar.append(b);
    if (r.status === 'running' && r.id === S.current) { const x = el('button', 'stopRun'); x.type = 'button'; x.textContent = 'stop run'; x.onclick = () => sendCmd(`stop ${r.id}`); bar.append(x); }
    if (r.status === 'interrupted' && r.id === S.current) { const x = el('button', 'stopRun'); x.type = 'button'; x.textContent = 'resume'; x.onclick = () => sendCmd(`resume ${r.id}`); bar.append(x); }
  }
}

function renderAccounts() {
  const list = $('#accList');
  const accs = [...S.accounts.values()].sort((a, b) => a.id.localeCompare(b.id));
  const ready = accs.filter((a) => a.enabled !== false && a.authed === true && !a.checking && a.status !== 'limited').length;
  $('#poolSum').textContent = `${ready}/${accs.length} ready`;
  list.innerHTML = '';
  const groups = {};
  for (const [enabled, title] of [[true, 'Using'], [false, 'Not using']]) {
    const group = el('section', 'account-group'); group.dataset.enabled = String(enabled);
    const count = accs.filter((a) => (a.enabled !== false) === enabled).length;
    group.innerHTML = `<div class="account-group-head"><span>${title}</span><span>${count}</span></div><div class="account-drop" data-enabled="${enabled}"></div>`;
    const drop = group.querySelector('.account-drop');
    if (!count) drop.append(el('div', 'account-drop-empty', enabled ? 'Move accounts here to use them' : 'Drag accounts here to exclude them'));
    group.addEventListener('dragover', (event) => { event.preventDefault(); group.classList.add('drag-over'); event.dataTransfer.dropEffect = 'move'; });
    group.addEventListener('dragleave', (event) => { if (!group.contains(event.relatedTarget)) group.classList.remove('drag-over'); });
    group.addEventListener('drop', (event) => { event.preventDefault(); group.classList.remove('drag-over'); const id = event.dataTransfer.getData('text/plain'); if (S.accounts.has(id)) sendCmd(`account ${enabled ? 'enable' : 'disable'} ${id}`); });
    groups[String(enabled)] = drop; list.append(group);
  }
  for (const a of accs) {
    const pips = Array.from({ length: S.settings.slots ?? 2 }, (_, i) => `<i class="${i < a.load ? 'on' : ''}"></i>`).join('');
    const d = el('div', `acc ${a.harness} ${a.status}`, `
      <div class="top"><span class="id">${esc(a.id)}</span><span class="account-state ${a.authed === true ? 'connected' : a.authed === false ? 'disconnected' : 'unknown'}">${a.checking ? 'Checking…' : a.authed === true ? 'Signed in' : a.authed === false ? 'Sign in needed' : 'Not checked'}</span></div>
      <div class="account-email">${esc(a.email ?? (a.checking ? 'Checking account details…' : a.authMessage ?? (a.authed === true ? 'Email not provided by provider' : a.authed === false ? 'Sign in to use this account' : 'Sign-in status unavailable')))}</div>
      <div class="task">${esc(a.task ?? (a.status === 'limited' ? 'Usage limit reached · excluded' : a.enabled === false ? 'Excluded from new assignments' : 'idle'))}</div>
      <div class="row"><span class="pips">${pips}</span><span>${a.done}✓ ${a.failed ? a.failed + '✗' : ''}</span>
        <span class="${a.status === 'limited' ? 'cd' : a.authed === false ? 'auth-no' : ''}" data-until="${a.limitedUntil || 0}">${a.authed === false ? 'not logged in' : ''}</span></div>`);
    d.draggable = true;
    d.setAttribute('role', 'button'); d.tabIndex = 0; d.setAttribute('aria-label', `${a.id}, open account details`);
    d.onclick = () => openAccount(a.id);
    d.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openAccount(a.id); } };
    d.ondragstart = (event) => { event.dataTransfer.setData('text/plain', a.id); event.dataTransfer.effectAllowed = 'move'; d.classList.add('dragging'); };
    d.ondragend = () => d.classList.remove('dragging');
    const move = el('button', 'account-move'); move.type = 'button'; move.textContent = a.enabled === false ? 'Use account' : 'Stop using';
    move.setAttribute('aria-label', `${a.enabled === false ? 'Use' : 'Stop using'} ${a.id}`);
    move.onclick = (event) => { event.stopPropagation(); sendCmd(`account ${a.enabled === false ? 'enable' : 'disable'} ${a.id}`); };
    d.append(move); groups[String(a.enabled !== false)].append(d);
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
  const placeholder = box.querySelector('.activity-empty'); if (placeholder) placeholder.remove();
  stick(box, () => {
    const row = el('div', `activity-entry ${f.kind}`);
    const head = el('div', 'activity-head');
    const who = el('button', `activity-agent ${f.harness}`); who.type = 'button'; who.textContent = f.id; who.onclick = () => openAgent(f.id);
    const type = el('span', 'activity-kind'); type.textContent = ({ tool: 'Tool call', err: 'Error', sys: 'Status', text: 'Update' })[f.kind] || 'Update';
    const time = el('time', 'activity-time'); if (f.t) { time.dateTime = new Date(f.t).toISOString(); time.textContent = new Date(f.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
    head.append(who, type, time);
    const tx = el('div', 'activity-text'); tx.textContent = f.text;
    row.append(head, tx); box.append(row);
    while (box.childElementCount > 500) box.firstChild.remove();
  });
}
function renderFeed() {
  const box = $('#feed'); box.replaceChildren();
  const r = cur(); const entries = (r?.feed ?? []).filter((f) => (!S.focus || f.id === S.focus) && (S.feedTools || f.kind !== 'tool')).slice(-400);
  if (!entries.length) { const empty = el('div', 'activity-empty'); empty.textContent = S.focus ? 'No updates for this agent yet.' : 'Agent updates will appear here as work progresses.'; box.append(empty); }
  else entries.forEach(addFeed);
}
$('#feedDetail').onclick = () => { S.feedTools = !S.feedTools; $('#feedDetail').textContent = S.feedTools ? 'Hide tool calls' : 'Show tool calls'; $('#feedDetail').setAttribute('aria-pressed', String(S.feedTools)); renderFeed(); };
function focusAgent(id) {
  const r = cur(); if (!r) return;
  S.focus = S.focus === id ? null : id; renderFeed();
  const f = $('#focus'); f.hidden = !S.focus; f.textContent = S.focus ? `Showing ${S.focus} · click to show all` : '';
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
const G = { layout: localStorage.getItem('ratchet.layout') || 'tree' };
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
      if (!ks.length) { out.set(id, { x: depth * 280, y: row++ * 84 }); return; }
      ks.forEach((k) => walk(k.id, depth + 1));
      const ys = ks.map((k) => out.get(k.id).y);
      out.set(id, { x: depth * 280, y: (Math.min(...ys) + Math.max(...ys)) / 2 });
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
  const padR = G.layout === 'tree' ? 116 : 70;
  return { x0: x0 - (G.layout === 'tree' ? 116 : 70), y0: y0 - 60, x1: x1 + padR, y1: y1 + 70 };
}
function fitTo(b, snap) {
  const ov = $('#overview'); const cvTop = cv.getBoundingClientRect().top;
  const top = (ov.hidden ? 44 : ov.getBoundingClientRect().bottom - cvTop) + 16; const bottom = 44; // room for the run strip and legend
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
  if (G.layout === 'radial') {
  ctx.strokeStyle = tint(COLORS.accent, 22); ctx.beginPath();
  const teeth = 28; const rot = t / 60000;
  for (let i = 0; i < teeth; i++) {
    const a0 = rot + (i / teeth) * Math.PI * 2; const a1 = a0 + (0.8 / teeth) * Math.PI * 2; const a2 = a0 + (1 / teeth) * Math.PI * 2;
    const ri = 34; const ro = 44;
    if (i === 0) ctx.moveTo(lead.x + Math.cos(a0) * ri, lead.y + Math.sin(a0) * ri);
    ctx.lineTo(lead.x + Math.cos(a1) * ro, lead.y + Math.sin(a1) * ro); ctx.lineTo(lead.x + Math.cos(a1) * ri, lead.y + Math.sin(a1) * ri); ctx.lineTo(lead.x + Math.cos(a2) * ri, lead.y + Math.sin(a2) * ri);
  }
  ctx.closePath(); ctx.stroke();
  }
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
    ctx.beginPath(); ctx.moveTo(p.x + (G.layout === 'tree' ? 102 : 0), p.y);
    if (G.layout === 'tree') { const mx = (p.x + n.x) / 2; ctx.bezierCurveTo(mx, p.y, mx, n.y, n.x - 102, n.y); }
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
    if (G.layout === 'tree') {
      const w = 204, h = 60, x = n.x - w / 2, y = n.y - h / 2;
      const isHovered = mw.x >= x && mw.x <= x + w && mw.y >= y && mw.y <= y + h;
      if (isHovered) hover = a;
      ctx.fillStyle = COLORS.panel; ctx.strokeStyle = isHovered ? col : tint(col, 40); ctx.lineWidth = isHovered ? 1.8 : 1;
      ctx.beginPath(); ctx.roundRect(x, y, w, h, 8); ctx.fill(); ctx.stroke();
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x + 14, y + 17, 4, 0, Math.PI * 2); ctx.fill();
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillStyle = COLORS.ink2; ctx.font = '600 11px system-ui, sans-serif';
      ctx.fillText(a.id === 'lead' ? 'LEAD' : String(a.role || 'agent').toUpperCase(), x + 25, y + 17, 96);
      ctx.fillStyle = a.status === 'failed' ? COLORS.bad : a.status === 'done' ? COLORS.ok : a.status === 'running' ? COLORS.accent : COLORS.muted;
      ctx.textAlign = 'right'; ctx.font = '10px system-ui, sans-serif'; ctx.fillText(a.status || 'queued', x + w - 12, y + 17, 68);
      ctx.textAlign = 'left'; ctx.fillStyle = COLORS.lead; ctx.font = '500 12px system-ui, sans-serif';
      ctx.fillText(a.title || a.id, x + 12, y + 38, w - 24);
      ctx.fillStyle = COLORS.muted; ctx.font = '10px system-ui, sans-serif';
      const meta = [a.account || (a.id === 'lead' ? a.id : 'waiting for account'), a.score != null && Number.isFinite(a.score) ? Number(a.score).toFixed(3) : ''].filter(Boolean).join(' · ');
      ctx.fillText(meta, x + 12, y + 52, w - 24); ctx.textBaseline = 'alphabetic';
      if (a.status === 'running' && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        ctx.fillStyle = col; ctx.globalAlpha = 0.45 + 0.5 * (0.5 + 0.5 * Math.sin(t / 550)); ctx.beginPath(); ctx.arc(x + 14, y + 17, 4, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
      }
      continue;
    }
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
        if (58 * view.k >= 30) ctx.fillText([a.account ?? (a.id === 'lead' ? '' : 'waiting'), score].filter(Boolean).join('  '), n.x + rad + 8 * u, n.y + 11 * u);
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
function spark(ag) {
  const sc = ag.filter((a) => a.score != null && Number.isFinite(a.score) && a.ok !== false).sort((x, y) => (x.born ?? 0) - (y.born ?? 0));
  if (sc.length < 2) return '';
  const vals = sc.map((a) => a.score); const minimize = vals[vals.length - 1] <= vals[0];
  let best = vals[0]; const curve = vals.map((v) => (best = minimize ? Math.min(best, v) : Math.max(best, v)));
  const lo = Math.min(...vals); const hi = Math.max(...vals); const w = 160; const h = 32;
  const pts = (arr) => arr.map((v, i) => `${((i / (arr.length - 1)) * w).toFixed(1)},${(h - ((v - lo) / (hi - lo || 1)) * h).toFixed(1)}`).join(' ');
  return `<div class="spark"><div><dt>best score</dt><dd>${curve[curve.length - 1].toFixed(3)}</dd></div><svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-label="best score over ${sc.length} tries"><polyline class="all" points="${pts(vals)}"/><polyline class="best" points="${pts(curve)}"/></svg></div>`;
}
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
    <dl class="ovStats"><div><dt>elapsed</dt><dd>${fmtAgo(elapsed)}</dd></div><div><dt>running</dt><dd>${c('running')}</dd></div><div><dt>waiting</dt><dd>${c('queued')}</dd></div><div><dt>done</dt><dd>${c('done')}</dd></div><div><dt>failed</dt><dd>${c('failed')}</dd></div>${cl.length ? `<div><dt>claims</dt><dd><span class="okc">${v('confirmed')}</span> / <span class="unc">${v('unclear') + v('pending')}</span> / <span class="badc">${v('refuted')}</span></dd></div>` : ''}<div><dt>busy accounts</dt><dd>${busy.length ? esc(busy.join(', ')) : 'none'}</dd></div></dl>${spark(ag)}`;
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
// Replayed events rebuild initial state; already applied events are ignored on reconnect.
connect();

// ---------- workspace picker ----------
const pk = { path: '', home: '' };
const api = (u) => fetch(u + (u.includes('?') ? '&' : '?') + 't=' + encodeURIComponent(token)).then((r) => r.json());
const sendCmd = (line) => fetch('/api/cmd', { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify({ line }) });

const pick = { dirs: [], meta: {}, sel: 0, filter: '' };
const tagHtml = (tags) => (tags ?? []).map((t) => `<span class="tag ${t}">${esc(t)}</span>`).join('');
async function browse(path) {
  const r = await api(`/api/ls?path=${encodeURIComponent(path ?? '')}${$('#pickHidden').checked ? '&hidden=1' : ''}`);
  if (r.error) { $('#pickDirs').innerHTML = `<div class="err">${esc(r.error)}</div>`; return; }
  pk.path = r.path; pk.home = r.home;
  Object.assign(pick, { dirs: r.dirs, meta: r.meta ?? {}, sel: 0, filter: '' });
  $('#pickFilter').value = '';
  $('#pickPath').textContent = r.path;
  $('#pickTags').innerHTML = tagHtml(r.here);
  const crumbs = $('#pickCrumbs'); crumbs.innerHTML = '';
  const parts = r.path.split('/').filter(Boolean);
  const root = el('button', 'crumb', '/'); root.type = 'button'; root.onclick = () => browse('/'); crumbs.append(root);
  parts.forEach((part, i) => {
    const b = el('button', `crumb${i === parts.length - 1 ? ' on' : ''}`); b.type = 'button'; b.textContent = part;
    b.onclick = () => browse(`/${parts.slice(0, i + 1).join('/')}`);
    crumbs.append(el('span', 'sep', '/'), b);
  });
  renderDirs();
}
function renderDirs() {
  const box = $('#pickDirs'); box.innerHTML = '';
  const q = pick.filter.toLowerCase();
  const shown = pick.dirs.filter((d) => d.toLowerCase().includes(q));
  pick.sel = Math.min(pick.sel, Math.max(0, shown.length));
  const up = el('div', `drow up${pick.sel === 0 ? ' on' : ''}`, '<span class="dn">..</span><span class="muted2">parent folder</span>');
  up.setAttribute('role', 'option'); up.onclick = () => browse(pk.path.replace(/\/[^/]+\/?$/, '') || '/');
  box.append(up);
  shown.forEach((d, i) => {
    const row = el('div', `drow${pick.sel === i + 1 ? ' on' : ''}`, `<span class="dn">${esc(d)}</span>${tagHtml(pick.meta[d])}`);
    row.setAttribute('role', 'option');
    row.onclick = () => browse(`${pk.path.replace(/\/$/, '')}/${d}`);
    box.append(row);
  });
  if (!shown.length) box.append(el('div', 'err', q ? 'no folder matches' : 'no subfolders'));
  box.querySelector('.on')?.scrollIntoView({ block: 'nearest' });
  pick.shown = shown;
}
function renderPlaces() {
  const box = $('#pickPlaces'); box.innerHTML = '';
  const add = (label, path, cls = '') => { const b = el('button', `place ${cls}`); b.type = 'button'; b.title = path; b.innerHTML = `<b>${esc(label)}</b><small>${esc(path)}</small>`; b.onclick = () => browse(path); box.append(b); };
  box.append(el('div', 'pl', 'Places'));
  add('Current workspace', S.settings.dir, 'cur');
  if (pk.home) add('Home', pk.home);
  const rec = (S.settings.recent ?? []).filter((d) => d !== S.settings.dir).slice(0, 6);
  if (rec.length) { box.append(el('div', 'pl', 'Recent')); rec.forEach((d) => add(d.split('/').filter(Boolean).pop() || '/', d)); }
}
async function openPicker() {
  $('#picker').hidden = false;
  await browse(S.settings.dir);
  renderPlaces();
  $('#pickFilter').focus();
}
const closePicker = () => { $('#picker').hidden = true; $('#termIn').focus(); };
$('#ws').onclick = openPicker;
$('#pickCancel').onclick = closePicker;
$('#picker').addEventListener('mousedown', (e) => { if (e.target.id === 'picker') closePicker(); });
$('#pickFilter').addEventListener('input', (e) => { pick.filter = e.target.value.startsWith('/') || e.target.value.startsWith('~') ? '' : e.target.value; pick.sel = pick.filter ? 1 : 0; renderDirs(); });
$('#pickFilter').addEventListener('keydown', (e) => {
  const n = (pick.shown?.length ?? 0) + 1;
  if (e.key === 'ArrowDown') { e.preventDefault(); pick.sel = (pick.sel + 1) % n; renderDirs(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); pick.sel = (pick.sel - 1 + n) % n; renderDirs(); }
  else if (e.key === 'Backspace' && !e.target.value) { e.preventDefault(); browse(pk.path.replace(/\/[^/]+\/?$/, '') || '/'); }
  else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $('#pickUse').click(); }
});
$('#pickForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const v = $('#pickFilter').value.trim();
  if (v.startsWith('/') || v.startsWith('~')) return browse(v);
  if (pick.sel === 0) return browse(pk.path.replace(/\/[^/]+\/?$/, '') || '/');
  const d = pick.shown?.[pick.sel - 1]; if (d) browse(`${pk.path.replace(/\/$/, '')}/${d}`);
});
$('#pickHidden').onchange = () => browse(pk.path);
$('#pickUse').onclick = async () => { await sendCmd(`dir ${pk.path}`); closePicker(); };
$('#pickNew').onclick = async () => {
  const name = prompt('New folder name (created inside the selected folder):');
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
  const header = el('div', 'agents-heading');
  const label = el('div', 'agents-heading-copy', `<strong>Run agents</strong><span>Inspect progress and troubleshoot failures</span>`);
  const add = el('button', 'dact addAgent', '+ Add agent'); add.type = 'button'; add.onclick = openAgentModal; header.append(label, add); box.append(header);
  if (!r || !r.agents.size) { box.append(el('div', 'agents-empty', 'No agents in this run yet. Start research or add an agent to begin.')); return; }
  const list = el('div', 'agent-list');
  [...r.agents.values()].forEach((a) => {
    const col = a.status === 'failed' ? COLORS.bad : COLORS[harnessOf(a)] ?? COLORS.none;
    const status = a.status || 'queued';
    const detail = a.status === 'failed' && a.error ? a.error : a.account ? `${a.account} · ${a.harness || 'assigned'}` : 'Waiting for an account';
    const row = el('button', 'arow', `<span class="agent-marker" style="background:${col}"></span><span class="agent-main"><span class="agent-title">${esc(a.title ?? a.role ?? a.id)}</span><span class="agent-meta">${esc(a.id)} · ${esc(detail)}</span></span><span class="pill ${esc(status)}">${esc(status)}</span>`);
    row.type = 'button'; row.setAttribute('aria-label', `${a.title ?? a.id}, ${status}. Open agent details`);
    row.onclick = () => openAgent(a.id); list.append(row);
  });
  box.append(list);
}
document.querySelectorAll('#side nav button').forEach((b) => b.addEventListener('click', renderAgents));

let drawer = null;
let detailReturnFocus = null;
function showDetails(type, id) {
  if (!drawer) detailReturnFocus = document.activeElement;
  drawer = { type, id };
  const isAccount = type === 'account';
  $('#drawer').hidden = isAccount;
  $('#agentDetails').hidden = !isAccount;
  refreshDrawer();
  $(isAccount ? '#agentDetailClose' : '#dclose').focus();
}
function openAccount(id) { showDetails('account', id); }
function openAgent(id) { showDetails('agent', id); }
function closeDetails() { drawer = null; $('#drawer').hidden = true; $('#agentDetails').hidden = true; if (detailReturnFocus?.isConnected) detailReturnFocus.focus(); detailReturnFocus = null; }
$('#dclose').onclick = closeDetails;
$('#agentDetailClose').onclick = closeDetails;
$('#agentDetails').onclick = (e) => { if (e.target === $('#agentDetails')) closeDetails(); };
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && drawer && $('#picker').hidden) { e.preventDefault(); closeDetails(); }
  if (e.key === 'Tab' && drawer?.type === 'account') {
    const items = [...$('#agentDetails').querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled])')];
    if (!items.length) return;
    if (e.shiftKey && document.activeElement === items[0]) { e.preventDefault(); items.at(-1).focus(); }
    else if (!e.shiftKey && document.activeElement === items.at(-1)) { e.preventDefault(); items[0].focus(); }
  }
});

const api2 = (u, b) => fetch(u, { method: 'POST', headers: { 'x-ratchet-token': token, 'content-type': 'application/json' }, body: JSON.stringify(b) });
const beginLogin = (id) => { S.logins[id] = { text: '', url: null, done: false }; api2('/api/login', { id }); };
function signInHtml(a) {
  const L = S.logins[a.id];
  const canAuto = a.harness === 'claude' || a.harness === 'codex' || (a.harness === 'antigravity' && !!a.configDir);
  const status = a.authed === true ? '<span class="pill done">signed in</span>' : a.authed === false ? '<span class="pill failed">not signed in</span>' : '<span class="pill">unknown</span>';
  let h = `<div class="dsec">Account access ${status}</div>`;
  if (a.harness === 'antigravity' && !a.configDir) return h + `<div class="muted2">This account has no private folder. It uses whichever Google account is signed in to the Antigravity app on this machine. Add another Antigravity account for a separate Google login.</div>${a.checking ? '<div class="muted2" style="margin-top:8px">Checking with a one-line test prompt…</div>' : ''}`;
  if (!canAuto) return h + `<div class="muted2">Automatic sign-in is not available for ${esc(a.harness)}. Run this in a terminal, then use check sign-in:</div><div class="dpre">${esc(a.hint || 'uses your existing CLI login')}</div><div class="dacts"><button class="dact" data-act="copy">copy command</button></div>`;
  h += `<div class="dacts"><button class="dact primary" data-act="signin">${L && !L.done ? 'Restart sign-in' : a.authed === true ? 'Sign in again' : 'Sign in'}</button>${L?.url ? '<button class="dact primary" data-act="open">open sign-in page</button>' : ''}</div>`;
  if (a.harness === 'claude' && L && !L.done) h += `<div class="muted2" style="margin-top:8px">After approving in the browser, paste the code it shows:</div><div class="codeRow"><input id="loginCode" spellcheck="false" placeholder="paste code"><button class="dact" data-act="send">submit</button></div>`;
  if (a.harness === 'antigravity' && L && !L.done) h += `<div class="muted2" style="margin-top:8px">Open the page and finish the Google sign-in within 60 seconds. If the page shows a code, paste it here:</div><div class="codeRow"><input id="loginCode" spellcheck="false" placeholder="paste code if shown"><button class="dact" data-act="send">submit</button></div>`;
  if (a.harness === 'codex' && L && !L.done) h += `<div class="muted2" style="margin-top:8px">Open the page, then enter the code shown below.</div>`;
  if (L) h += `<div class="dpre" style="margin-top:8px">${esc(L.text.trim() || 'waiting…')}</div>${L.done ? `<div class="${L.ok ? 'ok' : 'err'}" style="margin-top:4px">${L.ok ? 'Signed in. This account is ready.' : 'Sign-in did not finish. Try again.'}</div>` : ''}`;
  return h;
}
const kv = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
function refreshDrawer() {
  if (!drawer) return;
  const body = $(drawer.type === 'account' ? '#agentDetailBody' : '#dbody'); const keep = body.scrollTop; const typed = body.querySelector('#loginCode')?.value; const hadFocus = document.activeElement?.id === 'loginCode'; const modelInput = body.querySelector('#accountModel'); const modelTyped = modelInput?.value; const modelFocused = document.activeElement === modelInput;
  if (drawer.type === 'account') {
    const a = S.accounts.get(drawer.id);
    if (!a) { $('#dclose').click(); return; }
    $('#agentDetailTitle').textContent = a.id;
    const jobs = [];
    for (const r of S.runs.values()) for (const g of r.agents.values()) if (g.account === a.id || g.attempts?.includes(a.id)) jobs.push({ r, g });
    const left = a.limitedUntil && a.limitedUntil > Date.now() ? `${fmtDur(a.limitedUntil - Date.now())} (until ${new Date(a.limitedUntil).toLocaleTimeString()})` : '-';
    body.innerHTML = kv([
      ['provider', esc(a.harness)], ['activity', `<span class="pill ${a.status === 'working' ? 'running' : a.status === 'limited' ? 'failed' : 'done'}">${esc(a.status)}</span>`],
      ['sign-in status', a.checking ? 'Checking…' : a.authed === true ? 'Signed in' : a.authed === false ? 'Not signed in' : esc(a.authMessage ?? 'Not checked')],
      ['email', esc(a.email ?? (a.checking ? 'Checking…' : a.authed === false ? 'Sign in to see account details' : a.authed === true ? 'Not provided by this provider' : 'Sign-in status not checked'))], ['plan', esc(a.plan ?? (a.authed === true ? 'Not provided by this provider' : 'Unavailable'))],
      ['provider usage', 'Unavailable from the current integration'],
      ['slots in use', `${a.load} / ${S.settings.slots}`], ['jobs done', a.done], ['jobs failed', a.failed],
      ['limit resets in', esc(left)], ['current task', esc(a.task ?? '-')], ['config folder', esc(a.configDir ?? '(default login)')],
    ]) + (['claude', 'codex'].includes(a.harness) ? `<div class="dsec">Model</div><form id="modelForm" class="model-form"><label for="accountModel">Model name</label><div class="model-controls"><input id="accountModel" value="${esc(a.model ?? '')}" placeholder="Provider default" autocomplete="off" spellcheck="false"><button class="dact" type="submit">Save</button></div><span class="muted2">Leave blank to use the provider default. Applies to new assignments.</span><span class="err" id="modelError" role="status"></span></form>` : `<div class="dsec">Model</div><div class="muted2">Model selection is not available for this provider.</div>`) + signInHtml(a)
      + `<div class="dacts account-actions"><button class="dact" data-act="probe" ${a.checking ? 'disabled' : ''}>${a.checking ? 'Checking…' : 'Check sign-in'}</button><button class="dact" data-act="rm">Remove account</button></div>`
      + `<div class="dsec">Recent jobs (${jobs.length})</div><div class="dlist">${jobs.slice(-12).reverse().map(({ r, g }, i) => `<div data-job="${r.id}/${g.id}"><span>${esc(r.id)} · ${esc(g.id)} · ${esc(g.title ?? '')}</span><span class="pill ${g.status}">${g.status}</span></div>`).join('') || '<div class="muted2">none yet</div>'}</div>`;
    const modelForm = body.querySelector('#modelForm');
    if (modelForm) modelForm.onsubmit = async (event) => { event.preventDefault(); const field = body.querySelector('#accountModel'); const response = await api2('/api/account/model', { id: a.id, model: field.value.trim() }); if (!response.ok) body.querySelector('#modelError').textContent = await response.text() || 'Could not save model.'; };
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
  const modelBox = body.querySelector('#accountModel'); if (modelBox && modelFocused) { modelBox.value = modelTyped; modelBox.focus(); }
}

// ---------- add-account wizard: pick a provider, then sign in ----------
const PROVIDERS = {
  claude: { name: 'Claude Code', how: 'Claude subscription or Console account.', auto: true },
  codex: { name: 'Codex', how: 'ChatGPT account with device sign-in.', auto: true },
  antigravity: { name: 'Antigravity', how: 'Google account with browser sign-in.', auto: true },
  gemini: { name: 'Gemini CLI', how: 'Sign in through your terminal.' },
  opencode: { name: 'OpenCode', how: 'Sign in through your terminal.' },
};
const WZ = { step: 'pick', harness: null, id: null };
const wizOpen = () => !$('#addModal').hidden;
function renderWizard() {
  if (!wizOpen()) return;
  const body = $('#addBody'); const typed = body.querySelector('#loginCode')?.value; const hadFocus = document.activeElement?.id === 'loginCode';
  $('#addBack').hidden = WZ.step === 'pick'; $('#addDone').hidden = true;
  if (WZ.step === 'pick') {
    $('#addTitle').textContent = 'Add an account';
    const signature = JSON.stringify([S.harnesses, [...S.accounts.values()].map((a) => a.harness)]);
    if (body.dataset.pickerSignature === signature && body.querySelector('.providers')) return;
    body.dataset.pickerSignature = signature;
    body.innerHTML = '<p class="provider-intro">Choose the provider you want to connect. You can add more than one account.</p><div class="provider-list-heading">Available providers</div><div class="providers"></div>';
    const list = body.querySelector('.providers');
    (S.harnesses?.length ? S.harnesses : Object.keys(PROVIDERS)).forEach((h) => {
      const p = PROVIDERS[h] ?? { name: h, how: 'Custom harness from ~/.ratchet/harnesses.json. Sign in from a terminal.' };
      const have = [...S.accounts.values()].filter((a) => a.harness === h).length;
      const full = p.single && have > 0;
      const b = el('button', `provider ${h}`, `<span class="provider-dot" aria-hidden="true"></span><span class="provider-copy"><b>${esc(p.name)}</b><span>${esc(p.how)}</span></span><span class="provider-count">${have ? `${have} added` : 'Add'}${full ? ' · limit reached' : ''}</span><span class="provider-arrow" aria-hidden="true">›</span>`);
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
  delete body.dataset.pickerSignature;
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
  // The account stays in the local registry so sign-in can be completed later.
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
  [...S.accounts.values()].filter((a) => a.enabled !== false).filter((a) => ag.harness === 'any' || a.harness === ag.harness).sort((x, y) => x.id.localeCompare(y.id))
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

$('#accToggle').onclick = () => $('#accounts').classList.toggle('open');
document.addEventListener('mousedown', (e) => { const a = $('#accounts'); if (a.classList.contains('open') && !a.contains(e.target) && e.target.id !== 'accToggle') a.classList.remove('open'); });
