import { createServer } from 'node:http';
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync, mkdirSync, rmSync, appendFileSync, renameSync } from 'node:fs';
import { homedir, availableParallelism, setPriority } from 'node:os';
import { execFileSync } from 'node:child_process';
import { randomBytes, timingSafeEqual } from 'node:crypto';
import { spawn } from 'node:child_process';
import { dirname, join, extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { AccountPool, HOME, codexIdentity } from './accounts.mjs';
import { Swarm } from './swarm.mjs';
import { Optimizer } from './optimizer.mjs';
import { ForgeSwarm } from './forge.mjs';
import { HARNESSES, mergeEnv } from './harness.mjs';
import { MODES } from './modes.mjs';
const CLEAN = /\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[A-Za-z]/g;
const LOGIN_CMD = { claude: ['claude', ['auth', 'login']], codex: ['codex', ['login', '--device-auth']], antigravity: ['agy', ['--print=Reply with OK']] };
import { execCommand, COMMANDS } from './commands.mjs';

const SRC = dirname(fileURLToPath(import.meta.url));
const UI = join(SRC, '..', 'ui');
const req = createRequire(import.meta.url);
const pkgDir = (name) => { try { return dirname(req.resolve(`${name}/package.json`)); } catch { return null; } };
// browser-side vendor files served straight from node_modules (no bundler)
const VENDOR = {
  '/vendor/xterm.js': ['@xterm/xterm', 'lib/xterm.js'],
  '/vendor/xterm.css': ['@xterm/xterm', 'css/xterm.css'],
  '/vendor/addon-fit.js': ['@xterm/addon-fit', 'lib/addon-fit.js'],
};
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
const MAX_EVENTS = 4000;

/**
 * Keep the machine usable: Ratchet and everything it starts run at the lowest CPU priority,
 * numeric libraries default to one thread, and the whole process tree is pinned to a core
 * budget (children inherit the affinity of the server). Linux only; elsewhere it is a no-op.
 */
export function applyCpuLimit(cpus) {
  const total = availableParallelism();
  const n = Math.max(1, Math.min(total, Math.floor(cpus)));
  const list = `${total - n}-${total - 1}`; // the highest-numbered cores, away from core 0
  const pinned = [];
  const walk = (pid) => {
    try { execFileSync('taskset', ['-a', '-cp', list, String(pid)], { stdio: 'ignore' }); pinned.push(pid); } catch { /* not Linux or no taskset */ }
    let kids = [];
    try { kids = execFileSync('pgrep', ['-P', String(pid)], { encoding: 'utf8' }).trim().split('\n').filter(Boolean); } catch { /* no children */ }
    for (const k of kids) walk(Number(k));
  };
  walk(process.pid);
  return { cpus: n, list, pinned: pinned.length };
}
function quietDefaults() {
  try { setPriority(19); } catch { /* not permitted */ }
  for (const k of ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']) process.env[k] ??= '1';
}
const VERSION = JSON.parse(readFileSync(join(SRC, '..', 'package.json'), 'utf8')).version;
const STARTED = Date.now();
const LOG = join(HOME, 'ratchet.log');
// structured operations log: one JSON object per line, rotated at 5 MB
const LOGGED = new Set(['run.start', 'run.end', 'run.plan', 'run.score', 'agent.failover', 'agent.strike', 'account.limited']);
function oplog(ev) {
  if (!LOGGED.has(ev.kind) && !(ev.kind === 'agent.end' && !ev.ok)) return;
  try {
    if (existsSync(LOG) && statSync(LOG).size > 5e6) renameSync(LOG, `${LOG}.1`);
    const { kind, runId, agentId, status, note, error, reason, complexity, score, from } = ev;
    appendFileSync(LOG, `${JSON.stringify({ t: new Date().toISOString(), kind, runId, agentId, status, note, error, reason, complexity, score: score?.overall, from })}\n`);
  } catch { /* logging never breaks the app */ }
}
const STATE = join(HOME, 'state.json');
const loadRecent = () => { try { return JSON.parse(readFileSync(STATE, 'utf8')).recent ?? []; } catch { return []; } };
const expand = (p) => resolve(String(p).replace(/^~(?=$|\/)/, homedir()));

export async function startServer({ dir, port, mock, open, stableToken = false }) {
  // a service keeps one token across restarts so its bookmarked URL keeps working
  const tokenFile = join(HOME, 'token');
  let token = stableToken && existsSync(tokenFile) ? readFileSync(tokenFile, 'utf8').trim() : '';
  if (!/^[a-f0-9]{32}$/.test(token)) { token = randomBytes(16).toString('hex'); if (stableToken) writeFileSync(tokenFile, token, { mode: 0o600 }); }
  const clients = new Set();
  const history = [];
  let seq = 0;

  const emit = (ev) => {
    const e = { ...ev, seq: ++seq, t: ev.t ?? Date.now() };
    if (!ev.restored) oplog(e);
    history.push(e);
    if (history.length > MAX_EVENTS) history.splice(0, history.length - MAX_EVENTS);
    const data = `data: ${JSON.stringify(e)}\n\n`;
    for (const c of clients) c.write(data);
  };

  quietDefaults();
  const cpus = Number(process.env.RATCHET_CPUS) || Math.max(2, Math.floor(availableParallelism() / 4));
  const settings = { cpus, dir, mode: process.env.RATCHET_MODE === 'auto' ? 'auto' : 'yolo', mock, slots: 4, workers: 12, agentTimeoutMin: 30, rounds: 0, runAgentBudget: 80, runCostBudget: 0, recent: loadRecent().filter((d) => existsSync(d)) };
  const remember = (d) => {
    settings.recent = [d, ...settings.recent.filter((x) => x !== d)].slice(0, 8);
    try { writeFileSync(STATE, JSON.stringify({ recent: settings.recent })); } catch { /* best effort */ }
  };
  remember(dir);
  applyCpuLimit(settings.cpus);
  const pool = new AccountPool(emit, { mock });
  const swarm = new Swarm({ pool, emit, settings });
  const optimizer = new Optimizer(swarm);
  const forge = new ForgeSwarm(swarm);
  const restored = swarm.restore();
  const say = (text, cls = '') => emit({ kind: 'term', text, cls });

  // Check non-interactive provider status on startup; never send a test prompt automatically.
  if (!mock) for (const a of pool.list().filter((x) => ['claude', 'codex'].includes(x.harness))) {
    void execCommand(`account probe ${a.id}`, { say: () => {}, swarm, pool, settings, optimizer, remember });
  }

  const terminalCommands = new Map();
  const logins = new Map();
  const startLogin = (a) => {
    logins.get(a.id)?.kill();
    const def = LOGIN_CMD[a.harness];
    if (!def || (a.harness === 'antigravity' && !a.configDir)) { emit({ kind: 'login', id: a.id, text: `Sign in from a terminal: ${pool.loginHint(a)}`, done: true, ok: false }); return; }
    if (a.configDir) mkdirSync(a.configDir, { recursive: true });
    emit({ kind: 'login', id: a.id, reset: true, text: 'starting sign-in…' });
    const p = spawn(def[0], def[1], { env: mergeEnv(pool.env(a)), stdio: ['pipe', 'pipe', 'pipe'] });
    logins.set(a.id, p);
    const feed = (d) => {
      const text = String(d).replace(CLEAN, '').replace(/\r/g, '');
      if (!text.trim()) return;
      emit({ kind: 'login', id: a.id, text, url: /https?:\/\/[^\s\x1b]+/.exec(text)?.[0] ?? null });
    };
    p.stdout.on('data', feed); p.stderr.on('data', feed);
    p.on('error', (e) => emit({ kind: 'login', id: a.id, text: `could not start ${def[0]}: ${e.message}`, done: true, ok: false }));
    p.on('close', (code) => {
      logins.delete(a.id);
      a.authed = code === 0; if (a.harness === 'codex' && a.authed) Object.assign(a, codexIdentity(a.configDir)); pool.publish(a);
      emit({ kind: 'login', id: a.id, text: code === 0 ? 'signed in' : `sign-in ended (exit ${code})`, done: true, ok: code === 0 });
      if (code === 0 && ['claude', 'codex'].includes(a.harness)) void execCommand(`account probe ${a.id}`, { say: () => {}, swarm, pool, settings, optimizer, remember });
    });
  };

  // ---- real shell sessions (python pty bridge, streamed over SSE) ----
  const shells = new Map();
  let shellSeq = 0;
  const startShell = (cols, rows) => {
    const id = `sh${++shellSeq}`;
    const p = spawn('python3', ['-I', '-u', join(SRC, 'ratchet_pty.py'), String(cols), String(rows)], { cwd: settings.dir, env: { ...process.env, TERM: 'xterm-256color', COLORTERM: 'truecolor' }, stdio: ['pipe', 'pipe', 'pipe'] });
    const sh = { id, p, buf: [], size: 0, clients: new Set(), alive: true };
    const send = (obj) => { const line = `data: ${JSON.stringify(obj)}\n\n`; for (const c of sh.clients) c.write(line); };
    const onData = (d) => {
      const b64 = Buffer.from(d).toString('base64');
      sh.buf.push(b64); sh.size += b64.length;
      while (sh.size > 400000 && sh.buf.length > 1) sh.size -= sh.buf.shift().length;
      send({ d: b64 });
    };
    p.stdout.on('data', onData); p.stderr.on('data', onData);
    p.on('error', (e) => { sh.alive = false; send({ exit: true, error: `python3 is required for the shell tab: ${e.message}` }); });
    p.on('close', () => { sh.alive = false; send({ exit: true }); setTimeout(() => shells.delete(id), 60000); });
    shells.set(id, sh);
    return sh;
  };

  const guard = (req, res) => {
    const host = (req.headers.host ?? '').split(':')[0];
    if (host !== '127.0.0.1' && host !== 'localhost') { res.writeHead(403).end('bad host'); return false; }
    return true;
  };
  const tokenBuf = Buffer.from(token);
  const authed = (req, url) => {
    const given = Buffer.from(String(req.headers['x-ratchet-token'] ?? url.searchParams.get('t') ?? ''));
    return given.length === tokenBuf.length && timingSafeEqual(given, tokenBuf);
  };

  const server = createServer(async (req, res) => {
    if (!guard(req, res)) return;
    const url = new URL(req.url, 'http://x');
    res.setHeader('x-content-type-options', 'nosniff');
    res.setHeader('referrer-policy', 'no-referrer');
    res.setHeader('x-frame-options', 'DENY');

    // liveness for scripts and the service manager; reveals nothing sensitive
    if (url.pathname === '/api/health') {
      return void res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true, version: VERSION, uptimeSec: Math.round((Date.now() - STARTED) / 1000) }));
    }

    if (url.pathname === '/api/metrics') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      const runs = [...swarm.runs.values()];
      const agents = runs.flatMap((r) => [...r.agents.values()]);
      const accounts = pool.list().map((a) => ({ id: a.id, harness: a.harness, status: a.status, enabled: a.enabled !== false, authed: a.authed, load: a.load, done: a.done, failed: a.failed, tokens: a.usage?.tokens ?? 0, cost: a.usage?.cost ?? 0 }));
      return void res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({
        version: VERSION, uptimeSec: Math.round((Date.now() - STARTED) / 1000), workspace: settings.dir, mode: settings.mode,
        runs: { total: runs.length, running: runs.filter((r) => r.status === 'running').length },
        agents: { total: agents.length },
        usage: runs.reduce((u, r) => ({ tokens: u.tokens + (r.usage?.tokens ?? 0), cost: u.cost + (r.usage?.cost ?? 0) }), { tokens: 0, cost: 0 }),
        accounts,
      }));
    }

    if (url.pathname === '/api/events') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.write(`data: ${JSON.stringify({ kind: 'hello', settings, commands: COMMANDS, harnesses: HARNESSES, modes: Object.entries(MODES).map(([id, m]) => ({ id, label: m.label, summary: m.summary })), seq: 0, t: Date.now() })}\n\n`);
      for (const e of history) res.write(`data: ${JSON.stringify(e)}\n\n`);
      clients.add(res);
      pool.publishAll();
      if (!mock) for (const a of pool.list().filter((x) => ['claude', 'codex'].includes(x.harness) && x.authed == null && !x.checking)) {
        void execCommand(`account probe ${a.id}`, { say: () => {}, swarm, pool, settings, optimizer, remember });
      }
      const ka = setInterval(() => res.write(': ka\n\n'), 20000);
      req.on('close', () => { clearInterval(ka); clients.delete(res); });
      return;
    }

    if (url.pathname === '/api/ls') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      const path = expand(url.searchParams.get('path') || settings.dir);
      try {
        if (!statSync(path).isDirectory()) throw new Error('not a folder');
        const dirs = readdirSync(path, { withFileTypes: true })
          .filter((d) => d.isDirectory() && (url.searchParams.get('hidden') === '1' || !d.name.startsWith('.')))
          .map((d) => d.name).sort((a, b) => a.localeCompare(b));
        const MARK = [['.git', 'git'], ['package.json', 'node'], ['pyproject.toml', 'python'], ['requirements.txt', 'python'], ['Cargo.toml', 'rust'], ['go.mod', 'go'], ['ratchet.json', 'ratchet'], ['.ratchet', 'runs']];
        const meta = {};
        for (const d of dirs.slice(0, 400)) {
          const tags = MARK.filter(([f]) => existsSync(join(path, d, f))).map(([, tag]) => tag);
          if (tags.length) meta[d] = [...new Set(tags)];
        }
        const here = [...new Set(MARK.filter(([f]) => existsSync(join(path, f))).map(([, tag]) => tag))];
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ path, parent: dirname(path), dirs, meta, here, home: homedir() }));
      } catch (e) { res.writeHead(400, { 'content-type': 'application/json' }).end(JSON.stringify({ error: e.message, path })); }
      return;
    }

    if (url.pathname === '/api/account/model' && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = ''; for await (const c of req) { body += c; if (body.length > 4096) return void res.writeHead(413).end(); }
      let j; try { j = JSON.parse(body); } catch { return void res.writeHead(400).end(); }
      if (typeof j.id !== 'string' || typeof j.model !== 'string') return void res.writeHead(400).end();
      try { if (!pool.setModel(j.id, j.model.trim())) return void res.writeHead(404).end(); }
      catch (e) { return void res.writeHead(400).end(e.message); }
      return void res.writeHead(204).end();
    }

    if ((url.pathname === '/api/login' || url.pathname === '/api/login/input') && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = '';
      for await (const c of req) { body += c; if (body.length > 1e5) return void res.writeHead(413).end(); }
      let j = {};
      try { j = JSON.parse(body); } catch { return void res.writeHead(400).end(); }
      const a = pool.get(String(j.id));
      if (!a) return void res.writeHead(404).end();
      if (url.pathname === '/api/login') startLogin(a);
      else logins.get(a.id)?.stdin.write(`${String(j.text ?? '').trim()}\n`);
      return void res.writeHead(204).end();
    }

    if (url.pathname === '/api/forge/tasks' && req.method === 'GET') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      return void res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ tasks: forge.listTasks() }));
    }

    if (url.pathname === '/api/forge' && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = ''; for await (const c of req) { body += c; if (body.length > 20000) return void res.writeHead(413).end(); }
      let j; try { j = JSON.parse(body); } catch { return void res.writeHead(400).end('Invalid request'); }
      if (!Array.isArray(j.tasks) || !j.tasks.length || j.tasks.length > 12 || j.tasks.some((t) => typeof t !== 'string' || !t.trim()) || !['check', 'start'].includes(j.action)) return void res.writeHead(400).end('Select one or more tasks.');
      if (j.device != null && typeof j.device !== 'string') return void res.writeHead(400).end('Invalid device.');
      try {
        if (j.action === 'check') {
          if (j.tasks.length !== 1) return void res.writeHead(400).end('Check one task at a time.');
          const result = await forge.check(j.tasks[0], settings.dir, j.device);
          return void res.writeHead(result.ok ? 200 : 400, { 'content-type': 'text/plain; charset=utf-8' }).end(result.ok ? result.output : result.error);
        }
        const run = forge.start(j.tasks, j.device);
        return void res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ runId: run.id }));
      } catch (e) { return void res.writeHead(400, { 'content-type': 'text/plain; charset=utf-8' }).end(e.message); }
    }

    if (url.pathname === '/api/agent' && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = '';
      for await (const c of req) { body += c; if (body.length > 2e5) return void res.writeHead(413).end(); }
      let j = {};
      try { j = JSON.parse(body); } catch { return void res.writeHead(400).end(); }
      if (!String(j.prompt ?? '').trim()) return void res.writeHead(400).end('prompt required');
      if (!pool.list().length) return void res.writeHead(409).end('no accounts');
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(swarm.adhoc({ harness: String(j.harness ?? 'any'), account: j.account ? String(j.account) : null, title: String(j.title ?? '').slice(0, 80), prompt: String(j.prompt) })));
      return;
    }

    if (url.pathname === '/api/shell/stream') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      const sh = shells.get(url.searchParams.get('id') ?? '');
      if (!sh) return void res.writeHead(404).end();
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for (const b of sh.buf) res.write(`data: ${JSON.stringify({ d: b })}\n\n`);
      if (!sh.alive) res.write(`data: ${JSON.stringify({ exit: true })}\n\n`);
      sh.clients.add(res);
      const ka = setInterval(() => res.write(': ka\n\n'), 20000);
      req.on('close', () => { clearInterval(ka); sh.clients.delete(res); });
      return;
    }

    if (url.pathname.startsWith('/api/shell/') && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = '';
      for await (const c of req) { body += c; if (body.length > 1e6) return void res.writeHead(413).end(); }
      let j = {};
      try { j = JSON.parse(body || '{}'); } catch { return void res.writeHead(400).end(); }
      const op = url.pathname.slice('/api/shell/'.length);
      const clamp = (n, lo, hi, d) => (Number.isFinite(Number(n)) ? Math.min(hi, Math.max(lo, Math.floor(Number(n)))) : d);
      if (op === 'start') {
        const existing = j.id && shells.get(String(j.id));
        const sh = existing?.alive ? existing : startShell(clamp(j.cols, 20, 400, 120), clamp(j.rows, 5, 200, 30));
        return void res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ id: sh.id, cwd: settings.dir }));
      }
      const sh = shells.get(String(j.id ?? ''));
      if (!sh?.alive) return void res.writeHead(404).end();
      if (op === 'input') sh.p.stdin.write(String(j.data ?? ''));
      else if (op === 'resize') sh.p.stdin.write(`\x00\x1fR${clamp(j.cols, 20, 400, 120)},${clamp(j.rows, 5, 200, 30)}\n`);
      else if (op === 'kill') sh.p.kill('SIGTERM');
      else return void res.writeHead(404).end();
      return void res.writeHead(204).end();
    }

    if ((url.pathname === '/api/cmd' || url.pathname === '/api/cmd/interrupt') && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = '';
      for await (const c of req) { body += c; if (body.length > 1e5) return void res.writeHead(413).end(); }
      let request;
      try { request = JSON.parse(body); } catch { return void res.writeHead(400).end(); }
      const clientId = typeof request.clientId === 'string' && /^[a-zA-Z0-9-]{1,80}$/.test(request.clientId) ? request.clientId : null;
      if (url.pathname === '/api/cmd/interrupt') {
        if (!clientId) return void res.writeHead(400).end();
        const controller = terminalCommands.get(clientId);
        if (controller) controller.abort();
        return void res.writeHead(controller ? 204 : 404).end();
      }
      const line = String(request.line ?? '');
      if (clientId && terminalCommands.has(clientId)) return void res.writeHead(409).end('A command is already running');
      const controller = new AbortController();
      if (clientId) terminalCommands.set(clientId, controller);
      say(`❯ ${line}`, 'echo');
      try { await execCommand(line, { say, swarm, pool, settings, optimizer, applyCpuLimit, forge, remember, signal: controller.signal }); }
      catch (e) { say(`error: ${e.message}`, 'err'); }
      finally { if (clientId && terminalCommands.get(clientId) === controller) terminalCommands.delete(clientId); }
      return void res.writeHead(204).end();
    }

    if (VENDOR[url.pathname]) {
      const [pkg, rel] = VENDOR[url.pathname];
      const base = pkgDir(pkg);
      if (!base) return void res.writeHead(404).end('vendor package missing');
      res.writeHead(200, { 'content-type': `${MIME[extname(rel)] ?? 'text/plain'}; charset=utf-8`, 'cache-control': 'max-age=3600' });
      return void res.end(readFileSync(join(base, rel)));
    }

    const file = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
    if (file === 'index.html') res.setHeader('content-security-policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'");
    const p = join(UI, file);
    if (!p.startsWith(UI) || !existsSync(p)) return void res.writeHead(404).end('not found');
    res.writeHead(200, { 'content-type': `${MIME[extname(p)] ?? 'text/plain'}; charset=utf-8`, 'cache-control': 'no-store' });
    res.end(readFileSync(p));
  });

  await new Promise((ok, fail) => { server.once('error', fail); server.listen(port, '127.0.0.1', ok); });
  const link = `http://127.0.0.1:${port}/?t=${token}`;
  try { writeFileSync(join(HOME, 'url'), `${link}\n`, { mode: 0o600 }); } catch { /* best effort */ }
  console.log(`\n  RATCHET  ${mock ? '(mock agents)' : ''}\n  workspace  ${dir}\n  open       ${link}\n`);
  if (open) spawn('xdg-open', [link], { stdio: 'ignore', detached: true }).on('error', () => {}).unref();

  say(`workspace ${dir}`, 'dim');
  if (restored) say(`loaded ${restored} earlier run(s) from this workspace; interrupted ones can be resumed`, 'dim');
  say('type run <task> to start research, or help for commands', 'dim');
  const shutdown = () => {
    swarm.stop('all'); server.close();
    for (const sh of shells.values()) { try { sh.p.kill('SIGTERM'); } catch { /* gone */ } }
    for (const f of ['url', 'pid']) { try { if (f === 'url' && !readFileSync(join(HOME, 'url'), 'utf8').includes(token)) continue; rmSync(join(HOME, f), { force: true }); } catch { /* gone */ } }
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  return server;
}
