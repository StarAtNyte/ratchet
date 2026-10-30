import { createServer } from 'node:http';
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync, mkdirSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { dirname, join, extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { AccountPool, HOME, codexIdentity } from './accounts.mjs';
import { Swarm } from './swarm.mjs';
import { Optimizer } from './optimizer.mjs';
import { HARNESSES, mergeEnv } from './harness.mjs';
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
    history.push(e);
    if (history.length > MAX_EVENTS) history.splice(0, history.length - MAX_EVENTS);
    const data = `data: ${JSON.stringify(e)}\n\n`;
    for (const c of clients) c.write(data);
  };

  const settings = { dir, mode: process.env.RATCHET_MODE === 'auto' ? 'auto' : 'yolo', mock, slots: 2, workers: 6, agentTimeoutMin: 30, rounds: 0, recent: loadRecent().filter((d) => existsSync(d)) };
  const remember = (d) => {
    settings.recent = [d, ...settings.recent.filter((x) => x !== d)].slice(0, 8);
    try { writeFileSync(STATE, JSON.stringify({ recent: settings.recent })); } catch { /* best effort */ }
  };
  remember(dir);
  const pool = new AccountPool(emit, { mock });
  const swarm = new Swarm({ pool, emit, settings });
  const optimizer = new Optimizer(swarm);
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
  const authed = (req, url) => (req.headers['x-ratchet-token'] ?? url.searchParams.get('t')) === token;

  const server = createServer(async (req, res) => {
    if (!guard(req, res)) return;
    const url = new URL(req.url, 'http://x');

    if (url.pathname === '/api/events') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.write(`data: ${JSON.stringify({ kind: 'hello', settings, commands: COMMANDS, harnesses: HARNESSES, seq: 0, t: Date.now() })}\n\n`);
      for (const e of history) res.write(`data: ${JSON.stringify(e)}\n\n`);
      pool.publishAll();
      clients.add(res);
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
      try { await execCommand(line, { say, swarm, pool, settings, optimizer, remember, signal: controller.signal }); }
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
