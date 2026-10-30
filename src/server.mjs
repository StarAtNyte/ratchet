import { createServer } from 'node:http';
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync, mkdirSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { dirname, join, extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { AccountPool, HOME } from './accounts.mjs';
import { Swarm } from './swarm.mjs';
import { Optimizer } from './optimizer.mjs';
import { HARNESSES, mergeEnv } from './harness.mjs';
const CLEAN = /\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[A-Za-z]/g;
const LOGIN_CMD = { claude: ['claude', ['auth', 'login']], codex: ['codex', ['login', '--device-auth']], antigravity: ['agy', ['--print=Reply with OK']] };
import { execCommand, COMMANDS } from './commands.mjs';

const UI = join(dirname(fileURLToPath(import.meta.url)), '..', 'ui');
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
    const e = { ...ev, seq: ++seq, t: Date.now() };
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
      a.authed = code === 0; pool.publish(a);
      emit({ kind: 'login', id: a.id, text: code === 0 ? 'signed in' : `sign-in ended (exit ${code})`, done: true, ok: code === 0 });
    });
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
        res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ path, parent: dirname(path), dirs, home: homedir() }));
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

    if (url.pathname === '/api/cmd' && req.method === 'POST') {
      if (!authed(req, url)) return void res.writeHead(401).end();
      let body = '';
      for await (const c of req) { body += c; if (body.length > 1e5) return void res.writeHead(413).end(); }
      let line = '';
      try { line = String(JSON.parse(body).line ?? ''); } catch { return void res.writeHead(400).end(); }
      res.writeHead(204).end();
      say(`❯ ${line}`, 'echo');
      try { await execCommand(line, { say, swarm, pool, settings, optimizer, remember }); }
      catch (e) { say(`error: ${e.message}`, 'err'); }
      return;
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
  say('type a research task and press enter, or "help"', 'dim');
  const shutdown = () => {
    swarm.stop('all'); server.close();
    for (const f of ['url', 'pid']) { try { if (f === 'url' && !readFileSync(join(HOME, 'url'), 'utf8').includes(token)) continue; rmSync(join(HOME, f), { force: true }); } catch { /* gone */ } }
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  return server;
}
