import { spawn } from 'node:child_process';
import { existsSync, statSync, readdirSync, readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { homedir } from 'node:os';
import { mkdirSync } from 'node:fs';
import { HARNESSES, mergeEnv } from './harness.mjs';
import { TEMPLATE } from './optimizer.mjs';
import { writeFileSync } from 'node:fs';

export const COMMANDS = ['help', 'run', 'resume', 'opt', 'dir', 'accounts', 'account', 'runs', 'stop', 'report', 'mode', 'set', 'clear'];

const HELP = [
  ['run <task>', 'launch a swarm on the task (bare text does the same)'],
  ['opt init|start [goal]|apply', 'evolutionary optimise loop on ratchet.json (archive, ratchet, holdout)'],
  ['dir [path|n]', 'show/change workspace (also: click the path in the header); dir recent, dir mk <path>'],
  ['accounts', 'list the account pool'],
  ['account add <claude|codex|antigravity>', 'create an isolated account slot and show how to log in'],
  ['account login <id>', 'run the login flow for an account'],
  ['account probe [id]', 'check which accounts are logged in'],
  ['account rm <id>', 'remove an account'],
  ['runs', 'list runs (earlier sessions included)'],
  ['resume <runId>', 're-run an interrupted or failed research run; finished agents are restored, not re-run'],
  ['stop [runId|all]', 'abort a run'],
  ['report [runId]', 'print a run report'],
  ['mode <yolo|auto>', 'yolo (default) = full permissions: codex --yolo, claude/agy --dangerously-skip-permissions; auto = sandboxed workspace-write'],
  ['set slots|workers|timeout|rounds <n>', 'jobs per account / lines of attack / agent time limit (min) / research rounds (0 = until stopped)'],
  ['clear', 'clear the terminal'],
];

/** Execute one terminal line. `say(text, cls)` writes to the web terminal. */
export async function execCommand(line, { say, swarm, pool, settings, optimizer, remember }) {
  const text = line.trim();
  if (!text) return;
  const [cmd, ...rest] = text.split(/\s+/);
  const arg = rest.join(' ');

  switch (cmd) {
    case 'help':
      HELP.forEach(([c, d]) => say(`  ${c.padEnd(28)} ${d}`, 'dim'));
      return;

    case 'clear':
      return;

    case 'opt': {
      const [sub, ...g] = rest;
      if (sub === 'init') {
        const f = join(settings.dir, 'ratchet.json');
        if (existsSync(f)) return say(`${f} already exists`, 'warn');
        writeFileSync(f, JSON.stringify(TEMPLATE, null, 2));
        return say(`wrote ${f}: set targets, eval (last stdout line = {"score":n,"ok":true}), direction`, 'ok');
      }
      if (sub === 'apply') return say(optimizer.apply(), 'ok');
      if (sub === 'start') {
        if (!pool.list().length) return say('no accounts. try: account add claude', 'err');
        try { const r = await optimizer.start(g.join(' ')); say(`optimise run ${r.id} launched`, 'ok'); } catch (e) { say(e.message, 'err'); }
        return;
      }
      return say('usage: opt init | opt start [goal] | opt apply', 'err');
    }

    case 'dir': {
      if (rest[0] === 'recent') { settings.recent.forEach((d, i) => say(`  ${i + 1}  ${d}`, 'dim')); return; }
      if (arg) {
        const make = rest[0] === 'mk';
        const raw = make ? rest.slice(1).join(' ') : /^\d$/.test(arg) && settings.recent[Number(arg) - 1] ? settings.recent[Number(arg) - 1] : arg;
        const p = resolve(settings.dir, raw.replace(/^~(?=$|\/)/, homedir()));
        if (make) mkdirSync(p, { recursive: true });
        if (!existsSync(p) || !statSync(p).isDirectory()) return say(`not a folder: ${p}  (dir mk <path> creates it)`, 'err');
        settings.dir = p;
        remember?.(p);
        swarm.emit({ kind: 'settings', settings: { ...settings } });
        if ([...swarm.runs.values()].some((r) => r.status === 'running')) say('active runs keep their old folder; new runs use this one', 'warn');
      }
      say(settings.dir, 'ok');
      return;
    }

    case 'accounts':
      for (const a of pool.list()) {
        pool.refresh(a);
        const lim = a.limitedUntil ? ` back ${new Date(a.limitedUntil).toLocaleTimeString()}` : '';
        say(`  ${a.id.padEnd(10)} ${a.status.padEnd(8)} load ${a.load}  done ${a.done}  fail ${a.failed}${lim}  ${a.configDir ?? '(default login)'}`, a.status === 'limited' ? 'warn' : 'dim');
      }
      return;

    case 'account': {
      const [sub, a1] = rest;
      if (sub === 'add') {
        if (!HARNESSES.includes(a1)) return say(`usage: account add <${HARNESSES.join('|')}>`, 'err');
        const a = pool.add(a1);
        say(`created ${a.id}${a.configDir ? ` -> ${a.configDir}` : ''}`, 'ok');
        if (a.configDir) say(`sign in from the account panel, or:  ${pool.loginHint(a)}`, 'dim');
        return;
      }
      if (sub === 'rm') return say(pool.remove(a1) ? `removed ${a1}` : `no such account: ${a1}`, pool.get(a1) ? 'ok' : 'err');
      if (sub === 'login') {
        const a = pool.get(a1);
        if (!a) return say(`no such account: ${a1}`, 'err');
        if (a.harness !== 'codex') return say(`run in a shell:  ${pool.loginHint(a)}`, 'warn');
        say(`starting codex device login for ${a.id}…`, 'dim');
        const p = spawn('codex', ['login', '--device-auth'], { env: { ...process.env, ...pool.env(a) } });
        const pipe = (d) => String(d).split('\n').filter(Boolean).forEach((l) => say(`  ${l}`, 'dim'));
        p.stdout.on('data', pipe); p.stderr.on('data', pipe);
        p.on('error', (e) => say(`login failed: ${e.message}`, 'err'));
        p.on('close', (c) => say(c === 0 ? `${a.id} logged in` : `login exited ${c}`, c === 0 ? 'ok' : 'err'));
        return;
      }
      if (sub === 'probe') {
        const targets = a1 ? [pool.get(a1)].filter(Boolean) : pool.list();
        await Promise.all(targets.map((a) => new Promise((done) => {
          if (!['claude', 'codex', 'antigravity'].includes(a.harness)) { a.authed = null; say(`  ${a.id}: no status command; run a task to verify`, 'dim'); return done(); }
          // antigravity has no status command: a one-line prompt is the only reliable check
          const [bin, args] = a.harness === 'claude' ? ['claude', ['auth', 'status']] : a.harness === 'codex' ? ['codex', ['login', 'status']] : ['agy', ['--print=Reply with OK']];
          const p = spawn(bin, args, { env: mergeEnv(pool.env(a)), stdio: 'ignore' });
          const t = setTimeout(() => p.kill(), a.harness === 'antigravity' ? 90000 : 15000);
          p.on('error', () => { clearTimeout(t); a.authed = false; pool.publish(a); say(`  ${a.id}: cli missing`, 'err'); done(); });
          p.on('close', (c) => { clearTimeout(t); a.authed = c === 0; pool.publish(a); say(`  ${a.id}: ${a.authed ? 'logged in' : 'NOT logged in'}`, a.authed ? 'ok' : 'warn'); done(); });
        })));
        return;
      }
      return say('usage: account add|rm|login|probe', 'err');
    }

    case 'resume': {
      if (!arg) return say('usage: resume <runId>  (see: runs)', 'err');
      try { const { run, restored } = swarm.resume(arg); say(`resumed ${run.id}: ${restored} finished agents restored, the rest re-run`, 'ok'); }
      catch (e) { say(e.message, 'err'); }
      return;
    }

    case 'runs':
      if (!swarm.runs.size) return say('no runs yet', 'dim');
      for (const r of swarm.runs.values()) say(`  ${r.id}  ${r.status.padEnd(8)} ${r.agents.size} agents  ${r.claims.length} claims  ${r.task.slice(0, 60)}`, 'dim');
      return;

    case 'stop': {
      const n = swarm.stop(arg || 'all');
      return say(n ? `stopping ${arg || 'all'}…` : 'nothing to stop', n ? 'warn' : 'dim');
    }

    case 'report': {
      const r = arg ? swarm.runs.get(arg) : [...swarm.runs.values()].at(-1);
      if (r?.report) return r.report.split('\n').forEach((l) => say(l));
      const dir = join(settings.dir, '.ratchet', 'runs');
      if (existsSync(dir)) {
        const last = readdirSync(dir).sort().at(-1);
        const f = last && join(dir, last, 'REPORT.md');
        if (f && existsSync(f)) return readFileSync(f, 'utf8').split('\n').forEach((l) => say(l));
      }
      return say('no report yet', 'dim');
    }

    case 'mode':
      if (!['auto', 'yolo'].includes(arg)) return say(`mode is ${settings.mode}; usage: mode <auto|yolo>`, 'dim');
      settings.mode = arg;
      swarm.emit({ kind: 'settings', settings: { ...settings } });
      return say(`mode ${arg}${arg === 'yolo' ? '  (agents run without a sandbox)' : ''}`, arg === 'yolo' ? 'warn' : 'ok');

    case 'set': {
      const [k, v] = rest;
      const n = Number(v);
      if (k === 'rounds') { if (!(n >= 0 && n <= 1000)) return say('usage: set rounds <0-1000>  (0 = keep going until stopped)', 'err'); settings.rounds = n; swarm.emit({ kind: 'settings', settings: { ...settings } }); return say(n ? `research runs stop after ${n} round(s)` : 'research runs keep going until you stop them', 'ok'); }
      if (k === 'timeout') { if (!(n >= 1 && n <= 600)) return say('usage: set timeout <1-600 minutes>', 'err'); settings.agentTimeoutMin = n; swarm.emit({ kind: 'settings', settings: { ...settings } }); return say(`agents stop after ${n} min`, 'ok'); }
      if (!['slots', 'workers'].includes(k) || !(n >= 1 && n <= 32)) return say('usage: set slots|workers|timeout <n>', 'err');
      settings[k] = n;
      swarm.emit({ kind: 'settings', settings: { ...settings } });
      return say(`${k} = ${n}`, 'ok');
    }

    case 'run':
    default: {
      const task = cmd === 'run' ? arg : text;
      if (!task) return say('usage: run <task>', 'err');
      if (!pool.list().length) return say('no accounts. try: account add claude', 'err');
      const r = swarm.start(task);
      say(`run ${r.id} launched in ${settings.dir}${settings.rounds ? '' : '; it keeps going round after round until you type stop'}`, 'ok');
    }
  }
}
