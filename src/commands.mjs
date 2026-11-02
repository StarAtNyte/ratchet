import { codexIdentity } from './accounts.mjs';
import { spawn } from 'node:child_process';
import { existsSync, statSync, readdirSync, readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { homedir } from 'node:os';
import { mkdirSync } from 'node:fs';
import { HARNESSES, mergeEnv } from './harness.mjs';
import { TEMPLATE } from './optimizer.mjs';
import { MODES, MODE_IDS, parseMode } from './modes.mjs';
import { writeFileSync } from 'node:fs';

export const COMMANDS = ['help', 'program', 'scores', 'run', 'modes', 'research', 'math', 'score', 'build', 'follow', 'resume', 'opt', 'kernel', 'forge', 'dir', 'accounts', 'account', 'runs', 'stop', 'report', 'mode', 'set', 'clear'];

const HELP = [
  ['run <task>', 'launch a research swarm'],
  ['follow <instruction>', 'queue an instruction for the active swarm’s next agent turns'],
  ['cd [path]', 'change the current workspace directory'],
  ['<command>', 'run a non-interactive shell command in the workspace'],
  ['kernel list | check <task> | run <task> [task...]', 'run kernel optimization campaigns'],
  ['opt init|start [goal]|apply', 'evolutionary optimise loop on ratchet.json (archive, ratchet, holdout)'],
  ['dir [path|n]', 'show/change workspace (also: click the path in the header); dir recent, dir mk <path>'],
  ['accounts', 'list the account pool'],
  ['account add <claude|codex|antigravity>', 'create an isolated account slot and show how to log in'],
  ['account login <id>', 'run the login flow for an account'],
  ['account probe [id]', 'check which accounts are logged in'],
  ['account enable|disable <id>', 'include or exclude an account from new work'],
  ['account rm <id>', 'remove an account'],
  ['run [-m mode] <task>', `start a run in a mode (${MODE_IDS.join(', ')}); research is the default`],
  ['research | math | score | build <task>', 'shortcuts for run -m <mode>'],
  ['modes', 'what each mode does'],
  ['program [init]', 'show or create program.md, the research instructions every agent follows'],
  ['runs', 'list runs (earlier sessions included)'],
  ['scores', 'quality score of each research round (rubric judge), to compare rounds and settings'],
  ['resume <runId>', 're-run an interrupted or failed research run; finished agents are restored, not re-run'],
  ['stop [runId|all]', 'abort a run'],
  ['report [runId]', 'print a run report'],
  ['mode <yolo|auto>', 'yolo (default) = full permissions: codex --yolo, claude/agy --dangerously-skip-permissions; auto = sandboxed workspace-write'],
  ['set slots|workers|timeout|rounds|budget|cost|cpus <n>', 'jobs per account / lines of attack / agent time limit (min) / research rounds (0 = until stopped) / agents per run / USD per run / CPU cores Ratchet may use'],
  ['clear', 'clear console and activity output in this browser'],
];

/** Execute one terminal line. `say(text, cls)` writes to the web terminal. */
export async function execCommand(line, { say, swarm, pool, settings, optimizer, applyCpuLimit, forge, remember, signal }) {
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

    case 'kernel':
    case 'forge': {
      const [sub, ...args] = rest;
      let tasks = args; let device = null;
      if (args[0] === '--device') { device = args[1]; tasks = args.slice(2); if (!device) return say('usage: --device modal:<GPU> | local:<index>', 'err'); }
      if (!forge) return say('ForgeKernel integration is unavailable.', 'err');
      if (sub === 'list' && !tasks.length) { const available = forge.listTasks(); return say(available.length ? available.map((task) => `  ${task.value}`).join('\n') : 'No built-in kernel tasks found.', 'dim'); }
      if (sub === 'status' && !tasks.length) return say(`Ratchet kernel engine: ${forge.root} · runtime: ${forge.binary}`, 'dim');
      if (sub === 'setup' && !tasks.length) { try { await forge.ensureRuntime(); return say('Ratchet kernel runtime is ready.', 'ok'); } catch (e) { return say(e.message, 'err'); } }
      if (!tasks.length || !['run', 'start', 'check'].includes(sub)) return say('usage: kernel list | setup | status | check [--device <device>] <task> | run [--device <device>] <task> [task...]', 'err');
      if (sub === 'check') { try { const result = await forge.check(tasks.join(' '), settings.dir, device); return say(result.ok ? result.output : result.error, result.ok ? 'ok' : 'err'); } catch (e) { return say(e.message, 'err'); } }
      try { const run = forge.start(tasks, device); return say(`Kernel optimization swarm ${run.id} launched with ${tasks.length} campaign(s).`, 'ok'); } catch (e) { return say(e.message, 'err'); }
    }

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

    case 'cd':
    case 'dir': {
      if (cmd === 'dir' && rest[0] === 'recent') { settings.recent.forEach((d, i) => say(`  ${i + 1}  ${d}`, 'dim')); return; }
      if (arg || cmd === 'cd') {
        const make = cmd === 'dir' && rest[0] === 'mk';
        const dest = arg.replace(/^(?:"([^"]*)"|'([^']*)')$/, (_, a, b) => a ?? b);
        const raw = cmd === 'cd' && !arg ? homedir() : make ? rest.slice(1).join(' ') : cmd === 'dir' && /^\d$/.test(arg) && settings.recent[Number(arg) - 1] ? settings.recent[Number(arg) - 1] : dest;
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
      if (sub === 'enable' || sub === 'disable') {
        return say(pool.setEnabled(a1, sub === 'enable') ? `${a1} ${sub === 'enable' ? 'included in' : 'excluded from'} new assignments` : `no such account: ${a1}`, 'dim');
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
          a.checking = true; pool.publish(a);
          const p = spawn(bin, args, { env: mergeEnv(pool.env(a)), stdio: ['ignore', 'pipe', 'ignore'] });
          let output = ''; let timedOut = false; let finished = false;
          p.stdout.on('data', (chunk) => { if (output.length < 65536) output += chunk; });
          const t = setTimeout(() => { timedOut = true; p.kill(); }, a.harness === 'antigravity' ? 90000 : 15000);
          const finish = (code, missing = false) => {
            if (finished) return; finished = true; clearTimeout(t); a.checking = false;
            if (timedOut || missing) { a.authed = null; a.authMessage = missing ? 'Provider CLI not found' : 'Check timed out'; }
            else {
              a.authed = code === 0; a.authMessage = null;
              if (a.harness === 'codex') Object.assign(a, codexIdentity(a.configDir));
              if (a.harness === 'claude') {
                try { const info = JSON.parse(output); if (typeof info.loggedIn === 'boolean') a.authed = info.loggedIn; a.email = typeof info.email === 'string' ? info.email : null; a.plan = typeof info.subscriptionType === 'string' ? info.subscriptionType : null; } catch {}
              }
            }
            pool.publish(a); say(`  ${a.id}: ${a.authMessage || (a.authed ? 'signed in' : 'not signed in')}`, a.authed ? 'ok' : 'warn'); done();
          };
          p.on('error', () => finish(null, true));
          p.on('close', (code) => finish(code));
        })));
        return;
      }
      return say('usage: account add|rm|login|probe', 'err');
    }

    case 'follow': {
      if (!arg) return say('usage: follow <instruction>', 'err');
      try { const id = swarm.followup(arg); return say(`Instruction queued for ${id}. New agent turns will receive it; agents already running are not interrupted.`, 'ok'); } catch (e) { return say(e.message, 'err'); }
    }

    case 'resume': {
      if (!arg) return say('usage: resume <runId>  (see: runs)', 'err');
      try { const { run, restored } = swarm.resume(arg); say(`resumed ${run.id}: ${restored} finished agents restored, the rest re-run`, 'ok'); }
      catch (e) { say(e.message, 'err'); }
      return;
    }

    case 'scores': {
      let rows = [];
      try { rows = readFileSync(join(settings.dir, '.ratchet', 'scores.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)); } catch { /* none yet */ }
      if (!rows.length) return say('no scored rounds yet in this workspace', 'dim');
      say('  when              run        round  overall  accuracy  citations  agents  task', 'dim');
      for (const r of rows.slice(-20)) say(`  ${new Date(r.t).toLocaleString().padEnd(17)} ${String(r.run).padEnd(10)} ${String(r.round).padStart(5)}  ${Number(r.overall).toFixed(2).padStart(7)}  ${Number(r.accuracy ?? 0).toFixed(2).padStart(8)}  ${Number(r.citations ?? 0).toFixed(2).padStart(9)}  ${String(r.agents).padStart(6)}  ${String(r.task).slice(0, 40)}`);
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
      if (k === 'cpus') {
        if (!(n >= 1 && n <= 1024)) return say('usage: set cpus <n>  (cores Ratchet and its agents may use)', 'err');
        settings.cpus = n;
        const r = applyCpuLimit?.(n);
        swarm.emit({ kind: 'settings', settings: { ...settings } });
        return say(r ? `Ratchet and its agents now use cores ${r.list} (${r.pinned} processes pinned)` : `cpu budget ${n} (pinning unavailable on this system)`, 'ok');
      }
      if (k === 'budget') { if (!(n >= 0 && n <= 1000)) return say('usage: set budget <0-1000 agents per run>  (0 = no cap)', 'err'); settings.runAgentBudget = n; swarm.emit({ kind: 'settings', settings: { ...settings } }); return say(n ? `runs stop opening agents after ${n}` : 'no agent cap per run', 'ok'); }
      if (k === 'cost') { if (!(n >= 0 && n <= 10000)) return say('usage: set cost <USD per run>  (0 = no cap; only providers that report cost count)', 'err'); settings.runCostBudget = n; swarm.emit({ kind: 'settings', settings: { ...settings } }); return say(n ? `runs stop at $${n} of reported cost` : 'no cost cap per run', 'ok'); }
      if (k === 'timeout') { if (!(n >= 1 && n <= 600)) return say('usage: set timeout <1-600 minutes>', 'err'); settings.agentTimeoutMin = n; swarm.emit({ kind: 'settings', settings: { ...settings } }); return say(`agents stop after ${n} min`, 'ok'); }
      if (!['slots', 'workers'].includes(k) || !(n >= 1 && n <= 32)) return say('usage: set slots|workers|timeout <n>', 'err');
      settings[k] = n;
      swarm.emit({ kind: 'settings', settings: { ...settings } });
      return say(`${k} = ${n}`, 'ok');
    }

    case 'program': {
      const f = join(settings.dir, 'program.md');
      if (rest[0] === 'init') {
        if (existsSync(f)) return say(`${f} already exists`, 'warn');
        writeFileSync(f, `# Research program\n\nEdit this file to steer every agent Ratchet starts in this workspace.\n\n## Goal\nWhat counts as success, and the one number or result that measures it.\n\n## Constraints\n- What must not change (files, interfaces, data).\n- Time or compute limits per experiment.\n\n## Ideas to try\n- Directions you believe in, papers to read first.\n\n## Avoid\n- Approaches that are known dead ends here.\n`);
        return say(`wrote ${f}; every agent now reads it`, 'ok');
      }
      if (!existsSync(f)) return say('no program.md in this workspace. Create one with: program init', 'dim');
      readFileSync(f, 'utf8').split('\n').slice(0, 40).forEach((l) => say(l, 'dim'));
      return;
    }

    case 'modes':
      for (const [id, m] of Object.entries(MODES)) say(`  ${id.padEnd(9)} ${m.label}: ${m.summary}`, 'dim');
      say('  start one with: run -m <mode> <task>   (kernel campaigns on a task folder: kernel run <task>)', 'dim');
      return;

    case 'research': case 'math': case 'score': case 'build':
    case 'run': {
      let parsed;
      try { parsed = parseMode(cmd === 'run' ? arg : `-m ${cmd} ${arg}`); } catch (e) { return say(e.message, 'err'); }
      const { mode, task } = parsed;
      if (!task) return say(`usage: ${cmd === 'run' ? 'run [-m mode]' : cmd} <task>`, 'err');
      if (!pool.list().length) return say('no accounts. try: account add claude', 'err');
      // score mode hands over to the optimise loop once a fixed evaluator exists
      if (mode === 'score' && existsSync(join(settings.dir, 'ratchet.json'))) {
        try { const r = await optimizer.start(task); say(`ratchet.json found: optimise run ${r.id} launched (evolutionary search with explorers)`, 'ok'); } catch (e) { say(e.message, 'err'); }
        return;
      }
      const r = swarm.start(task, undefined, mode);
      say(`${MODES[mode].label} run ${r.id} launched in ${settings.dir}${settings.rounds ? '' : '; it keeps going round after round until you stop it'}`, 'ok');
      return;
    }
    default: {
      // Ordinary commands run in the workspace; only the explicit `run` command starts research.
      return await new Promise((done) => {
        const shell = process.env.SHELL || '/bin/sh';
        let finished = false;
        const p = spawn(shell, ['-c', text], { cwd: settings.dir, env: mergeEnv(), detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'] });
        const interrupt = () => { try { if (process.platform !== 'win32') process.kill(-p.pid, 'SIGINT'); else p.kill('SIGINT'); } catch {} };
        if (signal?.aborted) interrupt(); else signal?.addEventListener('abort', interrupt, { once: true });
        const output = (stream, cls) => {
          let pending = '';
          stream.setEncoding('utf8');
          stream.on('data', (chunk) => { pending += chunk; const lines = pending.split('\n'); pending = lines.pop(); for (const line of lines) say(line, cls); });
          stream.on('end', () => { if (pending) say(pending, cls); });
        };
        output(p.stdout, ''); output(p.stderr, 'err');
        const timer = setTimeout(() => { try { if (process.platform !== 'win32') process.kill(-p.pid, 'SIGTERM'); else p.kill('SIGTERM'); } catch {} say('Command timed out after 2 minutes.', 'warn'); }, 120000);
        const finish = () => { if (finished) return; finished = true; clearTimeout(timer); signal?.removeEventListener('abort', interrupt); done(); };
        p.on('error', (error) => { say(`Could not run command: ${error.message}`, 'err'); finish(); });
        p.on('close', (code, signal) => { if (code && !signal) say(`Command exited with status ${code}.`, 'err'); finish(); });
      });
    }
  }
}
