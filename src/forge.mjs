import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve, delimiter } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LIMIT_RE, parseReset } from './harness.mjs';
import { homedir } from 'node:os';
import { HOME } from './accounts.mjs';

// Explicit overrides affect one campaign only; omitted means the checkout's configuration.
export function deviceEnv(device) {
  if (!device || device === 'configured') return {};
  const [delivery, value, ...extra] = String(device).split(':');
  if (extra.length || !value) throw new Error('Device must be configured, modal:<GPU type>, or local:<device index>.');
  if (delivery === 'modal' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(value)) return { FK_DELIVERY: 'modal', FK_MODAL_GPU: value };
  if (delivery === 'local' && /^\d{1,4}$/.test(value)) return { FK_DELIVERY: 'local', FK_GPU: value };
  throw new Error('Device must be configured, modal:<GPU type>, or local:<device index>.');
}

export class ForgeSwarm {
  constructor(swarm, { binary } = {}) {
    this.swarm = swarm;
    this.root = dirname(dirname(fileURLToPath(import.meta.url)));
    this.binary = binary || join(HOME, 'kernel-venv', 'bin', 'python');
    this.injectedBinary = !!binary;
    this.setupPromise = null;
  }

  async ensureRuntime() {
    const ready = join(dirname(dirname(this.binary)), '.ratchet-ready');
    if (this.injectedBinary || (existsSync(this.binary) && existsSync(ready))) return;
    if (!this.setupPromise) this.setupPromise = (async () => {
      const run = (bin, args) => new Promise((done, fail) => {
        const child = spawn(bin, args, { cwd: this.root, stdio: ['ignore', 'ignore', 'pipe'] }); let error = '';
        child.stderr.on('data', (data) => { error = (error + data).slice(-4000); });
        child.on('error', fail); child.on('close', (code) => code === 0 ? done() : fail(new Error(error.trim() || `${bin} exited ${code}`)));
      });
      const venv = dirname(dirname(this.binary));
      await run('python3', ['-m', 'venv', venv]);
      await run(join(venv, 'bin', 'python'), ['-m', 'pip', 'install', 'modal>=1.6,<2']);
      writeFileSync(ready, 'Ratchet kernel runtime\n', { mode: 0o600 });
    })().catch((error) => { this.setupPromise = null; throw new Error(`Kernel runtime setup failed: ${error.message}`); });
    return this.setupPromise;
  }

  async command(args, { cwd, env = {}, signal, log = () => {} } = {}) {
    if (signal?.aborted) return { ok: false, error: 'stopped' };
    return new Promise((resolveResult) => {
      const childEnv = { ...process.env, PYTHONPATH: [join(this.root, 'src'), process.env.PYTHONPATH].filter(Boolean).join(delimiter), RATCHET_FORGE_ENV: join(HOME, 'forge.env'), ...env };
      delete childEnv.CLAUDECODE; delete childEnv.CLAUDE_CODE_ENTRYPOINT;
      const child = spawn(this.binary, this.injectedBinary ? args : ['-m', 'forgekernel', ...args], { cwd, env: childEnv, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'] });
      let output = ''; let error = ''; let settled = false;
      const onAbort = () => { try { if (process.platform === 'win32') child.kill('SIGTERM'); else process.kill(-child.pid, 'SIGTERM'); } catch {} };
      signal?.addEventListener('abort', onAbort, { once: true });
      const collect = (stream, kind) => {
        let pending = '';
        stream.setEncoding('utf8');
        stream.on('data', (chunk) => { if (kind === 'err') error = (error + chunk).slice(-10000); else output = (output + chunk).slice(-10000); pending += chunk; const lines = pending.split(/\r?\n/); pending = lines.pop(); for (const line of lines) if (line.trim()) log(kind, line.slice(0, 2000)); });
        stream.on('end', () => { if (pending.trim()) log(kind, pending.slice(0, 2000)); });
      };
      collect(child.stdout, 'text'); collect(child.stderr, 'err');
      const finish = (result) => { if (settled) return; settled = true; signal?.removeEventListener('abort', onAbort); resolveResult(result); };
      child.on('error', (e) => finish({ ok: false, error: `Could not launch ForgeKernel: ${e.message}`, output }));
      child.on('close', (code) => finish({ ok: code === 0 && !signal?.aborted, error: signal?.aborted ? 'stopped' : code === 0 ? null : (error.trim() || `ForgeKernel exited ${code}`), output }));
    });
  }

  listTasks() {
    const dir = join(this.root, 'tasks');
    if (!existsSync(dir)) return [];
    return readdirSync(dir, { withFileTypes: true }).filter((entry) => entry.isDirectory() && !entry.name.startsWith('_') && existsSync(join(dir, entry.name, 'task.toml'))).map((entry) => ({ name: entry.name, value: `builtin:${entry.name}` })).sort((a, b) => a.name.localeCompare(b.name));
  }

  resolveTask(task, cwd) {
    const raw = String(task);
    const value = raw.startsWith('@') ? raw.slice(1) : raw;
    if (!value) throw new Error('Provide a task folder after @.');
    if (value.startsWith('builtin:')) {
      const name = value.slice('builtin:'.length);
      if (!/^[a-zA-Z0-9_-]+$/.test(name)) throw new Error('Invalid built-in task name.');
      return join(this.root, 'tasks', name);
    }
    return resolve(cwd, value.replace(/^~(?=$|\/)/, homedir()));
  }

  async check(task, cwd, device = null) {
    const path = this.resolveTask(task, cwd);
    await this.ensureRuntime();
    return this.command(['new', path, '--dry-run'], { cwd, env: deviceEnv(device) });
  }

  start(tasks, device = null) {
    const cwd = this.swarm.settings.dir;
    const overrides = deviceEnv(device);
    if (!tasks.length) throw new Error('Usage: forge start <task-directory> [task-directory ...]');
    if (tasks.length > 12) throw new Error('At most 12 ForgeKernel campaigns can be started together.');
    const paths = tasks.map((task) => this.resolveTask(task, cwd));
    for (const task of paths) if (!existsSync(join(task, 'task.toml'))) throw new Error(`No task.toml in ${task}. Choose a kernel task folder containing task.toml, description.md, shapes.json, and a reference/ directory.`);
    const run = this.swarm.newRun(`ForgeKernel: ${paths.map((path) => basename(path)).join(', ')}`, 'forge');
    this.drive(run, paths, cwd, overrides).catch((error) => this.swarm.end(run, 'failed', error.message));
    return run;
  }

  async drive(run, tasks, cwd, overrides = {}) {
    const sw = this.swarm;
    sw.emit({ kind: 'agent.start', runId: run.id, id: 'lead', role: 'lead', title: 'ForgeKernel campaigns', parent: null });
    run.agents.set('lead', { id: 'lead', role: 'lead', title: 'ForgeKernel campaigns' });
    sw.emit({ kind: 'agent.end', runId: run.id, agentId: 'lead', ok: true });
    const results = await Promise.all(tasks.map(async (task, index) => {
      const id = `forge${index + 1}`; const title = basename(task); const artifactRoot = join(run.root, 'forge', id);
      mkdirSync(artifactRoot, { recursive: true });
      run.agents.set(id, { id, role: 'worker', title, parent: 'lead', harness: 'forge' });
      sw.emit({ kind: 'agent.start', runId: run.id, id, role: 'worker', title, parent: 'lead' });
      const log = (logKind, text) => sw.emit({ kind: 'agent.log', runId: run.id, agentId: id, logKind, text });
      let campaign = null; let lastError = null; const tried = new Set();
      try { log('sys', 'Preparing kernel runtime and waiting for a Claude account.'); await this.ensureRuntime(); }
      catch (error) { sw.emit({ kind: 'agent.end', runId: run.id, agentId: id, ok: false, error: error.message }); return { id, task, error: error.message }; }
      for (let attempt = 0; attempt < 6; attempt++) {
        let account;
        try { account = await sw.acquire('claude', run.abort.signal, (message) => log('sys', message), null, tried, true); }
        catch (error) { lastError = error.message; break; }
        let success = false;
        try {
          account.task = `kernel: ${title}`; sw.pool.refresh(account); sw.pool.publish(account);
          Object.assign(run.agents.get(id), { account: account.id, harness: 'claude' });
          sw.emit({ kind: 'agent.assign', runId: run.id, agentId: id, harness: 'claude', account: account.id, attempt });
          const accountAuth = account.configDir ? { ANTHROPIC_API_KEY: '', ANTHROPIC_AUTH_TOKEN: '', CLAUDE_CODE_OAUTH_TOKEN: '' } : {};
          const env = { ...accountAuth, ...sw.pool.env(account), ...overrides, FORGEKERNEL_RUNS_DIR: artifactRoot, RATCHET_KERNEL_MODEL: account.model || '' };
          let result;
          if (!campaign) {
            result = await this.command(['new', task], { cwd, env, signal: run.abort.signal, log });
            if (result.ok) {
              const dirs = readdirSync(artifactRoot, { withFileTypes: true }).filter((d) => d.isDirectory() && existsSync(join(artifactRoot, d.name, 'run.json')));
              if (dirs.length !== 1) result = { ok: false, error: 'ForgeKernel did not create exactly one campaign' };
              else { campaign = join(artifactRoot, dirs[0].name); log('sys', `Campaign: ${campaign}`); }
            }
          }
          if (campaign && (!result || result.ok)) result = await this.command(['run', campaign], { cwd, env, signal: run.abort.signal, log });
          const detail = `${result.error || ''}\n${result.output || ''}`;
          if (!result.ok && !run.abort.signal.aborted && LIMIT_RE.test(detail)) {
            tried.add(account.id); sw.pool.markLimited(account, parseReset(detail));
            log('err', `${account.id} reached a usage limit and was moved to Not using. Retrying with another Claude account.`);
            sw.emit({ kind: 'agent.failover', runId: run.id, agentId: id, from: account.id });
            lastError = result.error || 'Account usage limit'; continue;
          }
          success = result.ok;
          const reportFile = campaign && join(campaign, 'report.md');
          const report = reportFile && existsSync(reportFile) ? readFileSync(reportFile, 'utf8') : '';
          sw.emit({ kind: 'agent.output', runId: run.id, agentId: id, text: report.slice(0, 20000) });
          sw.emit({ kind: 'agent.end', runId: run.id, agentId: id, ok: result.ok, error: result.error });
          return { id, task, campaign, ok: result.ok, error: result.error, report };
        } finally { sw.release(account, success); }
      }
      lastError ||= 'No account available for the kernel campaign.';
      sw.emit({ kind: 'agent.end', runId: run.id, agentId: id, ok: false, error: lastError });
      return { id, task, campaign, error: lastError };
    }));
    const report = ['# ForgeKernel campaigns', '', ...results.flatMap((r) => [`## ${r.id} · ${r.task}`, `Status: ${r.ok ? 'target reached' : run.abort.signal.aborted ? 'stopped' : 'not completed successfully'}`, ...(r.campaign ? [`Artifacts: ${r.campaign}`] : []), ...(r.error ? [`Error: ${r.error}`] : []), '', r.report || 'No ForgeKernel report was produced.', ''])].join('\n');
    writeFileSync(join(run.root, 'REPORT.md'), report); run.report = report;
    sw.emit({ kind: 'run.report', runId: run.id, markdown: report, path: join(run.root, 'REPORT.md') });
    sw.end(run, run.abort.signal.aborted ? 'stopped' : results.every((r) => r.ok) ? 'done' : 'failed');
  }
}
