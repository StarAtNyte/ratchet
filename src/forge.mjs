import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { HOME } from './accounts.mjs';

export class ForgeSwarm {
  constructor(swarm, { binary = process.env.RATCHET_FORGE_BIN || join(HOME, 'forgekernel-venv', 'bin', 'forge') } = {}) {
    this.swarm = swarm;
    this.binary = binary;
  }

  async command(args, { cwd, env = {}, signal, log = () => {} } = {}) {
    if (signal?.aborted) return { ok: false, error: 'stopped' };
    return new Promise((resolveResult) => {
      const child = spawn(this.binary, args, { cwd, env: { ...process.env, ...env }, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'] });
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

  async check(task, cwd) {
    return this.command(['new', resolve(cwd, task), '--dry-run'], { cwd });
  }

  start(tasks) {
    const cwd = this.swarm.settings.dir;
    if (!tasks.length) throw new Error('Usage: forge start <task-directory> [task-directory ...]');
    if (tasks.length > 12) throw new Error('At most 12 ForgeKernel campaigns can be started together.');
    const paths = tasks.map((task) => resolve(cwd, task));
    for (const task of paths) if (!existsSync(join(task, 'task.toml'))) throw new Error(`No task.toml in ${task}`);
    const run = this.swarm.newRun(`ForgeKernel: ${paths.map((path) => basename(path)).join(', ')}`, 'forge');
    this.drive(run, paths, cwd).catch((error) => this.swarm.end(run, 'failed', error.message));
    return run;
  }

  async drive(run, tasks, cwd) {
    const sw = this.swarm;
    sw.emit({ kind: 'agent.start', runId: run.id, id: 'lead', role: 'lead', title: 'ForgeKernel campaigns', parent: null });
    run.agents.set('lead', { id: 'lead', role: 'lead', title: 'ForgeKernel campaigns' });
    sw.emit({ kind: 'agent.end', runId: run.id, agentId: 'lead', ok: true });
    const results = await Promise.all(tasks.map(async (task, index) => {
      const id = `forge${index + 1}`; const title = basename(task); const artifactRoot = join(run.root, 'forge', id);
      mkdirSync(artifactRoot, { recursive: true });
      run.agents.set(id, { id, role: 'worker', title, parent: 'lead', harness: 'forge' });
      sw.emit({ kind: 'agent.start', runId: run.id, id, role: 'worker', title, parent: 'lead' });
      sw.emit({ kind: 'agent.assign', runId: run.id, agentId: id, harness: 'forge', account: 'ForgeKernel', attempt: 0 });
      const log = (logKind, text) => sw.emit({ kind: 'agent.log', runId: run.id, agentId: id, logKind, text });
      const env = { FORGEKERNEL_RUNS_DIR: artifactRoot };
      const created = await this.command(['new', task], { cwd, env, signal: run.abort.signal, log });
      if (!created.ok) { sw.emit({ kind: 'agent.end', runId: run.id, agentId: id, ok: false, error: created.error }); return { id, task, error: created.error }; }
      const dirs = readdirSync(artifactRoot, { withFileTypes: true }).filter((d) => d.isDirectory() && existsSync(join(artifactRoot, d.name, 'run.json')));
      if (dirs.length !== 1) { const error = 'ForgeKernel did not create exactly one campaign'; sw.emit({ kind: 'agent.end', runId: run.id, agentId: id, ok: false, error }); return { id, task, error }; }
      const campaign = join(artifactRoot, dirs[0].name);
      log('sys', `Campaign: ${campaign}`);
      const result = await this.command(['run', campaign], { cwd, env, signal: run.abort.signal, log });
      const reportFile = join(campaign, 'report.md');
      const report = existsSync(reportFile) ? readFileSync(reportFile, 'utf8') : '';
      sw.emit({ kind: 'agent.output', runId: run.id, agentId: id, text: report.slice(0, 20000) });
      sw.emit({ kind: 'agent.end', runId: run.id, agentId: id, ok: result.ok, error: result.error });
      return { id, task, campaign, ok: result.ok, error: result.error, report };
    }));
    const report = ['# ForgeKernel campaigns', '', ...results.flatMap((r) => [`## ${r.id} · ${r.task}`, `Status: ${r.ok ? 'target reached' : run.abort.signal.aborted ? 'stopped' : 'not completed successfully'}`, ...(r.campaign ? [`Artifacts: ${r.campaign}`] : []), ...(r.error ? [`Error: ${r.error}`] : []), '', r.report || 'No ForgeKernel report was produced.', ''])].join('\n');
    writeFileSync(join(run.root, 'REPORT.md'), report); run.report = report;
    sw.emit({ kind: 'run.report', runId: run.id, markdown: report, path: join(run.root, 'REPORT.md') });
    sw.end(run, run.abort.signal.aborted ? 'stopped' : results.every((r) => r.ok) ? 'done' : 'failed');
  }
}
