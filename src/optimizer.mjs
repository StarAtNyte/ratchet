import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { cpSync, mkdirSync, existsSync, readFileSync, writeFileSync, appendFileSync, rmSync } from 'node:fs';
import { Archive } from './archive.mjs';
import { extractJson } from './swarm.mjs';

const SKIP = /(^|\/)(\.git|\.ratchet|node_modules|__pycache__|\.venv)$/;
const copyTree = (src, dst) => cpSync(src, dst, { recursive: true, filter: (p) => !SKIP.test(p) });

export const TEMPLATE = {
  objective: 'make the target faster while staying correct',
  targets: ['kernel.py'],
  eval: ['python check.py', 'python bench.py'],
  holdout: null,
  direction: 'min',
  parallel: 4,
  strongEvery: 4,
  maxIterations: 40,
  budgetMinutes: 60,
  timeoutSec: 300,
};

export function loadConfig(dir, mock) {
  const f = join(dir, 'ratchet.json');
  if (existsSync(f)) return { ...TEMPLATE, ...JSON.parse(readFileSync(f, 'utf8')) };
  if (mock) return { ...TEMPLATE, parallel: 3, maxIterations: 14 };
  throw new Error('no ratchet.json in workspace (run: opt init)');
}

/** Run a shell command in cwd; last stdout line is {"score":n,"ok":bool} or a bare number. */
export function runEval(cmd, cwd, timeoutSec) {
  return new Promise((resolve) => {
    const p = spawn('bash', ['-lc', cmd], { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = ''; let err = '';
    const t = setTimeout(() => p.kill('SIGKILL'), timeoutSec * 1000);
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    p.on('close', (code) => {
      clearTimeout(t);
      const last = out.trim().split('\n').at(-1) ?? '';
      let r = null;
      try { r = JSON.parse(last); if (typeof r === 'number') r = { score: r, ok: true }; } catch { const n = Number(last); if (Number.isFinite(n) && last) r = { score: n, ok: true }; }
      if (code !== 0 || !r || !Number.isFinite(r.score) || r.ok === false) resolve({ ok: false, score: NaN, error: (err || out).trim().slice(-300) || `exit ${code}` });
      else resolve({ ok: true, score: r.score });
    });
  });
}

export class Optimizer {
  constructor(swarm) { this.swarm = swarm; this.last = null; }

  async start(objective) {
    const { settings } = this.swarm;
    const cfg = loadConfig(settings.dir, settings.mock);
    if (objective) cfg.objective = objective;
    const run = this.swarm.newRun(`optimize: ${cfg.objective}`);
    const optDir = join(run.root, 'opt');
    mkdirSync(join(optDir, 'snap'), { recursive: true });
    const o = { run, cfg, optDir, archive: new Archive({ direction: cfg.direction, islands: Math.max(2, Math.min(cfg.parallel, 4)) }), inflight: new Set(), failures: [], n: 0, t0: Date.now(), evalLock: Promise.resolve() };
    this.last = o;
    this.drive(o).catch((e) => this.swarm.end(run, 'failed', e.message));
    return run;
  }

  /** Serialise evaluations: benchmarks need exclusive hardware. */
  evalExclusive(o, fn) {
    const next = o.evalLock.then(fn, fn);
    o.evalLock = next.catch(() => {});
    return next;
  }

  async evaluate(o, id, snapDir, isBase) {
    const { cfg } = o;
    if (this.swarm.settings.mock) {
      await new Promise((r) => setTimeout(r, 300));
      const base = o.archive.champion?.score ?? 100;
      const score = isBase ? 100 : base * (1 + (Math.random() - 0.55) * 0.08);
      return { ok: Math.random() > 0.12, score, holdout: score * (1 + (Math.random() - 0.5) * 0.01) };
    }
    const stage = join(o.optDir, 'stage', id);
    rmSync(stage, { recursive: true, force: true });
    copyTree(this.swarm.settings.dir, stage); // pristine evaluator + data
    for (const t of cfg.targets) if (existsSync(join(snapDir, t))) cpSync(join(snapDir, t), join(stage, t), { recursive: true }); // only declared targets come from the candidate
    let res = { ok: true, score: NaN };
    for (const cmd of [].concat(cfg.eval)) {
      res = await this.evalExclusive(o, () => runEval(cmd, stage, cfg.timeoutSec));
      if (!res.ok) break;
    }
    if (res.ok && cfg.holdout && o.archive.champion && o.archive.better(res.score, o.archive.champion.score)) {
      const h = await this.evalExclusive(o, () => runEval(cfg.holdout, stage, cfg.timeoutSec));
      res.holdout = h.score;
      if (!h.ok || (o.archive.champion.holdout != null && o.archive.better(o.archive.champion.holdout, h.score))) res.accepted = false;
    } else if (res.ok && cfg.holdout && !o.archive.champion) res.holdout = (await this.evalExclusive(o, () => runEval(cfg.holdout, stage, cfg.timeoutSec))).score;
    rmSync(stage, { recursive: true, force: true });
    return res;
  }

  prompt(o, id, workDir, parent, inspirations) {
    const { cfg, archive, failures } = o;
    const insp = inspirations.map((e) => `- ${e.id} score ${e.score} family ${e.family}: ${e.idea} (files: ${join(o.optDir, 'snap', e.id)})`).join('\n') || '(none yet)';
    const fails = failures.slice(-12).map((f) => `- ${f.idea}: ${f.reason}`).join('\n') || '(none yet)';
    return `[[ROLE:cand]]\nYou are one mutation worker in an evolutionary optimisation loop.\nObjective: ${cfg.objective}\nDirection: ${cfg.direction === 'min' ? 'LOWER' : 'HIGHER'} score is better. Current champion: ${archive.champion?.score ?? 'baseline pending'}. Parent (${parent?.id ?? 'baseline'}) score: ${parent?.score ?? 'n/a'}.\n\nWork ONLY in ${workDir}. Edit only these target files: ${cfg.targets.join(', ')}. Changes to any other file are discarded and count as cheating.\nYou may run the evaluator yourself to check: ${[].concat(cfg.eval).join(' && ')}\n\nTry ONE focused, meaningfully different idea. Diverse inspirations from other method families:\n${insp}\n\nFailed ideas, do not repeat:\n${fails}\n\nFinish with ONE JSON block: \`\`\`json\n{"idea":"one sentence","family":"short method-family tag"}\n\`\`\``;
  }

  async lane(o) {
    const { run, cfg, archive } = o;
    const sw = this.swarm;
    while (!run.abort.signal.aborted && o.n < cfg.maxIterations && Date.now() - o.t0 < cfg.budgetMinutes * 60000) {
      const iter = ++o.n; const id = `c${iter}`;
      const { parent, island, inspirations } = archive.sample(o.inflight);
      o.inflight.add(parent?.id ?? 'base');
      const work = join(o.optDir, 'work', id);
      if (!sw.settings.mock) {
        copyTree(sw.settings.dir, work);
        if (parent) for (const t of cfg.targets) if (existsSync(join(o.optDir, 'snap', parent.id, t))) cpSync(join(o.optDir, 'snap', parent.id, t), join(work, t), { recursive: true });
      }
      const res = await sw.agent(run, { id, role: 'cand', title: `gen ${iter}`, parent: parent && parent.id !== 'base' ? parent.id : 'lead', pref: iter % cfg.strongEvery === 0 ? 'claude' : 'any', cwd: work, prompt: this.prompt(o, id, work, parent, inspirations) });
      o.inflight.delete(parent?.id ?? 'base');
      const meta = res.ok ? extractJson(res.text) ?? {} : {};
      const idea = String(meta.idea ?? (res.ok ? 'unspecified change' : 'agent failed'));
      const family = String(meta.family ?? 'misc').toLowerCase().slice(0, 24);
      let ev = { ok: false, score: NaN, error: res.error };
      if (res.ok) {
        const snap = join(o.optDir, 'snap', id);
        if (!sw.settings.mock) { mkdirSync(snap, { recursive: true }); for (const t of cfg.targets) if (existsSync(join(work, t))) cpSync(join(work, t), join(snap, t), { recursive: true }); }
        ev = await this.evaluate(o, id, snap, false);
      }
      if (!sw.settings.mock) rmSync(work, { recursive: true, force: true });
      const entry = { id, parent: parent?.id ?? 'base', gen: iter, island, family, idea, score: ev.score, ok: ev.ok, holdout: ev.holdout, accepted: ev.accepted };
      const champ = archive.add(entry);
      if (!ev.ok) { const f = { ts: Date.now(), idea, reason: ev.error ?? 'failed evaluation' }; o.failures.push(f); appendFileSync(join(o.optDir, 'failures.jsonl'), JSON.stringify(f) + '\n'); }
      sw.emit({ kind: 'cand.result', runId: run.id, agentId: id, score: ev.score, ok: ev.ok, champion: champ, idea, family });
      writeFileSync(join(o.optDir, 'archive.json'), JSON.stringify(archive.entries, null, 2));
    }
  }

  async drive(o) {
    const { run, cfg, archive } = o;
    const sw = this.swarm;
    sw.emit({ kind: 'agent.start', runId: run.id, id: 'lead', role: 'lead', title: 'Baseline', parent: null, harness: null, account: null });
    run.agents.set('lead', { id: 'lead', role: 'lead', title: 'Baseline' });
    mkdirSync(join(o.optDir, 'snap', 'base'), { recursive: true });
    if (!sw.settings.mock) for (const t of cfg.targets) if (existsSync(join(sw.settings.dir, t))) cpSync(join(sw.settings.dir, t), join(o.optDir, 'snap', 'base', t), { recursive: true });
    const base = await this.evaluate(o, 'base', join(o.optDir, 'snap', 'base'), true);
    sw.emit({ kind: 'agent.end', runId: run.id, agentId: 'lead', ok: base.ok, error: base.error ?? null });
    if (!base.ok) return sw.end(run, 'failed', `baseline evaluation failed: ${base.error ?? ''}`);
    archive.add({ id: 'base', parent: null, gen: 0, island: 0, family: 'baseline', idea: 'unmodified workspace', score: base.score, ok: true, holdout: base.holdout });
    sw.emit({ kind: 'cand.result', runId: run.id, agentId: 'lead', score: base.score, ok: true, champion: true, idea: 'baseline', family: 'baseline' });
    await Promise.all(Array.from({ length: cfg.parallel }, () => this.lane(o)));
    this.finish(o);
  }

  finish(o) {
    const { run, archive, cfg } = o;
    const ch = archive.champion;
    const champDir = join(o.optDir, 'champion');
    if (ch && !this.swarm.settings.mock && ch.id !== 'base') copyTree(join(o.optDir, 'snap', ch.id), champDir);
    const base = archive.entries.find((e) => e.id === 'base');
    const gain = base && ch ? (((base.score - ch.score) / base.score) * 100 * (cfg.direction === 'min' ? 1 : -1)).toFixed(2) : '0';
    const top = archive.ranked().slice(0, 8).map((e) => `- **${e.id}** score ${Number(e.score).toFixed(4)} · ${e.family} · ${e.idea}`).join('\n');
    const md = `# Optimisation result\n\nObjective: ${cfg.objective}\n\n- baseline: ${base?.score}\n- champion: **${ch?.id}** at ${ch?.score} (${gain}% improvement)\n- candidates tried: ${archive.entries.length - 1}, failed: ${o.failures.length}\n- method families: ${archive.families().join(', ')}\n\n## Top programs\n${top}\n\n## Apply\nRun \`opt apply\` to copy the champion's target files into the workspace.\n`;
    run.report = md;
    writeFileSync(join(run.root, 'REPORT.md'), md);
    this.swarm.emit({ kind: 'run.report', runId: run.id, markdown: md, path: join(run.root, 'REPORT.md') });
    this.swarm.end(run, run.abort.signal.aborted ? 'stopped' : 'done');
  }

  apply() {
    const o = this.last;
    if (!o?.archive.champion || o.archive.champion.id === 'base') return 'no champion to apply';
    const id = o.archive.champion.id;
    for (const t of o.cfg.targets) { const s = join(o.optDir, 'snap', id, t); if (existsSync(s)) cpSync(s, join(this.swarm.settings.dir, t), { recursive: true }); }
    return `applied ${id} (score ${o.archive.champion.score}) to ${this.swarm.settings.dir}`;
  }
}
