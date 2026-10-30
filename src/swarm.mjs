import { join } from 'node:path';
import { mkdirSync, writeFileSync, readFileSync, appendFileSync } from 'node:fs';
import { runCli } from './harness.mjs';
import { runMock } from './mock.mjs';

const sleep = (ms, signal) => new Promise((resolve) => {
  const t = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => { clearTimeout(t); resolve(); }, { once: true });
});

export function extractJson(text) {
  const fenced = [...text.matchAll(/```(?:json)?\s*([\s\S]*?)```/g)].map((m) => m[1]).reverse();
  for (const f of fenced) { try { return JSON.parse(f); } catch { /* next */ } }
  const a = text.indexOf('{');
  const b = text.lastIndexOf('}');
  if (a >= 0 && b > a) { try { return JSON.parse(text.slice(a, b + 1)); } catch { /* none */ } }
  return null;
}


export class Swarm {
  constructor({ pool, emit, settings }) {
    this.pool = pool;
    this.emit = emit;
    this.settings = settings; // {dir, mode, mock, slots, workers}
    this.runs = new Map();
    this.seq = 0;
  }

  /** Pick the best account for a harness preference, waiting for slots / resets. */
  async acquire(pref, signal, onWait) {
    let announced = false;
    for (;;) {
      if (signal.aborted) throw new Error('aborted');
      const all = this.pool.list();
      all.forEach((a) => this.pool.refresh(a));
      const usable = all.filter((a) => a.limitedUntil <= Date.now());
      if (!all.length) throw new Error('no accounts registered (try: account add claude)');
      if (!usable.length) {
        const wake = Math.min(...all.map((a) => a.limitedUntil));
        if (!announced) { onWait(`all accounts limited; waiting until ${new Date(wake).toLocaleTimeString()}`); announced = true; }
        await sleep(Math.min(Math.max(wake - Date.now(), 1000), 15_000), signal);
        continue;
      }
      let candidates = pref && pref !== 'any' ? usable.filter((a) => a.harness === pref) : usable;
      if (!candidates.length) candidates = usable;
      const free = candidates.filter((a) => this.pool.free(a, this.settings.slots)).sort((x, y) => x.load - y.load || x.done - y.done);
      if (free.length) { const a = free[0]; a.load++; return a; }
      await sleep(400, signal);
    }
  }

  /** A harness different from `h` that has accounts, for cross-model checks; 'any' when none. */
  other(h) {
    const alts = [...new Set(this.pool.list().map((a) => a.harness))].filter((x) => x !== h);
    return alts.length ? alts[Math.floor(Math.random() * alts.length)] : 'any';
  }

  release(a, ok) {
    a.load = Math.max(0, a.load - 1);
    if (ok) a.done++;
    if (!a.load) a.task = null;
    this.pool.refresh(a);
    this.pool.publish(a);
  }

  /** Run one agent with automatic failover across accounts. */
  async agent(run, spec) {
    const id = spec.id;
    const rec = { id, role: spec.role, title: spec.title, parent: spec.parent ?? null, harness: null, account: null };
    run.agents.set(id, rec);
    this.emit({ kind: 'agent.start', runId: run.id, ...rec, pref: spec.pref });
    const log = (kind, text) => this.emit({ kind: 'agent.log', runId: run.id, agentId: id, logKind: kind, text });
    let avoid = null;

    for (let attempt = 0; attempt < 6; attempt++) {
      let acct;
      try { acct = await this.acquire(avoid ? this.other(avoid) : spec.pref, run.abort.signal, (m) => log('sys', m)); }
      catch (e) { return this.finish(run, rec, { ok: false, text: '', error: e.message }); }
      rec.harness = acct.harness; rec.account = acct.id;
      acct.task = `${spec.role}: ${spec.title}`;
      this.pool.refresh(acct); this.pool.publish(acct);
      this.emit({ kind: 'agent.assign', runId: run.id, agentId: id, harness: acct.harness, account: acct.id, attempt });
      log('sys', `running on ${acct.id}`);

      const runner = this.settings.mock ? runMock : runCli;
      const res = await runner({
        harness: acct.harness, account: acct, env: this.pool.env(acct), cwd: spec.cwd ?? this.settings.dir,
        prompt: spec.prompt, mode: this.settings.mode, signal: run.abort.signal, onLog: (l) => log(l.kind, l.text),
      });
      this.release(acct, res.ok);
      if (res.ok) return this.finish(run, rec, res);
      if (run.abort.signal.aborted) return this.finish(run, rec, { ...res, error: 'stopped' });

      acct.failed++;
      if (res.limited) {
        this.pool.markLimited(acct, res.resetAt);
        log('err', `${acct.id} hit its limit (back ${new Date(res.resetAt).toLocaleTimeString()}); failing over`);
        this.emit({ kind: 'agent.failover', runId: run.id, agentId: id, from: acct.id });
        avoid = null;
        continue;
      }
      this.pool.publish(acct);
      log('err', res.error ?? 'failed');
      avoid = acct.harness; // try the other harness next
    }
    return this.finish(run, rec, { ok: false, text: '', error: 'gave up after repeated failures' });
  }

  finish(run, rec, res) {
    this.emit({ kind: 'agent.end', runId: run.id, agentId: rec.id, ok: res.ok, error: res.error ?? null });
    return res;
  }

  newRun(task) {
    const id = `r${(++this.seq).toString().padStart(2, '0')}`;
    const root = join(this.settings.dir, '.ratchet', 'runs', `${Date.now().toString(36)}-${id}`);
    mkdirSync(join(root, 'findings'), { recursive: true });
    const run = { id, task, root, agents: new Map(), claims: [], abort: new AbortController(), status: 'running', report: null };
    this.runs.set(id, run);
    this.emit({ kind: 'run.start', runId: id, task, dir: this.settings.dir, root });
    return run;
  }

  start(task) {
    const run = this.newRun(task);
    this.drive(run).catch((e) => this.end(run, 'failed', e.message));
    return run;
  }

  stop(id) {
    const targets = id === 'all' ? [...this.runs.values()] : [this.runs.get(id)].filter(Boolean);
    targets.filter((r) => r.status === 'running').forEach((r) => r.abort.abort());
    return targets.length;
  }

  end(run, status, note) {
    if (run.status !== 'running') return;
    run.status = status;
    this.emit({ kind: 'run.end', runId: run.id, status, note: note ?? null });
  }

  memory() {
    try {
      const rows = readFileSync(join(this.settings.dir, '.ratchet', 'memory.jsonl'), 'utf8').trim().split('\n').slice(-25).map((l) => JSON.parse(l));
      return rows.length ? `Known dead ends from earlier runs (do not repeat):\n${rows.map((r) => `- ${r.text} (${r.note})`).join('\n')}\n\n` : '';
    } catch { return ''; }
  }

  remember(run) {
    const dead = run.claims.filter((c) => c.verdict === 'refuted');
    if (!dead.length) return;
    appendFileSync(join(this.settings.dir, '.ratchet', 'memory.jsonl'), dead.map((c) => JSON.stringify({ ts: Date.now(), text: c.text, note: c.note })).join('\n') + '\n');
  }

  ctx(run) {
    return `Workspace folder: ${this.settings.dir}\nRun notes folder: ${run.root}\nOverall research task: ${run.task}`;
  }

  async drive(run) {
    const signal = run.abort.signal;
    const maxWorkers = this.settings.workers;

    // 1. plan
    const plan = await this.agent(run, {
      id: 'lead', role: 'lead', title: 'Plan', pref: 'claude',
      prompt: `[[ROLE:plan]]\nYou are the lead of a research swarm.\n${this.ctx(run)}\n\n${this.memory()}Inspect the workspace briefly, then split the task into at most ${maxWorkers} independent lines of attack that different researchers can pursue in parallel. Cover distinct approaches, not rewordings. Each line gets a harness: "claude" for long reading/synthesis/writing, "codex" for code-heavy experiments, "any" otherwise.\nReply with ONLY a JSON block:\n\`\`\`json\n{"subtasks":[{"title":"short","prompt":"self-contained instructions","harness":"claude|codex|any"}]}\n\`\`\``,
    });
    if (signal.aborted) return this.end(run, 'stopped');
    const parsed = plan.ok ? extractJson(plan.text) : null;
    const subtasks = (parsed?.subtasks ?? []).slice(0, maxWorkers);
    if (!subtasks.length) return this.end(run, 'failed', plan.error ?? 'planner returned no subtasks');

    // 2. research
    const results = await Promise.all(subtasks.map((st, i) => this.worker(run, `w${i + 1}`, st, null, 1)));
    if (signal.aborted) return this.end(run, 'stopped');

    // 3. cross-verify: each batch is checked by the opposite harness
    const batches = new Map();
    for (const c of run.claims) {
      if (!batches.has(c.agent)) batches.set(c.agent, []);
      batches.get(c.agent).push(c);
    }
    await Promise.all([...batches.entries()].map(([aid, claims], i) => this.verify(run, `v${i + 1}`, aid, claims)));
    if (signal.aborted) return this.end(run, 'stopped');

    // 4. synthesize
    const report = await this.synth(run, results);
    run.report = report;
    this.remember(run);
    writeFileSync(join(run.root, 'REPORT.md'), report);
    this.emit({ kind: 'run.report', runId: run.id, markdown: report, path: join(run.root, 'REPORT.md') });
    this.end(run, signal.aborted ? 'stopped' : 'done');
  }

  async worker(run, id, st, parent, depth) {
    const notes = join(run.root, 'findings', `${id}.md`);
    const spawnRule = depth < 2
      ? 'If a part of your task deserves its own investigator, list up to 3 in "spawn" (title, prompt, harness). They run as sub-agents and their claims merge into yours.'
      : 'You are a sub-agent: do not spawn further agents. [[DEPTH:2]]';
    const res = await this.agent(run, {
      id, role: depth > 1 ? 'sub' : 'worker', title: st.title, parent: parent ?? 'lead', pref: st.harness,
      prompt: `[[ROLE:worker]]\nYou are a researcher in a swarm. Be rigorous; check every claim against primary evidence (papers, code you actually ran, data). Prefer experiments and proofs over opinion.\n${this.ctx(run)}\n\nYour assignment: ${st.title}\n${st.prompt}\n\nWrite working notes and artifacts under ${join(run.root, 'findings', id + '-work')}/ and a summary to ${notes}. Do not modify existing project files unless the task requires it.\n${spawnRule}\n\nFinish with a short prose summary, then ONE JSON block:\n\`\`\`json\n{"claims":[{"text":"one checkable claim","source":"where the evidence is (url, file, command)"}],"spawn":[]}\n\`\`\``,
    });
    const out = res.ok ? extractJson(res.text) : null;
    const rec = run.agents.get(id);
    const claims = (out?.claims ?? []).map((c, i) => ({ id: `${id}.c${i}`, agent: id, by: rec.account, harness: rec.harness, text: String(c.text), source: String(c.source ?? ''), verdict: 'pending', note: '' }));
    run.claims.push(...claims);
    this.emit({ kind: 'claims', runId: run.id, claims });
    this.persist(run);

    let children = [];
    if (depth < 2 && out?.spawn?.length) {
      children = await Promise.all(out.spawn.slice(0, 3).map((s, i) => this.worker(run, `${id}.${i + 1}`, s, id, 2)));
    }
    return { id, title: st.title, ok: res.ok, children };
  }

  async verify(run, id, authorAgent, claims) {
    const author = run.agents.get(authorAgent);
    const list = claims.map((c) => `- (id: ${c.id}) ${c.text}\n  evidence claimed: ${c.source}`).join('\n');
    const res = await this.agent(run, {
      id, role: 'verify', title: `Check ${authorAgent}`, parent: authorAgent, pref: this.other(author?.harness),
      prompt: `[[ROLE:verify]]\nYou are an independent verifier from a different model family than the author. Do not trust the author. Re-derive or re-run what you can; look for counterexamples.\n${this.ctx(run)}\nAuthor notes: ${join(run.root, 'findings', authorAgent + '.md')}\n\nClaims to check:\n${list}\n\nFor each claim give verdict "confirmed" (you reproduced it), "refuted" (you found it false), or "unclear" (cannot tell), with a one-line note.\nReply with ONLY:\n\`\`\`json\n{"verdicts":[{"id":"...","verdict":"confirmed|refuted|unclear","note":"..."}]}\n\`\`\``,
    });
    const out = res.ok ? extractJson(res.text) : null;
    const rec = run.agents.get(id);
    const updates = [];
    for (const v of out?.verdicts ?? []) {
      const c = run.claims.find((x) => x.id === v.id);
      if (!c || !['confirmed', 'refuted', 'unclear'].includes(v.verdict)) continue;
      Object.assign(c, { verdict: v.verdict, note: String(v.note ?? ''), checkedBy: rec.account });
      updates.push(c);
    }
    this.emit({ kind: 'claims', runId: run.id, claims: updates });
    this.persist(run);
  }

  async synth(run) {
    const ledger = run.claims.map((c) => `- [${c.verdict}] (${c.id}, by ${c.by}${c.checkedBy ? `, checked by ${c.checkedBy}` : ''}) ${c.text} | source: ${c.source}${c.note ? ` | verifier: ${c.note}` : ''}`).join('\n');
    const res = await this.agent(run, {
      id: 'synth', role: 'lead', title: 'Synthesis', parent: 'lead', pref: 'claude',
      prompt: `[[ROLE:synth]]\nWrite the final research report in Markdown.\n${this.ctx(run)}\n\nClaim ledger (verdicts come from an independent model):\n${ledger || '(no claims were produced)'}\n\nResearcher notes are in ${join(run.root, 'findings')}. Rules: state confirmed claims plainly; flag unclear ones as unverified; do NOT present refuted claims as results, list them separately as dead ends with the reason. Cite claim ids. End with open questions and the best next experiments.`,
    });
    if (res.ok) return res.text;
    return `# Report (fallback: synthesis failed)\n\n${res.error ?? ''}\n\n## Ledger\n${ledger}\n`;
  }

  persist(run) {
    try { writeFileSync(join(run.root, 'claims.json'), JSON.stringify(run.claims, null, 2)); } catch { /* best effort */ }
  }
}
