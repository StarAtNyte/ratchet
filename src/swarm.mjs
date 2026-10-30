import { join } from 'node:path';
import { mkdirSync, writeFileSync, readFileSync, appendFileSync, existsSync, readdirSync } from 'node:fs';
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
    this.emitRaw = emit;
    // every run event is also appended to the run's events.jsonl so a restart can show and resume it
    this.emit = (ev) => {
      emit(ev);
      const run = ev.runId && this.runs.get(ev.runId);
      if (run?.root) { try { appendFileSync(join(run.root, 'events.jsonl'), `${JSON.stringify({ ...ev, t: Date.now() })}\n`); } catch { /* best effort */ } }
    };
    this.settings = settings; // {dir, mode, mock, slots, workers}
    this.runs = new Map();
    this.seq = 0;
  }

  /** Pick the best account for a harness preference, waiting for slots / resets. */
  async acquire(pref, signal, onWait, pin = null, exclude = new Set()) {
    let announced = false;
    for (;;) {
      if (signal.aborted) throw new Error('aborted');
      const all = this.pool.list();
      all.forEach((a) => this.pool.refresh(a));
      // signed-out accounts and accounts that already failed this agent are skipped
      const eligible = all.filter((a) => (pin ? a.id === pin : a.authed !== false && !exclude.has(a.id)));
      if (!pin && !eligible.length) throw new Error('no other signed-in account left to try');
      const usable = eligible.filter((a) => a.limitedUntil <= Date.now());
      if (pin && !all.some((a) => a.id === pin)) throw new Error(`account ${pin} not found`);
      if (!all.length) throw new Error('no accounts registered (try: account add claude)');
      if (!usable.length) {
        const wake = Math.min(...eligible.map((a) => a.limitedUntil));
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
    this.emit({ kind: 'agent.start', runId: run.id, ...rec, pref: spec.pref, prompt: String(spec.prompt ?? '').slice(0, 6000) });
    const log = (kind, text) => this.emit({ kind: 'agent.log', runId: run.id, agentId: id, logKind: kind, text });
    const hit = run.cache.get(id);
    if (hit) {
      Object.assign(rec, { harness: hit.harness, account: hit.account });
      this.emit({ kind: 'agent.assign', runId: run.id, agentId: id, harness: hit.harness, account: hit.account, attempt: 0 });
      log('sys', 'restored from the previous session (not re-run)');
      return this.finish(run, rec, hit);
    }
    let avoid = null;
    const tried = new Set();

    for (let attempt = 0; attempt < 6; attempt++) {
      let acct;
      try { acct = await this.acquire(avoid ? this.other(avoid) : spec.pref, run.abort.signal, (m) => log('sys', m), spec.pin ?? null, tried); }
      catch (e) { return this.finish(run, rec, { ok: false, text: '', error: e.message }); }
      rec.harness = acct.harness; rec.account = acct.id;
      acct.task = `${spec.role}: ${spec.title}`;
      this.pool.refresh(acct); this.pool.publish(acct);
      this.emit({ kind: 'agent.assign', runId: run.id, agentId: id, harness: acct.harness, account: acct.id, attempt });
      log('sys', `running on ${acct.id}`);

      const runner = this.settings.mock ? runMock : runCli;
      const limitMs = (this.settings.agentTimeoutMin ?? 30) * 60000;
      const timer = AbortSignal.timeout(limitMs);
      let res = await runner({
        harness: acct.harness, account: acct, env: this.pool.env(acct), cwd: spec.cwd ?? this.settings.dir,
        prompt: spec.prompt, mode: this.settings.mode, signal: AbortSignal.any([run.abort.signal, timer]), onLog: (l) => log(l.kind, l.text),
      });
      if (!res.ok && timer.aborted && !run.abort.signal.aborted) res = { ...res, limited: false, error: `timed out after ${limitMs / 60000} min` };
      if (res.ok) {
        run.results[id] = { ok: true, text: res.text, harness: acct.harness, account: acct.id };
        try { writeFileSync(join(run.root, 'results.json'), JSON.stringify(run.results)); } catch { /* best effort */ }
      }
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
      tried.add(acct.id);
      if (/\b401\b|unauthori[sz]ed|not logged in|authentication required|please (log|sign) ?in|missing bearer|invalid api key/i.test(res.error ?? '')) {
        acct.authed = false;
        log('err', `${acct.id} is not signed in; skipping it until you sign it in`);
      } else log('err', res.error ?? 'failed');
      this.pool.publish(acct);
      avoid = spec.pin ? null : acct.harness; // try the other harness next unless pinned
    }
    return this.finish(run, rec, { ok: false, text: '', error: 'gave up after repeated failures' });
  }

  finish(run, rec, res) {
    this.emit({ kind: 'agent.end', runId: run.id, agentId: rec.id, ok: res.ok, error: res.error ?? null });
    return res;
  }

  newRun(task, kind = 'research', reuse = null) {
    let id = reuse?.id;
    if (!id) { do id = `r${Date.now().toString(36).slice(-4)}${++this.seq}`; while (this.runs.has(id)); }
    const root = reuse?.root ?? join(this.settings.dir, '.ratchet', 'runs', `${Date.now().toString(36)}-${id}`);
    mkdirSync(join(root, 'findings'), { recursive: true });
    const run = { id, task, kind, root, agents: new Map(), claims: [], abort: new AbortController(), status: 'running', report: null, cache: reuse?.cache ?? new Map(), results: reuse?.results ?? {} };
    this.runs.set(id, run);
    if (!reuse) writeFileSync(join(root, 'meta.json'), JSON.stringify({ id, task, kind, started: Date.now() }));
    this.emit({ kind: 'run.start', runId: id, task, dir: this.settings.dir, root });
    return run;
  }

  /** Show runs of this workspace from earlier sessions; unfinished ones are marked interrupted. */
  restore(limit = 8) {
    const dir = join(this.settings.dir, '.ratchet', 'runs');
    if (!existsSync(dir)) return 0;
    const folders = readdirSync(dir).filter((f) => existsSync(join(dir, f, 'events.jsonl'))).sort().slice(-limit);
    for (const f of folders) {
      const root = join(dir, f);
      let events;
      try { events = readFileSync(join(root, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l)); } catch { continue; }
      const start = events.find((e) => e.kind === 'run.start');
      if (!start) continue;
      let id = start.runId;
      if (this.runs.has(id)) id = `${id}-${f.slice(0, 4)}`;
      const ended = events.find((e) => e.kind === 'run.end');
      const meta = (() => { try { return JSON.parse(readFileSync(join(root, 'meta.json'), 'utf8')); } catch { return {}; } })();
      const report = events.find((e) => e.kind === 'run.report')?.markdown ?? null;
      const claims = [];
      for (const e of events) if (e.kind === 'claims') for (const c of e.claims) { const i = claims.findIndex((x) => x.id === c.id); if (i >= 0) claims[i] = c; else claims.push(c); }
      const run = { id, task: start.task, kind: meta.kind ?? 'research', root, agents: new Map(), claims, abort: new AbortController(), status: ended ? ended.status : 'interrupted', report, cache: new Map(), results: {}, restored: true };
      for (const e of events) if (e.kind === 'agent.start') run.agents.set(e.id, { id: e.id, role: e.role, title: e.title, parent: e.parent });
      this.runs.set(id, run);
      for (const e of events) this.emitRaw({ ...e, runId: id, restored: true });
      if (!ended) this.emitRaw({ kind: 'run.end', runId: id, status: 'interrupted', note: `Ratchet stopped mid-run. Type: resume ${id}` });
    }
    return folders.length;
  }

  /** Re-run an interrupted or failed run; agents that already succeeded are restored from disk, not re-run. */
  resume(id) {
    const old = this.runs.get(id);
    if (!old) throw new Error(`no run ${id}`);
    if (old.status === 'running') throw new Error(`${id} is still running`);
    if (old.kind !== 'research') throw new Error(`only research runs can be resumed (this one is ${old.kind})`);
    let results = {};
    try { results = JSON.parse(readFileSync(join(old.root, 'results.json'), 'utf8')); } catch { /* none saved */ }
    const cache = new Map(Object.entries(results));
    this.runs.delete(id);
    const run = this.newRun(old.task, 'research', { id, root: old.root, cache, results });
    this.drive(run).catch((e) => this.end(run, 'failed', e.message));
    return { run, restored: cache.size };
  }

  /** Start a research run. Rounds chain until the user stops them or the round limit is reached. */
  start(task, chain = { round: 1, stopped: false, prev: null }) {
    const run = this.newRun(chain.round > 1 ? `${task} (round ${chain.round})` : task);
    run.chain = chain; run.baseTask = task;
    this.emit({ kind: 'run.round', runId: run.id, round: chain.round });
    this.drive(run).catch((e) => this.end(run, 'failed', e.message)).finally(() => {
      const limit = this.settings.rounds ?? 0; // 0 = until stopped
      if (chain.stopped || run.status !== 'done' || (limit && chain.round >= limit)) return;
      setTimeout(() => { if (!chain.stopped) this.start(task, { round: chain.round + 1, stopped: false, prev: { id: run.id, report: run.report } }); chain.stopped = true; }, 5000);
    });
    return run;
  }

  /** One user-launched agent on a chosen harness/account; joins the running run or opens its own. */
  adhoc({ harness = 'any', account = null, title, prompt }) {
    let run = [...this.runs.values()].reverse().find((r) => r.status === 'running');
    const own = !run;
    if (own) {
      run = this.newRun('ad-hoc agent', 'adhoc');
      this.emit({ kind: 'agent.start', runId: run.id, id: 'lead', role: 'lead', title: 'You', parent: null, harness: null, account: null });
      run.agents.set('lead', { id: 'lead', role: 'lead', title: 'You' });
      this.emit({ kind: 'agent.end', runId: run.id, agentId: 'lead', ok: true, error: null });
    }
    const id = `m${(run.manual = (run.manual ?? 0) + 1)}`;
    this.agent(run, {
      id, role: 'worker', title: title || 'Ad-hoc agent', parent: 'lead', pref: harness, pin: account || null,
      prompt: `[[ROLE:adhoc]]\n${this.ctx(run)}\nNotes and artifacts belong under ${join(run.root, 'findings', id + '-work')}/.\n\nTask from the user:\n${prompt}`,
    }).then((res) => {
      this.emit({ kind: 'agent.output', runId: run.id, agentId: id, text: String(res.text ?? '').slice(0, 20000) });
      if (own) this.end(run, res.ok ? 'done' : 'failed', res.ok ? null : res.error);
    });
    return { runId: run.id, agentId: id };
  }

  stop(id) {
    const targets = id === 'all' ? [...this.runs.values()] : [this.runs.get(id)].filter(Boolean);
    targets.forEach((r) => { if (r.chain) r.chain.stopped = true; });
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
      prompt: `[[ROLE:plan]]\nYou are the lead of a research swarm.\n${this.ctx(run)}\n\n${this.memory()}${run.chain?.prev?.report ? `This is round ${run.chain.round}. The previous round (${run.chain.prev.id}) concluded:\n${String(run.chain.prev.report).slice(0, 6000)}\n\nDo not redo settled work. Go deeper: chase its open questions, unclear claims and best next experiments.\n\n` : ''}Inspect the workspace briefly, then split the task into at most ${maxWorkers} independent lines of attack that different researchers can pursue in parallel. Cover distinct approaches, not rewordings. Each line gets a harness: "claude" for long reading/synthesis/writing, "codex" for code-heavy experiments, "any" otherwise.\nReply with ONLY a JSON block:\n\`\`\`json\n{"subtasks":[{"title":"short","prompt":"self-contained instructions","harness":"claude|codex|any"}]}\n\`\`\``,
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
