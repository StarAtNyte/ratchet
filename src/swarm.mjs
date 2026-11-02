import { join, isAbsolute, resolve } from 'node:path';
import { mkdirSync, writeFileSync, readFileSync, appendFileSync, existsSync, readdirSync } from 'node:fs';
import { runCli } from './harness.mjs';
import { runMock } from './mock.mjs';
import { linesFor, difficultyOf, DIFFICULTY, brief, gateVerdict, missingLocalSource, applyStrikes, elo, pairings, cluster } from './methods.mjs';

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
  async acquire(pref, signal, onWait, pin = null, exclude = new Set(), strict = false) {
    let announced = false;
    for (;;) {
      if (signal.aborted) throw new Error('aborted');
      const all = this.pool.list();
      all.forEach((a) => this.pool.refresh(a));
      // signed-out accounts and accounts that already failed this agent are skipped
      const eligible = all.filter((a) => a.enabled !== false && (pin ? a.id === pin : a.authed !== false && !exclude.has(a.id)));
      if (pin && all.some((a) => a.id === pin && a.enabled === false)) throw new Error(`account ${pin} is not in use`);
      if (!pin && !eligible.length) throw new Error('no other signed-in account left to try');
      if (strict && pref && pref !== 'any' && !eligible.some((a) => a.harness === pref)) throw new Error(`no eligible ${pref} account for independent verification`);
      const usable = eligible.filter((a) => a.limitedUntil <= Date.now());
      if (pin && !all.some((a) => a.id === pin)) throw new Error(`account ${pin} not found`);
      if (!all.length) throw new Error('no accounts registered (try: account add claude)');
      if (!usable.length) {
        const wake = Math.min(...eligible.map((a) => a.limitedUntil));
        if (!announced) { onWait(`all accounts limited; waiting until ${new Date(wake).toLocaleTimeString()}`); announced = true; }
        await sleep(Math.min(Math.max(wake - Date.now(), 1000), 15_000), signal);
        continue;
      }
      const free = usable.filter((a) => this.pool.free(a, this.settings.slots));
      const preferred = pref && pref !== 'any' ? free.filter((a) => a.harness === pref) : free;
      let pool = preferred.length ? preferred : strict && pref && pref !== 'any' ? [] : free;
      // spread quota: when every preferred account is already busy and another provider has an idle account, use it
      if (!strict && !pin && pref && pref !== 'any' && preferred.length && Math.min(...preferred.map((a) => a.load)) >= 1) {
        const idle = free.filter((a) => a.load === 0 && a.harness !== pref);
        if (idle.length) pool = idle;
      }
      const candidates = pool.sort((x, y) => x.load - y.load || x.done - y.done);
      if (candidates.length) { const a = candidates[0]; a.load++; return a; }
      await sleep(400, signal);
    }
  }

  /** A harness different from `h` that has accounts, for cross-model checks; 'any' when none. */
  other(h) {
    const alts = [...new Set(this.pool.list().filter((a) => a.enabled !== false).map((a) => a.harness))].filter((x) => x !== h);
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
    // run budget: a hard ceiling on agents per run keeps runaway fan-out from draining every account
    const budget = this.settings.runAgentBudget ?? 80;
    run.spawned = (run.spawned ?? 0) + 1;
    if (budget && run.spawned > budget && !run.cache.get(id)) {
      run.agents.set(id, rec);
      this.emit({ kind: 'agent.start', runId: run.id, ...rec, pref: spec.pref, prompt: '' });
      return this.finish(run, rec, { ok: false, text: '', error: `run agent budget (${budget}) reached` });
    }
    run.agents.set(id, rec);
    this.emit({ kind: 'agent.start', runId: run.id, ...rec, pref: spec.pref, prompt: String(spec.prompt ?? '').slice(0, 6000) });
    const log = (kind, text) => {
      this.emit({ kind: 'agent.log', runId: run.id, agentId: id, logKind: kind, text });
      // isolation: researchers must not read other researchers' raw notes (only verified claims are shared)
      if (kind === 'tool' && (spec.role === 'worker' || spec.role === 'sub')) {
        const m = /findings\/(w\d+(?:\.\d+)?)(?:-work|\.md)/.exec(text);
        if (m && m[1] !== id && !m[1].startsWith(`${id}.`) && !id.startsWith(`${m[1]}.`)) this.strike(run, id, `read another researcher's notes (${m[1]})`);
      }
    };
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
      try { acct = await this.acquire(avoid ? this.other(avoid) : spec.pref, run.abort.signal, (m) => log('sys', m), spec.pin ?? null, tried, spec.role === 'verify'); }
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
      if (res.usage) this.addUsage(run, rec, acct, res.usage);
      if (res.ok) {
        acct.authed = true;
        run.results[id] = { ok: true, text: res.text, harness: acct.harness, account: acct.id };
        try { writeFileSync(join(run.root, 'results.json'), JSON.stringify(run.results)); } catch { /* best effort */ }
      }
      this.release(acct, res.ok);
      if (res.ok) return this.finish(run, rec, res);
      if (run.abort.signal.aborted) return this.finish(run, rec, { ...res, error: 'stopped' });

      acct.failed++;
      if (res.limited) {
        this.pool.markLimited(acct, res.resetAt);
        log('err', `${acct.id} hit its limit (resets ${new Date(res.resetAt).toLocaleString()}); moved to Not using and retrying elsewhere`);
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

  /** Graduated sanctions: strike 1 flags the researcher, strike 2 quarantines all its claims. */
  strike(run, agentId, reason) {
    run.strikes ??= new Map();
    const key = `${agentId}|${reason}`;
    run.struck ??= new Set();
    if (run.struck.has(key)) return;
    run.struck.add(key);
    const n = (run.strikes.get(agentId) ?? 0) + 1;
    run.strikes.set(agentId, n);
    this.emit({ kind: 'agent.strike', runId: run.id, agentId, strikes: n, reason });
    this.emit({ kind: 'agent.log', runId: run.id, agentId, logKind: 'err', text: `strike ${n}: ${reason}${n >= 2 ? ' (claims quarantined)' : ''}` });
    const ids = applyStrikes(run.strikes, run.claims);
    if (ids.length) {
      const updates = run.claims.filter((c) => ids.includes(c.id)).map((c) => Object.assign(c, { verdict: 'quarantined', note: `quarantined: ${reason}` }));
      this.emit({ kind: 'claims', runId: run.id, claims: updates });
      this.persist(run);
    }
  }

  /** Token and cost accounting per agent, run and account. */
  addUsage(run, rec, acct, u) {
    const tokens = (u.input_tokens ?? 0) + (u.output_tokens ?? 0) + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0);
    const cost = Number(u.cost_usd ?? 0) || 0;
    run.usage ??= { tokens: 0, cost: 0 };
    run.usage.tokens += tokens; run.usage.cost += cost;
    acct.usage ??= { tokens: 0, cost: 0 };
    acct.usage.tokens += tokens; acct.usage.cost += cost;
    this.emit({ kind: 'agent.usage', runId: run.id, agentId: rec.id, account: acct.id, tokens, cost, runTokens: run.usage.tokens, runCost: run.usage.cost });
    const cap = this.settings.runCostBudget ?? 0;
    if (cap > 0 && run.usage.cost >= cap && run.status === 'running') {
      if (run.chain) run.chain.stopped = true;
      this.emit({ kind: 'agent.log', runId: run.id, agentId: rec.id, logKind: 'err', text: `run cost budget $${cap} reached; stopping the run` });
      run.abort.abort();
    }
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
    try { run.followups = readFileSync(join(old.root, 'instructions.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line).text); } catch { run.followups = []; }
    this.drive(run).catch((e) => this.end(run, 'failed', e.message));
    return { run, restored: cache.size };
  }

  /** Start a research run. Rounds chain until the user stops them or the round limit is reached. */
  start(task, chain = { round: 1, stopped: false, prev: null }) {
    const run = this.newRun(chain.round > 1 ? `${task} (round ${chain.round})` : task);
    run.followups = [...(chain.followups ?? [])];
    if (run.followups.length) writeFileSync(join(run.root, 'instructions.jsonl'), run.followups.map((text) => JSON.stringify({ t: Date.now(), text })).join('\n') + '\n');
    run.chain = chain; run.baseTask = task;
    this.emit({ kind: 'run.round', runId: run.id, round: chain.round });
    this.drive(run).catch((e) => this.end(run, 'failed', e.message)).finally(() => {
      const limit = this.settings.rounds ?? 0; // 0 = until stopped
      if (chain.stopped || run.status !== 'done' || (limit && chain.round >= limit)) return;
      setTimeout(() => { if (!chain.stopped) this.start(task, { round: chain.round + 1, stopped: false, prev: { id: run.id, report: run.report, meta: run.meta, next: run.next }, followups: [...(run.followups ?? [])] }); chain.stopped = true; }, 5000);
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

  /** Queue user instructions for future agent turns and subsequent rounds. */
  followup(text, id = null) {
    const run = id ? this.runs.get(id) : [...this.runs.values()].reverse().find((r) => r.status === 'running' && r.kind === 'research');
    if (!run || run.status !== 'running' || run.kind !== 'research') throw new Error('No active research run to update.');
    const instruction = String(text ?? '').trim();
    if (!instruction) throw new Error('Usage: follow <instruction>');
    if (instruction.length > 10000) throw new Error('Instruction must be 10,000 characters or fewer.');
    (run.followups ??= []).push(instruction);
    appendFileSync(join(run.root, 'instructions.jsonl'), JSON.stringify({ t: Date.now(), text: instruction }) + '\n');
    this.emit({ kind: 'run.instruction', runId: run.id, text: instruction });
    return run.id;
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
    return `Workspace folder: ${this.settings.dir}\nRun notes folder: ${run.root}\nOverall research task: ${run.task}${run.followups?.length ? `\nAdditional user instructions for subsequent work:\n${run.followups.map((text, i) => `${i + 1}. ${text}`).join('\n')}` : ''}`;
  }

  async drive(run) {
    const signal = run.abort.signal;
    const maxWorkers = this.settings.workers;
    const prev = run.chain?.prev;

    // 1. plan, with explicit effort scaling (Anthropic): the lead classifies the question first
    const plan = await this.agent(run, {
      id: 'lead', role: 'lead', title: 'Plan', pref: 'claude',
      prompt: `[[ROLE:plan]]\nYou are the lead of a research swarm.\n${this.ctx(run)}\n\n${this.memory()}${prev?.report ? `This is round ${run.chain.round}. The previous round (${prev.id}) concluded:\n${String(prev.report).slice(0, 6000)}\n\n${prev.meta?.length ? `Its meta-review found these recurring weaknesses; fix them in this round's briefs:\n${prev.meta.map((m) => `- ${m}`).join('\n')}\n\n` : ''}${prev.next?.length ? `Suggested next lines:\n${prev.next.map((m) => `- ${m}`).join('\n')}\n\n` : ''}Do not redo settled work. Go deeper: chase its open questions, unclear claims and best next experiments.\n\n` : ''}Inspect the workspace briefly. First classify the question:\n- "simple": one fact or one check (1-2 lines of attack)\n- "comparison": compare a few options (2-4 lines)\n- "complex": open research (up to ${maxWorkers} lines with clearly divided responsibilities)\nThen split it into independent lines of attack that different researchers can pursue in parallel. Cover distinct approaches, not rewordings, and give each line explicit boundaries so researchers do not duplicate each other. When the user requests performance optimization, assign concrete implementation, correctness checks, benchmarking and iteration; when the user explicitly requests submission, include the documented submission and result-review workflow. Coordinate shared files and remote benchmarks to avoid conflicting concurrent work. Each line gets a harness: "claude" for long reading/synthesis/writing, "codex" for code-heavy experiments, "any" otherwise, and a difficulty: "low" (direct answer), "medium", or "high" (may split into sub-investigations).\nReply with ONLY a JSON block:\n\`\`\`json\n{"complexity":"simple|comparison|complex","subtasks":[{"title":"short","objective":"what this line must establish","prompt":"self-contained instructions","non_goals":["what this line must not do"],"output":"what to deliver","stop_when":"stop condition","harness":"claude|codex|any","difficulty":"low|medium|high"}]}\n\`\`\``,
    });
    if (signal.aborted) return this.end(run, 'stopped');
    const parsed = plan.ok ? extractJson(plan.text) : null;
    const complexity = ['simple', 'comparison', 'complex'].includes(parsed?.complexity) ? parsed.complexity : 'complex';
    const all = parsed?.subtasks ?? [];
    const subtasks = all.slice(0, linesFor(complexity, all.length, maxWorkers));
    if (!subtasks.length) return this.end(run, 'failed', plan.error ?? 'planner returned no subtasks');
    run.complexity = complexity;
    this.emit({ kind: 'run.plan', runId: run.id, complexity, lines: subtasks.map((s) => ({ title: s.title, difficulty: difficultyOf(s.difficulty), harness: s.harness ?? 'any' })) });

    // 2. research, each line with a structured brief naming its siblings
    const titles = subtasks.map((s) => s.title);
    const results = await Promise.all(subtasks.map((st, i) => this.worker(run, `w${i + 1}`, { ...st, siblings: titles.filter((_, j) => j !== i) }, null, 1)));
    if (signal.aborted) return this.end(run, 'stopped');

    // 3. cross-verify: each batch is checked by a different model family, evidence required
    const batches = new Map();
    for (const c of run.claims) {
      if (c.verdict === 'quarantined') continue;
      if (!batches.has(c.agent)) batches.set(c.agent, []);
      batches.get(c.agent).push(c);
    }
    await Promise.all([...batches.entries()].map(([aid, claims], i) => this.verify(run, `v${i + 1}`, aid, claims)));
    if (signal.aborted) return this.end(run, 'stopped');

    // 4. rank surviving findings in a pairwise tournament (co-scientist style Elo)
    await this.rank(run);
    if (signal.aborted) return this.end(run, 'stopped');

    // 5. synthesize, with a meta-review that feeds the next round
    const { report, meta, next } = await this.synth(run, results);
    run.report = report; run.meta = meta; run.next = next;
    this.remember(run);
    writeFileSync(join(run.root, 'REPORT.md'), report);
    this.emit({ kind: 'run.report', runId: run.id, markdown: report, path: join(run.root, 'REPORT.md'), meta, next });

    // 6. score the round with a rubric judge so rounds and harness changes can be compared
    await this.judge(run);
    this.end(run, signal.aborted ? 'stopped' : 'done');
  }

  async worker(run, id, st, parent, depth) {
    const notes = join(run.root, 'findings', `${id}.md`);
    const maxSpawn = depth < 2 ? DIFFICULTY[difficultyOf(st.difficulty)].maxSpawn : 0;
    const spawnRule = maxSpawn > 0
      ? `If a part of your task deserves its own investigator, list up to ${maxSpawn} in "spawn" (title, objective, prompt, harness, difficulty). They run as sub-agents and their claims merge into yours.`
      : `Do not spawn further agents.${depth >= 2 ? ' [[DEPTH:2]]' : ''}`;
    const res = await this.agent(run, {
      id, role: depth > 1 ? 'sub' : 'worker', title: st.title, parent: parent ?? 'lead', pref: st.harness,
      prompt: `[[ROLE:worker]]\nYou are a researcher in a swarm. Be rigorous; check every claim against primary evidence (papers, code you actually ran, data). Prefer experiments and proofs over opinion.\n${this.ctx(run)}\n\nYour assignment: ${st.title}\n${brief(st, st.siblings ?? [])}\n\nWork only from primary evidence and your own notes. Do not read other researchers' notes or work folders under ${join(run.root, 'findings')}; verified results are shared with you when relevant. Every claim must point to evidence another model can check: a URL, a file path with line, or a command and its output.\n\nWrite working notes and artifacts under ${join(run.root, 'findings', id + '-work')}/ and a summary to ${notes}. Do not modify existing project files unless the task requires it.\n${spawnRule}\nIf this assignment requests optimization, make concrete improvements, run correctness and performance checks, compare results to the baseline, and iterate using the measured outcome. If the user explicitly requested submissions, use the workspace's documented submission process, record the result and iterate; never claim a submission or benchmark succeeded without evidence. Coordinate shared files and benchmark resources with other workers.\n\nFinish with a short prose summary, then ONE JSON block:\n\`\`\`json\n{"claims":[{"text":"one checkable claim","source":"where the evidence is (url, file, command)"}],"spawn":[]}\n\`\`\``,
    });
    const out = res.ok ? extractJson(res.text) : null;
    const rec = run.agents.get(id);
    const claims = (out?.claims ?? []).map((c, i) => ({ id: `${id}.c${i}`, agent: id, by: rec.account, harness: rec.harness, text: String(c.text), source: String(c.source ?? ''), verdict: 'pending', note: '' }));
    run.claims.push(...claims);
    this.emit({ kind: 'claims', runId: run.id, claims });
    this.persist(run);

    let children = [];
    if (depth < 2 && out?.spawn?.length) {
      const sib = out.spawn.slice(0, maxSpawn).map((s) => s.title);
      children = await Promise.all(out.spawn.slice(0, maxSpawn).map((s, i) => this.worker(run, `${id}.${i + 1}`, { ...s, siblings: sib.filter((_, j) => j !== i) }, id, 2)));
    }
    return { id, title: st.title, ok: res.ok, children };
  }

  async verify(run, id, authorAgent, claims) {
    const author = run.agents.get(authorAgent);
    // evidence pointing at local files that do not exist is caught before any model looks at it
    const exists = (p) => { const abs = isAbsolute(p) ? p : resolve(this.settings.dir, p.replace(/^~\//, `${process.env.HOME}/`)); return existsSync(abs) || existsSync(resolve(run.root, p)); };
    for (const c of claims) {
      const missing = missingLocalSource(c.source, exists);
      if (missing) { c.flag = `cited file ${missing} does not exist`; this.strike(run, authorAgent, `cited a file that does not exist (${missing})`); }
    }
    const list = claims.map((c) => `- (id: ${c.id}) ${c.text}\n  evidence claimed: ${c.source}${c.flag ? `\n  note: ${c.flag}` : ''}`).join('\n');
    const res = await this.agent(run, {
      id, role: 'verify', title: `Check ${authorAgent}`, parent: authorAgent, pref: this.other(author?.harness),
      prompt: `[[ROLE:verify]]\nYou are an independent verifier from a different model family than the author. Do not trust the author. Re-derive or re-run what you can; look for counterexamples.\n${this.ctx(run)}\nAuthor notes: ${join(run.root, 'findings', authorAgent + '.md')}\n\nClaims to check:\n${list}\n\nFor each claim give verdict "confirmed" (you reproduced it), "refuted" (you found it false), or "unclear" (cannot tell), a one-line note, and "evidence": a short verbatim quote from the source you checked, or the command you ran and the decisive line of its output. A confirmation without evidence counts as unclear. Set "fabricated": true only when the cited evidence does not exist or says something different.\nReply with ONLY:\n\`\`\`json\n{"verdicts":[{"id":"...","verdict":"confirmed|refuted|unclear","note":"...","evidence":"...","fabricated":false}]}\n\`\`\``,
    });
    const out = res.ok ? extractJson(res.text) : null;
    const rec = run.agents.get(id);
    const updates = [];
    for (const v of out?.verdicts ?? []) {
      const c = run.claims.find((x) => x.id === v.id);
      const g = gateVerdict(v);
      if (!c || !g || c.verdict === 'quarantined') continue;
      Object.assign(c, { verdict: g.verdict, note: g.note, evidence: g.evidence, checkedBy: rec.account });
      if (g.fabricated) this.strike(run, authorAgent, `verifier found fabricated evidence for ${c.id}`);
      updates.push(c);
    }
    this.emit({ kind: 'claims', runId: run.id, claims: updates.filter((c) => c.verdict !== 'quarantined') });
    this.persist(run);
  }

  /** Pairwise tournament over surviving findings; near-duplicates are merged first. */
  async rank(run) {
    const pool = run.claims.filter((c) => c.verdict === 'confirmed' || c.verdict === 'unclear');
    if (pool.length < 3) { pool.forEach((c) => { c.elo = 1200; }); return; }
    const groups = cluster(pool.map((c) => c.text), 0.6);
    const reps = groups.map((g) => pool[g[0]]);
    groups.forEach((g) => g.slice(1).forEach((i) => { pool[i].dupOf = pool[g[0]].id; }));
    const pairs = pairings(reps.map((c) => c.id), 3).slice(0, 30);
    const byId = new Map(reps.map((c) => [c.id, c]));
    const res = await this.agent(run, {
      id: 'rank', role: 'judge', title: 'Rank findings', parent: 'lead', pref: 'any',
      prompt: `[[ROLE:rank]]\nYou judge a tournament between research findings for: ${run.task}\nFor each pair pick the finding that matters more for the task: better supported, more consequential, more novel. Judge only what is written; do not favour longer text.\n\n${pairs.map(([a, b], i) => `Match ${i + 1}\nA (${a}): ${byId.get(a).text}\nB (${b}): ${byId.get(b).text}`).join('\n\n')}\n\nReply with ONLY:\n\`\`\`json\n{"winners":[{"match":1,"winner":"A|B"}]}\n\`\`\``,
    });
    const out = res.ok ? extractJson(res.text) : null;
    const matches = [];
    for (const w of out?.winners ?? []) {
      const p = pairs[Number(w.match) - 1];
      if (!p || !['A', 'B'].includes(w.winner)) continue;
      matches.push(w.winner === 'A' ? [p[0], p[1]] : [p[1], p[0]]);
    }
    const ratings = elo(reps.map((c) => c.id), matches.sort(() => Math.random() - 0.5));
    for (const c of pool) c.elo = Math.round(ratings.get(c.dupOf ?? c.id) ?? 1200);
    this.emit({ kind: 'claims', runId: run.id, claims: pool });
    this.persist(run);
  }

  /** Rubric judge for the round (accuracy, citations, completeness, sources, efficiency). */
  async judge(run) {
    if (!run.report) return;
    const res = await this.agent(run, {
      id: 'judge', role: 'judge', title: 'Score round', parent: 'synth', pref: 'any',
      prompt: `[[ROLE:judge]]\nScore this research report for the task "${run.task}" from 0.0 to 1.0 on: factual accuracy (claims match the ledger verdicts), citation accuracy, completeness, source quality, efficiency (${run.agents.size} agents used). Then give an overall score.\n\nReport:\n${String(run.report).slice(0, 12000)}\n\nReply with ONLY:\n\`\`\`json\n{"accuracy":0.0,"citations":0.0,"completeness":0.0,"sources":0.0,"efficiency":0.0,"overall":0.0,"why":"one sentence"}\n\`\`\``,
    });
    const s = res.ok ? extractJson(res.text) : null;
    if (!s || typeof s.overall !== 'number') return;
    run.score = s;
    writeFileSync(join(run.root, 'score.json'), JSON.stringify(s, null, 2));
    try { appendFileSync(join(this.settings.dir, '.ratchet', 'scores.jsonl'), JSON.stringify({ t: Date.now(), run: run.id, task: run.baseTask ?? run.task, round: run.chain?.round ?? 1, ...s, agents: run.agents.size, tokens: run.usage?.tokens ?? 0, cost: run.usage?.cost ?? 0 }) + '\n'); } catch { /* best effort */ }
    this.emit({ kind: 'run.score', runId: run.id, score: s });
  }

  async synth(run) {
    const shown = run.claims.filter((c) => c.verdict !== 'quarantined').sort((a, b) => (b.elo ?? 0) - (a.elo ?? 0));
    const ledger = shown.map((c) => `- [${c.verdict}${c.elo ? `, rank ${c.elo}` : ''}] (${c.id}, by ${c.by}${c.checkedBy ? `, checked by ${c.checkedBy}` : ''}) ${c.text} | source: ${c.source}${c.note ? ` | verifier: ${c.note}` : ''}${c.dupOf ? ` | duplicate of ${c.dupOf}` : ''}`).join('\n');
    const quarantined = run.claims.filter((c) => c.verdict === 'quarantined').length;
    const res = await this.agent(run, {
      id: 'synth', role: 'lead', title: 'Synthesis', parent: 'lead', pref: 'claude',
      prompt: `[[ROLE:synth]]\nWrite the final research report in Markdown.\n${this.ctx(run)}\n\nClaim ledger, highest-ranked first (verdicts come from an independent model; rank comes from a pairwise tournament):\n${ledger || '(no claims were produced)'}\n${quarantined ? `\n${quarantined} claims were quarantined for integrity problems and must not be used.\n` : ''}\nResearcher notes are in ${join(run.root, 'findings')}. Rules: lead with the highest-ranked confirmed findings; state confirmed claims plainly; flag unclear ones as unverified; do NOT present refuted claims as results, list them separately as dead ends with the reason; merge duplicates. Cite claim ids. End with open questions and the best next experiments.\n\nAfter the report, add ONE JSON block for the next round:\n\`\`\`json\n{"meta_review":["recurring weakness in this round's research, phrased as an instruction for the next round"],"next_lines":["most valuable next line of attack"]}\n\`\`\``,
    });
    if (!res.ok) return { report: `# Report (fallback: synthesis failed)\n\n${res.error ?? ''}\n\n## Ledger\n${ledger}\n`, meta: [], next: [] };
    const j = extractJson(res.text);
    const report = res.text.replace(/```json\s*\{[\s\S]*?"meta_review"[\s\S]*?```\s*$/m, '').trim();
    return { report, meta: (j?.meta_review ?? []).map(String).slice(0, 8), next: (j?.next_lines ?? []).map(String).slice(0, 8) };
  }

  persist(run) {
    try { writeFileSync(join(run.root, 'claims.json'), JSON.stringify(run.claims, null, 2)); } catch { /* best effort */ }
  }
}
