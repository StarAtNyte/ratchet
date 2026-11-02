import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { linesFor, brief, gateVerdict, missingLocalSource, applyStrikes, elo, pairings, similarity, cluster, ucb1 } from '../src/methods.mjs';

test('effort scaling caps lines by complexity class', () => {
  assert.equal(linesFor('simple', 6, 12), 2);
  assert.equal(linesFor('comparison', 6, 12), 4);
  assert.equal(linesFor('complex', 9, 6), 6);
  assert.equal(linesFor('unknown', 3, 12), 3);
  assert.equal(linesFor('complex', 0, 12), 1);
});

test('briefs name siblings, boundaries and an effort budget', () => {
  const b = brief({ title: 't', objective: 'find X', non_goals: ['Y'], difficulty: 'low' }, ['other line']);
  assert.match(b, /Objective: find X/);
  assert.match(b, /do not duplicate: other line/);
  assert.match(b, /Out of scope.*Y/);
  assert.match(b, /low difficulty/);
});

test('confirmations without quoted evidence are downgraded', () => {
  assert.equal(gateVerdict({ verdict: 'confirmed', note: 'ok' }).verdict, 'unclear');
  assert.equal(gateVerdict({ verdict: 'confirmed', evidence: 'line 42: assert x == 3 passed' }).verdict, 'confirmed');
  assert.equal(gateVerdict({ verdict: 'refuted' }).verdict, 'refuted');
  assert.equal(gateVerdict({ verdict: 'bogus' }), null);
  assert.equal(gateVerdict({ verdict: 'refuted', fabricated: true }).fabricated, true);
});

test('missing local evidence files are detected, URLs are not', () => {
  const exists = (p) => p === 'src/real.py';
  assert.equal(missingLocalSource('see src/real.py:12', exists), null);
  assert.equal(missingLocalSource('see src/fake.py:12', exists), 'src/fake.py');
  assert.equal(missingLocalSource('https://example.com/a/b.html', exists), null);
  assert.equal(missingLocalSource('a paper', exists), null);
});

test('two strikes quarantine a researcher', () => {
  const claims = [{ id: 'w1.c0', agent: 'w1', verdict: 'confirmed' }, { id: 'w2.c0', agent: 'w2', verdict: 'confirmed' }];
  assert.deepEqual(applyStrikes(new Map([['w1', 1]]), claims), []);
  assert.deepEqual(applyStrikes(new Map([['w1', 2]]), claims), ['w1.c0']);
});

test('elo rewards winners and pairings never repeat', () => {
  const r = elo(['a', 'b', 'c'], [['a', 'b'], ['a', 'c'], ['b', 'c']]);
  assert.ok(r.get('a') > r.get('b') && r.get('b') > r.get('c'));
  const p = pairings(['a', 'b', 'c', 'd', 'e'], 4);
  const keys = p.map(([x, y]) => [x, y].sort().join('|'));
  assert.equal(new Set(keys).size, keys.length);
  assert.ok(p.every(([x, y]) => x !== y));
});

test('similarity and clustering merge near-duplicates', () => {
  assert.ok(similarity('the kernel is faster with tiling', 'the kernel is faster with tiling enabled') > 0.5);
  assert.ok(similarity('tiling speeds the kernel', 'warmup stabilises adam') < 0.1);
  const g = cluster(['softmax overflows for large logits', 'softmax overflows for large logits in fp16', 'adam needs warmup']);
  assert.equal(g.length, 2);
});

test('ucb1 tries every arm before exploiting', () => {
  assert.equal(ucb1([{ id: 'a', pulls: 3, reward: 3 }, { id: 'b', pulls: 0, reward: 0 }]).id, 'b');
  assert.equal(ucb1([{ id: 'a', pulls: 50, reward: 45 }, { id: 'b', pulls: 50, reward: 5 }]).id, 'a');
});

test('mock research round runs plan, research, verify, rank, synth and judge', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'ratchet-ws-'));
  process.env.RATCHET_HOME = mkdtempSync(join(tmpdir(), 'ratchet-home-'));
  const { AccountPool } = await import('../src/accounts.mjs');
  const { Swarm } = await import('../src/swarm.mjs');
  const events = [];
  const pool = new AccountPool((e) => events.push(e), { mock: true });
  for (const a of pool.list()) a.demoHit = true; // skip the simulated limit for speed
  const swarm = new Swarm({ pool, emit: (e) => events.push(e), settings: { dir, mock: true, mode: 'yolo', slots: 4, workers: 4, rounds: 1, runAgentBudget: 80 } });
  const run = swarm.newRun('does warmup help adam');
  await swarm.drive(run);
  assert.equal(run.status, 'done');
  assert.ok(events.some((e) => e.kind === 'run.plan' && e.complexity === 'complex'));
  assert.ok(run.claims.some((c) => typeof c.elo === 'number'), 'claims ranked');
  assert.ok(run.score && typeof run.score.overall === 'number', 'round scored');
  assert.ok(run.meta.length > 0, 'meta-review captured');
  assert.ok(!/meta_review/.test(run.report), 'meta JSON stripped from the report');
  assert.ok(run.usage.tokens > 0, 'usage accounted');
  assert.ok(existsSync(join(run.root, 'score.json')));
  assert.ok(JSON.parse(readFileSync(join(run.root, 'claims.json'), 'utf8')).length > 0);
});

test('archive counts offspring and prefers under-explored parents', async () => {
  const { Archive } = await import('../src/archive.mjs');
  let i = 0; const seq = [0.0, 0.9, 0.5]; // island 0, explore branch, then mid pick
  const a = new Archive({ direction: 'min', islands: 1, rand: () => seq[i++ % seq.length] });
  a.add({ id: 'p', island: 0, family: 'a', ok: true, score: 10 });
  a.add({ id: 'q', island: 0, family: 'b', ok: true, score: 11 });
  for (let k = 0; k < 6; k++) a.add({ id: `c${k}`, parent: 'p', island: 0, family: 'a', ok: false, score: NaN });
  assert.equal(a.entries.find((e) => e.id === 'p').children, 6);
  assert.equal(a.sample().parent.id, 'q', 'explore pick goes to the parent with fewer offspring');
});

test('scheduler spreads work to an idle provider when the preferred account is busy', async () => {
  const { Swarm } = await import('../src/swarm.mjs');
  const acc = (id, harness, load) => ({ id, harness, load, done: 0, limitedUntil: 0, authed: true, enabled: true });
  const accounts = [acc('codex-2', 'codex', 1), acc('claude-1', 'claude', 0)];
  const pool = { list: () => accounts, refresh: () => {}, free: (a, cap) => a.load < cap, publish: () => {} };
  const swarm = new Swarm({ pool, emit: () => {}, settings: { slots: 4 } });
  const ctl = new AbortController();
  assert.equal((await swarm.acquire('codex', ctl.signal, () => {})).id, 'claude-1');
  accounts[1].load = 1; accounts[0].load = 1;
  assert.equal((await swarm.acquire('codex', ctl.signal, () => {})).id, 'codex-2', 'with no idle alternative the preferred provider is used');
  accounts.forEach((a) => { a.load = 1; });
  assert.equal((await swarm.acquire('codex', ctl.signal, () => {}, null, new Set(), true)).harness, 'codex', 'verification stays strict');
});

test('real optimise path: staging outside the workspace, evaluator run and git champion commits', async () => {
  const { execFileSync } = await import('node:child_process');
  const { writeFileSync: wf, mkdirSync: md } = await import('node:fs');
  const dir = mkdtempSync(join(tmpdir(), 'ratchet-opt-ws-'));
  wf(join(dir, 'kernel.py'), 'X = 1\n');
  wf(join(dir, 'bench.py'), 'import json\nfrom kernel import X\nprint(json.dumps({"score": X, "ok": True}))\n');
  const g = (...a) => execFileSync('git', a, { cwd: dir, stdio: 'ignore' });
  g('init', '-q'); g('add', '.'); g('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'init');
  const { Optimizer } = await import('../src/optimizer.mjs');
  const events = [];
  const run = { id: 'rtest', root: join(dir, '.ratchet', 'runs', 'x'), agents: new Map(), abort: new AbortController(), status: 'running' };
  md(run.root, { recursive: true });
  const swarm = { settings: { dir, mock: false }, newRun: () => run, emit: (e) => events.push(e), end: (r, st) => { r.status = st; } };
  const opt = new Optimizer(swarm);
  // drive only the baseline and one manual champion through the real file paths
  wf(join(dir, 'ratchet.json'), JSON.stringify({ objective: 'lower X', targets: ['kernel.py'], eval: ['python3 bench.py'], direction: 'min', parallel: 0, maxIterations: 1 }));
  await opt.start();
  await new Promise((r) => setTimeout(r, 1500));
  assert.equal(run.status, 'done', 'baseline evaluated from a staging copy outside the workspace');
  const o = opt.last;
  assert.ok(!o.scratch.startsWith(dir), 'scratch is outside the workspace');
  assert.ok(readFileSync(join(o.optDir, 'results.tsv'), 'utf8').startsWith('gen\tid'), 'results log written');
  assert.equal(o.archive.champion.id, 'base');
  assert.match(String(execFileSync('git', ['branch', '--list', 'ratchet/*'], { cwd: dir })), /ratchet\/rtest/, 'git ratchet branch created');
});
