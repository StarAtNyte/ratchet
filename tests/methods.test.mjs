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
