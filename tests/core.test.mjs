import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Archive } from '../src/archive.mjs';
import { parseReset } from '../src/harness.mjs';
import { extractJson } from '../src/swarm.mjs';
import { runEval } from '../src/optimizer.mjs';

const e = (id, score, o = {}) => ({ id, island: 0, family: 'a', ok: true, score, ...o });

test('ratchet only turns forward (min)', () => {
  const a = new Archive({ direction: 'min' });
  assert.ok(a.add(e('base', 100)));
  assert.ok(!a.add(e('c1', 120)));
  assert.ok(a.add(e('c2', 90)));
  assert.ok(!a.add(e('c3', 50, { ok: false })), 'invalid candidates never win');
  assert.ok(!a.add(e('c4', 10, { accepted: false })), 'holdout-rejected never wins');
  assert.equal(a.champion.id, 'c2');
});

test('max direction and per-family cells', () => {
  const a = new Archive({ direction: 'max' });
  a.add(e('x', 1, { family: 'f1' })); a.add(e('y', 5, { family: 'f2' })); a.add(e('z', 3, { family: 'f1' }));
  assert.equal(a.champion.id, 'y');
  assert.equal(a.cells.get('0:f1').id, 'z');
  const s = a.sample();
  assert.ok(s.parent);
});

test('sampling avoids in-flight parents when possible', () => {
  const a = new Archive({ direction: 'min', islands: 1 });
  a.add(e('p', 10, { family: 'a' })); a.add(e('q', 12, { family: 'b' }));
  for (let i = 0; i < 20; i++) assert.equal(a.sample(new Set(['p'])).parent.id, 'q');
});

test('extractJson reads fenced and bare blocks', () => {
  assert.deepEqual(extractJson('hi\n```json\n{"a":1}\n```'), { a: 1 });
  assert.deepEqual(extractJson('x {"b":2} y'), { b: 2 });
  assert.equal(extractJson('nothing'), null);
});

test('parseReset handles relative and clock times', () => {
  const now = Date.UTC(2026, 0, 1, 0, 0, 0);
  assert.equal(parseReset('try again in 2 hours', now), now + 2 * 36e5);
  assert.ok(parseReset('resets at 3pm', now) > now);
});

test('runEval parses json and bare-number scores, rejects failures', async () => {
  assert.deepEqual(await runEval('echo noise; echo \'{"score":1.5,"ok":true}\'', '.', 10), { ok: true, score: 1.5 });
  assert.deepEqual(await runEval('echo 2.5', '.', 10), { ok: true, score: 2.5 });
  assert.equal((await runEval('echo \'{"score":1,"ok":false}\'', '.', 10)).ok, false);
  assert.equal((await runEval('exit 3', '.', 10)).ok, false);
});
