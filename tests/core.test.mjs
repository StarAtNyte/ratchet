import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Archive } from '../src/archive.mjs';
import { parseReset, LIMIT_RE } from '../src/harness.mjs';
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

test('research starts only with the run command', async () => {
  const { execCommand } = await import('../src/commands.mjs');
  const messages = []; const started = [];
  const ctx = { say: (s) => messages.push(s), pool: { list: () => [{}] }, settings: { dir: process.cwd(), rounds: 1 }, swarm: { start: (task) => { started.push(task); return { id: 'r1' }; } } };
  await execCommand('printf hello', ctx);
  assert.deepEqual(started, []);
  assert.ok(messages.includes('hello'));
  await execCommand('run investigate a bug', ctx);
  assert.deepEqual(started, ['investigate a bug']);
});


test('cd changes workspace without launching a run', async () => {
  const { execCommand } = await import('../src/commands.mjs');
  const { dirname } = await import('node:path');
  const settings = { dir: process.cwd(), recent: [] }; const messages = [];
  const ctx = { settings, say: (s) => messages.push(s), swarm: { emit: () => {}, runs: new Map() } };
  await execCommand('cd ..', ctx);
  assert.equal(settings.dir, dirname(process.cwd()));
  assert.equal(messages.at(-1), settings.dir);
});

test('accounts survive restarts, multiple instances, and an empty registry', async () => {
  const { mkdtempSync, rmSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const { execFileSync } = await import('node:child_process');
  const home = mkdtempSync(join(tmpdir(), 'ratchet-accounts-'));
  try {
    const script = `
      import assert from 'node:assert/strict';
      import { AccountPool } from './src/accounts.mjs';
      const a = new AccountPool(() => {}), b = new AccountPool(() => {});
      const first = a.add('gemini'); const second = b.add('opencode'); const codex = a.add('codex'); a.setModel(codex.id, 'gpt-example');
      const restored = new AccountPool(() => {});
      assert.ok(restored.get(first.id)); assert.ok(restored.get(second.id)); assert.equal(restored.get(codex.id).model, 'gpt-example'); assert.throws(() => restored.setModel(first.id, 'model'), /not supported/); assert.throws(() => restored.setModel(codex.id, 'bad model'), /valid model/);
      const reset = Date.now() + 3600000; restored.markLimited(restored.get(first.id), reset); const again = new AccountPool(() => {}); assert.equal(again.get(first.id).enabled, false); assert.equal(again.get(first.id).limitedUntil, reset); assert.equal(again.get(first.id).status, 'limited'); assert.equal(again.get(second.id).enabled, true);
      for (const account of [...restored.list()]) restored.remove(account.id);
      assert.equal(new AccountPool(() => {}).list().length, 0);
    `;
    execFileSync(process.execPath, ['--input-type=module', '-e', script], { cwd: process.cwd(), env: { ...process.env, RATCHET_HOME: home } });
  } finally { rmSync(home, { recursive: true, force: true }); }
});


test('excluded accounts cannot be assigned, including when pinned', async () => {
  const { Swarm } = await import('../src/swarm.mjs');
  const disabled = { id: 'claude-1', harness: 'claude', enabled: false, authed: true, limitedUntil: 0, load: 0 };
  const pool = { list: () => [disabled], refresh: () => {}, free: () => true };
  const swarm = new Swarm({ pool, emit: () => {}, settings: { slots: 1 } });
  await assert.rejects(swarm.acquire('any', new AbortController().signal, () => {}), /no other signed-in account/);
  await assert.rejects(swarm.acquire('any', new AbortController().signal, () => {}, disabled.id), /not in use/);
});

test('interrupt stops a running shell command and its child process', async () => {
  const { execCommand } = await import('../src/commands.mjs');
  const controller = new AbortController(); const lines = [];
  const command = execCommand('sleep 30', { say: (s) => lines.push(s), settings: { dir: process.cwd() }, signal: controller.signal });
  setTimeout(() => controller.abort(), 100);
  await Promise.race([command, new Promise((_, reject) => setTimeout(() => reject(new Error('interrupt did not stop command')), 3000))]);
});


test('weekly provider limit and dated time-zone reset are recognized', () => {
  const message = "You've hit your weekly limit · resets Nov 1, 7:45am (Asia/Kathmandu)";
  assert.match(message, LIMIT_RE);
  assert.equal(parseReset(message, Date.UTC(2026, 9, 30)), Date.UTC(2026, 10, 1, 2, 0));
});

test('follow-up instructions are saved and included in future agent context', async () => {
  const { Swarm } = await import('../src/swarm.mjs');
  const { mkdtempSync, readFileSync, rmSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const root = mkdtempSync(join(tmpdir(), 'ratchet-follow-'));
  try {
    const swarm = new Swarm({ pool: {}, emit: () => {}, settings: { dir: root } });
    const run = { id: 'r1', kind: 'research', status: 'running', root, task: 'research' };
    swarm.runs.set(run.id, run);
    assert.equal(swarm.followup('submit and iterate too'), 'r1');
    assert.match(swarm.ctx(run), /submit and iterate too/);
    assert.equal(JSON.parse(readFileSync(join(root, 'instructions.jsonl'), 'utf8')).text, 'submit and iterate too');
    run.status = 'stopped'; assert.throws(() => swarm.followup('more'), /No active research run/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
