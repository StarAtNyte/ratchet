import { homedir } from 'node:os';
import { join } from 'node:path';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { ENV_KEY } from './harness.mjs';

export const HOME = process.env.RATCHET_HOME ?? join(homedir(), '.ratchet');
const FILE = join(HOME, 'accounts.json');

const which = (bin) => {
  try { return execFileSync('which', [bin], { encoding: 'utf8' }).trim() || null; } catch { return null; }
};

const fresh = (a) => ({ ...a, status: 'idle', load: 0, limitedUntil: 0, task: null, done: 0, failed: 0, authed: null });

export function pub(a) {
  return {
    id: a.id, harness: a.harness, status: a.status, load: a.load, limitedUntil: a.limitedUntil,
    task: a.task, done: a.done, failed: a.failed, authed: a.authed, configDir: a.configDir,
  };
}

/** Registry of paid accounts plus their live runtime state. */
export class AccountPool {
  constructor(emit, { mock = false } = {}) {
    this.emit = emit;
    this.mock = mock;
    this.accounts = [];
    mkdirSync(HOME, { recursive: true });
    if (mock) {
      this.accounts = ['claude-1', 'claude-2', 'codex-1', 'codex-2', 'codex-3', 'antigravity-1']
        .map((id) => fresh({ id, harness: id.split('-')[0], configDir: null }));
    } else this.load();
  }

  load() {
    let saved = [];
    if (existsSync(FILE)) { try { saved = JSON.parse(readFileSync(FILE, 'utf8')); } catch { saved = []; } }
    if (!saved.length) {
      if (which('claude')) saved.push({ id: 'claude-1', harness: 'claude', configDir: null });
      if (which('codex')) saved.push({ id: 'codex-1', harness: 'codex', configDir: null });
      if (which('agy')) saved.push({ id: 'antigravity-1', harness: 'antigravity', configDir: null });
      writeFileSync(FILE, JSON.stringify(saved, null, 2));
    }
    this.accounts = saved.map(fresh);
  }

  save() {
    if (this.mock) return;
    const slim = this.accounts.map(({ id, harness, configDir }) => ({ id, harness, configDir }));
    writeFileSync(FILE, JSON.stringify(slim, null, 2));
  }

  list() { return this.accounts; }
  get(id) { return this.accounts.find((a) => a.id === id); }
  publish(a) { this.emit({ kind: 'account', account: { ...pub(a), hint: a.configDir ? this.loginHint(a) : '' } }); }
  publishAll() { this.accounts.forEach((a) => this.publish(a)); }

  add(harness) {
    let i = this.accounts.filter((a) => a.harness === harness).length + 1;
    while (this.get(`${harness}-${i}`)) i++;
    const id = `${harness}-${i}`;
    const configDir = join(HOME, 'accounts', id);
    mkdirSync(configDir, { recursive: true });
    const a = fresh({ id, harness, configDir });
    this.accounts.push(a);
    this.save();
    this.publish(a);
    return a;
  }

  remove(id) {
    const i = this.accounts.findIndex((a) => a.id === id);
    if (i < 0) return false;
    this.accounts.splice(i, 1);
    this.save();
    this.emit({ kind: 'account.remove', id });
    return true;
  }

  env(a) {
    if (!a?.configDir) return {};
    const key = ENV_KEY[a.harness];
    return key ? { [key]: a.configDir } : {};
  }

  loginHint(a) {
    if (a.harness === 'claude') return `CLAUDE_CONFIG_DIR=${a.configDir} claude   # then run /login`;
    if (a.harness !== 'codex') return `${ENV_KEY[a.harness] ? ENV_KEY[a.harness] + '=' + a.configDir + ' ' : ''}${a.harness === 'antigravity' ? 'agy' : a.harness}   # sign in; isolation of non-claude/codex accounts is experimental`;
    return `CODEX_HOME=${a.configDir} codex login --device-auth`;
  }

  free(a, cap) {
    return a.limitedUntil <= Date.now() && a.load < cap;
  }

  markLimited(a, resetAt) {
    a.limitedUntil = resetAt;
    a.status = 'limited';
    this.publish(a);
  }

  refresh(a) {
    if (a.limitedUntil && a.limitedUntil <= Date.now()) a.limitedUntil = 0;
    a.status = a.limitedUntil ? 'limited' : a.load > 0 ? 'working' : 'idle';
  }
}
