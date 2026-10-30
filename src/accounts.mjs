import { homedir } from 'node:os';
import { join } from 'node:path';
import { existsSync, mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { ENV_KEY } from './harness.mjs';

export const HOME = process.env.RATCHET_HOME ?? join(homedir(), '.ratchet');
const FILE = join(HOME, 'accounts.json');

const which = (bin) => {
  try { return execFileSync('which', [bin], { encoding: 'utf8' }).trim() || null; } catch { return null; }
};

const fresh = (a) => ({ ...a, enabled: a.enabled !== false, status: a.limitedUntil > Date.now() ? 'limited' : 'idle', load: 0, limitedUntil: a.limitedUntil > Date.now() ? a.limitedUntil : 0, task: null, done: 0, failed: 0, authed: null, email: null, plan: null });

export function pub(a) {
  return {
    id: a.id, harness: a.harness, enabled: a.enabled !== false, status: a.status, load: a.load, limitedUntil: a.limitedUntil,
    task: a.task, done: a.done, failed: a.failed, authed: a.authed, checking: !!a.checking, authMessage: a.authMessage ?? null, email: a.email, plan: a.plan, configDir: a.configDir,
  };
}

// Read only identity claims from the local Codex login; never publish tokens.
export function codexIdentity(configDir) {
  try {
    const auth = JSON.parse(readFileSync(join(configDir || process.env.CODEX_HOME || join(homedir(), '.codex'), 'auth.json'), 'utf8'));
    const token = auth.tokens?.id_token;
    if (typeof token !== 'string') return {};
    const claims = JSON.parse(Buffer.from(token.split('.')[1], 'base64url').toString('utf8'));
    const details = claims['https://api.openai.com/auth'] || {};
    return { email: typeof claims.email === 'string' ? claims.email : null, plan: typeof details.chatgpt_plan_type === 'string' ? details.chatgpt_plan_type : null };
  } catch { return {}; }
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
    let saved;
    if (existsSync(FILE)) {
      saved = JSON.parse(readFileSync(FILE, 'utf8'));
      if (!Array.isArray(saved)) throw new Error(`Invalid account registry: ${FILE}`);
    } else {
      saved = [];
      if (which('claude')) saved.push({ id: 'claude-1', harness: 'claude', configDir: null });
      if (which('codex')) saved.push({ id: 'codex-1', harness: 'codex', configDir: null });
      if (which('agy')) saved.push({ id: 'antigravity-1', harness: 'antigravity', configDir: null });
      this.writeRegistry(saved);
    }
    this.accounts = saved.map((entry) => { const a = fresh(entry); if (a.harness === 'codex') Object.assign(a, codexIdentity(a.configDir)); return a; });
  }

  writeRegistry(entries) {
    const temp = `${FILE}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
    writeFileSync(temp, JSON.stringify(entries, null, 2), { mode: 0o600 });
    renameSync(temp, FILE);
  }

  // Keep live state for known accounts while importing changes made by another instance.
  syncRegistry() {
    if (this.mock || !existsSync(FILE)) return;
    const saved = JSON.parse(readFileSync(FILE, 'utf8'));
    if (!Array.isArray(saved)) throw new Error(`Invalid account registry: ${FILE}`);
    const existing = new Map(this.accounts.map((a) => [a.id, a]));
    this.accounts = saved.map((entry) => {
      const a = existing.get(entry.id);
      if (a && a.harness === entry.harness && a.configDir === entry.configDir) { a.enabled = entry.enabled !== false; a.limitedUntil = Math.max(a.limitedUntil || 0, entry.limitedUntil || 0); return a; }
      const restored = fresh(entry);
      if (restored.harness === 'codex') Object.assign(restored, codexIdentity(restored.configDir));
      return restored;
    });
  }

  save() {
    if (this.mock) return;
    this.writeRegistry(this.accounts.map(({ id, harness, configDir, enabled, limitedUntil }) => ({ id, harness, configDir, enabled: enabled !== false, limitedUntil: limitedUntil > Date.now() ? limitedUntil : 0 })));
  }

  list() { return this.accounts; }
  get(id) { return this.accounts.find((a) => a.id === id); }
  publish(a) { this.emit({ kind: 'account', account: { ...pub(a), hint: a.configDir ? this.loginHint(a) : '' } }); }
  publishAll() { this.syncRegistry(); this.accounts.forEach((a) => this.publish(a)); }

  add(harness) {
    this.syncRegistry();
    let i = this.accounts.filter((a) => a.harness === harness).length + 1;
    while (this.get(`${harness}-${i}`)) i++;
    const id = `${harness}-${i}`;
    const configDir = join(HOME, 'accounts', id);
    mkdirSync(configDir, { recursive: true });
    if (harness === 'antigravity') mkdirSync(join(configDir, 'run'), { recursive: true, mode: 0o700 });
    const a = fresh({ id, harness, configDir });
    this.accounts.push(a);
    this.save();
    this.publish(a);
    return a;
  }

  remove(id) {
    this.syncRegistry();
    const i = this.accounts.findIndex((a) => a.id === id);
    if (i < 0) return false;
    this.accounts.splice(i, 1);
    this.save();
    this.emit({ kind: 'account.remove', id });
    return true;
  }

  setEnabled(id, enabled) {
    this.syncRegistry();
    const account = this.get(id);
    if (!account) return false;
    account.enabled = enabled;
    this.save();
    this.publish(account);
    return true;
  }

  env(a) {
    if (!a?.configDir) return {};
    // agy keeps its token in the system keyring; hiding the keyring makes it use a separate login
    if (a.harness === 'antigravity') return { HOME: a.configDir, XDG_RUNTIME_DIR: join(a.configDir, 'run'), DBUS_SESSION_BUS_ADDRESS: null };
    const key = ENV_KEY[a.harness];
    return key ? { [key]: a.configDir } : {};
  }

  loginHint(a) {
    if (a.harness === 'claude') return `CLAUDE_CONFIG_DIR=${a.configDir} claude   # then run /login`;
    if (a.harness === 'antigravity') return a.configDir ? 'sign in from the account panel (Google sign-in)' : 'uses the Antigravity app sign-in on this machine';
    if (a.harness !== 'codex') return `${ENV_KEY[a.harness] ? ENV_KEY[a.harness] + '=' + a.configDir + ' ' : ''}${a.harness}   # sign in once; isolation for this harness is unverified`;
    return `CODEX_HOME=${a.configDir} codex login --device-auth`;
  }

  free(a, cap) {
    return a.limitedUntil <= Date.now() && a.load < cap;
  }

  markLimited(a, resetAt) {
    this.syncRegistry();
    const account = this.get(a.id);
    if (!account) return;
    account.limitedUntil = Math.max(account.limitedUntil || 0, resetAt);
    account.enabled = false;
    account.status = 'limited';
    this.save();
    this.publish(account);
  }

  refresh(a) {
    if (a.limitedUntil && a.limitedUntil <= Date.now()) a.limitedUntil = 0;
    a.status = a.limitedUntil ? 'limited' : a.load > 0 ? 'working' : 'idle';
  }
}
