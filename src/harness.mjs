import { spawn } from 'node:child_process';

const LIMIT_RE = /(usage limit|limit reached|rate.?limit|quota|exceeded your|too many requests|\b429\b|credit balance|try again (?:in|at|later))/i;

/** Best-effort parse of "resets at 3pm" / "try again in 2 hours" into epoch ms. */
export function parseReset(text, now = Date.now()) {
  const rel = /(?:in|after)\s+(\d+)\s*(hour|hr|minute|min|day)s?/i.exec(text);
  if (rel) {
    const unit = rel[2].toLowerCase();
    const ms = unit.startsWith('d') ? 864e5 : unit.startsWith('h') ? 36e5 : 6e4;
    return now + Number(rel[1]) * ms;
  }
  const abs = /(\d{1,2})(?::(\d{2}))?\s*(am|pm)/i.exec(text);
  if (abs) {
    let h = Number(abs[1]) % 12;
    if (abs[3].toLowerCase() === 'pm') h += 12;
    const d = new Date(now);
    d.setHours(h, Number(abs[2] ?? 0), 0, 0);
    if (d.getTime() <= now) d.setDate(d.getDate() + 1);
    return d.getTime();
  }
  return now + 30 * 60 * 1000;
}

const short = (s, n = 200) => (s.length > n ? `${s.slice(0, n)}…` : s);

function toolLine(name, input) {
  const v = input?.command ?? input?.file_path ?? input?.path ?? input?.pattern ?? input?.query ?? input?.url ?? input?.description ?? '';
  return `${name} ${short(String(v).replace(/\s+/g, ' '), 140)}`;
}

function lineSplitter(onLine) {
  let buf = '';
  return (chunk) => {
    buf += chunk;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) { onLine(buf.slice(0, i)); buf = buf.slice(i + 1); }
  };
}

/**
 * Run one agent turn on a real CLI. Resolves {ok, text, limited, resetAt, error}.
 * onLog({kind:'text'|'tool'|'err'|'sys', text})
 */
export function runCli({ harness, env, cwd, prompt, mode, onLog, signal }) {
  if (harness === 'codex') return runCodexSdk({ env, cwd, prompt, mode, onLog, signal });
  if (TEXT[harness]) return runText(harness, { env, cwd, prompt, mode, onLog, signal });
  const cmd = 'claude';
  const args = ['-p', '--output-format', 'stream-json', '--verbose',
    '--permission-mode', mode === 'yolo' ? 'bypassPermissions' : 'acceptEdits',
    '--allowedTools', 'Read Write Edit Glob Grep WebSearch WebFetch Bash Task'];

  return new Promise((resolve) => {
    const childEnv = { ...process.env, ...env };
    delete childEnv.CLAUDECODE;
    delete childEnv.CLAUDE_CODE_ENTRYPOINT;
    const p = spawn(cmd, args, { cwd, env: childEnv, stdio: ['pipe', 'pipe', 'pipe'], signal });
    let final = '';
    let lastText = '';
    let errText = '';
    let isError = false;

    p.on('error', (e) => resolve({ ok: false, text: '', error: `spawn ${cmd}: ${e.message}`, limited: false }));
    p.stdin.on('error', () => {});
    p.stdin.end(prompt);

    p.stdout.setEncoding('utf8');
    p.stdout.on('data', lineSplitter((line) => {
      if (!line.trim()) return;
      let ev;
      try { ev = JSON.parse(line); } catch { onLog({ kind: 'sys', text: short(line) }); return; }
      {
        if (ev.type === 'assistant') {
          for (const b of ev.message?.content ?? []) {
            if (b.type === 'text' && b.text?.trim()) { lastText = b.text; onLog({ kind: 'text', text: short(b.text, 400) }); }
            else if (b.type === 'tool_use') onLog({ kind: 'tool', text: toolLine(b.name, b.input) });
          }
        } else if (ev.type === 'result') {
          final = ev.result ?? lastText;
          isError = !!ev.is_error;
          if (isError) errText += `${ev.result ?? ''}\n`;
        }
      }
    }));

    p.stderr.setEncoding('utf8');
    p.stderr.on('data', (d) => { errText += d; });

    p.on('close', (code) => {
      const failed = code !== 0 || isError || (!final && !lastText);
      if (!failed) return resolve({ ok: true, text: final || lastText });
      const combined = `${errText}\n${final}`;
      const limited = LIMIT_RE.test(combined);
      resolve({ ok: false, text: final || lastText, limited, resetAt: limited ? parseReset(combined) : 0, error: short(combined.trim() || `exit ${code}`, 500) });
    });
  });
}

import { readFileSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

/** Plain-text print-mode CLIs. `{prompt}` in args is replaced; envKey isolates an account's config dir (unverified for non-claude/codex). */
const BUILTIN_TEXT = {
  antigravity: { cmd: 'agy', args: (p, yolo) => [...(yolo ? ['--dangerously-skip-permissions'] : ['--mode', 'accept-edits']), `--print=${p}`], envKey: 'ANTIGRAVITY_GEMINI_DIR' },
  gemini: { cmd: 'gemini', args: (p, yolo) => ['-p', p, ...(yolo ? ['--yolo'] : [])], envKey: 'GEMINI_CLI_HOME' },
  opencode: { cmd: 'opencode', args: (p) => ['run', p], envKey: 'XDG_CONFIG_HOME' },
};
function userHarnesses() {
  const f = join(process.env.RATCHET_HOME ?? join(homedir(), '.ratchet'), 'harnesses.json');
  if (!existsSync(f)) return {};
  try {
    return Object.fromEntries(Object.entries(JSON.parse(readFileSync(f, 'utf8'))).map(([k, d]) => [k, {
      cmd: d.cmd, envKey: d.envKey, args: (p, yolo) => [...d.args, ...(yolo ? d.yoloArgs ?? [] : [])].map((a) => a.replace('{prompt}', p)),
    }]));
  } catch { return {}; }
}
export const TEXT = { ...BUILTIN_TEXT, ...userHarnesses() };
export const HARNESSES = ['claude', 'codex', ...Object.keys(TEXT)];
export const ENV_KEY = { claude: 'CLAUDE_CONFIG_DIR', codex: 'CODEX_HOME', ...Object.fromEntries(Object.entries(TEXT).map(([k, d]) => [k, d.envKey])) };

function runText(name, { env, cwd, prompt, mode, onLog, signal }) {
  const def = TEXT[name];
  return new Promise((resolve) => {
    const p = spawn(def.cmd, def.args(prompt, mode === 'yolo'), { cwd, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'], signal });
    let out = ''; let err = '';
    p.on('error', (e) => resolve({ ok: false, text: '', error: `spawn ${def.cmd}: ${e.message}`, limited: false }));
    p.stdout.setEncoding('utf8');
    p.stdout.on('data', lineSplitter((l) => { if (l.trim()) onLog({ kind: 'text', text: short(l, 300) }); }));
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.setEncoding('utf8');
    p.stderr.on('data', (d) => { err += d; });
    p.on('close', (code) => {
      if (code === 0 && out.trim()) return resolve({ ok: true, text: out });
      const combined = `${err}\n${out}`;
      const limited = LIMIT_RE.test(combined) || /capacity.?exhausted/i.test(combined);
      resolve({ ok: false, text: out, limited, resetAt: limited ? parseReset(combined) : 0, error: short(combined.trim() || `exit ${code}`, 500) });
    });
  });
}

/** Codex via the official SDK (same engine Evidra uses). One thread per call. */
async function runCodexSdk({ env, cwd, prompt, mode, onLog, signal }) {
  let final = ''; let errText = ''; let failed = false;
  try {
    const { Codex } = await import('@openai/codex-sdk');
    const codex = new Codex({ env: Object.fromEntries(Object.entries({ ...process.env, ...env }).filter(([, v]) => v != null)) });
    const thread = codex.startThread({
      workingDirectory: cwd, skipGitRepoCheck: true, approvalPolicy: 'never', networkAccessEnabled: true,
      sandboxMode: mode === 'yolo' ? 'danger-full-access' : 'workspace-write',
    });
    const { events } = await thread.runStreamed(prompt, { signal });
    for await (const ev of events) {
      const it = ev.item;
      if (ev.type === 'item.completed' && it) {
        if (it.type === 'agent_message') { final = it.text ?? ''; onLog({ kind: 'text', text: short(final, 400) }); }
        else if (it.type === 'command_execution') onLog({ kind: 'tool', text: `shell ${short(it.command ?? '', 140)}` });
        else if (it.type === 'web_search') onLog({ kind: 'tool', text: `web ${short(it.query ?? '', 140)}` });
        else if (it.type === 'file_change') onLog({ kind: 'tool', text: 'edit files' });
      } else if (ev.type === 'turn.failed') { failed = true; errText += `${ev.error?.message ?? 'turn failed'}\n`; }
      else if (ev.type === 'error') { failed = true; errText += `${ev.message ?? 'error'}\n`; }
    }
  } catch (e) { failed = true; errText += `${e.message}\n`; }
  if (!failed && final) return { ok: true, text: final };
  const limited = LIMIT_RE.test(errText);
  return { ok: false, text: final, limited, resetAt: limited ? parseReset(errText) : 0, error: short(errText.trim() || 'no output', 500) };
}
