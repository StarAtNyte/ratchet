#!/usr/bin/env node
import { resolve, join } from 'node:path';
import { existsSync, statSync, readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';

const HOME = process.env.RATCHET_HOME ?? join(homedir(), '.ratchet');
const UNIT = join(homedir(), '.config', 'systemd', 'user', 'ratchet.service');
const args = process.argv.slice(2);
const flag = (n) => args.includes(n);
const val = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };

if (flag('-v') || flag('--version')) {
  console.log(JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).version);
  process.exit(0);
}
if (flag('-h') || flag('--help')) {
  console.log(`ratchet [folder] [--port 7777] [--mock] [--no-open]
ratchet service install|uninstall|status|restart|logs
ratchet url | stop

  Ratchet runs in the background: closing the terminal or the browser does not stop it or its runs.
  It keeps going until you stop runs in the app, run  ratchet stop , or shut the computer down.

  folder     workspace the agent swarm works in (default: cwd; as a service: the last workspace)
  --mock     simulated agents, burns no quota (UI demo / testing)
  --no-open  do not launch the browser
  --foreground  stay attached to this terminal (logs here, Ctrl+C stops)
  service    run Ratchet in the background as a systemd user service (starts at login, restarts on crash)
  url        print the link to the running Ratchet`);
  process.exit(0);
}

const sh = (cmd, a, opts = {}) => execFileSync(cmd, a, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...opts });
const urlFile = join(HOME, 'url');

if (args[0] === 'url') {
  if (!existsSync(urlFile)) { console.error('Ratchet is not running (no url recorded). Start it with: ratchet  or  ratchet service install'); process.exit(1); }
  console.log(readFileSync(urlFile, 'utf8').trim());
  process.exit(0);
}

if (args[0] === 'stop') {
  let pid = 0;
  try { pid = Number(readFileSync(join(HOME, 'pid'), 'utf8')); } catch { /* none */ }
  if (!pid) { console.log('Ratchet is not running'); process.exit(0); }
  try { process.kill(pid, 'SIGTERM'); console.log(`stopped Ratchet (pid ${pid})`); } catch { console.log('Ratchet was not running'); }
  rmSync(join(HOME, 'pid'), { force: true }); rmSync(urlFile, { force: true });
  process.exit(0);
}

if (args[0] === 'service') {
  const sub = args[1] ?? 'status';
  const systemctl = (...a) => sh('systemctl', ['--user', ...a]);
  try { sh('systemctl', ['--user', '--version']); } catch { console.error('systemd user services are not available on this machine'); process.exit(1); }
  if (sub === 'install') {
    const port = Number(val('--port', 7777));
    const bin = fileURLToPath(import.meta.url);
    mkdirSync(join(homedir(), '.config', 'systemd', 'user'), { recursive: true });
    writeFileSync(UNIT, `[Unit]
Description=Ratchet swarm research web terminal
After=default.target

[Service]
Type=simple
ExecStart=${process.execPath} ${bin} --service --no-open --port ${port}
Restart=on-failure
RestartSec=3
WorkingDirectory=${homedir()}
Environment=PATH=${process.env.PATH}

[Install]
WantedBy=default.target
`);
    systemctl('daemon-reload');
    systemctl('enable', '--now', 'ratchet.service');
    console.log(`installed and started: ${UNIT}`);
    console.log('Ratchet now runs in the background and starts again at every login.');
    console.log('To keep it running even while you are logged out:  loginctl enable-linger $USER');
    setTimeout(() => { try { console.log(`open: ${readFileSync(urlFile, 'utf8').trim()}`); } catch { console.log('open: run  ratchet url  in a moment'); } process.exit(0); }, 2500);
  } else if (sub === 'uninstall') {
    try { systemctl('disable', '--now', 'ratchet.service'); } catch { /* not installed */ }
    rmSync(UNIT, { force: true });
    systemctl('daemon-reload');
    console.log('removed the ratchet service');
    process.exit(0);
  } else if (sub === 'restart') {
    systemctl('restart', 'ratchet.service'); console.log('restarted'); process.exit(0);
  } else if (sub === 'logs') {
    spawn('journalctl', ['--user', '-u', 'ratchet.service', '-n', '100', '-f'], { stdio: 'inherit' });
  } else {
    try { console.log(systemctl('status', 'ratchet.service', '--no-pager', '-n', '5')); } catch (e) { console.log(e.stdout || 'not installed. Run: ratchet service install'); }
    process.exit(0);
  }
} else if (!flag('--service') && !flag('--foreground')) {
  // default: run detached so the app outlives this terminal
  const alive = async () => { try { const u = readFileSync(urlFile, 'utf8').trim(); const r = await fetch(u, { signal: AbortSignal.timeout(1500) }); return r.ok ? u : null; } catch { return null; } };
  const openUrl = (u) => { if (!flag('--no-open')) spawn('xdg-open', [u], { stdio: 'ignore', detached: true }).on('error', () => {}).unref(); };
  const running = await alive();
  const wantDir = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--port')[0];
  if (running && !wantDir) { console.log(`Ratchet is already running: ${running}`); openUrl(running); process.exit(0); }
  if (running && wantDir) {
    let pid = 0; try { pid = Number(readFileSync(join(HOME, 'pid'), 'utf8')); } catch { /* none */ }
    console.log(`Ratchet is already running; switch folders from inside the app, or run  ratchet stop  first. ${running}`); openUrl(running); process.exit(0);
  }
  mkdirSync(HOME, { recursive: true });
  try { rmSync(urlFile, { force: true }); } catch { /* none */ }
  const { openSync } = await import('node:fs');
  const out = openSync(join(HOME, 'server.log'), 'a');
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...args, '--foreground', '--no-open'], { cwd: process.cwd(), detached: true, stdio: ['ignore', out, out] });
  child.unref();
  for (let i = 0; i < 40; i++) {
    await new Promise((r) => setTimeout(r, 150));
    const u = existsSync(urlFile) ? readFileSync(urlFile, 'utf8').trim() : '';
    if (u) { console.log(`Ratchet is running in the background (pid ${child.pid}).\n  open  ${u}\n  stop  ratchet stop\n  logs  ${join(HOME, 'server.log')}`); openUrl(u); process.exit(0); }
  }
  console.error(`Ratchet did not start; see ${join(HOME, 'server.log')}`); process.exit(1);
} else {
  const { startServer } = await import('../src/server.mjs');
  const service = flag('--service');
  mkdirSync(HOME, { recursive: true });
  const positional = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--port');
  writeFileSync(join(HOME, 'pid'), String(process.pid));
  let dir = positional[0] ? resolve(positional[0]) : process.cwd();
  if (service && !positional[0]) {
    // a service has no meaningful cwd: reopen the last workspace
    try { const last = JSON.parse(readFileSync(join(HOME, 'state.json'), 'utf8')).recent?.find((d) => existsSync(d)); if (last) dir = last; } catch { dir = homedir(); }
  }
  if (!existsSync(dir) || !statSync(dir).isDirectory()) { console.error(`not a folder: ${dir}`); process.exit(1); }
  await startServer({ dir, port: Number(val('--port', 7777)), mock: flag('--mock'), open: !flag('--no-open') && !service, stableToken: service });
}
