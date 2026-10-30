#!/usr/bin/env node
import { resolve } from 'node:path';
import { existsSync, statSync } from 'node:fs';
import { startServer } from '../src/server.mjs';

const args = process.argv.slice(2);
if (args.includes('-h') || args.includes('--help')) {
  console.log(`ratchet [folder] [--port 7777] [--mock] [--no-open]
  folder    workspace the agent swarm works in (default: cwd)
  --mock    simulated agents, burns no quota (UI demo / testing)
  --no-open do not launch the browser`);
  process.exit(0);
}
const flag = (n) => args.includes(n);
const val = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const positional = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--port');
const dir = resolve(positional[0] ?? process.cwd());
if (!existsSync(dir) || !statSync(dir).isDirectory()) { console.error(`not a folder: ${dir}`); process.exit(1); }

await startServer({ dir, port: Number(val('--port', 7777)), mock: flag('--mock'), open: !flag('--no-open') });
