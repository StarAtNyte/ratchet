<h1 align="center">RATCHET</h1>

<p align="center"><b>A swarm research harness with a live web terminal.</b><br>
Point it at a folder, hand it a task, and it fans the work out across many agents, harnesses and paid accounts.<br>
Claims get cross-checked by a different model family, and measurable targets are optimised by an evolutionary loop that only ever turns forward.</p>

---

## Why

A single coding agent in a single account hits two walls: it runs out of quota, and it checks its own work. Ratchet is built around removing both.

- **Many accounts, one pool.** Register as many Claude Code, Codex and Antigravity logins as you own. Ratchet spreads work across them, tracks which are limited, and fails a task over to another account when one runs dry.
- **Many harnesses, real cross-checking.** Different harnesses run different model families. Every claim a researcher makes is re-checked by an agent from a *different* harness, so a result is only as confirmed as an independent model could reproduce it.
- **A ratchet for measurable work.** For kernels and other things with a score, an AlphaEvolve-style loop mutates, evaluates and archives candidates. A change only becomes champion if it is valid, strictly better, and does not regress on a hidden holdout.

Ratchet is the wide, fast fan-out layer with a UI. It is the sister tool to [Evidra](../evidra), which is the durable, auditable controller for one campaign. Ratchet keeps a light claim ledger; heavyweight evidence gates belong in Evidra.

## Install

Requires **Node 22+** and at least one harness CLI (`claude`, `codex` or `agy`) that you are already logged in to.

```bash
curl -fsSL https://raw.githubusercontent.com/StarAtNyte/ratchet/main/install.sh | sh
ratchet --mock
```

The installer clones the repo, packs it, and installs `ratchet` as a global npm command (into the default npm prefix, or `~/.local` when that is not writable). Set `RATCHET_REF` to install a branch or tag. Or install straight from GitHub:

```bash
npm install --global https://github.com/StarAtNyte/ratchet.git
```

From a checkout instead:

```bash
git clone https://github.com/StarAtNyte/ratchet.git && cd ratchet && npm install && npm link
```

Update by re-running the same one-liner.

## Quick start

```bash
ratchet --mock                  # simulated agents, burns no quota: try the UI first
ratchet ~/path/to/workspace     # real agents; opens http://127.0.0.1:7777
```

Run it from anywhere. With no folder it uses the current directory; you can change the workspace at any time without restarting by clicking the folder path in the header (folder browser, recent list, new folder) or with `dir <path>`. Recent workspaces are remembered in `~/.ratchet/state.json`. A yellow **MOCK** badge in the header shows when simulated agents are on.

Flags: `--port <n>`, `--mock`, `--no-open`. The page URL carries a random token and the server binds to localhost only.

Type a research task into the terminal at the bottom and press enter. The swarm graph, account pool, live feed, claim ledger and report fill in as it runs. Type `help` for commands.

## The UI

| Area | What it shows |
| --- | --- |
| Account pool (left) | One card per account: status LED, concurrency pips, current task, jobs done/failed, reset countdown when limited. |
| Swarm graph (centre) | The lead, workers, sub-agents, verifiers and optimiser candidates as a live tree. Particles flow along edges of running agents; champions are ringed; click a node to filter the feed to it. |
| Terminal (bottom) | Every action is a command. History with the arrow keys, tab completion. |
| Live / Claims / Report (right) | Streaming agent activity; the claim ledger with verdicts; the final Markdown report. |

## Accounts and harnesses

An account is an isolated config folder plus a harness. Connect one from the UI:

1. In the **Accounts** panel click **+ Add** and pick a harness.
2. The account's details open. For Claude and Codex, sign-in starts automatically: click **open sign-in page**, approve in the browser, and for Claude paste the code it shows into the box (Codex shows a device code in the log). The account flips to *signed in*.
3. Click any account later to re-run **sign in**, **check login**, or **remove** it.

Antigravity (and other plain-text CLIs) cannot be signed in automatically; the drawer shows the exact command to run once in a terminal. The same actions exist as terminal commands:

```
account add claude | codex | antigravity
account login <id>         # codex device-code login
account probe              # which accounts are logged in
accounts                   # status, load, done/failed, reset times
account rm <id>
```

The first start auto-registers `claude-1`, `codex-1` and `antigravity-1` for the CLIs it finds, using your existing logins.

| Harness | How Ratchet drives it | Per-account isolation |
| --- | --- | --- |
| `claude` | Claude Code headless: `claude -p --output-format stream-json` | `CLAUDE_CONFIG_DIR` |
| `codex` | The official [`@openai/codex-sdk`](https://www.npmjs.com/package/@openai/codex-sdk), one thread per task | `CODEX_HOME` |
| `antigravity` | `agy --print` (plain text) | `ANTIGRAVITY_GEMINI_DIR` (unverified) |
| `gemini`, `opencode` | Plain-text CLIs, built in | Best-guess env var (untested) |
| your own | Entries in `~/.ratchet/harnesses.json` | Your `envKey` |

Custom harness:

```json
{ "aider": { "cmd": "aider", "args": ["--message", "{prompt}", "--yes"], "yoloArgs": [], "envKey": "AIDER_HOME" } }
```

**Failover.** If a run fails with a usage or rate limit, the account is marked limited until its parsed reset time (30 minutes if none can be parsed) and the task is retried on another account. Other failures retry on a different harness. If every account is limited, the task waits.

**Terms of service.** Running several paid subscriptions side by side may conflict with a provider's terms on circumventing usage limits. Check each provider's terms; that call is yours.

## Add an agent by hand

Open the **Agents** tab and click **+ Add agent**: pick `auto`, `claude`, `codex` or another harness, optionally pin a specific account, name it, and describe the task. It runs on the workspace, joins the current run (or opens its own), shows on the graph, and its result and log are in its details. A pinned account is never swapped for another on failure.

## Research swarm

```
run <your task>        # or just type the task
```

1. **Plan.** A lead inspects the workspace and splits the task into independent lines of attack, each tagged with a preferred harness. Known dead ends from earlier runs (`.ratchet/memory.jsonl`) are included so it does not repeat them.
2. **Research.** Workers run in parallel on the account pool. A worker may spawn up to three sub-agents (depth 2); their claims merge into its own.
3. **Verify.** Each worker's claims are checked by an agent from a different harness, which returns `confirmed`, `refuted` or `unclear` per claim.
4. **Synthesise.** The lead writes `REPORT.md`: confirmed claims stated plainly, unclear ones flagged, refuted ones listed as dead ends, with claim ids cited. Refuted claims are saved to memory.

Everything is written under `<workspace>/.ratchet/runs/<id>/`: `findings/`, `claims.json`, `REPORT.md`.

## Optimise loop: the ratchet

For targets with a number attached (kernel latency, a benchmark, a loss), modelled on [AlphaEvolve](https://arxiv.org/abs/2506.13131) and [KernelEvolve](https://arxiv.org/abs/2512.23236).

```
opt init       # write ratchet.json in the workspace
opt start      # evaluate the baseline, then run parallel mutate -> evaluate -> archive lanes
opt apply      # copy the champion's target files into the workspace
```

`ratchet.json`:

```json
{
  "objective": "make softmax faster",
  "targets": ["kernel.py"],
  "eval": ["python check.py", "python bench.py"],
  "holdout": "python holdout.py",
  "direction": "min",
  "parallel": 4,
  "strongEvery": 4,
  "maxIterations": 40,
  "budgetMinutes": 60,
  "timeoutSec": 300
}
```

`eval` is a command or a list of commands run in the workspace. The **last stdout line of the final command** must be `{"score": 1.23, "ok": true}` or a bare number; earlier commands only need to exit 0.

How it works:

- **Program archive.** Valid programs are kept in cells keyed by island and method family. Parents are rank-sampled (70% exploit, 30% explore), and prompts include inspirations from other families, to keep diversity.
- **The ratchet.** A candidate becomes champion only if it is valid and strictly better. If `holdout` is set, it must also not regress on it. Agents never see holdout scores.
- **Fixed evaluator.** Each mutation works in a private copy of the workspace. For scoring, only the declared `targets` are overlaid onto a clean copy, so edits to the evaluator, tests or data are ignored.
- **Evaluation cascade.** Cheap checks come first in `eval`; expensive stages run only if earlier ones pass. Evaluations run one at a time so benchmarks get exclusive hardware.
- **Model tiering.** Most mutations go to the fast lane; every `strongEvery`-th goes to a strong model.
- **Memory.** Failed ideas are logged to `failures.jsonl` and fed to later workers; parents currently being mutated are avoided to reduce duplicate work.

Output lives in `<workspace>/.ratchet/runs/<id>/opt/`: `archive.json`, `failures.jsonl`, `snap/<candidate>/`, and a `REPORT.md` with the improvement over baseline.

## Commands

| Command | Does |
| --- | --- |
| `run <task>` | Launch a research swarm (bare text does the same) |
| `opt init \| start [goal] \| apply` | Optimise loop |
| `dir [path\|n]`, `dir recent`, `dir mk <path>` | Show, change, list recent, or create-and-switch the workspace |
| `accounts`, `account add\|login\|probe\|rm` | Manage the pool |
| `runs`, `stop [id\|all]`, `report [id]` | Inspect and control runs |
| `mode auto\|yolo` | `auto` = workspace-write sandbox, `yolo` = no sandbox |
| `set slots\|workers <n>` | Jobs per account / max lines of attack per run |
| `clear` | Clear the terminal |

## Safety

- The server listens on `127.0.0.1` only, checks the `Host` header, and requires the per-launch token on both the event stream and command endpoint.
- Agents run in `auto` mode by default: Claude Code with `acceptEdits` plus a fixed tool allowlist, and Codex with a `workspace-write` sandbox. `yolo` removes the sandbox; use it only in a throwaway environment.
- Agents can run shell commands and read the web. Point Ratchet at folders you are comfortable with agents editing.

## Status

Early. Verified: the Claude Code headless and Antigravity paths with real calls, the Codex SDK integration up to authentication, the full research swarm and optimise loop in `--mock` mode, and the unit tests. Not yet verified: a real multi-account run with limit failover, a real kernel optimisation, `gemini`/`opencode` harnesses, and Antigravity per-account isolation. Parsing of provider limit messages and reset times is best-effort.

## Development

```bash
npm test          # node --test tests/*.test.mjs
npm run demo      # ratchet --mock
```

Plain ES modules with no build step. Layout: `src/harness.mjs` (CLI/SDK adapters), `src/accounts.mjs` (pool), `src/swarm.mjs` (research orchestration and failover), `src/archive.mjs` + `src/optimizer.mjs` (optimise loop), `src/server.mjs` + `src/commands.mjs` (web terminal backend), `ui/` (frontend).

## Roadmap

In priority order. Each item is scoped so it can be built and checked on its own.

**1. Reliability of real runs** (nothing else matters if real runs flake)
- Run a real two-account swarm end to end and fix what breaks; record real Claude and Codex limit messages and tighten the limit and reset-time parsers against them.
- Persist runs, claims and the archive so a restart can resume a run instead of losing it.
- Per-agent timeouts and a visible retry history; surface harness stderr in the agent details.
- Show real quota use per account where the CLIs expose it.

**2. Add-agent flow**
- Model and effort choice per agent, working-folder override, and saved templates.
- Launch several agents at once; send a follow-up message to a running agent and resume a finished one.
- Stop a single agent from its details panel.

**3. Research quality**
- Iterative rounds: plan, run, verify, then re-plan from what was refuted or left unclear.
- Memory that lasts across runs for confirmed facts as well as dead ends.
- Optimiser: per-family score curves and an archive view, meta-prompt evolution between rounds, and an Evidra campaign as a worker type.

**4. Look and layout**
- Review each screen at real data volumes (50+ agents, long logs) and fix density, label collisions on the graph, and narrow-width layouts.
- A browser-driven visual check of every drawer and modal, run before each release.
- Bundle the Console theme fonts so the intended typography renders everywhere.

## Prior art

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the keep-or-discard experiment loop, [AlphaEvolve](https://arxiv.org/abs/2506.13131) for the program database and model ensemble, and [KernelEvolve](https://arxiv.org/abs/2512.23236) for graph search with retrieval-augmented context.
