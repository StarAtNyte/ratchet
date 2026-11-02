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

Type `run <your task>` into the terminal at the bottom and press enter. The swarm graph, account pool, live feed, claim ledger and report fill in as it runs. Type `help` for commands.

## Running in the background

`ratchet` starts the app detached from your terminal and prints its link: closing the terminal or the browser tab does not stop it or its runs. Research runs keep going round after round, and the optimiser keeps mutating, until you stop them in the app (`stop run`, or `stop all`), run `ratchet stop`, or shut the computer down. `ratchet url` prints the link again; `ratchet --foreground` keeps it attached for debugging.

To have it start at every login and restart if it crashes:

```bash
ratchet service install      # systemd user service with a stable URL
ratchet service status | restart | logs | uninstall
loginctl enable-linger $USER # optional: keep running while logged out
```

After a restart Ratchet reloads the workspace's earlier runs. A run that was cut off shows as interrupted with a **resume** button: agents that already finished are restored from disk and only the rest run again.

## The UI

The screen is built to stay calm while dozens of agents run. Details appear only when you ask for them.

| Area | What it shows |
| --- | --- |
| Header | Workspace folder (click to change), and for the selected run: round, elapsed time, tokens and reported cost. Chips show mock mode, the permission level and the connection. |
| Accounts (left) | One compact row per account: provider colour, status (idle, working, limited, signed out), a load bar while it is busy and a countdown when it is limited. Drag a row to **Not using** to keep work off it. **+ Add** opens a two-step wizard: choose the provider, then sign in. |
| Run header and graph (centre) | The task, its status and one line of numbers (agents running, claims confirmed and refuted, quality score), with **Stop**, **Resume** and **Add instruction**. Below it the swarm as a tree of readable cards; scroll to zoom, drag to pan, **Fit** to see everything, click a card to inspect it. |
| Activity and Terminal (bottom) | **Activity** (the default) streams agent updates; tool calls are hidden until you ask. **Terminal** takes every command with suggestions and Tab completion, and **Open shell** gives a real shell in the workspace. |
| Agents, Claims, Report (right) | Agents grouped by phase (Plan, Research, Verify, Rank and report). Clicking one opens the **inspector**: its account card (provider, email, model, slots, jobs, sign-in state, limit reset), the attempts timeline across accounts, its claims with verdicts, activity, result and prompt. |

## Accounts and harnesses

An account is an isolated config folder plus a harness. Ratchet saves the account registry in `~/.ratchet/accounts.json` and isolated provider configuration in `~/.ratchet/accounts/`, so accounts are available across workspaces and restarts. Connect one from the UI:

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
run <your task>        # launch research; rounds continue until you stop them (set rounds <n> to cap)
follow <instruction>   # steer the agents that have not started yet, and later rounds
```

Each round runs six steps. The methods come from recent work on multi-agent research and are cited in `src/methods.mjs`.

1. **Plan with effort scaling.** The lead classifies the question as simple (1 to 2 lines of attack), comparison (2 to 4) or complex (up to `workers`), and gives every line an objective, non-goals, a deliverable, a stop condition, a preferred harness and a difficulty. Earlier dead ends, the previous round's report and its meta-review are fed in. (Anthropic, *How we built our multi-agent research system*.)
2. **Research with structured briefs.** Each researcher sees its siblings' titles so lines do not overlap, and gets a tool-call budget from its difficulty. Only high-difficulty lines may open sub-investigations (up to three). Researchers are told not to read each other's notes; a researcher whose tool calls touch another's notes gets a strike.
3. **Verify across model families, with evidence.** Each researcher's claims go to an agent from a different harness. A confirmation must quote its evidence (a source passage or a command and its decisive output) or it is downgraded to unclear. Cited local files that do not exist, or evidence a verifier marks as fabricated, give strikes; two strikes quarantine all of that researcher's claims. (Graduated sanctions from the 2026 study of cheating in research swarms, arXiv 2609.04170.)
4. **Rank in a tournament.** Near-duplicate findings are merged, then a judge compares surviving findings pairwise and Elo ratings order them. (Google AI co-scientist.)
5. **Synthesise with a meta-review.** The report leads with the highest-ranked confirmed findings, flags unverified ones and lists refuted ones as dead ends. A meta-review of recurring weaknesses and the best next lines goes to the next round's planner.
6. **Score the round.** A rubric judge rates accuracy, citations, completeness, sources and efficiency. `scores` lists every round so you can compare rounds and settings over time.

Everything is written under `<workspace>/.ratchet/runs/<id>/`: `events.jsonl`, `findings/`, `claims.json`, `REPORT.md`, `score.json`. Runs survive restarts: an interrupted run shows a **Resume** button, and agents that already finished are restored instead of re-run.

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

- **Program archive.** Valid programs are kept in cells keyed by island and method family. Parents are sampled by rank (70% exploit, 30% explore) weighted by 1/(1 + offspring), so good but under-explored programs get chances (ShinkaEvolve). Prompts include inspirations from other families and the parent's own evaluation output.
- **The ratchet.** A candidate becomes champion only if it is valid and strictly better. If `holdout` is set, it must also not regress on it. Agents never see holdout scores.
- **Fixed evaluator.** Each mutation works in a private copy of the workspace. For scoring, only the declared `targets` are overlaid onto a clean copy, so edits to the evaluator, tests or data are ignored.
- **Evaluation cascade.** Cheap checks come first in `eval`; expensive stages run only if earlier ones pass. Evaluations run one at a time so benchmarks get exclusive hardware.
- **Novelty gate.** A candidate that is a near-copy of a recent program is rejected before it costs an evaluation, and the rejection is logged as a failed idea.
- **Model choice.** Every `strongEvery`-th mutation goes to the strong model; the rest are assigned by a UCB1 bandit over the available harnesses, rewarded when a child beats its parent.
- **Memory.** Failed ideas are logged to `failures.jsonl` and fed to later workers; parents currently being mutated are avoided to reduce duplicate work.

Output lives in `<workspace>/.ratchet/runs/<id>/opt/`: `archive.json`, `failures.jsonl`, `snap/<candidate>/`, and a `REPORT.md` with the improvement over baseline.

## Commands

| Command | Does |
| --- | --- |
| `run <task>` | Launch a research swarm |
| `follow <instruction>` | Queue instructions for subsequent agents and rounds |
| `resume <run>` | Re-run an interrupted research run, restoring finished agents |
| `scores` | Quality of each research round in this workspace |
| `opt init \| start [goal] \| apply` | Optimise loop |
| `kernel list \| check \| run` | Kernel optimisation campaigns |
| `dir [path\|n]`, `dir recent`, `dir mk <path>` | Show, change, list recent, or create-and-switch the workspace |
| `accounts`, `account add\|login\|probe\|enable\|disable\|rm` | Manage the pool |
| `runs`, `stop [id\|all]`, `report [id]` | Inspect and control runs |
| `mode yolo\|auto` | `yolo` (default) = full permissions; `auto` = workspace-write sandbox |
| `set slots\|workers\|timeout\|rounds\|budget\|cost <n>` | Jobs per account, lines of attack, agent time limit in minutes, rounds per run (0 = until stopped), agents per run (default 80), USD per run (0 = no cap) |
| `clear` | Clear the dashboard view |

## Safety and operations

- The server listens on `127.0.0.1` only, checks the `Host` header and requires the per-launch token (compared in constant time) on every API and on the event stream. Pages carry a content security policy, `nosniff`, `no-referrer` and `frame-ancestors 'none'`.
- Agents run with **full permissions by default** (`yolo`): Codex with no sandbox and no approvals (the SDK equivalent of `codex --yolo`), Claude Code and Antigravity with `--dangerously-skip-permissions`. They can run any command and edit any file your user can. Type `mode auto` (or start with `RATCHET_MODE=auto`) for the sandboxed mode.
- Budgets: every run stops opening agents after `set budget` agents (default 80) and stops entirely when reported cost reaches `set cost` dollars. Each agent is stopped after `set timeout` minutes (default 30).
- `GET /api/health` returns `{ok, version, uptimeSec}` without a token for service managers. `GET /api/metrics?t=<token>` returns runs, agents, token and cost totals and per-account state.
- `~/.ratchet/ratchet.log` is a JSON-lines operations log (run starts and ends, plans, scores, failovers, strikes, agent failures), rotated at 5 MB.

## Status

Version 1.0. Verified: every unit and integration test in CI; the full research round (plan, research, verify, rank, synthesise, judge) and the optimise loop with mock agents; real Claude Code headless and Antigravity calls; a real multi-account research round; sign-in flows for Claude Code and Codex. Not yet verified end to end: a real kernel optimisation on a GPU, `gemini` and `opencode` harnesses, and isolation of a second Antigravity account. Provider limit messages and reset times are parsed on a best-effort basis.

## Development

```bash
npm test          # node --test tests/*.test.mjs
npm run demo      # ratchet --mock
```

Plain ES modules with no build step. Layout: `src/harness.mjs` (CLI/SDK adapters), `src/accounts.mjs` (pool), `src/swarm.mjs` (research orchestration and failover), `src/archive.mjs` + `src/optimizer.mjs` (optimise loop), `src/server.mjs` + `src/commands.mjs` (web terminal backend), `ui/` (frontend).

## Roadmap

- **Harness evolution.** Version the lead, worker and verifier prompts, compare each change against the previous version on a held-out question set with the rubric judge, and keep only changes that win (recursive harness self-improvement, arXiv 2607.15524; admission on fresh validation data, arXiv 2610.18813).
- **Embedding novelty.** Replace the word-shingle similarity in the novelty gate and duplicate merging with embeddings when a local embedding model is available.
- **Quota readouts.** Show provider quota where the CLIs expose it.
- **Fonts.** Bundle the interface fonts so the intended typography renders everywhere.

## Prior art

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the keep-or-discard experiment loop, [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) for effort scaling and delegation briefs, Google's AI co-scientist for tournament ranking and meta-review, [ShinkaEvolve](https://arxiv.org/abs/2509.19349) for novelty rejection and offspring-weighted sampling, arXiv 2609.04170 for quarantine and graduated sanctions, [AlphaEvolve](https://arxiv.org/abs/2506.13131) for the program database and model ensemble, and [KernelEvolve](https://arxiv.org/abs/2512.23236) for graph search with retrieval-augmented context.

## Kernel optimization campaigns

Ratchet includes its adapted ForgeKernel engine in `src/forgekernel` with its task templates, prompts, skills, and hardware knowledge base. Its code and resources are licensed under Apache 2.0; see `LICENSE-forgekernel` and `NOTICE-forgekernel`.

Use **Kernel** in the Agents panel to select a built-in task or enter your own task directory. You can also use the Terminal:

```
kernel list
kernel check builtin:muse_prefill_boundary
kernel run builtin:muse_prefill_boundary
kernel check --device modal:H200 /path/to/task
kernel run --device local:0 @/path/to/task
```

The existing `forge` commands remain available as aliases. `kernel run` accepts a task directory path, optionally prefixed with `@` (for example, `@/path/to/task` or `@tasks/my-task`). It requires a task directory containing `task.toml`, rather than an arbitrary project folder; tasks define the kernel, evaluator, and optimization budget.

The first check or campaign automatically prepares a Python virtual environment in `~/.ratchet/kernel-venv` and installs the Modal client. Python 3.11+ and the Claude CLI are required. Machine-specific Forge settings and credentials may be placed in `~/.ratchet/forge.env` using the format in `local.env.example`; the process environment takes precedence. Select **Use configured device**, a Modal GPU type, or a local CUDA device for each launch. Task directories specify architecture, shapes, reference, evaluator, and budgets. Match the task architecture to the selected GPU; a dry run checks configuration but does not validate GPU availability.

Each task runs an independent ForgeKernel campaign with an eligible Claude account reserved from Ratchet's account pool. Its selected model overrides the task's teacher and student models; leave the account model empty to use the task configuration. Campaign output appears in the graph and Activity, with artifacts and reports in the Ratchet run's `forge/` directory. ForgeKernel's GPU-job lock coordinates its own jobs. Remote-job cancellation, cross-host benchmark isolation, and end-to-end GPU execution have not yet been verified. `stop all` terminates local campaign processes; check your provider for any remote jobs they started.
