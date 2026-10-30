# Ratchet

Swarm research harness with a web terminal. Point it at a folder, give it a task, and it runs many agents across many paid accounts and several harnesses (Claude Code headless, Codex SDK, Antigravity, and others). Sister tool to [Evidra](../evidra): Evidra is the durable single-campaign controller, Ratchet is the wide, fast fan-out layer with a live UI.

## Install

Requires Node >= 22 and at least one harness CLI (`claude`, `codex`, `agy`).

```bash
gh repo clone StarAtNyte/ratchet ~/.local/share/ratchet
~/.local/share/ratchet/install.sh
```

Or without the script:

```bash
git clone git@github.com:StarAtNyte/ratchet.git && cd ratchet && npm install && npm link
```

Update: `git -C ~/.local/share/ratchet pull && ~/.local/share/ratchet/install.sh`

## Run

```bash
ratchet ~/path/to/workspace      # opens the web terminal on http://127.0.0.1:7777
ratchet --mock                   # simulated agents (no quota) to try the UI
```

The URL carries a random token; the server binds to localhost only.

## Accounts

Each account is an isolated config folder. In the web terminal:

```
account add claude        # then:  CLAUDE_CONFIG_DIR=<printed dir> claude   (run /login)
account add codex         # account login codex-2   (device-code login)
account add antigravity   # experimental isolation, see below
account probe             # which accounts are logged in
accounts                  # status, load, reset countdowns
```

Defaults (`claude-1`, `codex-1`, `antigravity-1`) use your existing logins. When an account hits a usage limit its task fails over to another account; the card shows a reset countdown.

| Harness | How it runs | Account isolation |
| --- | --- | --- |
| `claude` | `claude -p --output-format stream-json` (headless) | `CLAUDE_CONFIG_DIR` |
| `codex` | official `@openai/codex-sdk` threads | `CODEX_HOME` |
| `antigravity` | `agy --print` (plain text) | `ANTIGRAVITY_GEMINI_DIR` (unverified) |
| `gemini`, `opencode` | plain-text CLIs (built in, untested here) | env var best guess |
| custom | `~/.ratchet/harnesses.json` | your `envKey` |

Custom harness example:

```json
{ "aider": { "cmd": "aider", "args": ["--message", "{prompt}", "--yes"], "envKey": "AIDER_HOME" } }
```

## Research swarm

Type a task. A lead plans independent lines of attack, workers run in parallel (and may spawn sub-agents), every claim is checked by an agent from a different harness, and the lead writes `REPORT.md`. Refuted claims are saved to `.ratchet/memory.jsonl` and fed to later planners as dead ends. Everything lives in `<workspace>/.ratchet/runs/`.

## Optimise loop (the ratchet)

AlphaEvolve / KernelEvolve style evolutionary search for kernels and other measurable targets.

```
opt init          # writes ratchet.json
opt start         # baseline, then parallel mutate -> evaluate -> archive
opt apply         # copy the champion's target files into the workspace
```

`ratchet.json`:

```json
{ "objective": "make softmax faster", "targets": ["kernel.py"],
  "eval": ["python check.py", "python bench.py"], "holdout": "python holdout.py",
  "direction": "min", "parallel": 4, "strongEvery": 4, "maxIterations": 40,
  "budgetMinutes": 60, "timeoutSec": 300 }
```

The last stdout line of the final `eval` command is `{"score": 1.23, "ok": true}` (or a bare number).

- **Program archive**: islands x method-family cells keep the best valid program per cell; parents are rank-sampled, inspirations come from other families (AlphaEvolve's MAP-Elites + islands idea).
- **Ratchet**: a candidate becomes champion only if valid and strictly better, and the hidden `holdout` does not regress.
- **Evaluator is fixed**: candidates edit a copy; only declared `targets` are overlaid onto a pristine copy of the workspace before scoring, so edits to the evaluator are ignored.
- **Evaluation cascade**: `eval` is a list; later (expensive) stages run only if earlier ones pass. Evaluations are serialised for exclusive hardware.
- **Model tiering**: most mutations use the fast lane; every `strongEvery`-th uses a strong model (AlphaEvolve's Flash + Pro mix).
- **Failure memory**: failed ideas are fed back so workers do not repeat them; in-flight parents are avoided to reduce duplicate work.

## Commands

`help`, `run <task>`, `opt init|start|apply`, `dir`, `accounts`, `account add|login|probe|rm`, `runs`, `stop`, `report`, `mode auto|yolo`, `set slots|workers n`, `clear`.

`mode yolo` runs agents without a sandbox. Using several paid subscriptions may conflict with providers' terms; check them.

## Test

```bash
npm test
```

## Roadmap

Meta-prompt / harness evolution between rounds, shared claim board across machines, Evidra campaign as a worker type, richer archive view in the UI.
