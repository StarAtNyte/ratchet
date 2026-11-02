# Changelog

## 1.0.0

First production release.

### Research swarm
- Effort scaling: the lead classifies each question (simple, comparison, complex) and sizes the swarm to it; every line of attack carries a difficulty that sets its tool-call budget and whether it may open sub-investigations.
- Structured delegation briefs with objective, non-goals, sibling lines, deliverable and stop condition, so researchers do not duplicate each other.
- Evidence-gated verification by a different model family: confirmations must quote evidence or they count as unclear.
- Integrity controls: isolation between researchers, strikes for reading peers' notes, citing missing files or fabricated evidence, and quarantine of a researcher's claims after two strikes.
- Tournament ranking: near-duplicate findings are merged and the rest are ordered by Elo from pairwise judging.
- Meta-review: each round's recurring weaknesses and best next lines feed the next round's plan.
- Rubric judge per round (accuracy, citations, completeness, sources, efficiency) and a `scores` command to compare rounds.
- Continuous rounds until stopped, follow-up instructions, resume after restart with finished agents restored.

### Optimise loop
- Novelty gate that rejects near-duplicate programs before evaluation.
- Parent sampling weighted by rank and by 1/(1 + offspring).
- UCB1 bandit over harnesses for mutations, alongside the strong-model cadence.
- Parent evaluation output fed back into mutation prompts.

### Production
- Per-run agent budget and cost budget, per-agent time limit.
- Token and cost accounting per agent, run and account (from Claude Code and Codex usage reports).
- `GET /api/health` and token-protected `GET /api/metrics`.
- Constant-time token comparison, content security policy and hardening headers.
- Rotating JSON-lines operations log at `~/.ratchet/ratchet.log`.
- GitHub Actions CI: syntax checks and the full test suite.

### Interface
- Calm three-column layout from a Superdesign draft: compact accounts, run header with Stop, Resume and Add instruction, readable agent cards in the graph, Activity first.
- Agents grouped by phase, and an inspector with the agent's account card, attempts timeline across accounts, claims, activity, result and prompt.
- Header meters for round, elapsed time, tokens and cost.
