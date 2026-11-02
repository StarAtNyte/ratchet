# Ratchet design system
Operational desktop dashboard for coordinating coding-agent swarms. Calm by default, details on demand.

- Surfaces: neutral dark charcoal in three steps (paper-3 behind, paper for panels, paper-2 for cards), hairline rules, 6px card corners and 4px control corners. No gradients, glow or decorative dots.
- Colour: one warm amber accent for running work, selection and primary actions; semantic green for done or signed in, red for failed, limited or signed out. Each provider has a muted identity colour (Claude coral, Codex teal, Antigravity violet) shown as small squares and thin stripes.
- Type: Host Grotesk for interface text, Atkinson Hyperlegible Mono for measured values, ids, paths and logs, with system fallbacks. Labels are sentence case and untracked.
- Layout: header (workspace, run meters, mode, connection); left accounts rail of compact rows; centre run header (task, status, one line of numbers, Stop / Resume / Add instruction), swarm graph as a tree of readable agent cards, and a bottom panel with Activity first then Terminal; right column with Agents grouped by phase, Claims and Report. Selecting an agent opens an inspector over the right column with its account card, attempts timeline, claims, activity, result and prompt.
- Behaviour: progressive disclosure (tool calls, prompts and full logs are collapsed until asked for), keyboard focus always visible, reduced-motion respected.
