const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const TOPICS = [
  'prior art and known results', 'numerical experiments', 'structural decomposition',
  'counterexample search', 'literature of adjacent fields', 'formal / symbolic approach',
];

const roleOf = (prompt) => /\[\[ROLE:(\w+)\]\]/.exec(prompt)?.[1] ?? 'worker';

/** Simulated agent: no CLI, no quota. Mirrors the real result contract. */
export async function runMock({ account, prompt, onLog, signal }) {
  const role = roleOf(prompt);
  const step = async (kind, text, ms = 500 + Math.random() * 900) => {
    if (signal?.aborted) throw new Error('aborted');
    await sleep(ms);
    onLog({ kind, text });
  };

  if (account.id.endsWith('-2') && account.harness === 'claude' && !account.demoHit) {
    account.demoHit = true;
    await sleep(600);
    return { ok: false, limited: true, resetAt: Date.now() + 75_000, error: 'Claude usage limit reached. Your limit will reset at 3pm.' };
  }

  try {
    if (role === 'plan') {
      await step('text', 'Reading the workspace and decomposing the task into independent lines of attack.');
      await step('tool', 'Glob **/*');
      await step('text', 'Six lines of attack; mixing harnesses so every claim has a cross-model checker.');
      const subtasks = TOPICS.map((t, i) => ({
        title: t[0].toUpperCase() + t.slice(1),
        prompt: `Investigate: ${t}. Report concrete, checkable claims.`,
        harness: i % 2 ? 'codex' : 'claude',
      }));
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ subtasks })}\n\`\`\`` };
    }

    if (role === 'cand') {
      await step('tool', 'Read kernel.py');
      await step('text', 'Trying a tiled variant of the inner loop.');
      await step('tool', 'Bash python bench.py');
      const fam = ['tiling', 'fusion', 'vectorize', 'cache-blocking', 'unroll'][Math.floor(Math.random() * 5)];
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ idea: `${fam} variant of the hot loop`, family: fam })}\n\`\`\`` };
    }

    if (role === 'worker') {
      await step('tool', 'WebSearch survey of recent results');
      await step('text', 'Found two promising threads; testing the stronger one.');
      await step('tool', 'Bash python experiment.py --n 4096');
      await step('text', 'Experiment finished; writing up claims with sources.');
      const spawn = Math.random() < 0.35 && !/\[\[DEPTH:2\]\]/.test(prompt)
        ? [{ title: 'Sub-question: edge cases', prompt: 'Probe edge cases of the main claim.', harness: 'any' }] : [];
      const claims = [
        { text: `Finding A (${(Math.random() * 100).toFixed(0)}): the bound tightens under the stated constraint.`, source: 'experiment log + arXiv survey' },
        { text: 'Finding B: naive approach fails past n=2^12; structured variant holds.', source: 'workspace/bench.py' },
      ];
      return { ok: true, text: `Summary of findings.\n\`\`\`json\n${JSON.stringify({ claims, spawn })}\n\`\`\`` };
    }

    if (role === 'verify') {
      const ids = [...prompt.matchAll(/\(id: ([\w.]+)\)/g)].map((m) => m[1]);
      await step('tool', 'Read findings/…');
      await step('text', 'Re-deriving each claim independently.');
      await step('tool', 'Bash python recheck.py');
      const verdicts = ids.map((id) => {
        const r = Math.random();
        return { id, verdict: r < 0.65 ? 'confirmed' : r < 0.85 ? 'unclear' : 'refuted', note: r < 0.65 ? 'Reproduced independently.' : r < 0.85 ? 'Plausible, could not fully reproduce.' : 'Counterexample found at n=37.' };
      });
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ verdicts })}\n\`\`\`` };
    }

    await step('text', 'Merging findings, discarding refuted claims, flagging unclear ones.');
    await step('text', 'Drafting report.', 1200);
    return { ok: true, text: '# Report (mock)\n\n## Summary\nSimulated swarm run. Confirmed claims carry a cross-model check; refuted claims were dropped.\n\n## Confirmed\n- See the claim ledger.\n\n## Open questions\n- Everything marked unclear.\n' };
  } catch (e) {
    return { ok: false, text: '', error: e.message, limited: false };
  }
}
