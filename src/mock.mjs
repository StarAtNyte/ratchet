const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const TOPICS = [
  'prior art and known results', 'numerical experiments', 'structural decomposition',
  'counterexample search', 'literature of adjacent fields', 'formal / symbolic approach',
];

const roleOf = (prompt) => /\[\[ROLE:(\w+)\]\]/.exec(prompt)?.[1] ?? 'worker';

/** Simulated agent: no CLI, no quota. Mirrors the real result contract. */
export async function runMock(args) {
  const res = await runMockInner(args);
  // simulated token usage so the cost and token meters have something to show
  const tokens = 2000 + Math.floor(Math.random() * 18000);
  return { ...res, usage: { input_tokens: Math.floor(tokens * 0.8), output_tokens: Math.floor(tokens * 0.2), cost_usd: +(tokens * 0.000004).toFixed(4) } };
}

async function runMockInner({ account, prompt, onLog, signal }) {
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
        objective: `Establish what ${t} says about the question.`,
        prompt: `Investigate: ${t}. Report concrete, checkable claims.`,
        non_goals: ['re-deriving results another line covers'],
        harness: i % 2 ? 'codex' : 'claude',
        difficulty: ['low', 'medium', 'high'][i % 3],
      }));
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ complexity: 'complex', subtasks })}\n\`\`\`` };
    }

    if (role === 'cand') {
      await step('tool', 'Read kernel.py');
      await step('text', 'Trying a tiled variant of the inner loop.');
      await step('tool', 'Bash python bench.py');
      const fam = ['tiling', 'fusion', 'vectorize', 'cache-blocking', 'unroll'][Math.floor(Math.random() * 5)];
      const knob = ['block size', 'unroll factor', 'vector width', 'tile shape', 'prefetch distance'][Math.floor(Math.random() * 5)];
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ idea: `${fam} of the hot loop with ${knob} ${2 ** (2 + Math.floor(Math.random() * 6))}`, family: fam })}\n\`\`\`` };
    }

    if (role === 'worker') {
      await step('tool', 'WebSearch survey of recent results');
      await step('text', 'Found two promising threads; testing the stronger one.');
      await step('tool', 'Bash python experiment.py --n 4096');
      await step('text', 'Experiment finished; writing up claims with sources.');
      const spawn = Math.random() < 0.35 && !/\[\[DEPTH:2\]\]/.test(prompt)
        ? [{ title: 'Sub-question: edge cases', prompt: 'Probe edge cases of the main claim.', harness: 'any' }] : [];
      const claims = [
        { text: `Finding A (${(Math.random() * 100).toFixed(0)}): the bound tightens under the stated constraint.`, source: 'https://arxiv.org/abs/2506.13131 section 3' },
        { text: 'Finding B: naive approach fails past n=2^12; structured variant holds.', source: 'command: python bench.py --n 4096' },
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
        return { id, verdict: r < 0.65 ? 'confirmed' : r < 0.85 ? 'unclear' : 'refuted', note: r < 0.65 ? 'Reproduced independently.' : r < 0.85 ? 'Plausible, could not fully reproduce.' : 'Counterexample found at n=37.', evidence: r < 0.65 ? 'bench.py --n 4096: max error 3.1e-7 (structured), 0.42 (naive)' : '' };
      });
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ verdicts })}\n\`\`\`` };
    }

    if (role === 'rank') {
      const n = [...prompt.matchAll(/^Match (\d+)$/gm)].length;
      await step('text', `Judging ${n} pairwise matches.`);
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ winners: Array.from({ length: n }, (_, i) => ({ match: i + 1, winner: Math.random() < 0.5 ? 'A' : 'B' })) })}\n\`\`\`` };
    }

    if (role === 'judge') {
      await step('text', 'Scoring the report against the rubric.');
      const r = () => +(0.6 + Math.random() * 0.35).toFixed(2);
      const sc = { accuracy: r(), citations: r(), completeness: r(), sources: r(), efficiency: r() };
      return { ok: true, text: `\`\`\`json\n${JSON.stringify({ ...sc, overall: +((sc.accuracy + sc.citations + sc.completeness + sc.sources + sc.efficiency) / 5).toFixed(2), why: 'Mostly well supported; two claims rest on one source.' })}\n\`\`\`` };
    }

    await step('text', 'Merging findings, discarding refuted claims, flagging unclear ones.');
    await step('text', 'Drafting report.', 1200);
    return { ok: true, text: '# Report (mock)\n\n## Summary\nSimulated swarm run. Confirmed claims carry a cross-model check; refuted claims were dropped.\n\n## Confirmed\n- See the claim ledger.\n\n## Open questions\n- Everything marked unclear.\n\n\`\`\`json\n{"meta_review":["Several lines cited a single source; require two independent sources per claim."],"next_lines":["Re-run the structured variant at n=2^14 to test the bound."]}\n\`\`\`' };
  } catch (e) {
    return { ok: false, text: '', error: e.message, limited: false };
  }
}
