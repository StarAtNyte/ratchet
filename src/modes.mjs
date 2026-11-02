/**
 * Run modes: each is a playbook layered on the research swarm (plan, research, verify, rank,
 * synthesise, judge). The planner, researchers, verifiers and synthesis get mode-specific rules.
 */
export const MODES = {
  research: {
    label: 'Research',
    summary: 'Open questions: read the literature, invent and test new methods, falsify the leading ideas.',
    plan: '',
    worker: '',
    verify: '',
    synth: '',
  },
  math: {
    label: 'Math',
    summary: 'Theorems and conjectures: known results, proof attempts, computational counterexample search, proofs checked independently.',
    plan: 'This is a mathematics task. Lines must include: known results and the closest prior theorems with exact citations; experiments that compute small cases and search for counterexamples; proof attempts for the most promising conjectures (formal in Lean or another proof assistant when available in the workspace, otherwise fully rigorous written proofs); and an attack line that tries to break each proposed proof. Prefer several independent proof strategies over one.',
    worker: 'Mathematical standards: state definitions and every assumption; separate what is proved from what is conjectured; a proof must be complete enough that another mathematician can check every step; record computational evidence with the code and its output. Claim types: "finding" for a proved statement (source: the proof file), "hypothesis" for a conjecture with evidence, "invention" for a new construction or proof technique.',
    verify: 'Check proofs line by line: a confirmed proof has no gaps, every cited lemma applies, and edge cases are covered. Re-run computations. Quote the step you checked or the failing step as evidence.',
    synth: 'Present results as Theorem, Lemma, Conjecture and Counterexample, each with its status and claim id. Include full proofs of confirmed results and the precise gap in any proof that was not confirmed.',
  },
  kernel: {
    label: 'Kernel optimization',
    summary: 'GPU or CPU kernels: profile, model the hardware limits, read kernel papers, build faster kernels, prove they stay correct.',
    plan: 'This is a kernel optimization task. Lines must include: profiling of the current kernel and a roofline or bandwidth analysis that says how far it is from the hardware limit; a literature line on relevant kernel work (for example FlashAttention, CUTLASS and CuTe, Triton tutorials, recent kernel papers) with concrete techniques; at least one invention line that designs a new kernel strategy (tiling, fusion, memory layout, pipelining, precision) and implements it; and a correctness line that tests numerics against a reference across shapes and dtypes. Benchmarks must be repeatable and report the hardware.',
    worker: 'Kernel standards: every speed claim states the hardware, shapes, dtype, warmup and repetition count, and is compared against the same baseline in the same session; every new kernel passes the correctness check before it is timed. Use profilers where available and explain the speedup by the mechanism (for example memory traffic removed).',
    verify: 'Re-run the correctness check and the benchmark yourself with the same settings; a speedup is confirmed only if you reproduce it within noise. Quote the numbers you measured.',
    synth: 'Lead with the fastest correct kernel, its measured speedup, the hardware, and the mechanism. Include the roofline position before and after, and list approaches that were slower or incorrect.',
  },
  score: {
    label: 'Score improvement',
    summary: 'Benchmarks and competitions: lock the evaluator, set a baseline, analyse errors, try ideas and keep only what raises the score.',
    plan: 'This is a score-improvement task (a benchmark or a competition). If the workspace has no fixed evaluator, the first line must build one that prints {"score": <number>, "ok": true} as its last line and write it into ratchet.json (objective, targets, eval, holdout, direction), so the optimise loop can take over. Other lines: understand the metric and rules exactly; reproduce the baseline; error analysis of where the score is lost; a literature line on the strongest known approaches for this benchmark; invention lines that try new methods; and ablations. Never tune on a hidden test set; keep a holdout. Submit only if the user asked for submissions, using the documented process.',
    worker: 'Score standards: every score claim gives the exact command, the data split and the seed, and is compared against the baseline on the same split. Report variance when it matters. Do not edit the evaluator or its data.',
    verify: 'Re-run the scoring command and confirm the number within noise; check that the evaluator and its data were not modified. Quote the score line you saw.',
    synth: 'Lead with the best validated score, the change that produced it and its margin over the baseline, then the ablations. If ratchet.json exists, end by suggesting: opt start.',
  },
  build: {
    label: 'Build',
    summary: 'Engineering: design, implement and test a feature or tool in the workspace, with independent review.',
    plan: 'This is an engineering task. Lines: understand the existing code and constraints; design (with alternatives considered); implementation split into independent parts with clear interfaces; tests; and a review line that tries to break the result. Assign file ownership so lines do not edit the same files.',
    worker: 'Engineering standards: follow the existing code style; keep changes focused; write or update tests and run them; report the exact test command and its result.',
    verify: 'Run the tests and the feature yourself; review the diff for correctness, security and missing cases. Quote the test output.',
    synth: 'Summarise what was built, where it lives, how to run and test it, and the open issues found in review.',
  },
};

export const MODE_IDS = Object.keys(MODES);

/** Parse `[-m|--mode <id>] task` (or `<id>: task`); returns {mode, task}. */
export function parseMode(text, fallback = 'research') {
  let t = String(text ?? '').trim();
  let mode = fallback;
  const flag = /^(?:-m|--mode)\s+(\S+)\s*/.exec(t) ?? /^--mode=(\S+)\s*/.exec(t);
  if (flag) { mode = flag[1].toLowerCase(); t = t.slice(flag[0].length); }
  if (!MODES[mode]) throw new Error(`unknown mode "${mode}". Modes: ${MODE_IDS.join(', ')}`);
  return { mode, task: t.trim() };
}

export function modeRules(mode, part) {
  const m = MODES[mode] ?? MODES.research;
  const rule = m[part];
  return rule ? `\nMode: ${m.label}. ${rule}\n` : '';
}
