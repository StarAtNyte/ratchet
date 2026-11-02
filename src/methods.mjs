/**
 * Research-method primitives used by the swarm and the optimiser. Pure functions, no I/O.
 *
 * Sources: effort scaling and structured briefs (Anthropic, "How we built our multi-agent
 * research system"); Elo tournaments over hypotheses (Google AI co-scientist); quarantine and
 * graduated sanctions (arXiv 2609.04170, emergent cheating in research swarms); novelty
 * rejection and offspring-weighted parents (ShinkaEvolve, arXiv 2509.19349).
 */

/** Effort classes: how many lines the lead may open and how much each line may spend. */
export const COMPLEXITY = {
  simple: { maxLines: 2 },
  comparison: { maxLines: 4 },
  complex: { maxLines: Infinity },
};
export const DIFFICULTY = {
  low: { maxSpawn: 0, toolCalls: '5-15', note: 'answer directly; do not open sub-investigations' },
  medium: { maxSpawn: 1, toolCalls: '15-40', note: 'one focused sub-investigation at most' },
  high: { maxSpawn: 3, toolCalls: '40-120', note: 'split into sub-investigations when parts are independent' },
};

export function linesFor(complexity, requested, cap) {
  const max = COMPLEXITY[complexity]?.maxLines ?? COMPLEXITY.complex.maxLines;
  return Math.max(1, Math.min(requested, cap, max));
}

export function difficultyOf(value) {
  return DIFFICULTY[value] ? value : 'medium';
}

/** A delegation brief: objective, boundaries, siblings and stop conditions, so workers do not overlap. */
export function brief(st, siblings) {
  const d = DIFFICULTY[difficultyOf(st.difficulty)];
  const lines = [
    `Objective: ${st.objective ?? st.prompt ?? st.title}`,
    st.prompt && st.objective && st.prompt !== st.objective ? `Details: ${st.prompt}` : '',
    st.non_goals?.length ? `Out of scope (other lines or not wanted): ${st.non_goals.join('; ')}` : '',
    siblings.length ? `Other researchers are covering, do not duplicate: ${siblings.join('; ')}` : '',
    `Deliverable: ${st.output ?? 'checkable claims, each with the evidence that supports it'}`,
    `Stop when: ${st.stop_when ?? 'the objective is answered with evidence, or you can show it cannot be'}`,
    `Effort: ${difficultyOf(st.difficulty)} difficulty, roughly ${d.toolCalls} tool calls; ${d.note}.`,
  ];
  return lines.filter(Boolean).join('\n');
}

/** Verdicts must quote evidence; a confirmation without a quote is downgraded to unclear. */
export function gateVerdict(v) {
  const verdict = ['confirmed', 'refuted', 'unclear'].includes(v?.verdict) ? v.verdict : null;
  if (!verdict) return null;
  const evidence = String(v.evidence ?? '').trim();
  if (verdict === 'confirmed' && evidence.length < 12) {
    return { verdict: 'unclear', note: `${v.note ? `${v.note} ` : ''}(downgraded: no evidence quoted)`.trim(), evidence, fabricated: false };
  }
  return { verdict, note: String(v.note ?? ''), evidence, fabricated: v.fabricated === true };
}

/** Local file sources that do not exist are a red flag for fabricated evidence. */
export function missingLocalSource(source, exists) {
  const m = /(?:^|\s)((?:\.{0,2}\/|~\/)?[\w.-]+(?:\/[\w.-]+)+\.\w{1,6})(?::\d+)?/.exec(String(source ?? ''));
  if (!m || /^https?:/i.test(source)) return null;
  return exists(m[1]) ? null : m[1];
}

/**
 * Graduated sanctions per worker: strike 1 flags, strike 2 quarantines the worker's claims.
 * Returns the ids of claims that must be quarantined.
 */
export function applyStrikes(strikes, claims) {
  const out = [];
  for (const c of claims) if ((strikes.get(c.agent) ?? 0) >= 2 && c.verdict !== 'quarantined') out.push(c.id);
  return out;
}

/** Elo with K=24. `matches` are [winnerId, loserId]; order is randomised by the caller. */
export function elo(ids, matches, k = 24, base = 1200) {
  const r = new Map(ids.map((id) => [id, base]));
  for (const [w, l] of matches) {
    if (!r.has(w) || !r.has(l) || w === l) continue;
    const ew = 1 / (1 + 10 ** ((r.get(l) - r.get(w)) / 400));
    r.set(w, r.get(w) + k * (1 - ew));
    r.set(l, r.get(l) - k * (1 - ew));
  }
  return r;
}

/** Pairs for a tournament: each item meets `rounds` random opponents, no self-matches, no repeats. */
export function pairings(ids, rounds = 3, rand = Math.random) {
  const seen = new Set(); const out = [];
  for (let r = 0; r < rounds; r++) {
    const order = [...ids].sort(() => rand() - 0.5);
    for (let i = 0; i + 1 < order.length; i += 2) {
      const [a, b] = [order[i], order[i + 1]].sort();
      const key = `${a}|${b}`;
      if (!seen.has(key)) { seen.add(key); out.push([order[i], order[i + 1]]); }
    }
  }
  return out;
}

/** Word-shingle Jaccard similarity: a cheap, local stand-in for embedding similarity. */
export function similarity(a, b, n = 3) {
  const sh = (s) => {
    const w = String(s).toLowerCase().replace(/[^a-z0-9_]+/g, ' ').trim().split(/\s+/).filter(Boolean);
    const set = new Set();
    for (let i = 0; i + n <= w.length; i++) set.add(w.slice(i, i + n).join(' '));
    if (!set.size && w.length) set.add(w.join(' '));
    return set;
  };
  const A = sh(a); const B = sh(b);
  if (!A.size || !B.size) return 0;
  let inter = 0; for (const x of A) if (B.has(x)) inter++;
  return inter / (A.size + B.size - inter);
}

/** Merge near-duplicate texts (proximity step); returns groups of indices. */
export function cluster(texts, threshold = 0.6) {
  const groups = [];
  texts.forEach((t, i) => {
    const g = groups.find((grp) => similarity(texts[grp[0]], t) >= threshold);
    if (g) g.push(i); else groups.push([i]);
  });
  return groups;
}

/** UCB1 over arms {pulls, reward}; untried arms first. */
export function ucb1(arms) {
  const total = arms.reduce((s, a) => s + a.pulls, 0);
  let best = null; let bestV = -Infinity;
  for (const a of arms) {
    const v = a.pulls === 0 ? Infinity : a.reward / a.pulls + Math.sqrt((2 * Math.log(Math.max(total, 1))) / a.pulls);
    if (v > bestV) { bestV = v; best = a; }
  }
  return best;
}
