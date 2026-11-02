/**
 * Program database, AlphaEvolve-style: islands x method-family cells (MAP-Elites-lite).
 * Each cell keeps its best valid program; parents are sampled rank-biased from a random
 * island, and inspirations come from other cells to keep diversity.
 */
export class Archive {
  constructor({ direction = 'min', islands = 3, rand = Math.random } = {}) {
    this.direction = direction;
    this.islands = islands;
    this.rand = rand;
    this.entries = [];
    this.cells = new Map();
    this.champion = null;
  }

  /** Strictly better under the optimisation direction. */
  better(a, b) { return this.direction === 'min' ? a < b : a > b; }

  /** Add an entry; returns true when it becomes the new champion (the ratchet only turns forward). */
  add(e) {
    this.entries.push(e);
    // offspring counts drive ShinkaEvolve-style sampling: rarely-expanded parents get more chances
    const parent = e.parent && this.entries.find((x) => x.id === e.parent);
    if (parent) parent.children = (parent.children ?? 0) + 1;
    if (!e.ok) return false;
    const key = `${e.island}:${e.family}`;
    const cur = this.cells.get(key);
    if (!cur || this.better(e.score, cur.score)) this.cells.set(key, e);
    const accept = e.accepted !== false;
    if (accept && (!this.champion || this.better(e.score, this.champion.score))) { this.champion = e; e.champion = true; return true; }
    return false;
  }

  /** Entries sorted best-first, valid only. */
  ranked(island) {
    const v = [...this.cells.values()].filter((c) => island == null || c.island === island);
    return v.sort((a, b) => (a.score === b.score ? 0 : this.better(a.score, b.score) ? -1 : 1));
  }

  /** Pick {parent, island, inspirations}. Prefers parents not currently being mutated. */
  sample(inflight = new Set()) {
    const island = Math.floor(this.rand() * this.islands);
    let pool = this.ranked(island);
    if (!pool.length) pool = this.ranked();
    if (!pool.length) return { parent: null, island, inspirations: [] };
    const free = pool.filter((p) => !inflight.has(p.id));
    const use = free.length ? free : pool;
    const exploit = this.rand() < 0.7;
    // weight = rank preference (exploit only) x 1/(1 + offspring), so good but under-explored parents win
    const w = use.map((p, i) => (exploit ? 1 / (1 + i) : 1) / (1 + (p.children ?? 0)));
    let pick = this.rand() * w.reduce((a, b) => a + b, 0);
    let parent = use[use.length - 1];
    for (let i = 0; i < use.length; i++) { pick -= w[i]; if (pick <= 0) { parent = use[i]; break; } }
    const inspirations = this.ranked().filter((c) => c.id !== parent.id && c.family !== parent.family).slice(0, 2);
    return { parent, island, inspirations };
  }

  families() { return [...new Set(this.entries.map((e) => e.family).filter(Boolean))]; }
}
