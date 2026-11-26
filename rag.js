// DocMind RAG core: chunking, BM25, dense retrieval, Reciprocal Rank Fusion, MMR.
// Pure functions so they can be unit-tested in Node (see test/rag.test.mjs).

const STOP = new Set('a an and are as at be by for from has have he her his i in is it its of on or that the their them they this to was were will with you your not but what which who how why when where do does did can could should would'.split(' '));

export function tokenize(text) {
  return text.toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/).filter(t => t.length > 1 && !STOP.has(t));
}

// Sentence-aware chunking with overlap. size/overlap are in words.
export function chunkText(text, { size = 180, overlap = 40, source = 'doc', page = null } = {}) {
  const sents = text.replace(/\s+/g, ' ').match(/[^.!?]+[.!?]*\s*/g) || [];
  const chunks = []; let cur = []; let n = 0;
  const flush = () => {
    if (!cur.length) return;
    chunks.push({ id: `${source}#${chunks.length}`, source, page, text: cur.join('').trim() });
    // keep the tail sentences as overlap
    let keep = [], w = 0;
    for (let i = cur.length - 1; i >= 0 && w < overlap; i--) { keep.unshift(cur[i]); w += cur[i].split(' ').length; }
    cur = keep; n = w;
  };
  for (const s of sents) {
    const w = s.split(' ').length;
    if (n + w > size && n > overlap) flush();
    cur.push(s); n += w;
  }
  if (cur.length && (!chunks.length || n > overlap)) chunks.push({ id: `${source}#${chunks.length}`, source, page, text: cur.join('').trim() });
  return chunks;
}

export class BM25 {
  constructor(docs, { k1 = 1.5, b = 0.75 } = {}) {
    this.k1 = k1; this.b = b;
    this.toks = docs.map(d => tokenize(d.text));
    this.avg = this.toks.reduce((a, t) => a + t.length, 0) / Math.max(1, this.toks.length);
    this.df = new Map();
    for (const t of this.toks) for (const w of new Set(t)) this.df.set(w, (this.df.get(w) || 0) + 1);
    this.tf = this.toks.map(t => { const m = new Map(); for (const w of t) m.set(w, (m.get(w) || 0) + 1); return m; });
  }
  idf(w) { const N = this.toks.length, n = this.df.get(w) || 0; return Math.log(1 + (N - n + 0.5) / (n + 0.5)); }
  scores(query) {
    const q = tokenize(query);
    return this.tf.map((tf, i) => {
      let s = 0; const len = this.toks[i].length;
      for (const w of q) {
        const f = tf.get(w); if (!f) continue;
        s += this.idf(w) * (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + this.b * len / this.avg));
      }
      return s;
    });
  }
}

export const cosine = (a, b) => { let d = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / (Math.sqrt(x * y) || 1); };

export const rankOf = scores => scores.map((s, i) => [s, i]).sort((p, q) => q[0] - p[0]).map(p => p[1]);

// Reciprocal Rank Fusion (Cormack et al., 2009)
export function rrf(rankings, k = 60) {
  const fused = new Map();
  for (const r of rankings) r.forEach((idx, rank) => fused.set(idx, (fused.get(idx) || 0) + 1 / (k + rank + 1)));
  return [...fused.entries()].sort((a, b) => b[1] - a[1]);
}

// Maximal Marginal Relevance to diversify the final context window
export function mmr(candidates, qVec, vecs, k = 5, lambda = 0.7) {
  const picked = []; const pool = [...candidates];
  while (picked.length < k && pool.length) {
    let best = -Infinity, bi = 0;
    pool.forEach((idx, j) => {
      const rel = cosine(qVec, vecs[idx]);
      const red = picked.length ? Math.max(...picked.map(p => cosine(vecs[p], vecs[idx]))) : 0;
      const s = lambda * rel - (1 - lambda) * red;
      if (s > best) { best = s; bi = j; }
    });
    picked.push(pool.splice(bi, 1)[0]);
  }
  return picked;
}

// Extractive fallback answer: best-matching sentences, each tagged with its source chunk.
export function extractiveAnswer(query, chunks, maxSent = 3) {
  const q = new Set(tokenize(query));
  const sents = [];
  chunks.forEach((c, ci) => (c.text.match(/[^.!?]+[.!?]*/g) || []).forEach(s => {
    const t = tokenize(s); if (!t.length) return;
    const hit = t.filter(w => q.has(w)).length;
    sents.push({ s: s.trim(), score: hit / Math.sqrt(t.length), ci });
  }));
  return sents.sort((a, b) => b.score - a.score).slice(0, maxSent).filter(x => x.score > 0);
}

// Retrieval metrics for the built-in eval panel
export function recallAtK(retrievedIds, relevantIds, k) {
  const top = new Set(retrievedIds.slice(0, k));
  return relevantIds.length ? relevantIds.filter(id => top.has(id)).length / relevantIds.length : 0;
}
export function mrr(retrievedIds, relevantIds) {
  const i = retrievedIds.findIndex(id => relevantIds.includes(id));
  return i < 0 ? 0 : 1 / (i + 1);
}
