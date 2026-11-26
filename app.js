import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
import { chunkText, BM25, cosine, rankOf, rrf, mmr, extractiveAnswer, recallAtK, mrr } from './rag.js';

env.allowLocalModels = false;
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
const $ = id => document.getElementById(id);
const S = { chunks: [], vecs: [], bm25: null, embed: null, files: [] };

async function getEmbedder() {
  if (S.embed) return S.embed;
  $('istat').textContent = 'Loading MiniLM-L6 embedding model (≈23 MB, cached after first load)…';
  S.embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    progress_callback: p => { if (p.status === 'progress' && p.total) $('ibar').style.width = (100 * p.loaded / p.total).toFixed(0) + '%'; }
  });
  return S.embed;
}
const embedOne = async t => Array.from((await (await getEmbedder())(t, { pooling: 'mean', normalize: true })).data);

async function readFile(f) {
  if (f.name.toLowerCase().endsWith('.pdf')) {
    const pdf = await pdfjsLib.getDocument({ data: await f.arrayBuffer() }).promise;
    const pages = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const c = await (await pdf.getPage(i)).getTextContent();
      pages.push({ page: i, text: c.items.map(x => x.str).join(' ') });
    }
    return pages;
  }
  return [{ page: null, text: await f.text() }];
}

async function ingest(name, pages) {
  const newChunks = pages.flatMap(p => chunkText(p.text, { source: name, page: p.page }));
  const emb = await getEmbedder();
  for (let i = 0; i < newChunks.length; i++) {
    const out = await emb(newChunks[i].text, { pooling: 'mean', normalize: true });
    S.vecs.push(Array.from(out.data)); S.chunks.push(newChunks[i]);
    $('ibar').style.width = (100 * (i + 1) / newChunks.length).toFixed(0) + '%';
    $('istat').textContent = `Indexing ${name}: ${i + 1}/${newChunks.length} chunks`;
  }
  S.bm25 = new BM25(S.chunks);
  S.files.push({ name, n: newChunks.length });
  $('files').innerHTML = S.files.map(f => `<li><span>${esc(f.name)}</span><span class="stat">${f.n} chunks</span></li>`).join('');
  $('istat').textContent = `Index ready · ${S.chunks.length} chunks · 384-d vectors + BM25`;
  $('ask').disabled = $('runEval').disabled = false;
}

const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function retrieve(q, k, mode = 'hybrid') {
  const qv = await embedOne(q);
  const dense = rankOf(S.vecs.map(v => cosine(qv, v)));
  const lex = rankOf(S.bm25.scores(q));
  if (mode === 'dense') return dense.slice(0, k);
  if (mode === 'bm25') return lex.slice(0, k);
  const fused = rrf([dense.slice(0, 30), lex.slice(0, 30)]).map(x => x[0]);
  return mmr(fused.slice(0, 15), qv, S.vecs, k);
}

async function gemini(q, ctx) {
  const key = $('key').value.trim(); if (!key) throw new Error('Add a Gemini API key, or switch to Extractive.');
  const prompt = `Answer ONLY from the context. Cite sources inline like [1], [2]. If the answer is not in the context, say you don't know.\n\n` +
    ctx.map((c, i) => `[${i + 1}] (${c.source}${c.page ? ' p.' + c.page : ''})\n${c.text}`).join('\n\n') + `\n\nQuestion: ${q}`;
  const r = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent', {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }], generationConfig: { temperature: 0.2 } })
  });
  const j = await r.json(); if (!r.ok) throw new Error(j.error?.message || r.status);
  return j.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '(empty)';
}

$('ask').onclick = async () => {
  const q = $('q').value.trim(); if (!q) return;
  $('ans').textContent = 'Retrieving…'; $('srcs').innerHTML = '';
  const t0 = performance.now();
  const ids = await retrieve(q, +$('k').value || 4);
  const ctx = ids.map(i => S.chunks[i]);
  const ms = (performance.now() - t0).toFixed(0);
  $('srcs').innerHTML = ctx.map((c, i) => `<div class="src" id="s${i + 1}"><div class="meta"><b>[${i + 1}]</b> ${esc(c.source)}${c.page ? ' · p.' + c.page : ''}</div>${esc(c.text.slice(0, 600))}${c.text.length > 600 ? '…' : ''}</div>`).join('');
  try {
    let text;
    if ($('gen').value === 'gemini') { $('ans').textContent = 'Generating…'; text = await gemini(q, ctx); }
    else { const a = extractiveAnswer(q, ctx); text = a.length ? a.map(x => `${x.s} [${x.ci + 1}]`).join(' ') : 'No supporting sentence found in the top chunks.'; }
    $('ans').innerHTML = `<span class="pill">retrieval ${ms} ms</span><span class="pill">${$('gen').value}</span>\n` + esc(text).replace(/\[(\d+)\]/g, '<span class="cite" data-s="$1">[$1]</span>');
    document.querySelectorAll('.cite').forEach(e => e.onclick = () => $('s' + e.dataset.s)?.scrollIntoView({ behavior: 'smooth' }));
  } catch (e) { $('ans').textContent = '⚠ ' + e.message; }
};
$('q').onkeydown = e => { if (e.key === 'Enter' && !$('ask').disabled) $('ask').click(); };

$('runEval').onclick = async () => {
  const rows = $('evalq').value.split('\n').map(l => l.split('|').map(s => s.trim())).filter(r => r[0] && r[1]);
  if (!rows.length) return;
  const res = { bm25: [0, 0], dense: [0, 0], hybrid: [0, 0] };
  for (const [q, kw] of rows) {
    const rel = S.chunks.map((c, i) => c.text.toLowerCase().includes(kw.toLowerCase()) ? i : -1).filter(i => i >= 0);
    for (const m of Object.keys(res)) {
      const r = await retrieve(q, 5, m);
      res[m][0] += recallAtK(r, rel, 5) > 0 ? 1 : 0; res[m][1] += mrr(r, rel);
    }
  }
  const n = rows.length;
  $('evalOut').innerHTML = `<table><tr><th>Method</th><th>Hit@5</th><th>MRR</th></tr>` +
    Object.entries(res).map(([m, [h, r]]) => `<tr><td>${m}</td><td>${(100 * h / n).toFixed(0)}%</td><td>${(r / n).toFixed(3)}</td></tr>`).join('') + `</table>`;
};

$('gen').onchange = () => $('keybox').hidden = $('gen').value !== 'gemini';
try { $('key').value = localStorage.getItem('dm_key') || ''; } catch {}
$('key').onchange = () => { try { localStorage.setItem('dm_key', $('key').value); } catch {} };

const drop = $('drop');
drop.onclick = () => $('file').click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('hot'); };
drop.ondragleave = () => drop.classList.remove('hot');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('hot'); handle(e.dataTransfer.files); };
$('file').onchange = e => handle(e.target.files);
async function handle(list) { for (const f of list) await ingest(f.name, await readFile(f)); }

$('sample').onclick = async () => {
  $('sample').disabled = true;
  const t = await (await fetch('sample/ml-papers.md')).text();
  await ingest('ml-papers.md', [{ page: null, text: t }]);
  $('q').value = 'Why did the Transformer remove recurrence?';
  $('evalq').value = 'Why did the Transformer remove recurrence? | parallel\nWhat does BERT pre-train on? | masked\nHow does RAG combine memory? | non-parametric\nWhat is LoRA? | low-rank';
};
