# DocMind — hybrid-search RAG that runs in your browser


**Live demo → https://docmind-rag.vercel.app**

Drop in PDFs, ask questions, and get answers with clickable citations. Parsing, chunking, embedding and retrieval all happen client-side, so your documents never leave the device. Generation is either **extractive** (offline, no key) or **Gemini** (bring your own key, grounded prompt with inline citations).

## Pipeline

```
PDF/MD ──pdf.js──► page text ──► sentence-aware chunks (180 words, 40 overlap)
                                     │
                     ┌───────────────┴───────────────┐
                 BM25 index                MiniLM-L6-v2 embeddings (384-d, transformers.js / ONNX-WASM)
                     └──── top-30 each ──► Reciprocal Rank Fusion (k=60)
                                              │
                                   MMR re-selection (λ=0.7) → top-k context
                                              │
                         extractive answer  |  Gemini 2.0 Flash, "answer only from context, cite [n]"
```

**Why hybrid?** Dense vectors catch paraphrases ("drop recurrence" ≈ "dispensing with recurrence"). BM25 catches exact terms, acronyms and numbers that embeddings blur. RRF fuses the two rankings without score calibration, and MMR removes near-duplicate overlapping chunks so the context window isn't wasted.

## Built-in retrieval eval
The eval panel takes `question | keyword` pairs and reports **Hit@5** and **MRR** for BM25-only, dense-only and hybrid, so you can measure retrieval on your own corpus instead of guessing.

## Run locally
```bash
python3 -m http.server 8000   # open http://localhost:8000
npm test                      # unit tests for chunking, BM25, RRF, MMR, metrics
```

## Code map
| File | What |
|---|---|
| `rag.js` | Pure retrieval core: tokenizer, chunker, BM25, cosine, RRF, MMR, extractive answerer, Recall@k / MRR |
| `app.js` | UI, PDF parsing, embedding with transformers.js, Gemini call |
| `test/` | `node:test` unit tests (run in CI) |

## Roadmap
- Cross-encoder re-ranker (ms-marco-MiniLM) as an optional stage
- IndexedDB persistence of the vector index
- WebGPU embeddings when available
