import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tokenize, chunkText, BM25, rrf, mmr, extractiveAnswer, recallAtK, mrr, rankOf } from '../rag.js';

const docs = [
  { id: 'a', text: 'Transformers use self-attention to model token interactions.' },
  { id: 'b', text: 'BM25 is a lexical ranking function based on term frequency and inverse document frequency.' },
  { id: 'c', text: 'Convolutional networks are common in computer vision.' },
];

test('tokenize drops stopwords and punctuation', () => {
  assert.deepEqual(tokenize('The model, is GREAT!'), ['model', 'great']);
});

test('chunkText respects size and overlaps', () => {
  const text = Array.from({ length: 60 }, (_, i) => `Sentence number ${i} talks about topic ${i % 5}.`).join(' ');
  const ch = chunkText(text, { size: 50, overlap: 10, source: 't' });
  assert.ok(ch.length > 3);
  ch.forEach(c => assert.ok(c.text.split(' ').length <= 70));
  assert.equal(ch[0].id, 't#0');
});

test('BM25 ranks the lexical match first', () => {
  const s = new BM25(docs).scores('term frequency ranking');
  assert.equal(rankOf(s)[0], 1);
});

test('RRF fuses rankings', () => {
  const f = rrf([[0, 1, 2], [1, 0, 2]]);
  assert.equal(f[2][0], 2);
  assert.ok(Math.abs(f[0][1] - f[1][1]) < 1e-12);
});

test('MMR avoids near-duplicates', () => {
  const vecs = [[1, 0], [0.99, 0.01], [0.6, 0.8]];
  const picked = mmr([0, 1, 2], [1, 0.2], vecs, 2, 0.5);
  assert.equal(picked[1], 2); // the near-duplicate is skipped
});

test('extractive answer finds the relevant sentence', () => {
  const a = extractiveAnswer('what is self-attention', docs);
  assert.equal(a[0].ci, 0);
});

test('metrics', () => {
  assert.equal(recallAtK(['x', 'y', 'z'], ['y', 'q'], 2), 0.5);
  assert.equal(mrr(['x', 'y'], ['y']), 0.5);
});
