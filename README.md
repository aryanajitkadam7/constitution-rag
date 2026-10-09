# Constitution of India RAG: a retrieval system you can measure

This project answers everyday questions such as *"Can the police keep me locked up without bringing me before a judge?"* from the Constitution of India. Every answer comes with verified article citations.

The focus is **measured retrieval quality**. Every design choice below was tested on a hand-labelled evaluation set, failures were analysed, and the fix was validated on a **held-out set written after the fix was frozen**.

- **Corpus:** 465 articles plus the Preamble, split into 634 chunks (`data/constitution.json`, see `data/NOTICE.md`)
- **Eval set:** 140 labelled questions: 110 in everyday language and 30 in legal phrasing. There are also 12 out-of-scope questions for refusal testing, and 38 more held-out questions.
- **Stack:** ONNX Runtime embeddings (MiniLM-L6, int8, CPU only, no PyTorch) · BM25 · weighted hybrid fusion · glossary query expansion · LightGBM LambdaRank · Gemini for generation · FastAPI · Docker

---

## Results

### Retrieval: main set (140 questions; article-level; `make eval`)

| System | R@1 | R@3 | R@5 | R@10 | MRR@10 | R@5 (everyday) | p50 latency |
|---|---|---|---|---|---|---|---|
| BM25 | 0.443 | 0.564 | 0.636 | 0.786 | 0.536 | 0.536 | 0.9 ms |
| Dense (MiniLM-L6 int8) | 0.621 | 0.807 | 0.857 | 0.914 | 0.723 | 0.818 | 4.5 ms |
| Hybrid, Reciprocal Rank Fusion | 0.593 | 0.786 | 0.843 | 0.921 | 0.702 | 0.800 | 5.0 ms |
| Hybrid, weighted (α tuned by 5-fold CV) | 0.657 | 0.814 | 0.857 | 0.936 | 0.751 | 0.818 | 5.6 ms |
| Hybrid + learned reranker (LambdaRank, CV) | 0.600 | 0.750 | 0.814 | 0.921 | 0.695 | 0.764 | 10.8 ms |
| **Hybrid, weighted + glossary expansion** | **0.686** | **0.879** | **0.921** | **0.957** | **0.787** | **0.900** | 5.9 ms |

### Retrieval: held-out set (38 new everyday questions, written after the glossary was frozen)

| System | R@1 | R@3 | R@5 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.526 | 0.684 | 0.816 | 0.626 |
| Dense | 0.553 | 0.816 | 0.895 | 0.705 |
| Hybrid weighted (α = 0.8) | 0.632 | 0.842 | 0.947 | 0.756 |
| **Hybrid weighted + expansion** | **0.816** | **0.947** | **0.974** | **0.879** |

The numbers are in `reports/retrieval_results.json` and `.md`. Latency is measured on a 2-vCPU machine and includes query embedding.

### What the experiments showed

1. **Keyword search fails on everyday language.** On legal phrasing every system reaches R@5 = 1.0. On everyday wording BM25 drops to 0.54. *"Locked up without a judge"* shares almost no words with *"Protection against arrest and detention"*.
2. **Reciprocal Rank Fusion was worse than dense alone** (MRR 0.702 vs 0.723). RRF weights both retrievers equally, and here BM25 is much weaker. A **weighted** fusion with α chosen by cross-validation (α = 0.8, consistent across all 5 folds) beats both.
3. **The learned reranker didn't pay off.** A LambdaRank model over 8 interpretable features, trained on about 110 questions per fold, scored *below* the plain weighted hybrid (MRR 0.695). There isn't enough labelled data for it. I kept it in the report as a negative result.
4. **Failure analysis found the real problem: vocabulary.** Most v1 misses were Indian civic terms the Constitution never uses: *Lok Sabha* (→ House of the People), *Rajya Sabha*, *MP*, *village council* (→ Panchayat), *budget* (→ annual financial statement), *temporary law* (→ Ordinance). A general English embedding model doesn't know these mappings.
5. **The fix: a 40-term glossary for query expansion** (`src/crag/expansion.py`). On the main set, MRR rose from 0.751 to 0.787 and R@5 for everyday questions from 0.82 to 0.90. The glossary was written *after* seeing the main-set failures, so it was frozen and tested on 38 new questions: **R@1 rose from 0.632 to 0.816 and MRR from 0.756 to 0.879.**

**Caveats:** the held-out set is small (38) and was written by the same author who wrote the glossary, so treat it as directional. The legal-phrasing subset is too easy, since the questions echo article titles.

**Remaining misses** (17 of 140 outside the top 3; the list is in the results JSON):
- (a) answers buried in long multi-clause articles, e.g. Art. 19 for *"peaceful protest"*
- (b) conceptual paraphrases, e.g. *"one common family law"* for Art. 44 (Uniform Civil Code)
- (c) near-duplicate neighbouring articles: 132–136 on Supreme Court appeals, 251 vs 254 on law conflicts

These are the cases a cross-encoder or LLM reranker should fix. That is the next experiment (`make eval-ce`).

---

## Generation (Gemini) with verified citations

`src/crag/generate.py` sends the top 5 chunks to Gemini (`gemini-2.5-flash`, temperature 0) with these rules: answer only from the context, cite every rule as `[Art. N]`, and refuse with a fixed sentence when the answer isn't there. **Citations are then checked in code**: any cited article that wasn't in the retrieved context is returned as `unverified_citations`.

`make eval-gen` scores answers with deterministic checks only, no LLM-as-judge:

| Metric | What it checks |
|---|---|
| Citation hit rate | In-scope answers that cite at least one gold article |
| Citation validity | Answers whose citations all appear in the retrieved context |
| False refusal rate | In-scope questions the system refused |
| Out-of-scope refusal rate | Out-of-scope questions (e.g. *"current repo rate"*) correctly refused |

> **Status:** generation results are not yet reported because this needs a Gemini API key. Run `GEMINI_API_KEY=... make eval-gen`; it writes `reports/generation_results.json`. Its numbers will be added here once that run is done. No figures are claimed until then.

---

## Run it

```bash
pip install -r requirements.txt
make eval                  # retrieval benchmark, ~20 s, no API key
make test
make serve                 # http://localhost:8000/docs
curl "localhost:8000/search?q=How+many+members+can+the+Rajya+Sabha+have&k=3"
export GEMINI_API_KEY=...  # https://aistudio.google.com/apikey (free tier)
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "Can the police detain me without telling me why?"}'
```

Docker: `docker build -t constitution-rag . && docker run -p 8000:8000 -e GEMINI_API_KEY=$GEMINI_API_KEY constitution-rag`

The embedding model (`Xenova/all-MiniLM-L6-v2`, `onnx/model_quantized.onnx`, about 23 MB) is downloaded on first run. The reported numbers were produced with the file whose SHA-256 is `afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1`.

## Layout

```
src/crag/
  corpus.py           article loading + clause-aware chunking (220 words, 1-paragraph overlap)
  embed.py            ONNX Runtime sentence embedder with on-disk cache
  retrievers.py       BM25, dense, hybrid (RRF / weighted); chunk -> article aggregation
  expansion.py        Indian civic-vocabulary glossary for query expansion
  rerank.py           LambdaRank learned reranker; optional cross-encoder reranker
  evaluate.py         Recall@k, MRR, nDCG, question-level CV, held-out eval, failure analysis
  generate.py         Gemini prompting, citation parsing and verification
  eval_generation.py  deterministic end-to-end answer metrics
  pipeline.py, api.py production retriever + FastAPI service
scripts/              the eval sets (main + held-out) with gold-label validation
```

## Limitations

- The eval set was hand-written with AI assistance. Every gold label was checked against the article text, but it is a single annotator's judgement.
- Scores are min-max normalised per query, so a retrieval score alone can't detect out-of-scope questions. Refusal relies on the LLM (and is measured by `eval-gen`).
- The corpus has no amendments history, case law, or Schedules. It covers only what the articles themselves say. This is not legal advice.
