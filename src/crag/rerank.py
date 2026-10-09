"""Second-stage rerankers.

1. LearnedFusionReranker - a LightGBM LambdaRank model over cheap, interpretable
   features of each (query, candidate article) pair: BM25 score/rank, dense score/rank,
   RRF score, query-title word overlap, and query-title embedding similarity.
   It is trained on labelled questions and must be evaluated with question-level
   cross-validation (see evaluate.py) so that no test question is ever seen in training.

2. CrossEncoderReranker - optional, uses sentence-transformers' CrossEncoder
   (default cross-encoder/ms-marco-MiniLM-L-6-v2). Requires `pip install sentence-transformers`
   and internet access to download the model on first use.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from .retrievers import BM25Retriever, DenseRetriever, Hit, _ranks, tokenize

FEATURES = ["bm25", "bm25_rank", "dense", "dense_rank", "rrf", "title_overlap", "title_sim", "log_len"]


class CandidateFeaturizer:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, articles: list[dict], n_cand: int = 30):
        self.bm25, self.dense, self.n_cand = bm25, dense, n_cand
        self.chunks = bm25.chunks
        self.art_title = {a["article"]: a["title"] for a in articles}
        self.art_len = {a["article"]: len(a["text"].split()) for a in articles}
        self.art_ids = list(self.art_title)
        self.title_emb = dense.emb.encode_cached([self.art_title[a] for a in self.art_ids], tag="titles")
        self.title_idx = {a: i for i, a in enumerate(self.art_ids)}
        self.chunk_art = np.array([c.article for c in self.chunks])

    def _article_max(self, s: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for a, v in zip(self.chunk_art, s):
            if v > out.get(a, -np.inf):
                out[a] = float(v)
        return out

    def candidates(self, query: str):
        qv = self.dense.embed_query(query)
        s_b, s_d = self.bm25.scores(query), self.dense.scores(query, qvec=qv)
        rrf = 1 / (60 + _ranks(s_b)) + 1 / (60 + _ranks(s_d))
        ab, ad, ar = self._article_max(s_b), self._article_max(s_d), self._article_max(rrf)
        rb = {a: i + 1 for i, a in enumerate(sorted(ab, key=ab.get, reverse=True))}
        rd = {a: i + 1 for i, a in enumerate(sorted(ad, key=ad.get, reverse=True))}
        cand = list(dict.fromkeys(
            sorted(ab, key=ab.get, reverse=True)[: self.n_cand] + sorted(ad, key=ad.get, reverse=True)[: self.n_cand]))
        qt = set(tokenize(query))
        tsim = self.title_emb @ qv
        X = []
        for a in cand:
            tt = set(tokenize(self.art_title[a]))
            X.append([ab[a], rb[a], ad[a], rd[a], ar[a],
                      len(qt & tt) / max(len(qt), 1), float(tsim[self.title_idx[a]]), np.log1p(self.art_len[a])])
        return cand, np.asarray(X, dtype=np.float32)


class LearnedFusionReranker:
    name = "hybrid+ltr_rerank"

    def __init__(self, featurizer: CandidateFeaturizer, seed: int = 42):
        self.f = featurizer
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=15,
                                    min_child_samples=5, random_state=seed, verbose=-1)
        self._cache: dict[str, tuple] = {}

    def _cand(self, q):
        if q not in self._cache:
            self._cache[q] = self.f.candidates(q)
        return self._cache[q]

    def fit(self, questions: list[dict]):
        Xs, ys, groups = [], [], []
        for r in questions:
            cand, X = self._cand(r["question"])
            y = np.array([1 if a in r["gold"] else 0 for a in cand])
            if y.sum() == 0:  # gold not among candidates: nothing to learn from this query
                continue
            Xs.append(X), ys.append(y), groups.append(len(cand))
        self.model.fit(np.vstack(Xs), np.concatenate(ys), group=groups)
        return self

    def search(self, query: str, k: int = 10) -> list[Hit]:
        cand, X = self._cand(query)
        s = self.model.predict(X)
        order = np.argsort(-s)[:k]
        return [Hit(cand[i], float(s[i]), "") for i in order]


class CrossEncoderReranker:
    def __init__(self, first_stage, articles: list[dict], model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 n_cand: int = 30):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.first, self.n_cand = first_stage, n_cand
        self.ce = CrossEncoder(model)
        self.text = {c.chunk_id: c.text for c in first_stage.chunks}
        self.name = f"{first_stage.name}+cross_encoder"

    def search(self, query: str, k: int = 10) -> list[Hit]:
        hits = self.first.search(query, self.n_cand)
        scores = self.ce.predict([(query, self.text[h.chunk_id]) for h in hits])
        order = np.argsort(-scores)[:k]
        return [Hit(hits[i].article, float(scores[i]), hits[i].chunk_id) for i in order]
