"""Retrievers: BM25 (sparse), dense (embeddings), and hybrid fusion.

All retrievers score *chunks* and then aggregate to *articles* (max over an
article's chunks), because the unit a user cares about - and the unit we cite -
is the article.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Chunk
from .embed import OnnxEmbedder

STOP = set("""a an the of to in on for and or is are was were be by with as at from that this these those it its
which who whom what when where why how can do does did shall may any all such under any other than into not no
if i me my we our you your he his she her they them their there here about than then so also has have had will
would should could""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in toks:
        if t in STOP:
            continue
        # very light stemming: plural / -ing / -ed
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > 4 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


@dataclass
class Hit:
    article: str
    score: float
    chunk_id: str


def to_articles(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    best: dict[str, Hit] = {}
    for i in np.argsort(-scores):
        c = chunks[i]
        if c.article not in best:
            best[c.article] = Hit(c.article, float(scores[i]), c.chunk_id)
            if len(best) == k:
                break
    return list(best.values())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return to_articles(self.chunks, self.scores(query), k)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: OnnxEmbedder):
        self.chunks, self.emb = chunks, embedder
        self.matrix = embedder.encode_cached([c.text for c in chunks], tag="chunks")

        self._qcache: dict[str, np.ndarray] = {}

    def embed_query(self, query: str) -> np.ndarray:
        if query not in self._qcache:  # bounded in-memory cache of query embeddings
            if len(self._qcache) > 10_000:
                self._qcache.clear()
            self._qcache[query] = self.emb.encode([query])[0]
        return self._qcache[query]

    def scores(self, query: str, qvec: np.ndarray | None = None) -> np.ndarray:
        q = self.embed_query(query) if qvec is None else qvec
        return self.matrix @ q

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return to_articles(self.chunks, self.scores(query), k)


def _ranks(scores: np.ndarray) -> np.ndarray:
    r = np.empty_like(scores, dtype=np.float32)
    r[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return r


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) of BM25 and dense, or a convex combination of
    min-max normalised scores when `mode="weighted"`."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, mode: str = "rrf", k_rrf: int = 60,
                 alpha: float = 0.5):
        self.bm25, self.dense, self.mode, self.k_rrf, self.alpha = bm25, dense, mode, k_rrf, alpha
        self.chunks = bm25.chunks
        self.name = f"hybrid_{mode}" + (f"_a{alpha:g}" if mode == "weighted" else "")

    def scores(self, query: str) -> np.ndarray:
        s_b, s_d = self.bm25.scores(query), self.dense.scores(query)
        if self.mode == "rrf":
            return 1.0 / (self.k_rrf + _ranks(s_b)) + 1.0 / (self.k_rrf + _ranks(s_d))
        norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
        return self.alpha * norm(s_d) + (1 - self.alpha) * norm(s_b)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return to_articles(self.chunks, self.scores(query), k)
