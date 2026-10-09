"""Builds the production retriever: weighted hybrid (alpha from CV) + glossary expansion."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .corpus import load_articles, load_chunks
from .embed import OnnxEmbedder
from .expansion import ExpandedRetriever
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALPHA = 0.8  # median of the per-fold CV choices in reports/retrieval_results.json


def tuned_alpha() -> float:
    p = ROOT / "reports" / "retrieval_results.json"
    if p.exists():
        a = json.loads(p.read_text()).get("weighted_alpha_per_fold_expansion")
        if a:
            return float(sorted(a)[len(a) // 2])
    return DEFAULT_ALPHA


@lru_cache(maxsize=1)
def build():
    chunks = load_chunks()
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, OnnxEmbedder())
    retriever = HybridRetriever(ExpandedRetriever(bm25), ExpandedRetriever(dense), "weighted", alpha=tuned_alpha())
    retriever.name = "hybrid_weighted+expansion"
    articles = {a["article"]: a for a in load_articles()}
    return retriever, {c.chunk_id: c for c in chunks}, articles
