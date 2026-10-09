"""Retrieval evaluation on the hand-labelled question set.

    python -m crag.evaluate                    # all retrievers, writes reports/retrieval_results.{json,md}
    python -m crag.evaluate --cross-encoder    # also evaluate a cross-encoder reranker (needs sentence-transformers)

Metrics are computed at the *article* level: a question counts as answered at k if any
of its gold articles appears in the top-k. Out-of-scope questions are excluded here
(they are used in the generation eval instead).

Learned components (weighted-hybrid alpha, LTR reranker) are tuned with 5-fold
question-level cross-validation: each question is scored by a model that never saw it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .corpus import load_articles, load_chunks
from .embed import OnnxEmbedder
from .expansion import ExpandedRetriever
from .rerank import CandidateFeaturizer, CrossEncoderReranker, LearnedFusionReranker
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
QPATH = ROOT / "data" / "eval" / "questions.jsonl"
HELDOUT_PATH = ROOT / "data" / "eval" / "questions_heldout.jsonl"
REPORTS = ROOT / "reports"
KS = (1, 3, 5, 10)


def load_questions(include_oos: bool = False, path: Path = QPATH) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows if include_oos else [r for r in rows if r["type"] != "out_of_scope"]


def first_hit_rank(ranked: list[str], gold: list[str]) -> int | None:
    for i, a in enumerate(ranked, 1):
        if a in gold:
            return i
    return None


def ndcg_at_10(ranked, gold):
    dcg = sum(1 / np.log2(i + 1) for i, a in enumerate(ranked[:10], 1) if a in gold)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    return dcg / idcg


def score_runs(runs: dict[str, list[str]], questions: list[dict]) -> dict:
    ranks = [first_hit_rank(runs[q["id"]], q["gold"]) for q in questions]
    out = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in KS}
    out["mrr@10"] = float(np.mean([1 / r if r and r <= 10 else 0 for r in ranks]))
    out["ndcg@10"] = float(np.mean([ndcg_at_10(runs[q["id"]], q["gold"]) for q in questions]))
    return out


def run_retriever(retriever, questions, k=10, reset=None):
    """reset() is called (untimed) before each query, e.g. to clear embedding caches so
    the measured latency includes query encoding."""
    runs, lat = {}, []
    for q in questions:
        if reset:
            reset()
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k)
        lat.append((time.perf_counter() - t0) * 1000)
        runs[q["id"]] = [h.article for h in hits]
    return runs, lat


def folds(questions, n=5, seed=42):
    idx = np.random.default_rng(seed).permutation(len(questions))
    return [idx[i::n] for i in range(n)]


def cv_weighted_hybrid(bm25, dense, questions, alphas=np.linspace(0.1, 0.9, 9)):
    runs, chosen = {}, []
    for test_idx in folds(questions):
        test_ids = set(test_idx)
        train = [q for i, q in enumerate(questions) if i not in test_ids]
        best = max(alphas, key=lambda a: score_runs(
            run_retriever(HybridRetriever(bm25, dense, "weighted", alpha=a), train)[0], train)["mrr@10"])
        chosen.append(float(best))
        test = [questions[i] for i in test_idx]
        runs.update(run_retriever(HybridRetriever(bm25, dense, "weighted", alpha=best), test)[0])
    return runs, chosen


def cv_ltr(featurizer, questions, reset=None):
    runs, lat = {}, []
    for test_idx in folds(questions):
        test_ids = set(test_idx)
        train = [q for i, q in enumerate(questions) if i not in test_ids]
        rr = LearnedFusionReranker(featurizer).fit(train)
        r, l = run_retriever(rr, [questions[i] for i in test_idx], reset=reset)
        runs.update(r), lat.extend(l)
    return runs, lat


def main(cross_encoder: bool = False):
    articles, chunks = load_articles(), load_chunks()
    qs = load_questions()
    emb = OnnxEmbedder()
    t0 = time.perf_counter()
    bm25, dense = BM25Retriever(chunks), DenseRetriever(chunks, emb)
    index_s = time.perf_counter() - t0

    systems, latency = {}, {}
    cold = dense._qcache.clear
    for r in [bm25, dense, HybridRetriever(bm25, dense, "rrf")]:
        systems[r.name], latency[r.name] = run_retriever(r, qs, reset=cold)
    systems["hybrid_weighted_cv"], alphas = cv_weighted_hybrid(bm25, dense, qs)
    a_med = float(np.median(alphas))
    _, latency["hybrid_weighted_cv"] = run_retriever(HybridRetriever(bm25, dense, "weighted", alpha=a_med), qs, reset=cold)
    feat = CandidateFeaturizer(bm25, dense, articles)
    systems["hybrid+ltr_rerank_cv"], latency["hybrid+ltr_rerank_cv"] = cv_ltr(feat, qs, reset=cold)
    # v2: glossary query expansion (motivated by v1 failure analysis)
    bm25x, densex = ExpandedRetriever(bm25), ExpandedRetriever(dense)
    for r in [bm25x, densex]:
        systems[r.name], latency[r.name] = run_retriever(r, qs, reset=cold)
    systems["hybrid_weighted_cv+expansion"], alphas_x = cv_weighted_hybrid(bm25x, densex, qs)
    _, latency["hybrid_weighted_cv+expansion"] = run_retriever(
        HybridRetriever(bm25x, densex, "weighted", alpha=float(np.median(alphas_x))), qs, reset=cold)

    # held-out set written after the glossary was frozen; alpha fixed from the main-set CV
    held = load_questions(path=HELDOUT_PATH)
    held_systems = {
        "bm25": bm25, "dense": dense,
        f"hybrid_weighted_a{a_med:g}": HybridRetriever(bm25, dense, "weighted", alpha=a_med),
        "bm25+expansion": bm25x, "dense+expansion": densex,
        f"hybrid_weighted_a{a_med:g}+expansion": HybridRetriever(bm25x, densex, "weighted", alpha=a_med),
    }
    heldout_results = {n: score_runs(run_retriever(r, held)[0], held) for n, r in held_systems.items()}

    if cross_encoder:
        ce = CrossEncoderReranker(HybridRetriever(bm25, dense, "rrf"), articles)
        systems[ce.name], latency[ce.name] = run_retriever(ce, qs, reset=cold)

    results = {"n_questions": len(qs), "n_chunks": len(chunks), "n_articles": len(articles),
               "embedder": emb.name, "index_build_seconds": round(index_s, 2),
               "weighted_alpha_per_fold": alphas, "weighted_alpha_per_fold_expansion": alphas_x,
               "systems": {}}
    for name, runs in systems.items():
        res = {"all": score_runs(runs, qs)}
        for t in ("lay", "legal"):
            sub = [q for q in qs if q["type"] == t]
            res[t] = score_runs(runs, sub)
        if name in latency:
            res["latency_ms_p50"] = round(statistics.median(latency[name]), 2)
        results["systems"][name] = res

    # failure analysis for the best system
    best = max(results["systems"], key=lambda n: results["systems"][n]["all"]["mrr@10"])
    fails = []
    for q in qs:
        r = first_hit_rank(systems[best][q["id"]], q["gold"])
        if r is None or r > 3:
            fails.append({"id": q["id"], "question": q["question"], "gold": q["gold"], "gold_rank": r,
                          "top3": systems[best][q["id"]][:3],
                          "bm25_rank": first_hit_rank(systems["bm25"][q["id"]], q["gold"]),
                          "dense_rank": first_hit_rank(systems["dense"][q["id"]], q["gold"])})
    results["best_system"] = best
    results["heldout"] = {"n_questions": len(held), "systems": heldout_results}
    results["failures_best_system"] = fails

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrieval_results.json").write_text(json.dumps(results, indent=2))
    (REPORTS / "retrieval_results.md").write_text(to_markdown(results))
    print(to_markdown(results))
    return results


def to_markdown(res: dict) -> str:
    lines = [f"Questions: {res['n_questions']} (excluding out-of-scope) | chunks: {res['n_chunks']} | "
             f"embedder: {res['embedder']}\n",
             "| System | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | R@5 lay | R@5 legal | p50 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in res["systems"].items():
        a = r["all"]
        lines.append(f"| {n} | {a['recall@1']:.3f} | {a['recall@3']:.3f} | {a['recall@5']:.3f} | {a['recall@10']:.3f} | "
                     f"{a['mrr@10']:.3f} | {a['ndcg@10']:.3f} | {r['lay']['recall@5']:.3f} | {r['legal']['recall@5']:.3f} | "
                     f"{r.get('latency_ms_p50', '-')} |")
    if "heldout" in res:
        lines += [f"\nHeld-out set ({res['heldout']['n_questions']} questions written after the glossary was frozen):\n",
                  "| System | R@1 | R@3 | R@5 | R@10 | MRR@10 |", "|---|---|---|---|---|---|"]
        for n, a in res["heldout"]["systems"].items():
            lines.append(f"| {n} | {a['recall@1']:.3f} | {a['recall@3']:.3f} | {a['recall@5']:.3f} | "
                         f"{a['recall@10']:.3f} | {a['mrr@10']:.3f} |")
    lines.append(f"\nBest by MRR@10: **{res['best_system']}** - {len(res['failures_best_system'])} questions "
                 f"with the gold article outside the top 3.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-encoder", action="store_true")
    main(ap.parse_args().cross_encoder)
