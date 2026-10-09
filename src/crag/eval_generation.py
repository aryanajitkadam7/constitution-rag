"""End-to-end answer evaluation (needs GEMINI_API_KEY).

    python -m crag.eval_generation [--model gemini-3.8-flash] [--limit 50]

All checks are deterministic, no LLM-as-judge:
  * citation_hit_rate      - in-scope answers that cite at least one gold article
  * citation_validity      - answers whose citations are all inside the retrieved context
  * false_refusal_rate     - in-scope questions the system refused
  * oos_refusal_rate       - out-of-scope questions correctly refused
Writes reports/generation_results.json and reports/generation_samples.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .evaluate import REPORTS, load_questions
from .generate import GeminiGenerator, answer
from .pipeline import build


def main(model: str, limit: int | None, sleep: float):
    retriever, lookup, _ = build()
    llm = GeminiGenerator(model)
    qs = load_questions(include_oos=True)
    if limit:
        inscope = [q for q in qs if q["type"] != "out_of_scope"][:limit]
        qs = inscope + [q for q in qs if q["type"] == "out_of_scope"]
    samples, lat = [], []
    for q in qs:
        t0 = time.perf_counter()
        a = answer(q["question"], retriever, lookup, llm)
        lat.append(time.perf_counter() - t0)
        samples.append({**q, "answer": a.text, "cited": a.cited, "retrieved": a.retrieved,
                        "refused": a.refused, "invalid_citations": a.invalid_citations})
        time.sleep(sleep)  # stay under free-tier rate limits

    write(score_samples(samples, model, retriever.name, lat), samples)


def score_samples(samples: list[dict], model: str, retriever_name: str, lat: list[float]) -> dict:
    ins = [s for s in samples if s["type"] != "out_of_scope"]
    oos = [s for s in samples if s["type"] == "out_of_scope"]
    answered = [s for s in ins if not s["refused"]]
    return {
        "model": model, "retriever": retriever_name, "n_in_scope": len(ins), "n_out_of_scope": len(oos),
        "citation_hit_rate": float(np.mean([bool(set(s["cited"]) & set(s["gold"])) for s in ins])),
        "citation_validity": float(np.mean([not s["invalid_citations"] for s in answered])) if answered else None,
        "false_refusal_rate": float(np.mean([s["refused"] for s in ins])),
        "oos_refusal_rate": float(np.mean([s["refused"] for s in oos])) if oos else None,
        "latency_s_p50": float(np.median(lat)),
    }


def write(res: dict, samples: list[dict]):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "generation_results.json").write_text(json.dumps(res, indent=2))
    (REPORTS / "generation_samples.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples))
    print(json.dumps(res, indent=2))


def from_answers(path: str):
    """Score answers produced elsewhere (e.g. by calling Gemini from a browser with
    reports/generation_prompts.json). File format: {"<question id>": {"text": ..., "latency_s": ...}}"""
    from .generate import REFUSAL, parse_citations

    P = json.loads((REPORTS / "generation_prompts.json").read_text())
    ans = json.loads(open(path, encoding="utf-8").read())
    samples, lat = [], []
    for r in P["rows"]:
        a = ans[r["id"]]
        cited = parse_citations(a["text"])
        samples.append({k: r[k] for k in ("id", "type", "question", "gold", "retrieved")} | {
            "answer": a["text"], "cited": cited, "refused": REFUSAL.lower() in a["text"].lower(),
            "invalid_citations": [c for c in cited if c not in r["retrieved"]]})
        lat.append(a["latency_s"])
    res = score_samples(samples, P["model"], P["retriever"], lat)
    write(res, samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--from-answers", default=None, help="score a JSON file of pre-generated answers")
    a = ap.parse_args()
    if a.from_answers:
        from_answers(a.from_answers)
    else:
        main(a.model, a.limit, a.sleep)
