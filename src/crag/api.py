"""FastAPI service.

    uvicorn crag.api:app --port 8000
    GET  /search?q=...&k=5   -> ranked articles (no API key needed)
    POST /ask {"question": "..."} -> grounded answer with verified citations (needs GEMINI_API_KEY)
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .generate import GeminiGenerator, answer
from .pipeline import build

app = FastAPI(title="Constitution of India RAG", version="1.0.0")
_llm = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health():
    r, lookup, _ = build()
    return {"status": "ok", "retriever": r.name, "chunks": len(lookup)}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(default=5, ge=1, le=20)):
    r, lookup, arts = build()
    t0 = time.perf_counter()
    hits = r.search(q, k)
    return {
        "query": q,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "results": [{"article": h.article, "title": arts[h.article]["title"], "score": round(h.score, 4),
                     "snippet": lookup[h.chunk_id].body[:300]} for h in hits],
    }


@app.post("/ask")
def ask(req: AskRequest):
    global _llm
    try:
        _llm = _llm or GeminiGenerator()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    r, lookup, _ = build()
    a = answer(req.question, r, lookup, _llm, k=req.k)
    return {"answer": a.text, "citations": a.cited, "retrieved_articles": a.retrieved,
            "refused": a.refused, "unverified_citations": a.invalid_citations}
