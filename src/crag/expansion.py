"""Glossary-based query expansion for Indian civic vocabulary.

Failure analysis of v1 (see README) showed that most misses were *vocabulary gaps*:
people say "Lok Sabha", "MP", "village council" or "budget", while the Constitution says
"House of the People", "member of Parliament", "Panchayat" and "annual financial statement".
Neither BM25 nor a general-purpose English embedding model knows these equivalences.

The glossary below maps everyday / Hindi-origin terms to the constitutional terms. It was
written after the v1 failure analysis, so it is evaluated on a *separate held-out question
set written after the glossary was frozen* (data/eval/questions_heldout.jsonl), not only on
the set that motivated it.
"""
from __future__ import annotations

import re

GLOSSARY: dict[str, str] = {
    "lok sabha": "House of the People",
    "rajya sabha": "Council of States",
    "vidhan sabha": "Legislative Assembly",
    "vidhan parishad": "Legislative Council",
    "mp": "member of Parliament",
    "mps": "members of Parliament",
    "mla": "member of the Legislative Assembly",
    "mlas": "members of the Legislative Assembly",
    "pm": "Prime Minister",
    "cm": "Chief Minister",
    "centre": "Union",
    "central government": "Union Government of India",
    "village council": "Panchayat",
    "gram panchayat": "Panchayat",
    "village assembly": "Gram Sabha",
    "city corporation": "Municipality",
    "municipal corporation": "Municipality",
    "nagar palika": "Municipality",
    "budget": "annual financial statement",
    "cag": "Comptroller and Auditor-General",
    "upsc": "Union Public Service Commission",
    "ias": "All-India Services",
    "ips": "All-India Services",
    "gst": "goods and services tax",
    "president's rule": "failure of constitutional machinery in State",
    "temporary law": "Ordinance",
    "outcaste": "untouchability",
    "untouchable": "untouchability",
    "quota": "reservation",
    "top court": "Supreme Court",
    "apex court": "Supreme Court",
    "chief justice": "Chief Justice of India",
    "bharat": "India",
    "sacked": "dismissed removed",
    "fired": "dismissed removed",
    "jail": "custody detention",
    "protest": "assemble peaceably",
    "free speech": "freedom of speech and expression",
}

_PATTERNS = [(re.compile(rf"\b{re.escape(k)}\b", re.I), v) for k, v in
             sorted(GLOSSARY.items(), key=lambda kv: -len(kv[0]))]


def expand(query: str) -> str:
    """Append the constitutional term for every glossary term found in the query."""
    extra = []
    for pat, v in _PATTERNS:
        if pat.search(query) and v.lower() not in query.lower() and v not in extra:
            extra.append(v)
    return f"{query} ({'; '.join(extra)})" if extra else query


class ExpandedRetriever:
    """Wraps any retriever so that queries are expanded before search."""

    def __init__(self, base):
        self.base = base
        self.chunks = base.chunks
        self.name = f"{base.name}+expansion"

    def scores(self, query: str):
        return self.base.scores(expand(query))

    def search(self, query: str, k: int = 10):
        return self.base.search(expand(query), k)
