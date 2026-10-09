"""Load the Constitution of India and split it into retrievable chunks.

Chunking strategy
-----------------
* One article = one logical document. Most articles are short (median ~120 words),
  so they stay whole.
* Articles longer than `max_words` are split on clause boundaries ("(1)", "(2)", ...)
  and packed into windows of <= max_words with a one-paragraph overlap.
* Every chunk is prefixed with "Article <n>: <title>." so that both BM25 and the
  embedding model see the article heading, which carries most of the topic signal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "constitution.json"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    article: str
    title: str
    text: str  # includes the "Article n: title." header

    @property
    def body(self) -> str:
        return self.text.split(". ", 1)[-1]


def _clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def load_articles(path: Path = CORPUS_PATH) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arts = []
    for r in raw:
        art = "Preamble" if r["article"] == 0 else str(r["article"])
        arts.append({"article": art, "title": r["title"].strip(), "text": _clean(r["description"])})
    return arts


def chunk_articles(articles: list[dict], max_words: int = 220) -> list[Chunk]:
    chunks: list[Chunk] = []
    for a in articles:
        header = f"Article {a['article']}: {a['title']}." if a["article"] != "Preamble" else "Preamble."
        paras = [p.strip() for p in a["text"].split("\n") if p.strip()]
        windows, cur = [], []
        for p in paras:
            if cur and sum(len(x.split()) for x in cur) + len(p.split()) > max_words:
                windows.append(cur)
                cur = [cur[-1]]  # one-paragraph overlap
            cur.append(p)
        if cur:
            windows.append(cur)
        for i, w in enumerate(windows):
            chunks.append(Chunk(f"{a['article']}#{i}", a["article"], a["title"], f"{header} {' '.join(w)}"))
    return chunks


def load_chunks(max_words: int = 220) -> list[Chunk]:
    return chunk_articles(load_articles(), max_words)
