"""Grounded answer generation with Gemini + citation checking.

The model gets the top-k retrieved chunks, each tagged with its article number, and
must (a) answer only from them, (b) cite every claim as [Art. N], and (c) reply with
the exact refusal string when the context does not contain the answer.

After generation we *verify* the citations: every cited article must be one of the
retrieved ones. Answers citing articles that were not in the context are flagged.

Requires: pip install google-genai ; export GEMINI_API_KEY=...
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

REFUSAL = "I could not find this in the Constitution of India."
CITE_RE = re.compile(r"\[Art(?:icle)?\.?\s*([0-9]+[A-Z]*(?:\s?[A-Z]+)?|Preamble)\]", re.I)

SYSTEM = f"""You answer questions about the Constitution of India using ONLY the numbered context passages.
Rules:
- Use plain language a non-lawyer understands. Keep answers under 120 words.
- After every sentence that states a rule, cite the article like [Art. 21] or [Art. Preamble].
- Cite only articles that appear in the context.
- If the context does not contain the answer, reply exactly: "{REFUSAL}"
- Do not use outside knowledge, case law or current events."""


@dataclass
class Answer:
    text: str
    cited: list[str]
    retrieved: list[str]
    refused: bool
    invalid_citations: list[str] = field(default_factory=list)


def parse_citations(text: str) -> list[str]:
    out = []
    for m in CITE_RE.findall(text):
        a = "Preamble" if m.lower() == "preamble" else m.replace(" ", "").upper()
        if a not in out:
            out.append(a)
    return out


def build_prompt(question: str, chunks) -> str:
    ctx = "\n\n".join(f"[{i+1}] (Article {c.article}) {c.text}" for i, c in enumerate(chunks))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


class GeminiGenerator:
    def __init__(self, model: str = "gemini-3.8-flash", temperature: float = 0.0):
        from google import genai  # optional dependency

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY to use generation.")
        self.client = genai.Client(api_key=key)
        self.model, self.temperature = model, temperature

    def complete(self, prompt: str) -> str:
        from google.genai import types

        r = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=self.temperature))
        return (r.text or "").strip()


def answer(question: str, retriever, chunk_lookup: dict, llm, k: int = 5) -> Answer:
    hits = retriever.search(question, k)
    chunks = [chunk_lookup[h.chunk_id] for h in hits if h.chunk_id in chunk_lookup]
    text = llm.complete(build_prompt(question, chunks))
    cited = parse_citations(text)
    retrieved = [c.article for c in chunks]
    return Answer(text=text, cited=cited, retrieved=retrieved, refused=REFUSAL.lower() in text.lower(),
                  invalid_citations=[a for a in cited if a not in retrieved])
