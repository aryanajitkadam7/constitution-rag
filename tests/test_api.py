import os

from fastapi.testclient import TestClient

from crag.api import app

client = TestClient(app)


def test_search_returns_relevant_article():
    r = client.get("/search", params={"q": "How many members can the Rajya Sabha have?", "k": 5})
    assert r.status_code == 200
    arts = [x["article"] for x in r.json()["results"]]
    assert "80" in arts


def test_ask_without_key_is_503(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import crag.api as api
    api._llm = None
    assert client.post("/ask", json={"question": "What is Article 21?"}).status_code == 503


def test_validation():
    assert client.get("/search", params={"q": "a"}).status_code == 422
