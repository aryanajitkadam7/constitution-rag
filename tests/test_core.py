from crag.corpus import chunk_articles, load_articles, load_chunks
from crag.evaluate import first_hit_rank, load_questions, ndcg_at_10, score_runs
from crag.expansion import expand
from crag.generate import REFUSAL, answer, parse_citations


def test_corpus_loads_all_articles_and_chunks_keep_headers():
    arts = load_articles()
    assert len(arts) == 465 and arts[0]["article"] == "Preamble"
    chunks = load_chunks()
    assert all(c.text.startswith(("Article ", "Preamble")) for c in chunks)
    assert {c.article for c in chunks} == {a["article"] for a in arts}


def test_long_article_is_split_with_overlap():
    para = " ".join(["word"] * 100)
    art = {"article": "999", "title": "Test", "text": "\n".join([f"({i}) {para}" for i in range(5)])}
    chunks = chunk_articles([art], max_words=220)
    assert len(chunks) > 1
    assert chunks[1].body.split(" ")[0] == "(1)"  # previous paragraph repeated as overlap


def test_eval_set_is_well_formed():
    qs = load_questions(include_oos=True)
    assert len(qs) >= 150
    assert all(q["gold"] for q in qs if q["type"] != "out_of_scope")


def test_metrics():
    assert first_hit_rank(["5", "7", "9"], ["9"]) == 3
    assert first_hit_rank(["5"], ["9"]) is None
    assert ndcg_at_10(["9"], ["9"]) == 1.0
    qs = [{"id": "a", "gold": ["1"]}, {"id": "b", "gold": ["2"]}]
    m = score_runs({"a": ["1", "3"], "b": ["3", "2"]}, qs)
    assert m["recall@1"] == 0.5 and m["recall@3"] == 1.0 and m["mrr@10"] == 0.75


def test_expansion_maps_everyday_terms():
    assert "House of the People" in expand("How big is the Lok Sabha?")
    assert "Panchayat" in expand("who runs the village council")
    assert expand("What is Article 21 about?") == "What is Article 21 about?"
    assert "member of Parliament" in expand("Can an MP be arrested?")
    assert "Prime Minister" not in expand("pmt schedule")  # whole-word matching only


def test_parse_citations():
    t = "You can [Art. 21]. Also [Article 21A] and [art 243ZO] and [Art. Preamble]. Again [Art. 21]."
    assert parse_citations(t) == ["21", "21A", "243ZO", "Preamble"]


class FakeLLM:
    def __init__(self, reply): self.reply = reply
    def complete(self, prompt): return self.reply


def test_answer_flags_citations_outside_context():
    from crag.pipeline import build
    r, lookup, _ = build()
    a = answer("Can the police detain me without telling me why?", r, lookup, FakeLLM("No [Art. 22]. See [Art. 999]."))
    assert "22" in a.retrieved and a.invalid_citations == ["999"] and not a.refused
    a2 = answer("Who won the cricket world cup?", r, lookup, FakeLLM(REFUSAL))
    assert a2.refused
