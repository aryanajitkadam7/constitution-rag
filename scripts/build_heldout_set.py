"""Writes data/eval/questions_heldout.jsonl.

Written AFTER the v1 failure analysis and AFTER the expansion glossary (src/crag/expansion.py)
was frozen, so it measures whether query expansion generalises beyond the questions that
motivated it. Same labelling rules as build_eval_set.py.
"""
import json
from pathlib import Path

HELDOUT = [
 ("How many seats in the Lok Sabha are reserved for Scheduled Tribes?", ["330"]),
 ("Can an MLA be disqualified for holding another paid government post?", ["191"]),
 ("Who decides whether an MP has become disqualified?", ["103"]),
 ("Can the Rajya Sabha block a money bill?", ["109"]),
 ("What is the minimum age to be elected to the Vidhan Sabha?", ["173"]),
 ("Who appoints the Prime Minister and the other ministers?", ["75"]),
 ("Does the PM have to keep the President informed about government decisions?", ["78"]),
 ("Who is the chief law officer of the Indian government?", ["76"]),
 ("Where do the CAG's audit reports go after they are prepared?", ["151"]),
 ("How long can President's rule continue in a state?", ["356"]),
 ("Can someone become Chief Minister without being an MLA?", ["164"]),
 ("Who appoints the Chief Minister?", ["164"]),
 ("Can women and men be paid differently for the same work?", ["39"]),
 ("Is bonded labour banned?", ["23"]),
 ("Can I practise and spread my religion?", ["25"]),
 ("Can a religious group own property and run its own affairs?", ["26"]),
 ("Is carrying a kirpan by Sikhs protected as part of their religion?", ["25"]),
 ("Can a gram panchayat collect taxes?", ["243H"]),
 ("Who audits the accounts of a village council?", ["243J"]),
 ("Is there a body that reviews the finances of panchayats in each state?", ["243I"]),
 ("How long does a municipal corporation last before fresh elections?", ["243U"]),
 ("What qualifications are needed to become a Supreme Court judge?", ["124"]),
 ("How can a Supreme Court judge be removed?", ["124"]),
 ("In which city does the Supreme Court sit?", ["130"]),
 ("Can a retired judge be asked to sit in the Supreme Court again?", ["128"]),
 ("Can the central government give directions to state governments?", ["256", "257"]),
 ("Can states borrow money, and what limits apply?", ["293"]),
 ("Is there a cap on the tax states can charge on professions and jobs?", ["276"]),
 ("Can a state tax electricity used by the central government?", ["287"]),
 ("Who appoints UPSC members and how long do they serve?", ["316"]),
 ("Can a UPSC member be removed from office?", ["317"]),
 ("Can Parliament make a law on a state subject if the Rajya Sabha says it is in the national interest?", ["249"]),
 ("Can two or more states ask Parliament to make a law for them?", ["252"]),
 ("Can the Centre make laws to implement an international treaty?", ["253"]),
 ("Which language should two states use when writing to each other?", ["346"]),
 ("Can I send a complaint to the government in any language?", ["350"]),
 ("Do MPs get a salary?", ["106"]),
 ("Who fixes the salaries of Supreme Court judges?", ["125"]),
]


def main():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from crag.corpus import load_articles
    known = {a["article"] for a in load_articles()}
    main_qs = {json.loads(l)["question"].lower() for l in (root / "data/eval/questions.jsonl").read_text().splitlines()}
    rows = []
    for i, (q, g) in enumerate(HELDOUT):
        assert all(x in known for x in g), (q, g)
        assert q.lower() not in main_qs, q
        rows.append({"id": f"heldout-{i:03d}", "type": "lay", "question": q, "gold": g})
    out = root / "data" / "eval" / "questions_heldout.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} held-out questions -> {out}")


if __name__ == "__main__":
    main()
