"""Writes data/eval/questions.jsonl.

The eval set was written by hand (with AI assistance) and every gold label was checked
against the article text in data/constitution.json (see `verify` below).

types
  lay          - everyday wording a citizen would use; little word overlap with the article title
  legal        - uses legal/constitutional terminology, closer to the article wording
  out_of_scope - not answerable from the Constitution text; used to test refusal in generation
"""
import json
from pathlib import Path

LAY = [
 ("Can the police keep me locked up for days without bringing me before a judge?", ["22"]),
 ("Is it legal for a factory to hire a 12-year-old?", ["24"]),
 ("Do kids have a right to free schooling?", ["21A", "45"]),
 ("Can the government force people to work without pay?", ["23"]),
 ("Can I be punished twice for the same crime?", ["20"]),
 ("Can a shop refuse entry to someone because of their caste?", ["15", "17"]),
 ("Is treating someone as an outcaste because of birth a crime?", ["17"]),
 ("Can the government give a person an honour like 'Sir' or 'Raja'?", ["18"]),
 ("Am I allowed to hold a peaceful protest with my friends?", ["19"]),
 ("If the government violates my basic rights, which court can I approach directly?", ["32", "226"]),
 ("Can a minority community run its own school or college?", ["30"]),
 ("Can I be made to pay a tax that funds one particular religion?", ["27"]),
 ("Does a government-funded school have to make my child attend prayers?", ["28"]),
 ("Is there a promise of free lawyers for poor people?", ["39A"]),
 ("Is the government supposed to bring one common family law for all citizens?", ["44"]),
 ("Who is responsible for protecting forests and wildlife?", ["48A", "51A"]),
 ("What responsibilities does every citizen owe to the country?", ["51A"]),
 ("How old must you be to become President?", ["58"]),
 ("How long does a President stay in office?", ["56"]),
 ("How can a President be removed from office for violating the Constitution?", ["61"]),
 ("Can the President spare the life of a person sentenced to death?", ["72"]),
 ("Who takes over as President if the President dies suddenly?", ["65"]),
 ("Who advises the President when making decisions?", ["74"]),
 ("How many members can the Rajya Sabha have?", ["80"]),
 ("What is the maximum strength of the Lok Sabha?", ["81"]),
 ("After how many years does the Lok Sabha automatically end?", ["83"]),
 ("What is the minimum age to become a member of Parliament?", ["84"]),
 ("What happens if an MP stays away from Parliament for months without permission?", ["101"]),
 ("Can an MP be sued for something they said inside Parliament?", ["105"]),
 ("What happens when the two Houses of Parliament keep disagreeing on a bill?", ["108"]),
 ("Which kind of bill can only be introduced in the Lok Sabha?", ["109", "110"]),
 ("Can the President refuse to sign a bill passed by Parliament?", ["111"]),
 ("Who places the yearly budget before Parliament?", ["112"]),
 ("Can the government make a law when Parliament is not sitting?", ["123"]),
 ("In which languages can MPs speak in Parliament?", ["120"]),
 ("Can MPs discuss how a judge behaved while doing their job?", ["121"]),
 ("Who appoints the judges of the top court of the country?", ["124"]),
 ("Can the Supreme Court directly hear a fight between the Centre and a State?", ["131"]),
 ("Can I go to the Supreme Court even if the High Court refused to let me appeal?", ["136"]),
 ("Can the Supreme Court take a second look at its own decision?", ["137"]),
 ("Are lower courts bound to follow what the Supreme Court decides?", ["141"]),
 ("Can the President ask the Supreme Court for its opinion on a legal question?", ["143"]),
 ("Who checks the accounts of the central government?", ["148", "149"]),
 ("Who chooses the Governor of a State?", ["155"]),
 ("Can a Governor forgive a prisoner's sentence?", ["161"]),
 ("Who is the top lawyer for a State government?", ["165"]),
 ("Can a State have an upper house in its legislature?", ["168", "169"]),
 ("Can a Governor send a State bill to the President instead of signing it?", ["200", "201"]),
 ("Can a Governor make temporary laws when the assembly is not in session?", ["213"]),
 ("At what age does a High Court judge retire?", ["217"]),
 ("Can a retired High Court judge go back to practising as a lawyer?", ["220"]),
 ("Can a High Court judge be shifted to another state's High Court?", ["222"]),
 ("Can a High Court order the release of someone held illegally?", ["226"]),
 ("Who appoints district judges?", ["233"]),
 ("What is the meeting of all voters in a village called?", ["243A"]),
 ("Are seats kept aside for women in village councils?", ["243D"]),
 ("How many years does a village council last?", ["243E"]),
 ("Who runs elections for village councils?", ["243K"]),
 ("Are seats reserved in city corporations and municipal councils?", ["243T"]),
 ("Can a member of a co-operative society see its books and accounts?", ["243ZO"]),
 ("Who can make laws on a subject that is not in any of the lists?", ["248"]),
 ("Who has the power to make laws on GST?", ["246A"]),
 ("If a central law and a state law clash, which one wins?", ["254"]),
 ("How are fights between states over river water settled?", ["262"]),
 ("Can the government collect a tax without passing a law?", ["265"]),
 ("Is there an emergency kitty the government can use for unforeseen spending?", ["267"]),
 ("Who recommends how tax money is shared between the Centre and the States?", ["280"]),
 ("Can the government take my land without any law allowing it?", ["300A"]),
 ("Can goods move freely from one state to another?", ["301"]),
 ("Can a government employee be sacked without being heard?", ["311"]),
 ("Who creates services like the IAS and IPS?", ["312"]),
 ("Who conducts elections to Parliament and the state assemblies?", ["324"]),
 ("Can someone be left off the voter list because of their religion?", ["325"]),
 ("At what age can a citizen vote?", ["326"]),
 ("Can courts stop an election that is in progress?", ["329"]),
 ("Are Lok Sabha seats reserved for Scheduled Castes and Tribes?", ["330"]),
 ("What is the official language of the central government?", ["343"]),
 ("In which language are Supreme Court judgments written?", ["348"]),
 ("Should small children be taught in their mother tongue?", ["350A"]),
 ("When can a national emergency be declared?", ["352"]),
 ("What happens if a state government cannot be run as per the Constitution?", ["356"]),
 ("Can freedom of speech be put on hold during an emergency?", ["358", "359"]),
 ("When can the President declare a financial emergency?", ["360"]),
 ("Can a sitting President be taken to court for a crime?", ["361"]),
 ("How can the Constitution itself be changed?", ["368"]),
 ("What special status did Jammu and Kashmir have?", ["370"]),
 ("Are there special provisions for Nagaland?", ["371A"]),
 ("Can the State make special provisions for women and children?", ["15"]),
 ("Are government jobs equally open to all citizens?", ["16"]),
 ("Can someone be convicted for something that was not a crime when they did it?", ["20"]),
 ("Am I free to travel and settle anywhere in India?", ["19"]),
 ("Can I be forced to give evidence against myself?", ["20"]),
 ("Can my life or freedom be taken away arbitrarily?", ["21"]),
 ("Does an Indian who voluntarily takes another country's citizenship stay Indian?", ["9"]),
 ("Who has the power to make laws about citizenship?", ["11"]),
 ("Can Parliament create a new state or change a state's borders?", ["3"]),
 ("What other name does the Constitution give to India?", ["1"]),
 ("What ideals does the opening statement of the Constitution promise?", ["Preamble"]),
 ("Should judges be kept separate from the government's executive?", ["50"]),
 ("Should the State protect historic monuments?", ["49"]),
 ("Is the State supposed to improve nutrition and discourage alcohol?", ["47"]),
 ("Are workers supposed to get a decent living wage?", ["43"]),
 ("Should workers have a say in running industries?", ["43A"]),
 ("Should the State provide maternity relief?", ["42"]),
 ("Should the State stop wealth from piling up in a few hands?", ["39"]),
 ("Should the State improve cattle breeds and ban cow slaughter?", ["48"]),
 ("Can Parliament limit the rights of soldiers and police?", ["33"]),
 ("Is a law that takes away fundamental rights valid?", ["13"]),
 ("Can a Chief Minister hide government decisions from the Governor?", ["167"]),
 ("Who looks after the welfare of linguistic minorities?", ["350B"]),
]

LEGAL = [
 ("Issue of writs of habeas corpus, mandamus, prohibition, quo warranto and certiorari by the Supreme Court", ["32"]),
 ("Proclamation of Emergency on the ground of armed rebellion", ["352"]),
 ("Disqualification of Members of Parliament under the Tenth Schedule on ground of defection", ["102"]),
 ("Establishment of an Inter-State Council", ["263"]),
 ("Constitution and functions of the Finance Commission", ["280"]),
 ("Consolidated Fund of India and public accounts", ["266"]),
 ("Ordinance-making power of the President during recess of Parliament", ["123"]),
 ("Special leave to appeal from any judgment of any court or tribunal", ["136"]),
 ("Review of judgments or orders pronounced by the Supreme Court", ["137"]),
 ("Administrative tribunals for disputes regarding recruitment and conditions of service", ["323A"]),
 ("National Commission for Scheduled Tribes", ["338A"]),
 ("Public Service Commissions for the Union and for the States", ["315"]),
 ("Superintendence, direction and control of elections vested in the Election Commission", ["324"]),
 ("Residuary powers of legislation of Parliament", ["248"]),
 ("Powers and functions of the Gram Sabha", ["243A"]),
 ("Committee for Metropolitan planning", ["243ZE"]),
 ("Committee for district planning", ["243ZD"]),
 ("Power of the Governor to grant pardons, reprieves, respites or remissions of punishment", ["161"]),
 ("Tenure of office of civil servants at the pleasure of the President", ["310"]),
 ("Directive principle on a uniform civil code", ["44"]),
 ("Protection against double jeopardy and self-incrimination", ["20"]),
 ("Prohibition of traffic in human beings and begar", ["23"]),
 ("Power of superintendence of High Courts over all courts and tribunals", ["227"]),
 ("Borrowing by the Government of India upon the security of the Consolidated Fund", ["292"]),
 ("Joint sitting of both Houses of Parliament", ["108"]),
 ("Special provisions with respect to the State of Sikkim", ["371F"]),
 ("Official language of the Union: Hindi in Devanagari script", ["343"]),
 ("Annual financial statement laid before both Houses", ["112"]),
 ("Abolition or creation of Legislative Councils in States", ["169"]),
 ("Contingency Fund of India placed at the disposal of the President", ["267"]),
]

OUT_OF_SCOPE = [
 "What is the current GST rate on gold jewellery?",
 "Who is the current Prime Minister of India?",
 "What is the punishment for murder under the Indian Penal Code?",
 "How do I apply for an Indian passport?",
 "What are the income tax slabs for salaried employees this year?",
 "Which team won the 2011 Cricket World Cup?",
 "What is the speed limit on national highways?",
 "How many judges currently sit in the Bombay High Court?",
 "What did the Supreme Court hold in the Kesavananda Bharati case?",
 "What is the current repo rate set by the RBI?",
 "How do I file an RTI application online?",
 "What is the fine for riding a two-wheeler without a helmet?",
]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"
    rows = []
    for i, (q, g) in enumerate(LAY):
        rows.append({"id": f"lay-{i:03d}", "type": "lay", "question": q, "gold": g})
    for i, (q, g) in enumerate(LEGAL):
        rows.append({"id": f"legal-{i:03d}", "type": "legal", "question": q, "gold": g})
    for i, q in enumerate(OUT_OF_SCOPE):
        rows.append({"id": f"oos-{i:03d}", "type": "out_of_scope", "question": q, "gold": []})
    verify(rows)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"wrote {len(rows)} questions -> {out}")


def verify(rows):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from crag.corpus import load_articles
    known = {a["article"] for a in load_articles()}
    bad = [(r["id"], g) for r in rows for g in r["gold"] if g not in known]
    assert not bad, f"gold articles missing from corpus: {bad}"
    qs = [r["question"].lower() for r in rows]
    assert len(set(qs)) == len(qs), "duplicate questions"


if __name__ == "__main__":
    main()
