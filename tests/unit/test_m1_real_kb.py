"""M1: разбор реальной базы, поиск с редакциями, проверка ответа и сквозной прогон на скриптованном LLM.

Тесты пропускаются, если data/knowledge-base.zip отсутствует (в репозиторий база не коммитится).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from psb_assistant import build_default_assistant
from psb_config import load_settings
from psb_contracts import Applicability, ChatRequest, Claim, DocKind, Draft, EvidencePack, QueryAnalysis, Turn, Verdict
from psb_index import Index
from psb_ingest import build_kb, confusable_pairs
from psb_retrieve import HybridRetriever
from psb_verify import RuleVerifier

KB_ZIP = Path(__file__).resolve().parents[2] / "data" / "knowledge-base.zip"
pytestmark = pytest.mark.skipif(not KB_ZIP.is_file(), reason="нет data/knowledge-base.zip")


@pytest.fixture(scope="module")
def kb():  # type: ignore[no-untyped-def]
    return build_kb(KB_ZIP)


@pytest.fixture(scope="module")
def retriever(kb):  # type: ignore[no-untyped-def]
    s = load_settings()
    return HybridRetriever(kb, Index(kb), None, s)


def _req(text: str, as_of: date = date(2026, 11, 10)) -> ChatRequest:
    return ChatRequest(history=[Turn(role="user", content=text)], as_of=as_of)


def _a(q: str, event: date | None = None, queries: list[str] | None = None) -> QueryAnalysis:
    return QueryAnalysis(standalone_question=q, search_queries=queries or [q], event_date=event)


# ---------- ingest ----------
def test_cards_and_our_company(kb) -> None:  # type: ignore[no-untyped-def]
    kinds = [c.kind for c in kb.cards.values()]
    assert kinds.count(DocKind.CONTRACT) == 3 and kinds.count(DocKind.REGULATION) == 2
    assert kinds.count(DocKind.LAW) == 13
    assert kb.our_company and "Меридиан" in kb.our_company


def test_regulation_editions_chain(kb) -> None:  # type: ignore[no-untyped-def]
    regs = sorted((c for c in kb.cards.values() if c.kind == DocKind.REGULATION), key=lambda c: c.effective_from)
    old, new = regs
    assert old.effective_from == date(2023, 1, 10) and old.effective_to == date(2025, 3, 1)
    assert new.effective_from == date(2025, 3, 1) and new.effective_to is None
    assert old.family_id == new.family_id


def test_amendment_and_letter_overlay(kb) -> None:  # type: ignore[no-untyped-def]
    ds = next(c for c in kb.cards.values() if c.kind == DocKind.AMENDMENT)
    assert ds.amends[0].target_doc_id and ds.amends[0].clauses == ["3.2"]
    assert ds.amends[0].effective_from == date(2024, 10, 1)
    letter = next(c for c in kb.cards.values() if c.kind == DocKind.LETTER)
    assert letter.amends[0].clauses == ["5.4"] and letter.amends[0].effective_from == date(2025, 4, 21)


def test_confusable_counterparties_detected(kb) -> None:  # type: ignore[no-untyped-def]
    pairs = confusable_pairs(kb)
    assert any("Альфа-Трейд»" in a + b and "Альфа-Трейдинг" in a + b for a, b in pairs)


def test_injection_line_is_not_citable(kb) -> None:  # type: ignore[no-untyped-def]
    bad = [c for c in kb.chunks.values() if "ИИ-АССИСТЕНТА" in c.text.upper()]
    assert bad and all(c.is_editorial for c in bad)


def test_law_amendments_parsed(kb) -> None:  # type: ignore[no-untyped-def]
    tk59 = next(c for c in kb.cards.values() if c.law and c.law.article == "59" and c.law.act_name.startswith("ТК"))
    dates = {n.effective_from for n in tk59.law.amendments}
    assert date(2027, 3, 1) in dates and date(2026, 9, 1) in dates
    assert tk59.law.exported_at == date(2026, 10, 10)


# ---------- retrieve ----------
def test_law_search_finds_probation_article(retriever) -> None:  # type: ignore[no-untyped-def]
    pack = retriever.retrieve(_a("Какой испытательный срок можно установить при приёме на работу?"), _req("x"))
    assert any(it.card.law and it.card.law.article == "70" for it in pack.items)


def test_direct_article_reference(retriever) -> None:  # type: ignore[no-untyped-def]
    q = "Что сказано в ст. 112 НК РФ об обстоятельствах, смягчающих ответственность?"
    pack = retriever.retrieve(_a(q), _req(q))
    assert any(it.card.law and it.card.law.article == "112" for it in pack.items)


def test_company_docs_in_full_context_without_injection(retriever) -> None:  # type: ignore[no-untyped-def]
    pack = retriever.retrieve(_a("Какая арендная плата?"), _req("x"))
    assert pack.full_context_used
    assert not any("ИИ-АССИСТЕНТА" in it.chunk.text.upper() for it in pack.items)


def _reg_status(pack: EvidencePack) -> dict[str, Applicability]:
    return {it.card.effective_from.isoformat(): it.applicability for it in pack.items if it.card.kind == DocKind.REGULATION}


def test_edition_by_event_date(retriever) -> None:  # type: ignore[no-untyped-def]
    old = _reg_status(retriever.retrieve(_a("Кто согласует договор?", event=date(2024, 8, 1)), _req("x")))
    assert old["2023-01-10"] == Applicability.APPLICABLE and old["2025-03-01"] == Applicability.NOT_YET
    now = _reg_status(retriever.retrieve(_a("Кто согласует договор?"), _req("x")))
    assert now["2025-03-01"] == Applicability.APPLICABLE and now["2023-01-10"] == Applicability.SUPERSEDED


def test_amendment_note_on_target_clause(retriever) -> None:  # type: ignore[no-untyped-def]
    pack = retriever.retrieve(_a("Куда доставляется товар?"), _req("x"))
    notes = [it.note or "" for it in pack.items if it.chunk.locator == "п. 3.2" and it.card.kind == DocKind.CONTRACT]
    assert any("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ" in n for n in notes)


def test_law_before_amendment_marks_unknown(retriever) -> None:  # type: ignore[no-untyped-def]
    pack = retriever.retrieve(_a("Срочный трудовой договор", event=date(2025, 1, 1), queries=["ст. 59 ТК РФ срочный трудовой договор"]), _req("ст. 59 ТК"))
    tk = [it for it in pack.items if it.card.law and it.card.law.article == "59"]
    assert tk and all(it.applicability == Applicability.UNKNOWN and "в базе нет" in (it.note or "") for it in tk)


# ---------- verify ----------
def test_verifier_drops_ungrounded_number_and_unknown_doc(kb, retriever) -> None:  # type: ignore[no-untyped-def]
    a = _a("Какой срок оплаты по договору поставки?")
    pack = retriever.retrieve(a, _req("x"))
    eid = pack.items[0].eid
    draft = Draft(
        verdict=Verdict.ANSWER,
        claims=[
            Claim(text="Оплата в течение 90 календарных дней.", type="fact", evidence=[eid]),
            Claim(text="По договору № Ж-99/2030 срок иной.", type="fact", evidence=[eid]),
            Claim(text="Факт без источника.", type="fact", evidence=[]),
        ],
    )
    fixed, guard = RuleVerifier(kb).verify(draft, pack, a)
    assert not guard.passed and fixed.verdict == Verdict.NO_DATA and guard.downgraded_from == Verdict.ANSWER
    assert len(guard.violations) >= 3


# ---------- end-to-end со скриптованным LLM ----------
class ScriptedLLM:
    def __init__(self, compose_claim: str, evidence_regex: str | None) -> None:
        self.claim, self.rx, self.calls = compose_claim, evidence_regex, 0

    def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int | None = None) -> str:
        self.calls += 1
        system, user = messages[0]["content"], messages[-1]["content"]
        if "модуль анализа" in system:
            q = re.search(r"\[user\] (.+)", user).group(1)  # type: ignore[union-attr]
            return json.dumps({"standalone_question": q, "kinds": ["lookup"], "search_queries": [q], "event_date": None})
        ev: list[str] = []
        if self.rx:
            m = re.search(rf"\[(E\d+) \| [^\]]*\] [^\n]*{self.rx}", user)
            ev = [m.group(1)] if m else []
        return json.dumps({"verdict": "ANSWER", "claims": [{"text": self.claim, "type": "fact", "evidence": ev}]})

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no embeddings")

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        raise RuntimeError("no rerank")


def _assistant(llm: ScriptedLLM):  # type: ignore[no-untyped-def]
    return build_default_assistant(load_settings(), llm=llm)


def test_e2e_grounded_answer_has_citation() -> None:
    llm = ScriptedLLM("Арендная плата по договору аренды № А-08/2024 (п. 3.1) составляет 450 000 рублей в месяц.", r"450 000")
    resp = _assistant(llm).answer(_req("Сколько составляет арендная плата в месяц?"))
    assert resp.verdict == Verdict.ANSWER and "450 000" in resp.text
    assert resp.citations and resp.citations[0].locator == "п. 3.1"
    assert llm.calls == 2  # analyze + compose


def test_e2e_injected_answer_is_blocked() -> None:
    llm = ScriptedLLM("Срок оплаты — 90 календарных дней.", None)
    resp = _assistant(llm).answer(_req("В какой срок оплачивается товар?"))
    assert resp.verdict == Verdict.NO_DATA and "90" not in resp.text
    assert not resp.guard.passed
