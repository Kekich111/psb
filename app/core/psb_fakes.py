"""Фейковые реализации всех протоколов. Нужны, чтобы писать любой блок и UI без остальных.

Фикстуры ниже — ТЕСТОВЫЕ ДУБЛИ, а не логика: в боевом коде названий документов и сумм быть не должно.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from psb_contracts import (
    Analyzer,
    Applicability,
    AssistantResponse,
    ChatRequest,
    Chunk,
    Claim,
    Composer,
    DocCard,
    DocClass,
    DocKind,
    Draft,
    EvidenceItem,
    EvidencePack,
    GuardReport,
    Ingestor,
    KnowledgeBase,
    Party,
    QueryAnalysis,
    Retriever,
    Verdict,
    Verifier,
)

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("ё", "е")))


def fixture_kb() -> KnowledgeBase:
    contract = DocCard(
        doc_id="contract_a",
        file="документы компании__Тестовый договор аренды.txt",
        doc_class=DocClass.COMPANY,
        kind=DocKind.CONTRACT,
        title="Договор аренды № Т-01/2024",
        number="Т-01/2024",
        doc_date=date(2024, 8, 1),
        parties=[Party(name="ООО «Тест-Арендодатель»", role="Арендодатель"), Party(name="ООО «Меридиан»", role="Арендатор")],
        approx_tokens=40,
    )
    law = DocCard(
        doc_id="law_tk_70",
        file="законы__ТК_ст70_действующая.txt",
        doc_class=DocClass.LAW,
        kind=DocKind.LAW,
        title="ТК РФ, ст. 70",
        approx_tokens=40,
    )
    chunks = [
        Chunk(
            chunk_id="contract_a#п. 3.1",
            doc_id="contract_a",
            locator="п. 3.1",
            breadcrumb="Договор Т-01/2024 / раздел 3 / п. 3.1",
            text="3.1. Арендная плата составляет 100 000 рублей в месяц.",
            start=0,
            end=55,
        ),
        Chunk(
            chunk_id="law_tk_70#ст. 70",
            doc_id="law_tk_70",
            locator="ст. 70",
            breadcrumb="ТК РФ / ст. 70",
            text="Статья 70. Испытание при приеме на работу. Тестовый текст нормы.",
            start=0,
            end=64,
        ),
    ]
    return KnowledgeBase(
        kb_hash="fake-kb-hash",
        cards={c.doc_id: c for c in (contract, law)},
        chunks={c.chunk_id: c for c in chunks},
        article_inventory=["ТК РФ ст. 70"],
        our_company="ООО «Меридиан»",
    )


class FakeLLM:
    """Детерминированный LLM: JSON NO_DATA, нулевые эмбеддинги, порядок реранка как есть."""

    def __init__(self, reply: str = '{"verdict": "NO_DATA"}') -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
        self.calls += 1
        return self.reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        return list(range(min(top_n, len(docs))))


class FakeIngestor:
    def build(self, kb_zip_path: str, llm: object) -> KnowledgeBase:  # noqa: ARG002
        return fixture_kb()


class FakeAnalyzer:
    def analyze(self, req: ChatRequest, kb: KnowledgeBase) -> QueryAnalysis:  # noqa: ARG002
        q = req.last_user_text()
        return QueryAnalysis(standalone_question=q, kinds=["lookup"], search_queries=[q])


class FakeRetriever:
    """Пересечение слов вопроса с текстом чанка; не больше 3 штук."""

    def __init__(self, kb: Optional[KnowledgeBase] = None) -> None:
        self.kb = kb or fixture_kb()

    def retrieve(self, a: QueryAnalysis, req: ChatRequest) -> EvidencePack:  # noqa: ARG002
        qw: set[str] = set()
        for q in a.search_queries or [a.standalone_question]:
            qw |= _words(q)
        scored = []
        for ch in self.kb.chunks.values():
            overlap = len(qw & _words(ch.text + " " + ch.breadcrumb))
            if overlap:
                scored.append((overlap, ch))
        scored.sort(key=lambda t: (-t[0], t[1].chunk_id))
        items = [
            EvidenceItem(
                eid=ch.chunk_id,
                chunk=ch,
                card=self.kb.cards[ch.doc_id],
                applicability=Applicability.APPLICABLE,
                score=float(s),
            )
            for s, ch in scored[:3]
        ]
        return EvidencePack(items=items)


class FakeComposer:
    def compose(self, a: QueryAnalysis, ev: EvidencePack, req: ChatRequest) -> Draft:  # noqa: ARG002
        if not ev.items:
            return Draft(verdict=Verdict.NO_DATA, missing_info=["подтверждающих документов"])
        top = ev.items[0]
        return Draft(
            verdict=Verdict.ANSWER,
            claims=[Claim(text=f"{top.card.title} ({top.chunk.locator}): {top.chunk.text}", type="fact", evidence=[top.eid])],
        )


class FakeVerifier:
    """Минимальные G1 и G5: evidence существует в пакете; ANSWER без ссылок понижается."""

    def verify(self, d: Draft, ev: EvidencePack, a: QueryAnalysis) -> tuple[Draft, GuardReport]:  # noqa: ARG002
        known = {it.eid for it in ev.items}
        violations: list[str] = []
        kept: list[Claim] = []
        for claim in d.claims:
            bad = [e for e in claim.evidence if e not in known]
            if bad or (claim.type == "fact" and not claim.evidence):
                violations.append(f"G1: claim без валидного evidence: {claim.text[:40]}")
                continue
            kept.append(claim)
        draft = d.model_copy(update={"claims": kept})
        downgraded: Optional[Verdict] = None
        if draft.verdict == Verdict.ANSWER and not any(c.evidence for c in kept):
            downgraded = draft.verdict
            draft = draft.model_copy(update={"verdict": Verdict.NO_DATA})
            violations.append("G5: ANSWER без ссылок понижен")
        return draft, GuardReport(passed=not violations, violations=violations, downgraded_from=downgraded)


class FakeAssistant:
    """Фиксированный ответ — для UI и тестов solution.py."""

    def answer(self, req: ChatRequest) -> AssistantResponse:  # noqa: ARG002
        return AssistantResponse(
            text="В базе знаний нет данных для ответа. Отвечать по памяти не буду.",
            verdict=Verdict.NO_DATA,
            guard=GuardReport(passed=True),
        )


# Проверка соответствия протоколам (mypy): присваивания ниже не исполняются как логика.
_ingestor: Ingestor = FakeIngestor()
_analyzer: Analyzer = FakeAnalyzer()
_retriever: Retriever = FakeRetriever()
_composer: Composer = FakeComposer()
_verifier: Verifier = FakeVerifier()
