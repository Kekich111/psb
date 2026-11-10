"""Инварианты I1, I6, I8 и правило «только плоские psb_* модули» (CLAUDE.md §2.3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from psb_assistant import PipelineAssistant, build_fake_assistant
from psb_contracts import ChatRequest, Draft, EvidencePack, Turn, Verdict
from psb_fakes import FakeAnalyzer, FakeComposer, FakeRetriever, FakeVerifier, fixture_kb

CORE = Path(__file__).resolve().parents[2] / "app" / "core"


def _req(text: str) -> ChatRequest:
    return ChatRequest(history=[Turn(role="user", content=text)])


def test_i1_i6_answer_has_citations_and_known_evidence() -> None:
    resp = build_fake_assistant().answer(_req("Какая арендная плата по договору аренды?"))
    assert resp.verdict == Verdict.ANSWER
    assert resp.citations, "I6: ANSWER требует >=1 цитаты"
    kb = fixture_kb()
    assert all(c.doc_id in kb.cards for c in resp.citations), "I3: документ должен быть в KB"


def test_no_data_when_nothing_found() -> None:
    resp = build_fake_assistant().answer(_req("Какой размер пособия по беременности?"))
    assert resp.verdict == Verdict.NO_DATA
    assert resp.text.strip()


class _Boom:
    def analyze(self, *a: object, **k: object) -> object:
        raise RuntimeError("boom")

    def retrieve(self, *a: object, **k: object) -> object:
        raise RuntimeError("boom")

    def compose(self, *a: object, **k: object) -> object:
        raise RuntimeError("boom")

    def verify(self, *a: object, **k: object) -> object:
        raise RuntimeError("boom")


@pytest.mark.parametrize("broken", ["analyzer", "retriever", "composer", "verifier", "all"])
def test_i8_any_block_failure_degrades_safely(broken: str) -> None:
    kb = fixture_kb()
    blocks = {
        "analyzer": FakeAnalyzer(),
        "retriever": FakeRetriever(kb),
        "composer": FakeComposer(),
        "verifier": FakeVerifier(),
    }
    for name in blocks if broken == "all" else [broken]:
        blocks[name] = _Boom()  # type: ignore[assignment]
    assistant = PipelineAssistant(kb, **blocks)  # type: ignore[arg-type]
    resp = assistant.answer(_req("Какая арендная плата?"))
    assert resp.text.strip(), "I8: ответ не пустой"
    assert resp.verdict in set(Verdict)


def test_g1_unknown_evidence_is_dropped() -> None:
    from psb_contracts import Claim, QueryAnalysis

    draft = Draft(verdict=Verdict.ANSWER, claims=[Claim(text="Выдумка", type="fact", evidence=["нет_такого"])])
    fixed, guard = FakeVerifier().verify(draft, EvidencePack(), QueryAnalysis(standalone_question="q"))
    assert not guard.passed
    assert fixed.verdict == Verdict.NO_DATA


def test_core_modules_are_flat_and_use_flat_imports() -> None:
    """Платформа принимает только корневые .py: никаких пакетных и относительных импортов."""
    bad = re.compile(r"^\s*(from\s+(app|\.)|import\s+app\b)", re.MULTILINE)
    for path in CORE.glob("*.py"):
        assert path.name.startswith("psb_"), f"{path.name}: модуль ядра должен начинаться с psb_"
        assert not bad.search(path.read_text(encoding="utf-8")), f"{path.name}: пакетный импорт запрещён"
    assert not [p for p in CORE.iterdir() if p.is_dir() and p.name != "__pycache__"], "вложенные пакеты запрещены"
