from __future__ import annotations

from pathlib import Path

import pytest
from psb_contracts import (
    AssistantResponse,
    ChatRequest,
    Draft,
    EvidencePack,
    KnowledgeBase,
    QueryAnalysis,
    Turn,
)
from psb_fakes import FakeAnalyzer, FakeComposer, FakeIngestor, FakeRetriever, FakeVerifier, fixture_kb

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"


def _req(text: str = "Какая арендная плата?") -> ChatRequest:
    return ChatRequest(history=[Turn(role="user", content=text)])


def test_fake_blocks_return_contract_types() -> None:
    kb = FakeIngestor().build("ignored.zip", llm=None)  # type: ignore[arg-type]
    assert isinstance(kb, KnowledgeBase)
    req = _req()
    analysis = FakeAnalyzer().analyze(req, kb)
    assert isinstance(analysis, QueryAnalysis)
    pack = FakeRetriever(kb).retrieve(analysis, req)
    assert isinstance(pack, EvidencePack)
    draft = FakeComposer().compose(analysis, pack, req)
    assert isinstance(draft, Draft)
    draft2, guard = FakeVerifier().verify(draft, pack, analysis)
    assert isinstance(draft2, Draft) and guard.passed


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("chat_request", ChatRequest),
        ("knowledge_base", KnowledgeBase),
        ("query_analysis", QueryAnalysis),
        ("evidence_pack", EvidencePack),
        ("draft", Draft),
        ("assistant_response", AssistantResponse),
    ],
)
def test_examples_validate(name: str, model: type) -> None:
    raw = (EXAMPLES / f"{name}.json").read_text(encoding="utf-8")
    assert model.model_validate_json(raw) is not None  # type: ignore[attr-defined]


def test_fixture_kb_is_consistent() -> None:
    kb = fixture_kb()
    for chunk in kb.chunks.values():
        assert chunk.doc_id in kb.cards
