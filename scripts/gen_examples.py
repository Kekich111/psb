"""Генерирует contracts/examples/*.json из фейков. Запуск: make examples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "core"))

from psb_assistant import build_fake_assistant  # noqa: E402
from psb_contracts import ChatRequest, Turn  # noqa: E402
from psb_fakes import FakeAnalyzer, FakeComposer, FakeRetriever, fixture_kb  # noqa: E402


def main() -> None:
    out = ROOT / "contracts" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    kb = fixture_kb()
    req = ChatRequest(history=[Turn(role="user", content="Какая арендная плата?")])
    analysis = FakeAnalyzer().analyze(req, kb)
    pack = FakeRetriever(kb).retrieve(analysis, req)
    draft = FakeComposer().compose(analysis, pack, req)
    resp = build_fake_assistant().answer(req)
    for name, model in {
        "chat_request": req,
        "knowledge_base": kb,
        "query_analysis": analysis,
        "evidence_pack": pack,
        "draft": draft,
        "assistant_response": resp,
    }.items():
        (out / f"{name}.json").write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print("written", name)


if __name__ == "__main__":
    main()
