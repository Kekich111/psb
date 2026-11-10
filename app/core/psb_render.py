"""S8 Render: детерминированная сборка текста из claims и структурированного ответа для UI.

M0-версия: минимальная, но всегда возвращает непустой текст (инвариант I8).
Стиль (раздел 8 ARCHITECTURE): вердикт первым, документ + пункт, один абзац.
"""

from __future__ import annotations

from typing import Optional

from psb_contracts import (
    AssistantResponse,
    Citation,
    Draft,
    EvidencePack,
    GuardReport,
    Verdict,
)

MAX_ANSWER_CHARS = 10_000

NO_DATA_TEXT = (
    "В базе знаний нет данных, достаточных для надёжного ответа на этот вопрос. "
    "Отвечать по памяти не буду."
)
OFF_TOPIC_TEXT = (
    "Этот вопрос выходит за рамки моей компетенции: я отвечаю на юридические вопросы "
    "по документам предоставленной базы знаний."
)
REFUSE_TEXT = "Помогать с этим я не могу, так как это противоречит требованиям закона."


def _collapse(text: str) -> str:
    return " ".join(text.split())


def build_citations(draft: Draft, ev: EvidencePack) -> list[Citation]:
    by_id = {it.eid: it for it in ev.items}
    seen: list[str] = []
    for claim in draft.claims:
        for eid in claim.evidence:
            if eid in by_id and eid not in seen:
                seen.append(eid)
    out: list[Citation] = []
    for eid in seen:
        it = by_id[eid]
        out.append(
            Citation(
                eid=eid,
                doc_id=it.chunk.doc_id,
                doc_title=it.card.title,
                locator=it.chunk.locator,
                start=it.chunk.start,
                end=it.chunk.end,
                quote=it.chunk.text[:300],
            )
        )
    return out


def render_text(draft: Draft) -> str:
    claims = " ".join(c.text.strip() for c in draft.claims if c.text.strip())
    v = draft.verdict
    if v == Verdict.CLARIFY and draft.clarifying_question:
        return draft.clarifying_question.strip()
    if v == Verdict.OFF_TOPIC:
        return claims or OFF_TOPIC_TEXT
    if v == Verdict.REFUSE_ILLEGAL:
        return claims or REFUSE_TEXT
    if v == Verdict.NO_DATA or not claims:
        missing = "; ".join(m.strip() for m in draft.missing_info if m.strip())
        base = f"В базе знаний нет: {missing}. Отвечать по памяти не буду." if missing else NO_DATA_TEXT
        return f"{claims} {base}".strip() if claims else base
    if v == Verdict.PARTIAL and draft.missing_info:
        missing = "; ".join(m.strip() for m in draft.missing_info if m.strip())
        if missing:
            return f"{claims} В базе нет: {missing}."
    return claims


def render(
    draft: Draft,
    ev: EvidencePack,
    guard: GuardReport,
    trace: Optional[dict] = None,
) -> AssistantResponse:
    text = _collapse(render_text(draft)) or NO_DATA_TEXT
    if len(text) > MAX_ANSWER_CHARS:
        text = text[: MAX_ANSWER_CHARS - 1].rstrip() + "…"
    return AssistantResponse(
        text=text,
        verdict=draft.verdict,
        citations=build_citations(draft, ev),
        clarifying_question=draft.clarifying_question,
        guard=guard,
        trace=trace or {},
    )
