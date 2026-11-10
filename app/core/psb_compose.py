"""S6 Reason+Compose (LLM #2): EvidencePack -> Draft (claims с evidence ids). Текст собирает psb_render."""

from __future__ import annotations

from typing import Any

from psb_config import Settings
from psb_contracts import ChatRequest, Claim, Draft, EvidencePack, PremiseCheck, QueryAnalysis, Verdict
from psb_llm import chat_json
from psb_prompts import build_compose_messages
from psb_retrieve import doc_note


def _strs(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()][:limit]


class LLMComposer:
    def __init__(self, llm: Any, settings: Settings) -> None:
        self.llm = llm
        self.s = settings

    def compose(self, a: QueryAnalysis, ev: EvidencePack, req: ChatRequest) -> Draft:
        ref = a.event_date or req.as_of or self.s.as_of
        aliases = {it.eid: f"E{i}" for i, it in enumerate(ev.items, 1)}
        back = {v: k for k, v in aliases.items()}
        notes: dict[str, str] = {}
        for it in ev.items:
            notes.setdefault(it.chunk.doc_id, doc_note(it.card, it.applicability, ref))
        messages = build_compose_messages(a, ev, req, req.as_of or self.s.as_of, aliases, notes)
        raw = chat_json(self.llm, messages, max_tokens=max(self.s.max_tokens, 800))
        if raw is None:
            return Draft(verdict=Verdict.NO_DATA, missing_info=["ответ модели не удалось разобрать"])
        return self._parse(raw, back, req)

    @staticmethod
    def _parse(raw: dict, back: dict[str, str], req: ChatRequest) -> Draft:
        try:
            verdict = Verdict(str(raw.get("verdict", "")).strip().upper())
        except ValueError:
            verdict = Verdict.NO_DATA

        def eids(value: Any) -> list[str]:
            return [back.get(x, x) for x in _strs(value, 12)]

        claims: list[Claim] = []
        for c in raw.get("claims") or []:
            if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                kind = "inference" if str(c.get("type")) == "inference" else "fact"
                claims.append(Claim(text=c["text"].strip(), type=kind, evidence=eids(c.get("evidence"))))
        checks = [
            PremiseCheck(premise=str(p.get("premise", "")), status=str(p.get("status", "unverifiable")), evidence=eids(p.get("evidence")))
            for p in (raw.get("premise_checks") or [])
            if isinstance(p, dict) and p.get("premise")
        ]
        question = raw.get("clarifying_question")
        question = question.strip() if isinstance(question, str) and question.strip() else None
        if verdict == Verdict.CLARIFY and req.mode.value == "batch":
            verdict = Verdict.PARTIAL if claims else Verdict.NO_DATA  # в BATCH переспрашивать нельзя
            question = None
        return Draft(
            verdict=verdict,
            claims=claims,
            premise_checks=checks,
            missing_info=_strs(raw.get("missing_info")),
            clarifying_question=question,
            need_more_queries=_strs(raw.get("need_more_queries"), 2),
        )
