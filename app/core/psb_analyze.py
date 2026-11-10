"""S4 Analyze (LLM #1): диалог -> QueryAnalysis. Любой сбой деградирует в минимальный анализ без LLM."""

from __future__ import annotations

from datetime import date
from typing import Any

from psb_config import Settings
from psb_contracts import ChatRequest, KnowledgeBase, QueryAnalysis
from psb_llm import chat_json
from psb_prompts import build_analyze_messages
from psb_text import INJECTION_RE, iter_dates

_KINDS = {"lookup", "multi_doc", "date", "off_topic", "illegal", "adversarial"}
# Дополнительный детерминированный pre-guard к LLM-классификации.
_PRESSURE_RE = INJECTION_RE.pattern[:-1] + r"|исправь\s+(свой\s+)?ответ|ты\s+теперь|новые\s+инструкции|режим\s+разработчика)"


def _strs(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()][:limit]


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


class LLMAnalyzer:
    def __init__(self, llm: Any, settings: Settings) -> None:
        self.llm = llm
        self.s = settings

    def analyze(self, req: ChatRequest, kb: KnowledgeBase) -> QueryAnalysis:
        import re

        question = req.last_user_text().strip()
        user_text = " ".join(t.content for t in req.history if t.role == "user")
        as_of = req.as_of or self.s.as_of
        raw = chat_json(self.llm, build_analyze_messages(req, kb, as_of), max_tokens=700)

        adversarial = bool(re.search(_PRESSURE_RE, question, re.IGNORECASE))
        if raw is None:  # деградация: вопрос как есть, дата только если явно написана
            dates = iter_dates(question)
            return QueryAnalysis(
                standalone_question=question,
                kinds=["lookup"] + (["adversarial"] if adversarial else []),
                event_date=dates[0][1] if dates else None,
                search_queries=[question],
            )

        standalone = str(raw.get("standalone_question") or question).strip() or question
        kinds = [k for k in _strs(raw.get("kinds")) if k in _KINDS]
        if adversarial and "adversarial" not in kinds:
            kinds.append("adversarial")
        entities: dict[str, list[str]] = {}
        if isinstance(raw.get("entities"), dict):
            entities = {str(k): _strs(v) for k, v in raw["entities"].items()}
        # Дата события должна быть подтверждена текстом пользователя: год обязан встречаться в его репликах.
        event = _parse_date(raw.get("event_date"))
        if event and str(event.year) not in user_text:
            event = None
        queries = _strs(raw.get("search_queries"), 3) or [standalone]
        return QueryAnalysis(
            standalone_question=standalone,
            kinds=kinds or ["lookup"],
            entities=entities,
            event_date=event,
            premises=_strs(raw.get("premises")),
            ambiguities=_strs(raw.get("ambiguities")),
            search_queries=queries,
        )
