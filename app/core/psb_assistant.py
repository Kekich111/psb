"""Оркестратор: единая точка входа Assistant.answer для solution.py и веба.

Любой сбой любой стадии деградирует в безопасный ответ (I8): никогда не исключение
наружу и никогда не пустая строка. Бюджет: analyze + compose, дальше по условию
(второй hop ≤ 1, rescue при NO_DATA ≤ 1, критик внутри verifier).
"""

from __future__ import annotations

import time
from typing import Callable, Optional, TypeVar

from psb_config import Settings, load_settings, log
from psb_contracts import (
    Analyzer,
    AssistantResponse,
    ChatRequest,
    Composer,
    Draft,
    EvidencePack,
    GuardReport,
    KnowledgeBase,
    QueryAnalysis,
    Retriever,
    Verdict,
    Verifier,
)
from psb_render import NO_DATA_TEXT, render

T = TypeVar("T")


def safe_response(reason: str, trace: Optional[dict] = None) -> AssistantResponse:
    """Безопасный ответ-шаблон: используется при любом сбое."""
    return AssistantResponse(
        text=NO_DATA_TEXT,
        verdict=Verdict.NO_DATA,
        guard=GuardReport(passed=False, violations=[f"degraded: {reason}"]),
        trace=trace or {},
    )


def merge_packs(a: EvidencePack, b: EvidencePack) -> EvidencePack:
    seen = {it.eid for it in a.items}
    items = list(a.items) + [it for it in b.items if it.eid not in seen]
    gaps = list(a.gaps) + [g for g in b.gaps if g not in a.gaps]
    return EvidencePack(items=items, gaps=gaps, full_context_used=a.full_context_used or b.full_context_used)


class PipelineAssistant:
    def __init__(
        self,
        kb: KnowledgeBase,
        analyzer: Analyzer,
        retriever: Retriever,
        composer: Composer,
        verifier: Verifier,
        settings: Optional[Settings] = None,
    ) -> None:
        self.kb = kb
        self.analyzer = analyzer
        self.retriever = retriever
        self.composer = composer
        self.verifier = verifier
        self.settings = settings or load_settings()

    # -- публичный контракт --
    def answer(self, req: ChatRequest) -> AssistantResponse:
        trace: dict = {"stages": {}}
        try:
            return self._answer(req, trace)
        except Exception as exc:  # noqa: BLE001 — I8
            log(f"assistant: необработанный сбой {type(exc).__name__}")
            return safe_response(type(exc).__name__, trace)

    # -- внутреннее --
    def _stage(self, trace: dict, name: str, fn: Callable[[], T], fallback: Callable[[], T]) -> T:
        t0 = time.monotonic()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            log(f"stage {name}: {type(exc).__name__}")
            trace.setdefault("errors", []).append(f"{name}:{type(exc).__name__}")
            return fallback()
        finally:
            trace["stages"][name] = round(time.monotonic() - t0, 3)

    def _answer(self, req: ChatRequest, trace: dict) -> AssistantResponse:
        if req.as_of is None:
            req = req.model_copy(update={"as_of": self.settings.as_of})
        question = req.last_user_text()

        def fallback_analysis() -> QueryAnalysis:
            return QueryAnalysis(standalone_question=question, search_queries=[question])

        analysis = self._stage(trace, "analyze", lambda: self.analyzer.analyze(req, self.kb), fallback_analysis)
        ev = self._stage(trace, "retrieve", lambda: self.retriever.retrieve(analysis, req), EvidencePack)
        draft = self._stage(
            trace,
            "compose",
            lambda: self.composer.compose(analysis, ev, req),
            lambda: Draft(verdict=Verdict.NO_DATA),
        )

        # Второй hop — только по запросу reasoner'а, один раз.
        if draft.need_more_queries:
            more = analysis.model_copy(update={"search_queries": draft.need_more_queries})
            ev = merge_packs(ev, self._stage(trace, "retrieve2", lambda: self.retriever.retrieve(more, req), EvidencePack))
            draft = self._stage(trace, "compose2", lambda: self.composer.compose(analysis, ev, req), lambda: draft)

        # Rescue: перед отказом один повторный проход с переформулировкой.
        if draft.verdict == Verdict.NO_DATA and "off_topic" not in analysis.kinds:
            queries = list(dict.fromkeys([analysis.standalone_question, *analysis.search_queries]))
            rescue = analysis.model_copy(update={"search_queries": queries})
            ev_r = self._stage(trace, "rescue_retrieve", lambda: self.retriever.retrieve(rescue, req), EvidencePack)
            if ev_r.items and {it.eid for it in ev_r.items} - {it.eid for it in ev.items}:
                ev = merge_packs(ev, ev_r)
                d2 = self._stage(trace, "rescue_compose", lambda: self.composer.compose(analysis, ev, req), lambda: draft)
                if d2.verdict != Verdict.NO_DATA:
                    draft = d2
                    trace["rescued"] = True

        try:
            draft, guard = self.verifier.verify(draft, ev, analysis)
        except Exception as exc:  # noqa: BLE001
            # Проверить не смогли — безопаснее отказ, чем непроверенный ответ.
            log(f"stage verify: {type(exc).__name__}")
            trace.setdefault("errors", []).append(f"verify:{type(exc).__name__}")
            return safe_response("verify_failed", trace)

        trace["verdict"] = draft.verdict.value
        trace["retrieved"] = [it.eid for it in ev.items]
        return render(draft, ev, guard, trace)


def build_fake_assistant(settings: Optional[Settings] = None) -> PipelineAssistant:
    from psb_fakes import FakeAnalyzer, FakeComposer, FakeRetriever, FakeVerifier, fixture_kb

    kb = fixture_kb()
    return PipelineAssistant(kb, FakeAnalyzer(), FakeRetriever(kb), FakeComposer(), FakeVerifier(), settings)


def build_default_assistant(settings: Optional[Settings] = None, llm: object = None) -> PipelineAssistant:
    """Боевая сборка: разбор базы -> индекс -> LLM-анализ -> гибридный поиск -> LLM-ответ -> guard.

    Если базу знаний найти нельзя — безопасная сборка на фейках (ответы NO_DATA-подобные), ошибка в stderr.
    `llm` можно подставить в тестах (протокол LLM).
    """
    from psb_analyze import LLMAnalyzer
    from psb_compose import LLMComposer
    from psb_index import Index
    from psb_ingest import build_kb, find_kb_zip
    from psb_llm import LLMClient
    from psb_retrieve import HybridRetriever
    from psb_verify import RuleVerifier

    settings = settings or load_settings()
    zip_path = find_kb_zip(settings.data_dir)
    if zip_path is None:
        log("База знаний не найдена (knowledge-base.zip): сборка на фейках")
        return build_fake_assistant(settings)
    kb = build_kb(zip_path)
    client = llm if llm is not None else LLMClient(settings)
    index = Index(kb, client, settings)
    return PipelineAssistant(
        kb,
        LLMAnalyzer(client, settings),
        HybridRetriever(kb, index, client, settings),
        LLMComposer(client, settings),
        RuleVerifier(kb, settings, client),
        settings,
    )
