"""S7 Guard+Verify: детерминированные проверки G1–G6 (бесплатно, всегда).

G1 evidence существует в пакете; G2 числа/даты ⊆ evidence ∪ вопрос; G3 «№/ст./п.» существуют;
G4 названия в «ёлочках» есть в базе; G5 ANSWER без ссылок -> NO_DATA; G6 нет фрагментов системного промпта.
Нарушившее claim удаляется. LLM-критик (VERIFY_MODE=full) добавляется отдельным шагом (M3).
"""

from __future__ import annotations

import re
from typing import Any

from psb_config import Settings
from psb_contracts import Claim, Draft, EvidencePack, GuardReport, KnowledgeBase, QueryAnalysis, Verdict
from psb_prompts import PROMPT_FINGERPRINTS
from psb_text import iter_dates

_NO_EVIDENCE_VERDICTS = {Verdict.NO_DATA, Verdict.OFF_TOPIC, Verdict.REFUSE_ILLEGAL, Verdict.CLARIFY}
_ABSENCE_RE = re.compile(r"(?i)(в базе( знаний)? нет|не предусмотрен|не найден|отсутствует|отсутствуют|нет данных|не содержит)")
_REF_STRIP_RE = re.compile(
    r"(?i)(?:№\s*\S+|\bпп?\.\s*[\d.]+|\bст\.\s*[\d.]+|\bч\.\s*\d+|\bабз\.\s*\d+|"
    r"\b(?:пункт\w*|подпункт\w*|стать[яиеюй]\w*|част[ьиюе]\w*|раздел\w*)\s+[\d.,\s и-]*\d)"
)
_MULT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(тыс|млн|млрд)\w*", re.IGNORECASE)
_NUM_RE = re.compile(r"\d{1,3}(?:[\s ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_MULT = {"тыс": 1_000, "млн": 1_000_000, "млрд": 1_000_000_000}


def norm(s: str) -> str:
    return " ".join(s.casefold().replace("ё", "е").replace(" ", " ").split())


def _num(tok: str) -> str:
    t = re.sub(r"[\s ]", "", tok).replace(",", ".")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t.lstrip("0") or "0"


def extract_numbers(text: str, strip_refs: bool = False) -> tuple[set[str], set[str]]:
    """-> (числа, даты в ISO). strip_refs=True убирает ссылки «п. 3.1», «№ А-08/2024» (их проверяет G3)."""
    if strip_refs:
        text = _REF_STRIP_RE.sub(" ", text)
    dates = {d.isoformat() for _, d in iter_dates(text)}
    text = re.sub(r"\b\d{2}\.\d{2}\.\d{4}\b", " ", text)
    text = re.sub(r"\d{1,2}\s+(?:%s)\s+\d{4}" % "|".join(
        ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    ), " ", text)
    nums: set[str] = set()
    for m in _MULT_RE.finditer(text):
        base = float(m.group(1).replace(",", "."))
        nums.add(_num(str(int(base * _MULT[m.group(2).lower()]))))
        nums.add(_num(m.group(1)))
    text = _MULT_RE.sub(" ", text)
    for m in _NUM_RE.finditer(text):
        nums.add(_num(m.group(0)))
    return nums, dates


class RuleVerifier:
    def __init__(self, kb: KnowledgeBase, settings: Settings | None = None, llm: Any = None) -> None:
        self.kb = kb
        self.s = settings
        self.llm = llm
        self.corpus = norm(" ".join([c.text for c in kb.chunks.values() if not c.is_editorial] + [c.title for c in kb.cards.values()]))
        self.articles = {c.law.article for c in kb.cards.values() if c.law}

    def _allowed(self, ev: EvidencePack, a: QueryAnalysis) -> tuple[str, set[str], set[str]]:
        parts = [it.chunk.text + " " + it.chunk.breadcrumb + " " + it.card.title for it in ev.items]
        parts += [a.standalone_question, *a.premises, *[v for vs in a.entities.values() for v in vs], *a.search_queries[:1]]
        parts += [g.description for g in ev.gaps]
        text = " ".join(parts)
        nums, dates = extract_numbers(text)
        return norm(text), nums, dates

    def _check_claim(self, c: Claim, allowed_norm: str, nums: set[str], dates: set[str], q_norm: str, known: set[str],
                     verdict: Verdict) -> str | None:
        text = c.text
        low = norm(text)
        if any(fp in low for fp in PROMPT_FINGERPRINTS):
            return "G6: фрагмент системного промпта"
        needs_evidence = verdict not in _NO_EVIDENCE_VERDICTS and not _ABSENCE_RE.search(text)
        if needs_evidence and not [e for e in c.evidence if e in known]:
            return "G1: нет валидного evidence"
        c_nums, c_dates = extract_numbers(text, strip_refs=True)
        bad_n = sorted(n for n in c_nums if n not in nums)
        if bad_n:
            return f"G2: числа не из источников: {', '.join(bad_n)}"
        bad_d = sorted(d for d in c_dates if d not in dates)
        if bad_d:
            return f"G2: даты не из источников: {', '.join(bad_d)}"
        for m in re.finditer(r"№\s*([^\s,;)»]+)", text):
            tok = norm(m.group(1).rstrip(".,;:"))
            if tok and tok not in allowed_norm and tok not in self.corpus:
                return f"G3: № {m.group(1)} не найден в базе"
        for m in re.finditer(r"(?i)\bст(?:\.|атья|атьи|атье|атью|атьёй|атьей)\s*(\d+(?:\.\d+)?)", text):
            n = m.group(1)
            if n in self.articles or re.search(rf"(?:стать\w*|ст\.)\s*{re.escape(n)}\b", allowed_norm):
                continue
            return f"G3: ст. {n} нет в базе"
        for m in re.finditer(r"(?i)\b(?:пп?\.|пункт\w*)\s*(\d+(?:\.\d+)*)", text):
            n = m.group(1)
            if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", allowed_norm) and not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", q_norm):
                return f"G3: п. {n} нет в источниках"
        for m in re.finditer(r"«([^»]{2,80})»", text):
            name = norm(m.group(1))
            if name not in self.corpus and name not in allowed_norm:
                return f"G4: «{m.group(1)}» нет в базе"
        return None

    def verify(self, d: Draft, ev: EvidencePack, a: QueryAnalysis) -> tuple[Draft, GuardReport]:
        known = {it.eid for it in ev.items}
        allowed_norm, nums, dates = self._allowed(ev, a)
        q_norm = norm(a.standalone_question)
        violations: list[str] = []
        kept: list[Claim] = []
        for c in d.claims:
            c = c.model_copy(update={"evidence": [e for e in c.evidence if e in known]})
            problem = self._check_claim(c, allowed_norm, nums, dates, q_norm, known, d.verdict)
            if problem:
                violations.append(f"{problem} | «{c.text[:60]}»")
            else:
                kept.append(c)
        draft = d.model_copy(update={"claims": kept})
        downgraded: Verdict | None = None
        grounded = any(c.evidence for c in kept)
        if draft.verdict not in _NO_EVIDENCE_VERDICTS and not grounded:  # G5
            downgraded = draft.verdict
            draft = draft.model_copy(update={"verdict": Verdict.NO_DATA})
            violations.append(f"G5: {downgraded.value} без подтверждённых ссылок понижен до NO_DATA")
            if not kept:
                draft = draft.model_copy(update={"missing_info": draft.missing_info or []})
        return draft, GuardReport(passed=not violations, violations=violations, downgraded_from=downgraded)
