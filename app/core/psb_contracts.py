"""Контракты между блоками (источник истины). CONTRACT_VERSION = 1.1.0.

Менять только отдельным PR: bump версии, запись в contracts/CHANGELOG.md,
обновление psb_fakes.py и contracts/examples/.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional, Protocol

from pydantic import BaseModel, Field

CONTRACT_VERSION = "1.1.0"


# ---------- домен ----------
class DocClass(str, Enum):
    LAW = "law"
    COMPANY = "company"


class DocKind(str, Enum):
    LAW = "law"
    CONTRACT = "contract"
    AMENDMENT = "amendment"
    REGULATION = "regulation"
    LETTER = "letter"
    MEMO = "memo"
    OTHER = "other"


class Party(BaseModel):
    name: str
    inn: Optional[str] = None
    role: Optional[str] = None


class LawAmendmentNote(BaseModel):
    effective_from: date
    act: str  # «ФЗ от 26.07.2026 № 246-ФЗ»
    scope: Optional[str] = None  # «ч. 1», «п. 3», «пп. 2»; None = вся статья
    repealed: bool = False  # «абзац утратил силу с ...»


class LawMeta(BaseModel):
    act_name: str
    article: str
    article_title: Optional[str] = None  # «Статья 59. Срочный трудовой договор»
    alias: Optional[str] = None  # короткое имя из файла: «НК», «ТК», «МРОТ»
    exported_at: Optional[date] = None
    amendments: list[LawAmendmentNote] = Field(default_factory=list)


class AmendTarget(BaseModel):
    target_doc_id: Optional[str] = None
    clauses: list[str] = Field(default_factory=list)
    effective_from: Optional[date] = None


class DocCard(BaseModel):
    doc_id: str
    file: str
    doc_class: DocClass
    kind: DocKind
    title: str
    number: Optional[str] = None
    doc_date: Optional[date] = None
    family_id: Optional[str] = None  # редакции одного документа
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None  # None = действует
    parties: list[Party] = Field(default_factory=list)
    amends: list[AmendTarget] = Field(default_factory=list)
    law: Optional[LawMeta] = None
    approx_tokens: int = 0


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    locator: str  # «п. 3.1» | «ч. 1» | «ст. 59»
    breadcrumb: str
    text: str
    start: int  # offsets в исходном файле
    end: int
    is_editorial: bool = False  # пометки не из нормы и подозрительные «инструкции» — не цитировать


class KnowledgeBase(BaseModel):
    kb_hash: str
    cards: dict[str, DocCard]
    chunks: dict[str, Chunk]
    article_inventory: list[str] = Field(default_factory=list)  # какие статьи/акты есть в базе
    our_company: Optional[str] = None


# ---------- запрос ----------
class RunMode(str, Enum):
    BATCH = "batch"  # CSV-платформа: нельзя переспрашивать
    CHAT = "chat"  # веб: можно CLARIFY


class Turn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    history: list[Turn]  # последняя реплика user — последняя
    mode: RunMode = RunMode.BATCH
    as_of: Optional[date] = None  # «сегодня» для расчётов

    def last_user_text(self) -> str:
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.content
        return ""


class QueryAnalysis(BaseModel):
    standalone_question: str
    kinds: list[str] = Field(default_factory=list)  # lookup|multi_doc|date|off_topic|illegal|adversarial
    entities: dict[str, list[str]] = Field(default_factory=dict)
    event_date: Optional[date] = None
    premises: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)  # 1–3 переформулировки


# ---------- evidence ----------
class Applicability(str, Enum):
    APPLICABLE = "applicable"
    SUPERSEDED = "superseded"
    NOT_YET = "not_yet_in_force"
    UNKNOWN = "unknown"


class EvidenceItem(BaseModel):
    eid: str  # = chunk_id
    chunk: Chunk
    card: DocCard
    applicability: Applicability = Applicability.UNKNOWN
    note: Optional[str] = None  # «ред. от 10.01.2023», «изменён ДС №1 с …»
    score: float = 0.0


class KnowledgeGap(BaseModel):
    description: str  # «в базе нет ст. 6, 10, 11 152-ФЗ, на них ссылается ст. 9»


class EvidencePack(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    full_context_used: bool = False


# ---------- ответ ----------
class Verdict(str, Enum):
    ANSWER = "ANSWER"
    PARTIAL = "PARTIAL"
    NO_DATA = "NO_DATA"
    FALSE_PREMISE = "FALSE_PREMISE"
    CLARIFY = "CLARIFY"
    REFUSE_ILLEGAL = "REFUSE_ILLEGAL"
    OFF_TOPIC = "OFF_TOPIC"


class Claim(BaseModel):
    text: str
    type: str  # fact | inference
    evidence: list[str] = Field(default_factory=list)  # eids; пусто нельзя для fact


class PremiseCheck(BaseModel):
    premise: str
    status: str  # confirmed | contradicted | unverifiable
    evidence: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    verdict: Verdict
    claims: list[Claim] = Field(default_factory=list)
    premise_checks: list[PremiseCheck] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    need_more_queries: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    eid: str
    doc_id: str
    doc_title: str
    locator: str
    start: int
    end: int
    quote: str  # для подсветки в UI


class GuardReport(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    downgraded_from: Optional[Verdict] = None


class AssistantResponse(BaseModel):
    text: str  # финальный текст (в CSV идёт он)
    verdict: Verdict
    citations: list[Citation] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    guard: GuardReport
    trace: dict = Field(default_factory=dict)  # тайминги, llm_calls, retrieved ids
    contract_version: str = CONTRACT_VERSION


# ---------- интерфейсы блоков ----------
class LLM(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]: ...


class Ingestor(Protocol):
    def build(self, kb_zip_path: str, llm: LLM) -> KnowledgeBase: ...


class Retriever(Protocol):
    def retrieve(self, a: QueryAnalysis, req: ChatRequest) -> EvidencePack: ...


class Analyzer(Protocol):
    def analyze(self, req: ChatRequest, kb: KnowledgeBase) -> QueryAnalysis: ...


class Composer(Protocol):
    def compose(self, a: QueryAnalysis, ev: EvidencePack, req: ChatRequest) -> Draft: ...


class Verifier(Protocol):
    def verify(
        self, d: Draft, ev: EvidencePack, a: QueryAnalysis
    ) -> tuple[Draft, GuardReport]: ...


class Assistant(Protocol):  # ЕДИНАЯ точка входа для solution.py и веба
    def answer(self, req: ChatRequest) -> AssistantResponse: ...
