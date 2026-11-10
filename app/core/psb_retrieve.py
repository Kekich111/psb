"""S5 Retrieve: все документы компании (если влезают) с метками редакций + гибридный поиск по законам.

Порядок: entity routing -> full-context tier (компания) -> resolve редакций по дате события ->
BM25 ∪ dense -> RRF -> (опц.) реранкер -> прямые ссылки «ст. N» -> KnowledgeGap по кросс-ссылкам.
Реранкер и эмбеддинги опциональны: сбой = работаем на том, что есть.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from psb_config import Settings, log
from psb_contracts import (
    Applicability,
    ChatRequest,
    Chunk,
    DocCard,
    DocClass,
    EvidenceItem,
    EvidencePack,
    KnowledgeBase,
    KnowledgeGap,
    QueryAnalysis,
)
from psb_index import Index, rrf

LAW_TOP = 8
LAW_CHAR_BUDGET = 14_000
_ARTICLE_RE = re.compile(r"(?:\bст\.?|\bстать[яиеюй]\w*)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_XREF_RE = re.compile(
    r"стать[ьяиеюей]{1,2}\s+(\d+(?:\.\d+)?(?:\s*(?:,|и)\s*\d+(?:\.\d+)?)*)\s+настоящ\w+\s+(?:Федеральн\w+\s+закон\w*|Кодекс\w*)"
)


def fmt(d: date | None) -> str:
    return d.strftime("%d.%m.%Y") if d else "?"


# ---------- статусы редакций ----------
def company_status(card: DocCard, ref: date) -> tuple[Applicability, str]:
    if card.effective_from and ref < card.effective_from:
        return Applicability.NOT_YET, f"вступает в силу {fmt(card.effective_from)}"
    if card.effective_to and ref >= card.effective_to:
        return Applicability.SUPERSEDED, f"утратила силу с {fmt(card.effective_to)}"
    if card.effective_from:
        return Applicability.APPLICABLE, f"действует с {fmt(card.effective_from)}"
    return Applicability.APPLICABLE, ""


def doc_note(card: DocCard, appl: Applicability, ref: date) -> str:
    """Подпись документа для промпта: редакция, период, статус на дату `ref`."""
    if card.doc_class == DocClass.LAW and card.law:
        exp = f", текст выгружен {fmt(card.law.exported_at)}" if card.law.exported_at else ""
        return f"закон в действующей редакции{exp}"
    return company_status(card, ref)[1]


def _scope_matches(scope: str | None, locator: str) -> bool | None:
    """True/False для п.-уровня, None — не умеем сопоставить (ч., пп., абзац)."""
    if scope is None:
        return True
    m = re.match(r"п\. ([\d.]+)$", scope)
    if not m:
        return None
    return re.search(rf"(?:^|, )п\. {re.escape(m.group(1))}(?:,|$| \()", locator) is not None


def law_status(card: DocCard, chunk: Chunk, event_date: date | None, ref: date) -> tuple[Applicability, str | None]:
    meta = card.law
    if meta is None:
        return Applicability.APPLICABLE, None
    appl = Applicability.APPLICABLE
    notes: list[str] = []
    for am in meta.amendments:
        hit = _scope_matches(am.scope, chunk.locator)
        if not hit:
            continue
        future = bool(meta.exported_at and am.effective_from > meta.exported_at)
        if future:
            if ref >= am.effective_from:
                notes.append(f"изменение с {fmt(am.effective_from)} вступило в силу после выгрузки; его текста в базе нет")
            else:
                notes.append(f"с {fmt(am.effective_from)} вносятся изменения; содержание будущей редакции в базе неизвестно")
        elif event_date and event_date < am.effective_from:
            appl = Applicability.UNKNOWN
            notes.append(
                f"текст в базе — редакция после изменения с {fmt(am.effective_from)}; редакции на дату события {fmt(event_date)} в базе нет"
            )
    return appl, ("; ".join(dict.fromkeys(notes)) or None)


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, index: Index, llm: Any, settings: Settings) -> None:
        self.kb = kb
        self.index = index
        self.llm = llm
        self.s = settings
        self.company_tokens = sum(c.approx_tokens for c in kb.cards.values() if c.doc_class == DocClass.COMPANY)
        self.fullctx = self.company_tokens <= settings.company_fullctx_tokens
        self.use_rerank = bool(llm is not None and settings.api_base and settings.rerank_key and settings.reranker_model)
        self.law_idx = index.indices_where(lambda c: kb.cards[c.doc_id].doc_class == DocClass.LAW)
        self.company_idx = index.indices_where(lambda c: kb.cards[c.doc_id].doc_class == DocClass.COMPANY)
        self._overlays = self._build_overlays()
        self._inventory = set(kb.article_inventory)

    # ---------- компания ----------
    def _build_overlays(self) -> list[tuple[str, str, DocCard]]:
        out = []
        for card in self.kb.cards.values():
            for am in card.amends:
                if am.target_doc_id:
                    for cl in am.clauses:
                        out.append((am.target_doc_id, cl, card))
        return out

    def _overlay_note(self, chunk: Chunk, ref: date) -> str | None:
        m = re.match(r"п\. ([\d.]+)", chunk.locator)
        if not m:
            return None
        notes = []
        for target, clause, src in self._overlays:
            if target == chunk.doc_id and (m.group(1) == clause or m.group(1).startswith(clause + ".")):
                eff = src.effective_from
                tail = "" if not eff or ref >= eff else " — на дату события ещё не применяется"
                notes.append(f"затронут документом «{src.title}» (с {fmt(eff)}){tail}")
        return "; ".join(notes) or None

    def _mentioned_docs(self, a: QueryAnalysis) -> list[str]:
        hay = " ".join([a.standalone_question, *a.search_queries, *[v for vs in a.entities.values() for v in vs]]).casefold()
        scored = []
        for card in self.kb.cards.values():
            if card.doc_class != DocClass.COMPANY:
                continue
            hit = 0
            if card.number and card.number.casefold() in hay:
                hit += 2
            hit += sum(1 for p in card.parties if p.name.casefold().split("«")[-1].strip("» ") in hay)
            if hit:
                scored.append((-hit, card.doc_id))
        return [d for _, d in sorted(scored)]

    def _company_items(self, a: QueryAnalysis, ref: date, event_date: date | None) -> list[EvidenceItem]:
        forced = self._mentioned_docs(a)
        if self.fullctx:
            chosen = [c for i, c in enumerate(self.index.chunks) if i in self.company_idx]
        else:
            top = self.index.bm25_top(" ".join([a.standalone_question, *a.search_queries]), self.company_idx, 12)
            chosen = [self.index.chunks[i] for i in top]
            for did in forced:
                chosen += [c for c in self.index.chunks if c.doc_id == did and c not in chosen]
        order = {d: i for i, d in enumerate(forced)}
        chosen.sort(key=lambda c: (order.get(c.doc_id, len(order)), c.doc_id, c.start))
        items = []
        for ch in chosen:
            card = self.kb.cards[ch.doc_id]
            appl, _ = company_status(card, ref)
            items.append(
                EvidenceItem(eid=ch.chunk_id, chunk=ch, card=card, applicability=appl, note=self._overlay_note(ch, ref))
            )
        return items

    # ---------- законы ----------
    def _direct_article_chunks(self, text: str) -> list[int]:
        out: list[int] = []
        low = text.casefold()
        for num in dict.fromkeys(_ARTICLE_RE.findall(text)):
            cards = [c for c in self.kb.cards.values() if c.law and c.law.article == num]
            hinted = [
                c
                for c in cards
                if c.law
                and (c.law.act_name.casefold().split()[0] in low or (c.law.alias and c.law.alias.casefold() in low))
            ]
            for card in hinted or cards:
                idxs = self.index.indices_where(lambda c, d=card.doc_id: c.doc_id == d)
                top = self.index.bm25_top(text, idxs, 4) or sorted(idxs)[:3]
                out += [i for i in top if i not in out]
        return out

    def _law_candidates(self, a: QueryAnalysis, req: ChatRequest) -> list[int]:
        queries = list(dict.fromkeys(q.strip() for q in [a.standalone_question, *a.search_queries] if q.strip()))[:4]
        rankings = [self.index.bm25_top(q, self.law_idx, 30) for q in queries]
        for qv in self.index.embed_queries(queries):
            rankings.append(self.index.dense_top(qv, self.law_idx, 30))
        fused = rrf([r for r in rankings if r])[:24]
        direct = self._direct_article_chunks(" ".join([req.last_user_text(), a.standalone_question]))
        return direct + [i for i in fused if i not in direct]

    def _rerank(self, question: str, cands: list[int]) -> list[int]:
        pool = cands[:20]
        if not self.use_rerank or len(pool) <= LAW_TOP:
            return cands
        try:
            docs = [f"{self.index.chunks[i].breadcrumb}\n{self.index.chunks[i].text}"[:1500] for i in pool]
            order = self.llm.rerank(question, docs, LAW_TOP)
            picked = [pool[j] for j in order if 0 <= j < len(pool)]
            return picked + [i for i in cands if i not in picked]
        except Exception as exc:  # noqa: BLE001 — реранкер опционален
            log(f"rerank: {type(exc).__name__}")
            return cands

    def _gaps(self, idxs: list[int]) -> list[KnowledgeGap]:
        missing: dict[str, list[str]] = {}
        for i in idxs[:5]:
            ch = self.index.chunks[i]
            card = self.kb.cards[ch.doc_id]
            if not card.law:
                continue
            for m in _XREF_RE.finditer(ch.text):
                for num in re.findall(r"\d+(?:\.\d+)?", m.group(1)):
                    if num != card.law.article and f"{card.law.act_name} ст. {num}" not in self._inventory:
                        bucket = missing.setdefault(card.law.act_name, [])
                        if num not in bucket:
                            bucket.append(num)
        gaps = []
        for act, nums in missing.items():
            nums = sorted(nums, key=lambda x: float(x))[:6]
            gaps.append(
                KnowledgeGap(description=f"в базе нет ст. {', '.join(nums)} ({act}), на них ссылаются найденные нормы")
            )
        return gaps

    # ---------- публичный контракт ----------
    def retrieve(self, a: QueryAnalysis, req: ChatRequest) -> EvidencePack:
        ref = a.event_date or req.as_of or self.s.as_of
        items = self._company_items(a, ref, a.event_date)

        cands = self._law_candidates(a, req)
        cands = self._rerank(a.standalone_question, cands)
        picked: list[int] = []
        budget = 0
        for i in cands:
            ch = self.index.chunks[i]
            if i in picked:
                continue
            if len(picked) >= LAW_TOP or budget + len(ch.text) > LAW_CHAR_BUDGET:
                continue
            picked.append(i)
            budget += len(ch.text)
        for i in picked:
            ch = self.index.chunks[i]
            card = self.kb.cards[ch.doc_id]
            appl, note = law_status(card, ch, a.event_date, ref)
            items.append(EvidenceItem(eid=ch.chunk_id, chunk=ch, card=card, applicability=appl, note=note, score=0.0))
        return EvidencePack(items=items, gaps=self._gaps(picked), full_context_used=self.fullctx)
