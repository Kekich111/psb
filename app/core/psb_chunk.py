"""S2 Chunk: документы компании — по пунктам, законы — по пунктам/подпунктам.

chunk_id = {doc_id}#{locator}, offsets указывают в исходный файл (для подсветки в UI).
Редакционные пометки и «инструкции для ИИ» получают is_editorial=True (не цитируются, не индексируются).
Часть/абзац закона не угадываем: без нумерации — ссылка на статью целиком.
"""

from __future__ import annotations

import re

from psb_contracts import Chunk, DocCard, DocClass
from psb_text import AMEND_NOTE_RE, INJECTION_RE, LAW_EDITORIAL_RE

MAX_CHUNK_CHARS = 1800
_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(\S.*)$")  # 3.1. текст
_SECTION_RE = re.compile(r"^(\d+)\.\s+([^.]{1,100})$")  # 3. Арендная плата
_LAW_P_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(\S.*)$")  # 1. текст / 5.1. текст
_LAW_SUB_RE = re.compile(r"^(\d+)\)\s+(\S.*)$")  # 1) текст


class _Builder:
    def __init__(self, card: DocCard, text: str) -> None:
        self.card = card
        self.text = text
        self.pos = 0
        self.chunks: list[Chunk] = []
        self.ids: set[str] = set()

    def locate(self, line: str) -> tuple[int, int]:
        i = self.text.find(line, self.pos)
        if i < 0:
            i = self.text.find(line)
        if i < 0:
            return 0, 0
        self.pos = i + len(line)
        return i, i + len(line)

    def add(self, locator: str, breadcrumb: str, lines: list[tuple[str, int, int]], editorial: bool = False) -> None:
        if not lines:
            return
        body = "\n".join(ln for ln, _, _ in lines)
        # крупные куски режем по строкам
        parts: list[list[tuple[str, int, int]]] = [[]]
        size = 0
        for item in lines:
            if size + len(item[0]) > MAX_CHUNK_CHARS and parts[-1]:
                parts.append([])
                size = 0
            parts[-1].append(item)
            size += len(item[0]) + 1
        if len(parts) == 1:
            self._emit(locator, breadcrumb, body, lines[0][1], lines[-1][2], editorial)
            return
        for k, part in enumerate(parts, 1):
            self._emit(
                f"{locator} (фрагмент {k})",
                breadcrumb,
                "\n".join(ln for ln, _, _ in part),
                part[0][1],
                part[-1][2],
                editorial,
            )

    def _emit(self, locator: str, breadcrumb: str, text: str, start: int, end: int, editorial: bool) -> None:
        cid = f"{self.card.doc_id}#{locator}"
        n = 2
        while cid in self.ids:
            cid = f"{self.card.doc_id}#{locator}~{n}"
            n += 1
        self.ids.add(cid)
        self.chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=self.card.doc_id,
                locator=locator,
                breadcrumb=breadcrumb,
                text=text,
                start=start,
                end=end,
                is_editorial=editorial or bool(INJECTION_RE.search(text)),
            )
        )


def _chunk_company(card: DocCard, text: str) -> list[Chunk]:
    b = _Builder(card, text)
    section = ""
    cur_loc: str | None = None
    cur_bc = ""
    cur_lines: list[tuple[str, int, int]] = []
    free_idx = 0

    def flush() -> None:
        nonlocal cur_loc, cur_lines
        if cur_loc:
            b.add(cur_loc, cur_bc, cur_lines)
        cur_loc, cur_lines = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        s, e = b.locate(line)
        m_c = _CLAUSE_RE.match(line)
        m_s = _SECTION_RE.match(line)
        if m_c:
            flush()
            cur_loc = f"п. {m_c.group(1)}"
            cur_bc = f"{card.title} / {section + ' / ' if section else ''}{cur_loc}"
            cur_lines = [(line, s, e)]
        elif m_s and not line.endswith((":", ";")):
            flush()
            section = f"раздел {m_s.group(1)} {m_s.group(2).strip()}"
            free_idx = 0
            # заголовок сам по себе не чанк; текст раздела без пунктов попадёт в «абз.»
        elif cur_loc and cur_loc.startswith("п. "):
            cur_lines.append((line, s, e))  # продолжение пункта (подпункты, перечни)
        else:
            flush()
            free_idx += 1
            tag = f"{section}, " if section else ""
            cur_loc = f"{tag}абз. {free_idx}" if section else f"абз. {len(b.chunks) + 1}"
            cur_bc = f"{card.title} / {cur_loc}"
            cur_lines = [(line, s, e)]
            flush()
    flush()
    return b.chunks


def _chunk_law(card: DocCard, text: str) -> list[Chunk]:
    b = _Builder(card, text)
    law = card.law
    assert law is not None
    art = law.article
    head_bc = f"{card.title} / {law.article_title or 'Статья ' + art}"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 1) преамбула до «Статья N.» — редакционная
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^Статья\s+{re.escape(art)}\.", ln)), None)
    if start is None:
        start = next((i for i, ln in enumerate(lines) if ln.startswith("Статья")), 0)
    pre = [(ln, *b.locate(ln)) for ln in lines[:start]]
    b.add("примечание", head_bc, pre, editorial=True)

    cur_p: str | None = None
    lead = ""
    cur_loc = f"ст. {art}"
    cur_bc = head_bc
    cur_lines: list[tuple[str, int, int]] = []

    def flush() -> None:
        nonlocal cur_lines
        if cur_lines:
            b.add(cur_loc, cur_bc, cur_lines)
        cur_lines = []

    for ln in lines[start + 1 :]:
        s, e = b.locate(ln)
        if AMEND_NOTE_RE.match(ln):
            continue  # пометка об изменении: метаданные (LawMeta), не норма
        if LAW_EDITORIAL_RE.match(ln):
            flush()
            b.add("примечание", head_bc, [(ln, s, e)], editorial=True)
            continue
        m_sub = _LAW_SUB_RE.match(ln)
        m_p = _LAW_P_RE.match(ln)
        if m_p and not m_sub:
            flush()
            cur_p = m_p.group(1)
            lead = ln
            cur_loc = f"ст. {art}, п. {cur_p}"
            cur_bc = f"{head_bc} / п. {cur_p}"
            cur_lines = [(ln, s, e)]
        elif m_sub:
            flush()
            tag = f"п. {cur_p}, " if cur_p else ""
            cur_loc = f"ст. {art}, {tag}пп. {m_sub.group(1)}"
            lead_txt = lead if lead else ""
            cur_bc = f"{head_bc} / {tag}пп. {m_sub.group(1)}" + (f" ({lead_txt[:200]})" if lead_txt else "")
            cur_lines = [(ln, s, e)]
        else:
            cur_lines.append((ln, s, e))
            if ln.endswith(":") and not cur_p:
                lead = ln
    flush()
    return b.chunks


def chunk_document(card: DocCard, text: str) -> list[Chunk]:
    if card.doc_class == DocClass.LAW and card.law is not None:
        return _chunk_law(card, text)
    return _chunk_company(card, text)
