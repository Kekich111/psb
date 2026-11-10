"""S0/S1 Ingest: чтение zip, карточки документов (DocCard), метаданные законов, реестр сторон.

Всё детерминированно (regex), без LLM: поля берутся только из текста документа, поэтому
выдуманных значений быть не может. Никаких названий документов/номеров/сумм в логике нет.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

from psb_chunk import chunk_document
from psb_config import log
from psb_contracts import (
    AmendTarget,
    DocCard,
    DocClass,
    DocKind,
    KnowledgeBase,
    LawAmendmentNote,
    LawMeta,
    Party,
)
from psb_text import (
    AMEND_NOTE_RE,
    DATE_DOT,
    MONTH_ALT,
    MONTHS,  # noqa: F401
    first_date,
    iter_dates,  # noqa: F401
    mkdate,
    nonempty_lines as _lines,
)

KB_NAMES = ("knowledge-base.zip", "knowledge_base.zip", "kb.zip")

PARTY_RE = re.compile(r"\b(ООО|АО|ПАО|ЗАО|ИП)\s+«([^»]+)»(?:\s*\(ИНН\s*(\d{10,12})\))?")
ROLE_RE = re.compile(
    r"[«\"(]\s*(Арендодатель|Арендатор|Поставщик|Покупатель|Заказчик|Исполнитель|Продавец|Подрядчик)\b"
)


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", name).strip("_").lower()[:60]


def find_kb_zip(data_dir: Path) -> Path | None:
    here = Path(__file__).resolve().parent
    for root in (data_dir, here / data_dir, Path.cwd(), here, here.parent, here.parent.parent):
        for name in KB_NAMES:
            if (root / name).is_file():
                return root / name
    return None


def read_kb_files(zip_path: str | Path) -> dict[str, str]:
    files: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = unicodedata.normalize("NFC", Path(info.filename).name)
            if not name.lower().endswith((".txt", ".md")):
                continue
            raw = z.read(info)
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp1251", errors="replace")
            files[name] = text.replace("\r\n", "\n").replace("\r", "\n")
    return files


# ---------- карточки документов компании ----------
def _parties(text: str) -> list[Party]:
    seen: dict[str, Party] = {}
    for m in PARTY_RE.finditer(text):
        key = m.group(2).casefold()
        role = None
        rm = ROLE_RE.search(text[m.end() : m.end() + 200])
        if rm:
            role = rm.group(1)
        cur = seen.get(key)
        if cur is None:
            seen[key] = Party(name=f"{m.group(1)} «{m.group(2)}»", inn=m.group(3), role=role)
        else:
            cur.inn = cur.inn or m.group(3)
            cur.role = cur.role or role
    return list(seen.values())


def _number(title: str) -> str | None:
    m = re.search(r"№\s*([^\s,;]+)", title)
    return m.group(1).rstrip(".") if m else None


def _company_card(doc_id: str, fname: str, text: str) -> DocCard:
    lines = _lines(text)
    banner_to: date | None = None
    body = lines
    if lines and re.match(r"(?i)редакция\s+утратила\s+силу", lines[0]):
        banner_to = first_date(lines[0])
        body = lines[1:]
    if not body:
        return DocCard(doc_id=doc_id, file=fname, doc_class=DocClass.COMPANY, kind=DocKind.OTHER, title=fname)
    head = " ".join(body[:6])
    up0 = body[0].upper()
    kind, title, title_lines = DocKind.OTHER, body[0], 1
    if "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ" in up0:
        kind, title, title_lines = DocKind.AMENDMENT, " ".join(body[:2]), 2
    elif up0.startswith("ДОГОВОР"):
        kind = DocKind.CONTRACT
    elif "РЕГЛАМЕНТ" in up0 or up0.startswith("ПОЛОЖЕНИЕ"):
        kind = DocKind.REGULATION
    elif re.search(r"Исх\.\s*№", head):
        kind = DocKind.LETTER
        m = re.search(r"Исх\.\s*№\s*(\S+)\s+от\s+(\d{2}\.\d{2}\.\d{4})", head)
        org = body[1] if len(body) > 1 and len(body[1]) < 60 and not re.search(r"\d", body[1]) else ""
        title = f"Письмо № {m.group(1)} от {m.group(2)}" + (f" ({org})" if org else "") if m else body[0]
    elif "СЛУЖЕБНАЯ ЗАПИСКА" in head.upper():
        kind = DocKind.MEMO
        idx = next((i for i, ln in enumerate(body[:6]) if "СЛУЖЕБНАЯ ЗАПИСКА" in ln.upper()), 0)
        title = body[idx] + (f" — {body[idx + 1]}" if idx + 1 < len(body) and len(body[idx + 1]) < 90 else "")

    number = _number(title) if kind != DocKind.REGULATION else None
    doc_date: date | None = None
    eff_from: date | None = None
    eff_to: date | None = banner_to
    family: str | None = None

    if kind == DocKind.REGULATION:
        family = _slug(re.sub(r"\(.*?\)", "", body[0]))
        m = re.search(r"Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})", text)
        doc_date = first_date(m.group(1)) if m else None
        m2 = re.search(r"Действует\s+с\s+(\d{2}\.\d{2}\.\d{4})", text)
        eff_from = first_date(m2.group(1)) if m2 else doc_date
        if doc_date:
            title = f"{body[0]} (ред. от {doc_date.strftime('%d.%m.%Y')})"
    else:
        dot = DATE_DOT.search(title)
        if dot:
            doc_date = mkdate(*dot.groups())
        if doc_date is None:
            doc_date = first_date(" ".join(body[title_lines : title_lines + 3]))
        eff_from = doc_date

    return DocCard(
        doc_id=doc_id,
        file=fname,
        doc_class=DocClass.COMPANY,
        kind=kind,
        title=title,
        number=number,
        doc_date=doc_date,
        family_id=family,
        effective_from=eff_from,
        effective_to=eff_to,
        parties=_parties(text),
        approx_tokens=len(text) // 3,
    )


def _link_company(cards: dict[str, DocCard], texts: dict[str, str]) -> None:
    """Допсоглашения и письма -> на какой документ/пункты влияют; цепочка редакций регламента."""
    # цепочка редакций: effective_to = effective_from следующей редакции
    families: dict[str, list[DocCard]] = {}
    for c in cards.values():
        if c.family_id:
            families.setdefault(c.family_id, []).append(c)
    for eds in families.values():
        eds.sort(key=lambda c: c.effective_from or date.min)
        for cur, nxt in zip(eds, eds[1:], strict=False):
            if cur.effective_to is None and nxt.effective_from:
                cur.effective_to = nxt.effective_from

    by_number = {c.number: c for c in cards.values() if c.number and c.kind == DocKind.CONTRACT}
    regs = sorted((c for c in cards.values() if c.kind == DocKind.REGULATION), key=lambda c: c.effective_from or date.min)
    for c in cards.values():
        text = texts[c.doc_id]
        if c.kind == DocKind.AMENDMENT:
            m = re.search(r"к\s+договору[^№]*№\s*([^\s,;]+)", text)
            target = by_number.get(m.group(1).rstrip(".")) if m else None
            clauses = list(dict.fromkeys(re.findall(r"[Пп]ункт[а-я]*\s+(\d+(?:\.\d+)*)", text)))[:6]
            eff = None
            for sent in re.split(r"(?<=[.;])\s+", text):
                if re.search(r"применя", sent):
                    eff = first_date(sent)
                    if eff:
                        break
            eff = eff or c.doc_date
            c.effective_from = eff
            c.amends = [AmendTarget(target_doc_id=target.doc_id if target else None, clauses=clauses, effective_from=eff)]
        elif c.kind == DocKind.LETTER:
            m = re.search(r"п\.\s*(\d+(?:\.\d+)*)\s+Регламент", text)
            target = None
            if m and regs:
                ed = re.search(r"ред\.\s*от\s*(\d{2}\.\d{2}\.\d{4})", text[m.start() : m.start() + 200])
                ed_date = first_date(ed.group(1)) if ed else None
                target = next((r for r in regs if ed_date and r.doc_date == ed_date), regs[-1])
            sm = re.search(r"[Сс]\s+(\d{1,2}\s+(?:%s)\s+\d{4})" % MONTH_ALT, text)
            eff = first_date(sm.group(1)) if sm else c.doc_date
            c.effective_from = eff
            if m:
                c.amends = [AmendTarget(target_doc_id=target.doc_id if target else None, clauses=[m.group(1)], effective_from=eff)]


# ---------- законы ----------
def _law_names(fname: str, source: str) -> tuple[str, str, str]:
    """-> (act_name, short_alias, article)."""
    stem = fname.split("__", 1)[-1]
    am = re.match(r"([^_]+)_ст([\d.]+?)_", stem)
    token = am.group(1) if am else ""
    article = am.group(2) if am else ""
    if not article:
        m = re.search(r"Статья\s+(\d+(?:\.\d+)?)", source)
        article = m.group(1) if m else "?"
    if "кодекс" in source.lower() and token:
        act = f"{token} РФ"
    else:
        nm = re.search(r"N\s*(\d+)-ФЗ", source)
        act = f"{nm.group(1)}-ФЗ" if nm else (token or "закон")
    return act, token, article


def parse_law_amendments(text: str) -> list[LawAmendmentNote]:
    notes: list[LawAmendmentNote] = []
    for ln in _lines(text):
        m = AMEND_NOTE_RE.match(ln)
        if not m:
            continue
        dt = first_date(m.group(4))
        if not dt:
            continue
        kind, rest = m.group(1), m.group(2).strip()
        num = re.match(r"([\d.]+)", rest)
        scope = None
        if kind.startswith("Пункт") and num:
            scope = f"п. {num.group(1)}"
        elif kind.startswith("Подпункт") and num:
            scope = f"пп. {num.group(1)}"
        elif kind.startswith("Часть") and num:
            scope = f"ч. {num.group(1)}"
        elif kind.startswith(("Абзац", "Раздел")):
            scope = kind.lower()
        notes.append(
            LawAmendmentNote(
                effective_from=dt,
                act=m.group(5).strip()[:160],
                scope=scope,
                repealed=m.group(3).startswith("утратил"),
            )
        )
    return notes


def _law_card(doc_id: str, fname: str, text: str) -> DocCard:
    src = re.search(r"^Источник:\s*(.+)$", text, re.M)
    source = src.group(1) if src else ""
    act, token, article = _law_names(fname, source)
    exp = re.search(r"Дата выгрузки:\s*(\d{2}\.\d{2}\.\d{4})", text)
    exported = first_date(exp.group(1)) if exp else None
    head = re.search(rf"^Статья\s+{re.escape(article)}\.\s*(.*)$", text, re.M)
    art_title = head.group(0).strip() if head else f"Статья {article}"
    meta = LawMeta(
        act_name=act,
        article=article,
        article_title=art_title,
        alias=token or None,
        exported_at=exported,
        amendments=parse_law_amendments(text),
    )
    norm = lambda x: re.sub(r"[^0-9A-Za-zА-Яа-я]", "", x).casefold()  # noqa: E731
    alias = f" ({token})" if token and norm(token) not in norm(act) else ""
    return DocCard(
        doc_id=doc_id,
        file=fname,
        doc_class=DocClass.LAW,
        kind=DocKind.LAW,
        title=f"{act}{alias}, ст. {article}",
        number=article,
        doc_date=None,
        family_id=None,
        effective_from=None,
        parties=[],
        law=meta,
        approx_tokens=len(text) // 3,
    )


# ---------- сборка KB ----------
def _doc_class(fname: str, text: str) -> DocClass:
    low = fname.lower()
    if low.startswith("законы"):
        return DocClass.LAW
    if low.startswith("документы компании"):
        return DocClass.COMPANY
    return DocClass.LAW if ("Дата выгрузки:" in text[:2000] and "Источник:" in text[:2000]) else DocClass.COMPANY


def build_kb(zip_path: str | Path) -> KnowledgeBase:
    files = read_kb_files(zip_path)
    h = hashlib.sha256()
    cards: dict[str, DocCard] = {}
    texts: dict[str, str] = {}
    for fname in sorted(files):
        text = files[fname]
        h.update(fname.encode())
        h.update(text.encode())
        doc_id = _slug(fname.split("__", 1)[-1].rsplit(".", 1)[0]) or _slug(fname)
        while doc_id in cards:
            doc_id += "_"
        card = _law_card(doc_id, fname, text) if _doc_class(fname, text) == DocClass.LAW else _company_card(doc_id, fname, text)
        cards[doc_id] = card
        texts[doc_id] = text
    _link_company({k: v for k, v in cards.items() if v.doc_class == DocClass.COMPANY}, texts)

    chunks = {}
    for doc_id, card in cards.items():
        for ch in chunk_document(card, texts[doc_id]):
            chunks[ch.chunk_id] = ch

    # наша компания = самая частая сторона в документах компании
    cnt: Counter[str] = Counter()
    variants: dict[str, Counter[str]] = {}
    for c in cards.values():
        if c.doc_class != DocClass.COMPANY:
            continue
        for key in {p.name.casefold() for p in c.parties}:
            cnt[key] += 1
        for p in c.parties:
            variants.setdefault(p.name.casefold(), Counter())[p.name] += 1
    our = None
    if cnt:
        top = cnt.most_common(1)[0][0]
        names = variants[top]
        our = next((n for n, _ in names.most_common() if not n.isupper()), names.most_common(1)[0][0])

    inventory = sorted(
        f"{c.law.act_name} ст. {c.law.article}" for c in cards.values() if c.law is not None
    )
    kb = KnowledgeBase(kb_hash=h.hexdigest(), cards=cards, chunks=chunks, article_inventory=inventory, our_company=our)
    log(f"KB: документов {len(cards)}, чанков {len(chunks)}, наша компания: {our}")
    return kb


class RuleIngestor:
    """Реализация протокола Ingestor."""

    def build(self, kb_zip_path: str, llm: object = None) -> KnowledgeBase:  # noqa: ARG002
        return build_kb(kb_zip_path)


def confusable_pairs(kb: KnowledgeBase, threshold: float = 80.0) -> list[tuple[str, str]]:
    """Пары разных контрагентов с похожими названиями («Альфа-Трейд» / «Альфа-Трейдинг»)."""
    try:
        from rapidfuzz import fuzz

        ratio = fuzz.ratio
    except ImportError:  # rapidfuzz есть в requirements; запасной путь для окружений без него
        from difflib import SequenceMatcher

        def ratio(x: str, y: str) -> float:
            return 100.0 * SequenceMatcher(None, x, y).ratio()

    names = sorted({p.name for c in kb.cards.values() for p in c.parties})
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a.casefold() != b.casefold() and ratio(a.casefold(), b.casefold()) >= threshold:
                out.append((a, b))
    return out
