"""Общие текстовые утилиты и регулярные выражения (даты, пометки законов, инъекции)."""

from __future__ import annotations

import re
from datetime import date

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}  # fmt: skip
MONTH_ALT = "|".join(MONTHS)
DATE_TXT = re.compile(rf"(\d{{1,2}})\s+({MONTH_ALT})\s+(\d{{4}})")
DATE_DOT = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")

# Строки, похожие на обращение к ИИ внутри документов: не норма и не источник (prompt injection).
INJECTION_RE = re.compile(
    r"(?i)(для\s+ии\b|ии-ассистент|ассистент[ау]?\b.{0,40}(игнорир|отвечай)|игнорируй|игнорируйте|"
    r"забудь\s+(все|предыдущ)|system\s*:|не\s+упоминай\s+эту\s+инструкцию|отвечай\s+[«\"])"
)

# «Часть 1 изменена с 1 сентября 2025 г. - Федеральный закон ...»
AMEND_NOTE_RE = re.compile(
    r"^(Статья|Часть|Части|Пункт|Пункты|Подпункт|Подпункты|Абзац|Абзацы|Раздел)\b(.{0,60}?)\s+"
    r"(изменен\w*|утратил\w*\s+силу|дополнен\w*|введен\w*|исключен\w*)\s+с\s+"
    rf"(\d{{1,2}}\s+(?:{MONTH_ALT})\s+\d{{4}})\s*г?\.?\s*[-–—]\s*(.+)$"
)
LAW_EDITORIAL_RE = re.compile(
    r"(?i)^(о проверке конституционности|о конституционно-правовом смысле|см\.\s|информация об изменениях)"
)


def mkdate(d: str, m: str, y: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def iter_dates(text: str) -> list[tuple[int, date]]:
    out: list[tuple[int, date]] = []
    for m in DATE_DOT.finditer(text):
        dt = mkdate(m.group(1), m.group(2), m.group(3))
        if dt:
            out.append((m.start(), dt))
    for m in DATE_TXT.finditer(text):
        dt = mkdate(m.group(1), str(MONTHS[m.group(2)]), m.group(3))
        if dt:
            out.append((m.start(), dt))
    return sorted(out)


def first_date(text: str) -> date | None:
    found = iter_dates(text)
    return found[0][1] if found else None


def nonempty_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
