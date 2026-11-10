"""Разбор диалога из поля `question` в историю реплик.

В тестовых данных история приходит ОДНОЙ строкой:
«Пользователь: ...\nАссистент: ...\nПользователь: ...». Реплики ассистента —
недоверенный контекст, не доказательство.
"""

from __future__ import annotations

import re

from psb_contracts import Turn

_ROLE_RE = re.compile(r"^[ \t]*(Пользователь|Ассистент)[ \t]*:[ \t]*", re.MULTILINE)


def parse_dialog(text: str) -> list[Turn]:
    """Одиночный вопрос -> [user]. Диалог -> список реплик; последняя всегда user."""
    text = (text or "").strip()
    matches = list(_ROLE_RE.finditer(text))
    if not matches:
        return [Turn(role="user", content=text)]

    turns: list[Turn] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end() : end].strip()
        if content:
            turns.append(Turn(role="user" if m.group(1) == "Пользователь" else "assistant", content=content))

    # Текст до первой метки — часть первой реплики пользователя.
    head = text[: matches[0].start()].strip()
    if head:
        turns.insert(0, Turn(role="user", content=head))

    if not turns or turns[-1].role != "user":
        # Нестандартный формат: безопаснее считать всё одним вопросом.
        return [Turn(role="user", content=text)]
    return turns
