"""Точка входа платформы: questions.csv -> Assistant.answer -> submission.csv.

Тонкая обёртка без юридической логики. Гарантии: submission.csv создаётся сразу (заглушки),
перезаписывается атомарно по мере готовности и в конце; все id, непустые ответы, без дублей,
колонки ровно id,answer, UTF-8, не более 10 000 символов; мягкий дедлайн DEADLINE_SEC.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "app" / "core"):  # на платформе psb_*.py лежат рядом, локально — в app/core
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAX_ANSWER_CHARS = 10_000
# Дубль строки из psb_render: solution.py обязан писать файл даже если импорт ядра сломался.
FALLBACK = "В базе знаний нет данных, достаточных для надёжного ответа. Отвечать по памяти не буду."


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def find_file(*names: str) -> Path:
    data_dir = os.getenv("DATA_DIR", "data").strip() or "data"
    for root in (Path(data_dir), _HERE / data_dir, Path.cwd(), _HERE):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Не найден один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = [(str(r["id"]).strip(), str(r["question"]).strip()) for r in reader]
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for qid, text in rows:
        if qid and qid not in seen:  # без дублей id
            seen.add(qid)
            unique.append((qid, text))
    if not unique:
        raise ValueError("В questions.csv нет вопросов")
    return unique


def clean_answer(text: object) -> str:
    s = " ".join(str(text or "").split())
    if len(s) > MAX_ANSWER_CHARS:
        s = s[: MAX_ANSWER_CHARS - 1].rstrip() + "…"
    return s or FALLBACK


def write_submission(path: Path, questions: list[tuple[str, str]], answers: dict[str, str]) -> None:
    """Атомарная запись: сначала временный файл, затем os.replace."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for qid, _ in questions:
            writer.writerow([qid, clean_answer(answers.get(qid))])
    os.replace(tmp, path)


def main() -> None:
    started = time.monotonic()
    questions = load_questions(find_file("questions.csv"))
    out_path = Path(os.getenv("OUTPUT_PATH", "").strip() or "submission.csv")
    answers: dict[str, str] = {qid: FALLBACK for qid, _ in questions}
    write_submission(out_path, questions, answers)  # заглушки сразу
    log(f"Вопросов: {len(questions)}; заглушки записаны в {out_path}")

    try:
        from psb_assistant import build_default_assistant
        from psb_config import load_settings
        from psb_contracts import ChatRequest, RunMode
        from psb_dialog import parse_dialog

        settings = load_settings()
        assistant = build_default_assistant(settings)
    except Exception as exc:  # noqa: BLE001 — остаёмся с заглушками, но файл валиден
        log(f"Ядро не загрузилось: {type(exc).__name__}: {exc}")
        return

    deadline = started + settings.deadline_sec

    def work(qid: str, text: str) -> tuple[str, str]:
        if time.monotonic() > deadline:
            return qid, FALLBACK
        try:
            req = ChatRequest(history=parse_dialog(text), mode=RunMode.BATCH, as_of=settings.as_of)
            return qid, clean_answer(assistant.answer(req).text)
        except Exception as exc:  # noqa: BLE001
            log(f"Вопрос {qid}: {type(exc).__name__}")
            return qid, FALLBACK

    hard_exit = False
    pool = ThreadPoolExecutor(max_workers=settings.max_workers)
    futures = [pool.submit(work, qid, text) for qid, text in questions]
    try:
        for done, fut in enumerate(as_completed(futures, timeout=max(1.0, deadline - time.monotonic() + 60)), 1):
            qid, answer = fut.result()
            answers[qid] = answer
            if done % 10 == 0 or done == len(futures):
                write_submission(out_path, questions, answers)
                log(f"Обработано {done}/{len(futures)}")
    except FutureTimeout:
        log("Дедлайн: оставшиеся вопросы получают безопасный ответ")
        hard_exit = True
    pool.shutdown(wait=not hard_exit, cancel_futures=True)
    write_submission(out_path, questions, answers)
    log(f"Готово: {out_path}, {len(questions)} ответов, {time.monotonic() - started:.1f} с")
    if hard_exit:
        sys.stderr.flush()
        os._exit(0)  # зависшие потоки не должны держать процесс


if __name__ == "__main__":
    main()
