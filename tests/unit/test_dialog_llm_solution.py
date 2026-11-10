from __future__ import annotations

import csv
from pathlib import Path

import pytest
from psb_dialog import parse_dialog
from psb_fakes import FakeLLM
from psb_llm import chat_json, parse_json_tolerant

import solution


# ---------- диалог ----------
def test_single_question() -> None:
    turns = parse_dialog("Какая арендная плата?")
    assert [t.role for t in turns] == ["user"]


def test_dialog_roles_and_last_is_user() -> None:
    text = (
        "Пользователь: Какой гарантийный срок?\n"
        "Ассистент: Уточните поставщика.\n"
        "Пользователь: «Альфа-Трейдинг», на сколько гарантия?"
    )
    turns = parse_dialog(text)
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[-1].content.startswith("«Альфа-Трейдинг»")


def test_dialog_ending_with_assistant_is_treated_as_single_question() -> None:
    turns = parse_dialog("Пользователь: вопрос\nАссистент: ответ")
    assert len(turns) == 1 and turns[0].role == "user"


def test_demo_dialogs_parse() -> None:
    demo = Path(__file__).resolve().parents[2] / "app" / "eval" / "demo" / "demo_questions.csv"
    with demo.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18
    multi = [r for r in rows if r["question"].startswith("Пользователь:")]
    assert len(multi) == 5
    for r in multi:
        turns = parse_dialog(r["question"])
        assert turns[-1].role == "user" and len(turns) >= 3


# ---------- JSON ----------
@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '<think>размышление</think>{"a": 1}',
        'Вот ответ: {"a": 1,}',
        '{"a": 1',  # обрезан
        '{"a": "текст',  # обрезан внутри строки
    ],
)
def test_parse_json_tolerant(raw: str) -> None:
    parsed = parse_json_tolerant(raw)
    assert parsed is not None and parsed["a"] in (1, "текст")


def test_parse_json_rejects_garbage() -> None:
    assert parse_json_tolerant("совсем не json") is None
    assert parse_json_tolerant("") is None


def test_chat_json_retries_once_then_none() -> None:
    llm = FakeLLM(reply="не json")
    assert chat_json(llm, [{"role": "user", "content": "x"}]) is None
    assert llm.calls == 2


# ---------- solution.py ----------
def test_solution_writes_valid_complete_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "questions.csv").write_text(
        '﻿"id","question"\n'
        '"1","Какая арендная плата по договору аренды?"\n'
        '"2","Пользователь: Привет?\nАссистент: Здравствуйте.\nПользователь: Какая арендная плата?"\n'
        '"2","дубль id"\n'
        '"3","Какой размер пособия по беременности?"\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_PATH", str(out))
    solution.main()
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["id", "answer"]
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]
    assert all(r[1].strip() and len(r[1]) <= 10_000 and "\n" not in r[1] for r in rows[1:])
