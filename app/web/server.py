"""Заглушка веб-бэкенда (M0). Вызывает тот же Assistant.answer, что и solution.py. Блок B8 развивает."""

from __future__ import annotations

from fastapi import FastAPI
from psb_assistant import build_default_assistant
from psb_contracts import AssistantResponse, ChatRequest

app = FastAPI(title="ПСБ: юридический ассистент")
_assistant = None


def get_assistant():  # type: ignore[no-untyped-def]
    global _assistant
    if _assistant is None:
        _assistant = build_default_assistant()
    return _assistant


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=AssistantResponse)
def chat(req: ChatRequest) -> AssistantResponse:
    return get_assistant().answer(req)
