"""Настройки из окружения. Ничего не хардкодим: модели, ключи, лимиты — из env."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


def log(msg: str) -> None:
    """Логи только в stderr. Не передавать сюда ключи и полные промпты."""
    print(msg, file=sys.stderr, flush=True)


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


def parse_date(value: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_base: str
    llm_key: str
    emb_key: str
    rerank_key: str
    llm_model: str
    emb_model: str
    reranker_model: str
    top_k: int
    reranker_top_n: int
    chunker: str
    temperature: float
    max_tokens: int
    max_workers: int
    verify_mode: str  # full | lite | off
    deadline_sec: int
    as_of: date
    company_fullctx_tokens: int
    http_timeout: float
    max_retries: int
    max_rps: float  # 0 = без ограничения
    max_concurrent_calls: int


def load_settings() -> Settings:
    """Читает env при каждом вызове (удобно для тестов)."""
    llm_key = env("LLM_API_KEY") or env("OPENAI_API_KEY")
    verify = env("VERIFY_MODE", "full").lower()
    return Settings(
        data_dir=Path(env("DATA_DIR", "data")),
        api_base=(env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/"),
        llm_key=llm_key,
        emb_key=env("EMBEDDER_API_KEY") or llm_key,
        rerank_key=env("RERANKER_API_KEY") or llm_key,
        llm_model=env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash"),
        emb_model=env("EMB_MODEL", "baai/bge-m3"),
        reranker_model=env("RERANKER_MODEL", "cohere/rerank-5-fast"),
        top_k=env_int("TOP_K", 4, 1, 64),
        reranker_top_n=env_int("RERANKER_TOP_N", 2, 1, 32),
        chunker=env("CHUNKER", "common"),
        temperature=env_float("TEMPERATURE", 0.0),
        max_tokens=env_int("MAX_TOKENS", 1024, 128, 8192),
        max_workers=env_int("MAX_WORKERS", 2, 1, 8),
        verify_mode=verify if verify in {"full", "lite", "off"} else "full",
        deadline_sec=env_int("DEADLINE_SEC", 900, 30, 1140),
        as_of=parse_date(env("AS_OF_DATE")) or date.today(),
        company_fullctx_tokens=env_int("COMPANY_FULLCTX_TOKENS", 12000, 0, 200000),
        http_timeout=env_float("HTTP_TIMEOUT", 90.0, 5.0, 300.0),
        max_retries=env_int("MAX_RETRIES", 3, 0, 6),
        max_rps=env_float("MAX_RPS", 0.0, 0.0, 100.0),
        max_concurrent_calls=env_int("MAX_CONCURRENT_CALLS", 8, 1, 32),
    )
