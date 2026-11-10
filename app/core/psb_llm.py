"""Единственное место, где есть сеть: chat / embeddings / rerank.

Таймауты, ограниченные ретраи (429/5xx/сеть) с экспоненциальной паузой и джиттером,
общий rate-limiter. httpx импортируется лениво: без него модуль всё равно загружается.
Поле `reasoning_content` игнорируется, читается только `content`.
"""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from typing import Any, Optional

from psb_config import Settings, log


class LLMError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


# ---------- толерантный JSON ----------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _balance(s: str) -> str:
    """Дозакрывает строку и скобки у обрезанного JSON."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    tail = '"' if in_str else ""
    return s + tail + "".join(reversed(stack))


def parse_json_tolerant(text: Optional[str]) -> Optional[dict]:
    """Срезает ```json, <think>, чинит хвостовые запятые и обрезанный вывод. None = не вышло."""
    if not text:
        return None
    s = _THINK_RE.sub("", text).strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    candidates = [s]
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        candidates.append(s[i : j + 1])
    if i != -1:
        candidates.append(_balance(s[i:]))
    for cand in candidates:
        for fixed in (cand, _TRAILING_COMMA_RE.sub(r"\1", cand)):
            try:
                value = json.loads(fixed)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                return value
    return None


def chat_json(llm: Any, messages: list[dict], *, max_tokens: Optional[int] = None) -> Optional[dict]:
    """JSON-ответ модели: один вызов + один ретрай, затем None (вызывающий деградирует)."""
    for _ in range(2):
        try:
            raw = llm.chat(messages, json_mode=True, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — сбой LLM не должен ронять вопрос
            log(f"chat_json: {type(exc).__name__}")
            continue
        parsed = parse_json_tolerant(raw)
        if parsed is not None:
            return parsed
    return None


# ---------- клиент ----------
class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next)
            self._next = at + self._interval
        if at > now:
            time.sleep(at - now)


class LLMClient:
    """Реализация протокола LLM поверх OpenAI-совместимого прокси организаторов."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._limiter = _RateLimiter(settings.max_rps)
        self._sem = threading.Semaphore(settings.max_concurrent_calls)
        self._count_lock = threading.Lock()
        self._client: Any = None
        self.calls = 0

    # -- внутреннее --
    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise LLMError("httpx не установлен") from exc
            self._client = httpx.Client(timeout=self._s.http_timeout)
        return self._client

    def _post(self, path: str, key: str, payload: dict) -> dict:
        if not self._s.api_base or not key:
            raise LLMError("API не настроен (нет LLM_BASE_URL или ключа)")
        url = self._s.api_base + path
        last: Optional[Exception] = None
        for attempt in range(self._s.max_retries + 1):
            self._limiter.wait()
            retry = False
            try:
                with self._sem:
                    resp = self._http().post(url, headers={"Authorization": "Bearer " + key}, json=payload)
                with self._count_lock:
                    self.calls += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = LLMError(f"HTTP {resp.status_code}", resp.status_code)
                    retry = True
                elif resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}", resp.status_code)
                else:
                    data = resp.json()
                    if not isinstance(data, dict):
                        raise LLMError("ответ не JSON-объект")
                    return data
            except LLMError:
                if not retry:
                    raise
            except Exception as exc:  # noqa: BLE001 — сетевые ошибки httpx, битый JSON
                last = LLMError(type(exc).__name__)
                retry = True
            if retry and attempt < self._s.max_retries:
                time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))
        raise last or LLMError("неизвестная ошибка")

    # -- протокол LLM --
    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._s.llm_model,
            "messages": messages,
            "temperature": self._s.temperature,
            "max_tokens": max_tokens or self._s.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post("/chat/completions", self._s.llm_key, payload)
        except LLMError as exc:
            if json_mode and exc.status == 400:  # прокси не знает response_format
                payload.pop("response_format", None)
                data = self._post("/chat/completions", self._s.llm_key, payload)
            else:
                raise
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("неожиданная форма ответа chat") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("пустой ответ модели")
        return content

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            data = self._post(
                "/embeddings",
                self._s.emb_key,
                {"model": self._s.emb_model, "input": batch, "encoding_format": "float"},
            )
            vectors: list[Optional[list[float]]] = [None] * len(batch)
            for pos, item in enumerate(data.get("data", [])):
                idx = int(item.get("index", pos))
                if not 0 <= idx < len(batch):
                    raise LLMError("некорректный индекс embedding")
                vec = [float(x) for x in item["embedding"]]
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vectors[idx] = [x / norm for x in vec]
            if any(v is None for v in vectors):
                raise LLMError("получены не все embeddings")
            out.extend(v for v in vectors if v is not None)
        return out

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        if not docs:
            return []
        data = self._post(
            "/rerank",
            self._s.rerank_key,
            {
                "model": self._s.reranker_model,
                "query": query,
                "documents": docs,
                "top_n": min(top_n, len(docs)),
            },
        )
        items = data.get("results", data.get("data", []))  # у провайдеров по-разному
        seen: list[int] = []
        for item in items:
            idx = int(item["index"])
            if 0 <= idx < len(docs) and idx not in seen:
                seen.append(idx)
        return seen[:top_n]
