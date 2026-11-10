"""S3 Index: BM25 (своя реализация, лёгкая нормализация русского) + dense bge-m3 с кэшем.

Dense опционален: любая ошибка эмбеддингов -> работаем на одном BM25. Кэш эмбеддингов по sha256(текста):
`emb_cache.npz` рядом с решением (можно положить в ZIP) и копия во временном каталоге.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from psb_config import Settings, log
from psb_contracts import Chunk, KnowledgeBase

_STOP = frozenset(
    "и в во не на с со к по о об от до за из у а но или как что это для при он она они бы же ли то так "
    "быть был была были есть его ее их также если может могут который которая которые этот эта эти все вся "
    "свой при над под про между через без".split()
)
_TOKEN_RE = re.compile(r"[a-zа-я]+|\d+(?:[.,]\d+)?")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if tok[0].isdigit():
            out.append(tok.replace(",", "."))
        elif len(tok) >= 2 and tok not in _STOP:
            out.append(tok[:6])  # грубое усечение основы: «налогообложение» ~ «налоговый»
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = [len(d) for d in docs]
        self.avg = (sum(self.dl) / self.n) if self.n else 1.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.post[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, query: list[str]) -> dict[int, float]:
        acc: dict[int, float] = defaultdict(float)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.post[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg)
                acc[i] += idf * tf * (self.k1 + 1) / denom
        return acc


def _chunk_text(ch: Chunk) -> str:
    return f"{ch.breadcrumb}\n{ch.text}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_paths() -> list[Path]:
    here = Path(__file__).resolve().parent
    paths = [here / "emb_cache.npz", Path.cwd() / "emb_cache.npz"]
    paths.append(Path(os.getenv("EMB_CACHE") or Path(tempfile.gettempdir()) / "psb_emb_cache.npz"))
    return paths


def _load_cache() -> dict[str, Any]:
    cache: dict[str, Any] = {}
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return cache
    for path in _cache_paths():
        try:
            if path.is_file():
                data = np.load(path, allow_pickle=False)
                for k, v in zip(data["keys"].tolist(), data["vecs"], strict=False):
                    cache[str(k)] = v
        except Exception as exc:  # noqa: BLE001 — битый кэш не страшен
            log(f"emb cache {path.name}: {type(exc).__name__}")
    return cache


def _save_cache(cache: dict[str, Any]) -> None:
    try:
        import numpy as np

        keys = np.array(list(cache.keys()))
        vecs = np.stack([cache[k] for k in cache])
        np.savez_compressed(_cache_paths()[-1], keys=keys, vecs=vecs)
    except Exception as exc:  # noqa: BLE001
        log(f"emb cache save: {type(exc).__name__}")


class Index:
    """Индекс по всем цитируемым чанкам (is_editorial исключены)."""

    def __init__(self, kb: KnowledgeBase, llm: Any = None, settings: Settings | None = None) -> None:
        self.chunks: list[Chunk] = sorted((c for c in kb.chunks.values() if not c.is_editorial), key=lambda c: c.chunk_id)
        self.pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25([tokenize(_chunk_text(c)) for c in self.chunks])
        self.vecs: Any = None
        self.llm = llm
        if llm is not None and settings is not None and settings.api_base and settings.emb_key:
            self._build_dense()

    # -- dense --
    def _build_dense(self) -> None:
        try:
            import numpy as np

            texts = [_chunk_text(c) for c in self.chunks]
            keys = [_sha(t) for t in texts]
            cache = _load_cache()
            missing = [i for i, k in enumerate(keys) if k not in cache]
            if missing:
                log(f"эмбеддинги: считаем {len(missing)} из {len(texts)}")
                new = self.llm.embed([texts[i] for i in missing])
                for i, v in zip(missing, new, strict=True):
                    cache[keys[i]] = np.asarray(v, dtype="float32")
                _save_cache(cache)
            self.vecs = np.stack([cache[k] for k in keys])
        except Exception as exc:  # noqa: BLE001
            log(f"dense отключён: {type(exc).__name__}")
            self.vecs = None

    def dense_top(self, qvec: Any, allowed: set[int], k: int) -> list[int]:
        if self.vecs is None:
            return []
        import numpy as np

        sims = self.vecs @ np.asarray(qvec, dtype="float32")
        order = np.argsort(-sims)
        return [int(i) for i in order if int(i) in allowed][:k]

    def bm25_top(self, query: str, allowed: set[int] | None, k: int) -> list[int]:
        sc = self.bm25.scores(tokenize(query))
        items = [(s, i) for i, s in sc.items() if allowed is None or i in allowed]
        items.sort(key=lambda t: (-t[0], t[1]))
        return [i for _, i in items[:k]]

    def indices_where(self, pred: Any) -> set[int]:
        return {i for i, c in enumerate(self.chunks) if pred(c)}

    def embed_queries(self, queries: list[str]) -> list[Any]:
        if self.vecs is None or self.llm is None:
            return []
        try:
            return list(self.llm.embed(queries))
        except Exception as exc:  # noqa: BLE001
            log(f"embed query: {type(exc).__name__}")
            return []


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] += 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda t: (-t[1], t[0]))]
