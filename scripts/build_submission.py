"""Собирает dist/submission.zip: всё строго в корне архива, без вложенных папок.

Состав: solution.py, app/core/psb_*.py, requirements.txt, (emb_cache.npz). Без .env, KB и вопросов.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "submission.zip"
LIMIT = 128 * 1024 * 1024
FORBIDDEN = {".env", "knowledge-base.zip", "kb.zip", "questions.csv", "submission.csv"}
EXCLUDE: set[str] = set()  # например {"psb_fakes.py"}, когда фейки перестанут быть нужны на платформе


def collect() -> list[Path]:
    files = [ROOT / "solution.py", *sorted((ROOT / "app" / "core").glob("psb_*.py")), ROOT / "requirements.txt"]
    cache = ROOT / "emb_cache.npz"
    if cache.is_file():
        files.append(cache)
    return [f for f in files if f.name not in EXCLUDE]


def main() -> int:
    files = collect()
    names = [f.name for f in files]
    problems: list[str] = []
    for f in files:
        if not f.is_file():
            problems.append(f"нет файла: {f}")
    if len(set(names)) != len(names):
        problems.append("повторяющиеся имена файлов в корне архива")
    if any(n in FORBIDDEN for n in names):
        problems.append("в архив попал запрещённый файл")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    size = OUT.stat().st_size
    if size > LIMIT:
        print(f"Архив {size} байт > 128 МБ", file=sys.stderr)
        return 1
    print(f"{OUT} ({size / 1024:.1f} КБ): {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
