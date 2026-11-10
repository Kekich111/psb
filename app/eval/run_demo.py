"""Прогон 18 демо-вопросов через Assistant и сравнение с эталонами организаторов.

Эталоны demo_output.csv — ТОЛЬКО для оценки (не для логики). Судья пока не реализован (блок B7):
скрипт пишет dist/demo_run.csv (id, вопрос, наш ответ, эталон, вердикт) для ручного просмотра.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "core"))

from psb_assistant import build_default_assistant  # noqa: E402
from psb_config import load_settings  # noqa: E402
from psb_contracts import ChatRequest, RunMode  # noqa: E402
from psb_dialog import parse_dialog  # noqa: E402


def read(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r.get("question") or r.get("answer") or "" for r in csv.DictReader(f)}


def main() -> None:
    demo = ROOT / "app" / "eval" / "demo"
    questions, golden = read(demo / "demo_questions.csv"), read(demo / "demo_output.csv")
    settings = load_settings()
    assistant = build_default_assistant(settings)
    out = ROOT / "dist" / "demo_run.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "question", "answer", "golden", "verdict", "llm_stages_s"])
        for qid, q in questions.items():
            r = assistant.answer(ChatRequest(history=parse_dialog(q), mode=RunMode.BATCH, as_of=settings.as_of))
            w.writerow([qid, q, r.text, golden.get(qid, ""), r.verdict.value, r.trace.get("stages", {})])
    print(f"Записано {out}. Парный судья и метрики — блок B7 (пока не реализован).")


if __name__ == "__main__":
    main()
