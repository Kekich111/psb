PY ?= python3
export PYTHONPATH := app/core:.

.PHONY: install check lint typecheck test eval run web examples submission clean

install:
	$(PY) -m pip install -r requirements.txt -r requirements-web.txt -r requirements-dev.txt

check: lint typecheck test

lint:
	ruff check .

typecheck:
	mypy app/core solution.py

test:
	$(PY) -m pytest -q

examples:
	$(PY) scripts/gen_examples.py

eval:
	$(PY) app/eval/run_demo.py

run:
	$(PY) solution.py

web:
	uvicorn app.web.server:app --host 0.0.0.0 --port 8080 --reload

submission: check
	$(PY) scripts/build_submission.py

clean:
	rm -rf dist/submission.zip submission.csv .pytest_cache .mypy_cache .ruff_cache
