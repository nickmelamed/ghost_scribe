# Standard targets. Adjust the tool commands to this repo, keeping the names,
# since CLAUDE.md, the Stop hook, and CI refer to them.
.PHONY: setup lint typecheck test test-fast style numbers agent-check ci

PY ?= python3

setup:
	$(PY) -m pip install -r requirements.txt
	pre-commit install

lint:
	ruff check . --exclude ghost

typecheck:
	mypy --ignore-missing-imports --explicit-package-bases --exclude ghost .

test:
	pytest -q

test-fast:
	pytest -x -q -m "not slow"

style:
	$(PY) scripts/agent/check_style.py --changed .

numbers:
	$(PY) scripts/agent/check_numbers.py README.md --sources results/phase2_metrics.json results/phase4_metrics.json

# The fast checks the Stop hook runs before Claude may end a turn. Keep the
# whole target under a few minutes.
agent-check: lint typecheck test-fast style

ci: lint typecheck test style
	$(PY) scripts/agent/check_tests.py --warn
