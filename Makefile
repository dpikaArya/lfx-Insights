.PHONY: install install-standards install-api test test-all lint fmt typecheck ci \
       api

install:
	uv sync --extra dev

install-standards:
	uv sync --extra dev --extra standards

install-api:
	uv sync --extra dev

test:
	uv run pytest -m "not live"

test-all:
	uv run pytest

test-live:
	uv run pytest -m live   # needs Perspicacité running on :8002 (+ real LLM for generation)

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src

docs:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build --strict

ci: lint typecheck test

api:
	@echo "Starting LFX Insights API on http://127.0.0.1:8000 ..."
	set PYTHONPATH=src && python -m lfx_insights.api
