# Contributing

## Setup

```bash
uv sync --extra dev --extra standards --extra mcp
uv run pre-commit install
```

## Workflow

- **TDD** — write a failing test first, then the minimal implementation.
- `make ci` (ruff + `mypy --strict` + pytest) must pass before pushing.
- Keep files focused and small; split by responsibility.
- **Never reimplement Perspicacité-owned capabilities** (search, RAG, claim/citation graphs) —
  delegate them.
- Generated claims/citations must pass the `verify_quote` grounding gate — no fabricated references.
- Scores must expose components/weights/uncertainty — no bare magic numbers.

## Tests

```bash
make test                 # fast suite (excludes live)
uv run pytest -m unit
uv run pytest -m live     # needs a running Perspicacité on :8002 (+ real LLM for generation)
```

## Docs

```bash
uv sync --extra docs
uv run mkdocs serve       # local preview
uv run mkdocs build --strict
```

Specs and plans under `docs/superpowers/` are local working notes and are not committed or part of
the built site.
