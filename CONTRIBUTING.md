# Contributing to Consilium

## Development setup

```bash
uv sync --extra dev --extra standards
uv run pre-commit install
```

The `standards` extra installs the local sibling packages (`../indicium`,
`../astra-spec`, `../AgenticScienceBuilder/asb-schema`) as editable.

## Workflow

- **TDD.** Write a failing test first, then the minimal implementation.
- `make ci` must pass (ruff + mypy --strict + pytest) before pushing.
- Keep files focused and small; split by responsibility.
- Never reimplement Perspicacité-owned capabilities (search, RAG, claim/citation graphs).
- Generated claims/citations must pass the `verify_quote` grounding gate — no fabricated references.
- Scores must expose components/weights/uncertainty — no bare magic numbers.

## Tests

```bash
make test          # fast suite (excludes live)
uv run pytest -m unit
uv run pytest -m live   # needs a running Perspicacité + real LLM
```
