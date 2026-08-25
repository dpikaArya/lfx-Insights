# Installation

Consilium uses [uv](https://docs.astral.sh/uv/) and targets **Python ≥ 3.12**.

```bash
git clone https://github.com/HolobiomicsLab/consilium
cd consilium
uv sync --extra dev
```

## Optional extras

| Extra | Brings | When you need it |
|---|---|---|
| `dev` | pytest, ruff, mypy, pre-commit, scipy, statsmodels | development & tests |
| `standards` | `indicium`, `astra-spec`, `asb-schema` (local editable siblings) | standards-conformant export + `verify_quote` grounding |
| `mcp` | `fastmcp` | running Consilium [as an MCP server](mcp.md) |
| `docs` | `mkdocs-material` | building this site |

```bash
uv sync --extra dev --extra standards   # claims/insights export validated against the real schemas
uv sync --extra dev --extra mcp         # `consilium serve`
```

!!! note "The `standards` extra needs local siblings"
    `indicium`, `astra-spec`, and `asb-schema` are private sibling repos referenced via
    `[tool.uv.sources]` at `../indicium`, `../astra-spec`, `../AgenticScienceBuilder/asb-schema`.
    Without them, the exporters still produce standards-shaped output and the grounding gate falls
    back to a normalized-substring match; conformance/round-trip tests skip.

## Perspicacité

Consilium delegates all literature work to a running **Perspicacité** MCP server (default
`http://localhost:8002/mcp`). If it is unreachable, Consilium fails loudly rather than guessing —
it never falls back to home-grown search. Use `--offline` on any command for a network/LLM-free
demo backed by in-memory fakes.

## Verify

```bash
make ci        # ruff + mypy --strict + pytest
make test      # fast test suite (excludes live)
make test-live # live tests against a running Perspicacité on :8002
```
