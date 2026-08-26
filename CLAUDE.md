# lfx Insights — agent guide

## What this is
Research strategy & authoring copilot **layered on Perspicacité**. It consumes a
Perspicacité knowledge base (via MCP) and produces forward-looking, grounded artifacts:
themes, gaps, novelty/opportunity, hypotheses, questions, manuscript/grant drafts,
reviewer simulation, study design, statistics, protocols, reproducibility, dashboards.

## Hard rules
- **Never reimplement Perspicacité-owned capabilities**: literature search, RAG/passage
  retrieval, claim extraction, citation/claim graphs, PRISMA screening. Delegate them.
- **Ground everything**: generated claims/citations must pass `indicium.verify.verify_quote`
  against real corpus passages. Drop/flag unverifiable citations — never assert them.
- **Honest scoring**: every `Score` exposes components, weights, normalization,
  interpretation, uncertainty. No bare magic numbers.
- **Correct domain logic**: statistics use scipy/statsmodels with golden-value tests; name
  the STATO/OBI/EDAM term.

## Standards (internal models + validated exporters)
- Claims/evidence → `indicium` (Claim/Evidence/CitationLink, ECO/CiTO/SEPIO/DoCO/FaBiO + Bucur).
- Findings/insights/provenance → `astra` (ASTRA `Insight`/`Evidence`).
- Agentic runs → `asb_schema` (SciTask Card/Capsule).
Wired as editable siblings via `[tool.uv.sources]`; imports are guarded.

## Commands
- `make install` / `make install-standards`
- `make test` (fast) · `make ci` (lint+typecheck+test) · `uv run pytest -m unit`
- `lfx-insights serve` runs lfx Insights as an MCP server (needs `--extra mcp`); 19 tools.
- Default LLM: `ollama/qwen2.5-coder:7b` (local, no API key). Set `LFX_INSIGHTS_LLM__MODEL` to use a different model.

## Layout
`src/lfx_insights/`: `config` `errors` `logging` `models` · `llm/` · `sources/` (Perspicacité
adapter + Protocol + fake) · `themes/` `scoring/` `generation/` `lifescience/` `reporting/`
`projects/` · `standards/` (exporters + grounding) · `io/` · `pipeline.py` · `cli.py`.

Specs/plans live in `docs/superpowers/` and are **not committed** (local working notes).
