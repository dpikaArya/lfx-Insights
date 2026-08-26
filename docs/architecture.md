# Architecture

lfx Insights is an in-process pipeline of stages over lean pydantic models, with literature supplied
by Perspicacité and outputs exported to the Holobiomics standards.

## Package layout

```
src/lfx_insights/
├── cli.py            # click CLI: per-capability commands + `run` + `serve`
├── config.py         # pydantic-settings + YAML (init > env > yaml)
├── logging.py        # structlog (logs to stderr)
├── models.py         # Paper, Corpus, Theme, Score, Insight, Hypothesis, …
├── corpus_features.py# shared max_year / theme_papers / theme_years / keyword_homogeneity
├── context.py        # RunContext + build_context (real backends or offline fakes)
├── llm/              # litellm client (structured output, disk cache, fallback) + MockLLM
├── sources/          # RetrievalBackend Protocol → Perspicacité MCP adapter + FakeBackend
├── themes/           # embedding clustering + LLM labeling
├── scoring/          # gap/novelty/evidence_strength/opportunity/funding/meta-analysis (+ common kernel)
├── generation/       # hypotheses/questions/manuscript/grant/reviewer-sim (+ grounding common)
├── lifescience/      # study_design/statistics/bioinformatics/protocols/reproducibility/datasets
├── reporting/        # markdown renderers (themes/insights/generation/aggregate/life-science)
├── projects/         # cross-run project database + research memory
├── standards/        # indicium/ASTRA/asb exporters + verify_quote grounding gate
├── aggregate.py      # load this run's artifacts for the aggregation stages
├── pipeline.py       # the stage DAG + run()
└── mcp/              # FastMCP server (lfx-insights serve)
```

## Data flow

1. `build_context` constructs a `RunContext` (settings, retrieval backend, LLM client, embedder,
   output store).
2. `run(topic, ctx, stages)` builds the corpus once via the backend
   (`build_or_select_kb` → Perspicacité `search_literature`), then runs the selected stages.
3. Stages consume the in-memory `Corpus`/`Theme`s and write artifacts (`*.md`, `*.astra.json`,
   `*.indicium.json`) via the `OutputStore`, which **tracks the files written this run** so the
   aggregation stages bundle only the current run (no cross-topic contamination).
4. Aggregation stages (`kb_snapshot`, `explainability`, `dashboard`, `brief`, `capsule`,
   `project`, `memory`) read those tracked artifacts.

## Principles

- **Grounded generation** — see [Standards & grounding](standards.md).
- **Honest scoring** — `Score` carries components, weights, normalization, interpretation, and
  uncertainty; values are clamped to 0–1; "no signal" yields a neutral 0.5 rather than invented
  structure.
- **Correct domain logic** — statistics use scipy/statsmodels with golden-value tests;
  bioinformatics omics detection is assay-aware (RNA-seq ≠ genomics).
- **Resilient I/O** — every external call has a timeout, bounded retries, and a loud failure mode.

## Quality bar

`ruff` (E,F,I,N,W,UP,B,SIM,TCH,RUF), `mypy --strict`, `pytest` with markers
(`unit`/`integration`/`e2e`/`live`/`slow`), coverage gate, pre-commit, and CI.
