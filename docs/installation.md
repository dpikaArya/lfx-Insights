# Installation

## Clone & install

```bash
git clone https://github.com/dpikaArya/lfx-Insights.git
cd lfx-Insights
make install        # uv sync
```

Optional: install the standards exporters (private Holobiomics repos):

```bash
uv sync --extra standards
make install-standards
```

## Run (offline / demo)

```bash
uv run lfx-insights run --topic "deep learning for protein structure" --quick
```

This uses the built-in fake backend: 25–40 synthetic papers, zero network, deterministic local
LLM (MockLLM), and **all** stages complete successfully (including contradiction detection and
research gap validation). The `--quick` flag is shorthand for `--only themes,scoring,gap_validation,contradiction_detection,kb_snapshot,dashboard,brief`.

## Run (live)

1. Start Perspicacité MCP server (default `http://localhost:8002/mcp`)
2. Set `LFX_INSIGHTS_PERSPICACITE__URL=http://localhost:8002/mcp`
3. Set `LFX_INSIGHTS_LLM__MODEL=ollama/qwen2.5-coder:7b` (local Ollama — no API key needed)
4. Run:

```bash
uv run lfx-insights run --topic "immunotherapy resistance mechanisms"
```

## Serve

```bash
uv run lfx-insights serve                         # stdio
uv run lfx-insights serve --transport http --port 8100
uv run lfx-insights serve --offline                # fake backend, no network/LLM
```

## CLI overview

```bash
uv run lfx-insights run --topic "…" [--quick|--life-science] [--skip …] [--until …]
uv run lfx-insights themes --topic "…"
uv run lfx-insights novelty --topic "…"
uv run lfx-insights hypotheses --topic "…"
uv run lfx-insights manuscript --topic "…"
uv run lfx-insights grant --topic "…"
uv run lfx-insights study-design --topic "…"
uv run lfx-insights bioinformatics --topic "…"
uv run lfx-insights reproducibility --topic "…"
uv run lfx-insights datasets --topic "…"
uv run lfx-insights dashboard --topic "…"
uv run lfx-insights brief --topic "…"
uv run lfx-insights review --topic "…"
```

Each prints a JSON summary to stdout and writes artifacts to `outputs/<run>/`.

## What the output contains

| artifact | description |
|---|---|
| Executive summary | TL;DR, key insights, recommended actions |
| Gap report | Claimed gaps → Confirmed / Uncertain / Not Supported |
| Evidence matrix | What supports or contradicts each claimed gap |
| Novelty report | Conceptually novel → highly saturated |
| Opportunity map | High-confidence open research directions |
| Hypothesis bank | Structured hypotheses with priority and IV/DV |
| Research questions | Ranked by novelty, feasibility, funding potential, translational impact |
| Manuscript draft | Introduction, Methods, Discussion with citations |
| Grant concepts | Three proposal ideas, aims, summaries |
| Review simulation | Weak arguments, missing citations, unsupported claims |
| Study design | Design recommendations, sample-size estimates, stats guidance |
| Bioinformatics | Omics detection, datasets, tools, repositories, pipelines |
| Protocols | Wet-lab (PCR, WB) and bioinformatics (RNA-seq, variant calling) checklists |
| Knowledge base snapshot | `kb_snapshot.json` — everything the pipeline found and concluded this run |
| Research memory | `research_memory.json` — trends across topics, stop-research list |
| Dashboard | Aggregated project + memory overview |
