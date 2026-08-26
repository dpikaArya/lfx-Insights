# CLI usage

The `lfx-insights` command groups every capability. Add `--offline` to any topic command for a
network/LLM-free demo. Outputs are written under `outputs/<run>/` and printed as JSON to stdout
(logs go to stderr, so stdout stays pipeable).

## Pipeline

```bash
lfx-insights run --topic "deep learning drug discovery"            # all stages
lfx-insights run --topic "…" --quick                               # themes + scoring + aggregation
lfx-insights run --topic "…" --life-science                        # + study design/stats/omics/...
lfx-insights run --topic "…" --only themes,novelty,hypotheses      # pick stages
lfx-insights run --topic "…" --skip datasets --until novelty       # trim the pipeline
```

`--until`/`--skip` apply to the full pipeline even when no stage-set flag is given; an unknown
stage name is a hard error (not silently dropped).

## Individual capabilities

```bash
lfx-insights themes          --topic "…"
lfx-insights gaps            --topic "…" --gap "no work on X" --gap "Y unexplored"
lfx-insights novelty         --topic "…"
lfx-insights opportunities   --topic "…"
lfx-insights evidence-strength --topic "…"
lfx-insights funding         --topic "…"
lfx-insights meta-analysis   --topic "…"
lfx-insights hypotheses      --topic "…"
lfx-insights questions       --topic "…"
lfx-insights manuscript      --topic "…"
lfx-insights grant           --topic "…"
lfx-insights review          --topic "…"
lfx-insights study-design    --topic "…"
lfx-insights bioinformatics  --topic "…"
lfx-insights reproducibility --topic "…"
lfx-insights datasets        --topic "…"
lfx-insights dashboard       --topic "…"
lfx-insights brief           --topic "…"
```

## Parameter-driven (no corpus needed)

```bash
lfx-insights stats --design two_sample_t --effect-size 0.5    # → n per group = 64
lfx-insights stats --design correlation --effect-size 0.3     # → total n = 85
lfx-insights protocol --kind rna_seq                          # rna_seq | variant_calling | pcr | western_blot
```

## Serve over MCP

```bash
lfx-insights serve                         # stdio
lfx-insights serve --transport http --port 8100
```

See [MCP server](mcp.md).
