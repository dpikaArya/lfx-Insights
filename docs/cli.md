# CLI usage

The `consilium` command groups every capability. Add `--offline` to any topic command for a
network/LLM-free demo. Outputs are written under `outputs/<run>/` and printed as JSON to stdout
(logs go to stderr, so stdout stays pipeable).

## Pipeline

```bash
consilium run --topic "deep learning drug discovery"            # all stages
consilium run --topic "…" --quick                               # themes + scoring + aggregation
consilium run --topic "…" --life-science                        # + study design/stats/omics/...
consilium run --topic "…" --only themes,novelty,hypotheses      # pick stages
consilium run --topic "…" --skip datasets --until novelty       # trim the pipeline
```

`--until`/`--skip` apply to the full pipeline even when no stage-set flag is given; an unknown
stage name is a hard error (not silently dropped).

## Individual capabilities

```bash
consilium themes          --topic "…"
consilium gaps            --topic "…" --gap "no work on X" --gap "Y unexplored"
consilium novelty         --topic "…"
consilium opportunities   --topic "…"
consilium evidence-strength --topic "…"
consilium funding         --topic "…"
consilium meta-analysis   --topic "…"
consilium hypotheses      --topic "…"
consilium questions       --topic "…"
consilium manuscript      --topic "…"
consilium grant           --topic "…"
consilium review          --topic "…"
consilium study-design    --topic "…"
consilium bioinformatics  --topic "…"
consilium reproducibility --topic "…"
consilium datasets        --topic "…"
consilium dashboard       --topic "…"
consilium brief           --topic "…"
```

## Parameter-driven (no corpus needed)

```bash
consilium stats --design two_sample_t --effect-size 0.5    # → n per group = 64
consilium stats --design correlation --effect-size 0.3     # → total n = 85
consilium protocol --kind rna_seq                          # rna_seq | variant_calling | pcr | western_blot
```

## Serve over MCP

```bash
consilium serve                         # stdio
consilium serve --transport http --port 8100
```

See [MCP server](mcp.md).
