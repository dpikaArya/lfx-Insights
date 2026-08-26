# MCP server

lfx Insights can be *served* over MCP so other agents consume it the way it consumes Perspicacité.

```bash
uv sync --extra mcp
lfx-insights serve                          # stdio transport
lfx-insights serve --transport http --port 8100
lfx-insights serve --offline                # in-memory fakes (demo / no network / no LLM)
```

## Tools

`build_server(offline=…)` registers 19 tools, each returning structured JSON:

| Tool | Returns |
|---|---|
| `themes` | discovered + labeled themes |
| `validate_gaps` | gap-validation insights (Confirmed/Uncertain/Not Supported) |
| `novelty`, `opportunities`, `evidence_strength`, `funding_alignment`, `meta_analysis_readiness` | scoring insights |
| `hypotheses` | grounded hypotheses (quote-verified citations) |
| `questions` | ranked research questions |
| `manuscript`, `grant` | drafted sections |
| `reviewer_simulation` | review comments |
| `study_design`, `bioinformatics`, `reproducibility`, `datasets` | life-science insights |
| `sample_size` | power / sample-size recommendation (design, effect_size, alpha, power, groups) |
| `protocol`, `list_protocols` | protocol checklist / available templates |

## Consuming it

```python
from fastmcp import Client
from lfx_insights.mcp import build_server

async with Client(build_server(offline=True)) as client:
    result = await client.call_tool("sample_size", {"design": "two_sample_t", "effect_size": 0.5})
    assert result.data["n_per_group"] == 64
```

Corpus-backed tools accept `max_papers` (default 30) and require a reachable Perspicacité server
unless the server was built with `offline=True`.
