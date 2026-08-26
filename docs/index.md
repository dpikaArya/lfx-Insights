# lfx Insights

**A research strategy & authoring copilot, layered on Perspicacité.**

> **Perspicacité** answers *"What does the literature say about X — with sources?"*
> **lfx Insights** answers *"Given what the literature says, what should I do next — and help me design, plan, and write it."*

lfx Insights sits on top of a Perspicacité knowledge base and produces the forward-looking,
generative, decision-support layer: themes, research gaps, novelty and opportunity scores,
grounded hypotheses and research questions, manuscript and grant drafts, peer-review simulation,
study-design and statistics advice, protocols, reproducibility audits, dashboards, and project
tracking — every artifact grounded in real literature and exported to open standards.

## What makes it different

- **Delegates to Perspicacité** for all literature retrieval, RAG, claim extraction, and
  citation graphs (over MCP). lfx Insights does not reimplement search — it *consumes* a KB.
- **Grounded, not fabricated.** Generated citations carry a verbatim quote that
  [`verify_quote`](standards.md) confirms is present in the cited paper; ungrounded citations are
  dropped.
- **Honest scoring.** Every score exposes its components, weights, normalization, interpretation
  band, and uncertainty — no bare magic numbers.
- **Standards-native.** Hypotheses → indicium `Claim`s; findings → ASTRA `Insight`s; runs →
  asb-schema SciTask Capsules. Backed by ECO/CiTO/SEPIO/DoCO/FaBiO + the Bucur SuperPattern, plus
  STATO/OBI/EDAM in the life-science modules.
- **Composable two ways.** lfx Insights *consumes* Perspicacité over MCP, and can itself be
  [*served* over MCP](mcp.md) (19 tools) for other agents.

## Capability map

A run flows **themes → scoring → generation → life-science → aggregation**:

| Layer | Capabilities |
|---|---|
| Themes | discovery + LLM labeling + evolution |
| Scoring (deterministic) | gap validation · novelty · evidence strength · opportunity · funding alignment · meta-analysis readiness |
| Generation (LLM, grounded) | hypotheses · research questions · manuscript · grant · reviewer simulation |
| Life-science | study design (OBI) · statistics (STATO, scipy/statsmodels) · bioinformatics (EDAM) · protocols · reproducibility · datasets |
| Aggregation | knowledge-base snapshot · explainability trace · dashboard · brief · asb SciTask Capsule · project + memory |

See [Installation](installation.md) and [CLI usage](cli.md) to get started.
