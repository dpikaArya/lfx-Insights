# Agent Log

## 2026-06-12 — Phase 1 (skeleton + vertical slice)
Bootstrapped `lfx_insights` from scratch as a peer of Perspicacité/ASB. Scaffold, tooling,
config/logging/models, litellm client (+mock), Perspicacité MCP adapter (+fake), theme
discovery+labeling, standards grounding (`verify_quote`) + indicium/ASTRA exporters,
reporting, pipeline, CLI. See `docs/superpowers/` for spec + plan.

## 2026-06-12 — Phase 2 (scoring layer)
Added the honest-scoring kernel (`scoring/common.py`) and six deterministic scoring modules
(evidence_strength, novelty, gap_validation, opportunity, funding_alignment,
meta_analysis_readiness) — built in parallel via a workflow, each self-verified. Wired pipeline
stages + CLI subcommands. Scores expose components/weights/uncertainty (no magic numbers);
findings export to ASTRA. 90 tests, mypy --strict clean, coverage 90%. Old-tool deep-review
guardrails captured in docs/superpowers/notes/.

## 2026-06-12 — Phase 3 (generation layer)
LLM-backed, corpus-grounded generation built in parallel via a workflow: hypotheses (→ indicium
draft Claims), questions (scored, no RNG), manuscript, grant, reviewer-sim. Added generation
kernel (APA, citation verification dropping hallucinated refs, grounding, output-leak detection),
new models + indicium Claim exporter, pipeline stages + CLI. Fixed logs→stderr. 134 tests,
mypy --strict clean, coverage 90%.

## 2026-06-12 — Phase 4 (life-science advisory)
Correctness-critical layer built in parallel via a workflow: statistics (scipy/statsmodels,
golden-value tested — d=0.5→64/group, r=0.3→85; domain guards), study_design (OBI, conjunctive
maturity gate vs the old OR bug), bioinformatics (assay-aware: RNA-seq→transcriptomics not
genomics), protocols, reproducibility (weighted 6-dim), datasets (accession discovery). Moved
scipy/statsmodels to core deps. Pipeline stages + CLI (incl. param-driven stats/protocol). 218
tests, mypy --strict clean, coverage 92%.

## 2026-06-12 — Phase 5 (aggregation + projects)
Aggregation layer over a run's artifacts: kb_snapshot (knowledge_base.json), explainability
trace, dashboard, brief, and asb-schema SciTaskCapsule export; cross-run project database +
research memory. Shared run-artifact loader (aggregate.py). Pipeline now has 17 stages; quick &
life-science sets end with aggregation. 221 tests, mypy --strict clean, coverage 91%.

## 2026-06-12 — Live Perspicacité integration
Hardened the Perspicacité backend to speak real MCP streamable-HTTP (initialize -> mcp-session-id
-> notifications/initialized -> tools/call; SSE parsing; unwrap structuredContent.result JSON
string). Corrected tool arg names (kb_name, k) discovered by probing the live server. Added live
tests (tests/live/, -m live) that pass against Perspicacité on :8002: search_literature returns
real papers, scoring runs on the real corpus, passages call path works. Unit suite mocks the full
handshake. 223 unit + 3 live tests green; mypy --strict clean.

## 2026-06-12 — Post-review improvements
Three improvement rounds (each its own workflow + integration): (1) aggregation contamination
fixed at root (OutputStore tracks written; load_run reads only this run); (2) DRY shared
corpus_features adopted across 5 scoring/life-science modules (behavior-preserving); (3) genuine
quote-grounding in generation — LLM emits a verbatim supporting quote per citation, verify_quote
drops ungrounded ones (hypotheses Evidence carries the verified quote). 317 unit + 3 live tests;
mypy --strict; coverage 92%. Tagged v0.1.1.

## 2026-06-12 — MCP server (v0.2.0)
lfx Insights now serves itself over MCP: src/lfx_insights/mcp/server.py (FastMCP, 19 tools wrapping the
pipeline capabilities, structured JSON returns, offline mode), `lfx-insights serve` CLI (stdio/http).
fastmcp added as the public `mcp` extra (wired into CI). In-memory Client tests. Fixed a real bug
(validate_gaps tool shadowed the scoring import -> would self-recurse). 337 unit + 3 live; mypy
--strict; coverage 93%. Tagged v0.2.0.

## 2026-06-12 — Live-audit fixes (v0.2.1)
Ran a live benchmark (scRNA-seq tumor microenvironment, 15 real Perspicacité papers) and fixed
the findings: gap_validation now corpus-scoped + Out-of-Scope off-topic gate (nonsense gap ->
"Out of Scope" not "Confirmed very high", verified live); datasets/reproducibility accept
full_texts + pipeline full_text flag (abstract-only disclosed); theme k capped for small corpora
(7->2 themes live) + evidence-strength caveat for <=3-paper themes. 340 unit + 3 live; mypy
--strict; coverage 92%. Tagged v0.2.1.

## 2026-06-13 — Live full-text audit fixes (v0.2.2)
Verified the full_text path live (Perspicacité paper_content returns real full text, e.g. 59k
chars). Found + fixed a real recall bug: datasets only matched GSE; added GSM/GDS (GEO), dbGaP
(phs), EGA (EGAS/EGAD); reproducibility data-availability broadened too. Live proof: GSM9116755
found via full text (abstract-only=0). 346 unit + 3 live; mypy --strict; coverage 92%. Tagged v0.2.2.

## 2026-06-13 — gap off-topic gate relativised + generalist embedding tier (v0.3.0)
gap_validation: off-topic floor now relative to corpus coherence (median per-paper max-sim),
embed-once, embedder-agnostic; verdict flips with coherence (tested). Embedding tiers: added
LiteLLMEmbedder (text-embedding-3-large / <provider>/<model>) routed by name like Perspicacité;
HF org/model stays local sentence-transformers (fixed a "/"-routing bug caught by a test). 350
unit + 3 live; mypy --strict; coverage 93%. Tagged v0.3.0.
