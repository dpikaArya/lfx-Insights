# Consilium

**A research strategy & authoring copilot, layered on [Perspicacité](https://github.com/HolobiomicsLab).**

> Perspicacité answers *"What does the literature say about X — with sources?"*
> **Consilium answers *"Given what the literature says, what should I do next — and help me design, plan, and write it."***

Consilium is the forward-looking layer that sits on top of a Perspicacité knowledge
base: it discovers themes, validates research gaps, scores novelty and opportunity,
generates grounded hypotheses and research questions, drafts manuscript and grant
sections, simulates peer review, advises on study design and statistics, and tracks
projects — every artifact grounded in real literature and exported to open standards.

## What makes it different

- **Delegates to Perspicacité** for all literature retrieval, RAG, claim extraction,
  and citation graphs. Consilium does *not* reimplement search — it consumes a
  Perspicacité knowledge base over the MCP interface.
- **Grounded, not fabricated.** Every generated claim/citation is verified against a
  real corpus quote via indicium's `verify_quote` gate. Unverifiable citations are
  dropped, never asserted.
- **Honest scoring.** No magic numbers: every score exposes its components, weights,
  normalization, interpretation band, and uncertainty.
- **Standards-native outputs.** Hypotheses → indicium `Claim`s; findings → ASTRA
  `Insight`s; runs → asb-schema SciTask Cards/Capsules. Backed by ECO/CiTO/SEPIO/
  DoCO/FaBiO + the Bucur SuperPattern, plus STATO/OBI/EDAM in the life-science modules.

## Capabilities

A run flows **themes → scoring → generation → life-science → aggregation** (17 stages):

- **Themes** — embedding clustering (seeded) + LLM labeling + evolution.
- **Scoring** (deterministic, honest) — gap validation, novelty, evidence strength,
  opportunity ranking, funding alignment, meta-analysis readiness. Every score exposes
  its components, weights, normalization, and uncertainty.
- **Generation** (LLM, grounded) — hypotheses (→ indicium draft Claims), research
  questions, manuscript & grant sections, reviewer simulation. Citations are verified
  against the corpus; hallucinated references are dropped.
- **Life-science** (correctness-first) — statistics (scipy/statsmodels power & sample
  size, golden-value tested), study design (OBI), bioinformatics omics→repository
  mapping (EDAM, assay-aware), protocols, reproducibility audit, dataset discovery.
- **Aggregation** — knowledge-base snapshot, explainability trace, dashboard, research
  brief, asb-schema SciTask Capsule, cross-run project database + research memory.

Outputs are exported to the Holobiomics standards: **indicium** (claims/evidence),
**ASTRA** (insights/provenance), **asb-schema** (agentic capsules).

## Status

v0.1.0 — full pipeline implemented, tested (220+ tests, mypy --strict, ~91% coverage).
See `docs/superpowers/` for the design spec and plans.

## Local setup (no API keys required)

Consilium runs fully locally with Ollama for the LLM and local sentence-transformers
for embeddings. No external API keys are needed.

### Prerequisites

1. **Ollama** — [install](https://ollama.com/download), then pull a model:
   ```bash
   ollama serve                                  # start Ollama (if not running as a service)
   ollama pull qwen2.5-coder:7b                  # or any model you prefer
   ollama list                                   # verify your installed models
   ```
2. **Perspicacité** — running locally at `http://localhost:8002/mcp` (its own setup).
3. **Python >=3.12** and **uv**.

### Quick start

```bash
uv sync --extra dev
consilium run --topic "deep learning drug discovery" --config config.local.yml
```

### Configuration

Copy `config.local.yml` to `config.yml` or pass `--config config.local.yml`. Key settings:

```yaml
llm:
  model: ollama/qwen2.5-coder:7b     # local Ollama model
  fallback: []                         # no external fallback
  ollama_base_url: "http://localhost:11434"

embedding:
  model: all-MiniLM-L6-v2            # local sentence-transformers (default)

perspicacite:
  url: "http://localhost:8002/mcp"   # local Perspicacité
```

Override any setting via environment variables:

```bash
CONSILIUM_LLM__MODEL=ollama/llama3.2 consilium run --topic "..."
```

### How local mode differs from --offline

| Mode | LLM | Embeddings | Literature retrieval | API keys |
|---|---|---|---|---|
| **Local** (default) | Ollama (local) | sentence-transformers (local) | Perspicacité MCP (local) | None |
| **--offline** | MockLLM (deterministic fake) | TF-IDF (deterministic) | FakeBackend (4 hardcoded papers) | None |

Local mode uses real models and real retrieval. Offline mode uses fakes for CI/testing.

## Install

```bash
uv sync --extra dev               # core
uv sync --extra dev --extra standards   # + indicium/ASTRA/asb-schema (local siblings)
uv sync --extra dev --extra docx        # + python-docx (for `consilium export-docx`)
```

## Usage

```bash
# Full pipeline
consilium run --topic "deep learning drug discovery"                 # all 17 stages
consilium run --topic "…" --quick                                    # themes + scoring + aggregation
consilium run --topic "…" --life-science                             # + study design/stats/omics/...
consilium run --topic "…" --only themes,novelty,hypotheses           # pick stages

# Individual capabilities (each also accepts --offline for a network/LLM-free demo)
consilium themes        --topic "…"
consilium gaps          --topic "…" --gap "no work on X" --gap "Y unexplored"
consilium hypotheses    --topic "…"
consilium manuscript    --topic "…"
consilium dashboard     --topic "…"

# Export a drafted manuscript/grant to an APA .docx (needs the [docx] extra; decoupled
# from generation — reads the run's <artifact>.sections.json, no LLM/corpus needed)
consilium manuscript --topic "…" --offline --output-dir outputs
consilium export-docx --run outputs/default --artifact manuscript    # → outputs/default/manuscript.docx

# Parameter-driven (no corpus needed)
consilium stats --design two_sample_t --effect-size 0.5              # → n per group = 64
consilium protocol --kind rna_seq
```

Requires a running Perspicacité MCP server (default `http://localhost:8002/mcp`) unless
`--offline` is used. If it is unreachable, Consilium fails loudly rather than guessing.

## Embeddings

Consilium's own embedder (theme clustering + gap validation) is set by `embedding.model`:

| `embedding.model` | Tier | API key? |
|---|---|---|
| `tfidf` | deterministic local TF-IDF (no network; offline/CI) | No |
| `all-MiniLM-L6-v2` *(default)* | fast local sentence-transformers | No |
| `allenai/specter2`, `malteos/scincl`, `NeuML/pubmedbert-base-embeddings` | stronger **local** scientific models (recommended for papers) | No |
| `text-embedding-3-large` (or `cohere/…`, `voyage/…`) | **generalist hosted tier** via LiteLLM — best cross-domain | Yes (provider key) |

(Literature *retrieval* embeddings are Perspicacité's concern, configured there.)

## Consilium as an MCP server

Consilium can also be *served* over MCP, so other agents consume it the way it consumes
Perspicacité. It exposes 19 tools (themes, gaps, novelty, opportunities, hypotheses,
questions, manuscript, grant, study-design, sample-size, protocol, …) returning structured
JSON.

```bash
uv sync --extra mcp
consilium serve                       # stdio transport
consilium serve --transport http --port 8100
consilium serve --offline             # in-memory fakes (demo, no network/LLM)
```

## Evaluation (ScholarQABench)

Consilium includes a harness to evaluate the **joint Perspicacité→Consilium pipeline**
on a [ScholarQABench](https://github.com/AkariAsai/ScholarQABench)-style task (Asai et
al. 2024): retrieve literature, synthesise a citation-grounded answer, and score both
citation faithfulness and answer quality. The headline experiment is a **retrieval
ablation** — `null` (closed-book) vs `tfidf` (generic baseline) vs `perspicacite` (open
retrieval) — that isolates Perspicacité's lift while holding synthesis fixed.

```bash
consilium eval scholarqa --offline --conditions null,tfidf --judge lexical   # CI demo
consilium eval scholarqa --dataset path/to/scholarqa.jsonl \
  --conditions null,tfidf,perspicacite --judge llm --max-cases 50            # real run
```

The loader auto-detects three SOTA shapes — **ScholarQABench**, **ExpertQA** (expert
multi-domain attributed QA), and **LitSearch** (scientific-retrieval, gold corpus ids) —
so one harness covers the joint pipeline, expert-attribution breadth, and intrinsic
retrieval. Metrics: citation precision/recall/F1 (AutoAIS scheme, pluggable judge:
lexical / indicium-grounding / LLM), `match` + ROUGE-L correctness, a Prometheus-style LLM
quality judge, and `recall@k`/`nDCG@k` retrieval. It is a scaffold, not a paper
reproduction — see [`docs/eval.md`](docs/eval.md) for the disclosed caveats, and
[`docs/eval-handson.md`](docs/eval-handson.md) for a turnkey runbook (data downloads,
recommended flags, the independent-judge cross-check) to run a full eval on another machine.

## License

Apache-2.0.
