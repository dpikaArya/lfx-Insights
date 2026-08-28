# lfx Insights

A local AI-powered research intelligence platform for literature discovery, evidence synthesis, research gap analysis, hypothesis generation, manuscript development, citation validation, self-evaluation, and adaptive learning — all running entirely on your machine with Ollama.

## Key Features

### Literature Discovery & Theme Analysis
- **Multi-Source Literature Retrieval** — Search papers across Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, and CORE in parallel. Deduplicates by DOI and semantic title similarity.
- **Adaptive Theme Discovery** — Clusters papers into research themes using MiniLM embeddings. Falls back to multi-method consensus (NMF + hierarchical + fixed-k) for small corpora.
- **Evidence Extraction & Synthesis** — Extracts objectives, methods, results, limitations from abstracts. Compares findings across studies to detect consensus and disagreement.
- **Citation Intelligence** — Builds directed citation graphs via OpenAlex API. Computes PageRank, HITS, identifies foundational papers and hidden gems.
- **Scientific Claim Graph** — Extracts claims from abstracts and builds a directed evidence graph (supporting / contradictory) for RAG applications.

### Scoring & Validation (Deterministic)
- **Research Gap Validation** — Validates claimed gaps by searching the corpus with semantic similarity. Assigns confidence scores (Confirmed / Uncertain / Not Supported).
- **Contradiction Detection** — Detects opposing claims across the literature using 15 semantic opposition pairs.
- **Novelty Scoring** — Estimates topic saturation, publication density, and emerging concept presence to classify themes from Highly Novel to Highly Saturated.
- **Evidence Strength Assessment** — Evaluates the quality and consistency of evidence supporting each theme.
- **Opportunity Ranking** — Identifies high-confidence open research directions with actionable next steps.
- **Funding Alignment** — Maps research opportunities to funding priorities and grant potential.
- **Meta-Analysis Readiness** — Assesses whether the corpus contains comparable studies suitable for quantitative synthesis.

### Generation & Writing (LLM-Powered, Evidence-Grounded)
- **Hypothesis Generation** — Produces structured, reproducible hypothesis banks with priority scoring, IV/DV specification, and methodology suggestions.
- **Research Question Optimization** — Generates 20 research questions per topic ranked by novelty, feasibility, funding potential, and translational impact.
- **Manuscript Generation** — Drafts introduction, literature review, methods, and discussion sections with APA-formatted inline citations grounded in the knowledge base.
- **Grant Proposal Generation** — Drafts grant concepts, specific aims, and project summaries from research gaps and opportunity rankings.
- **Reviewer Simulation** — Evaluates manuscript drafts for weak arguments, missing citations, unsupported claims, and methodological concerns.

### Citation Validation & Evidence Grounding
- **Citation Accuracy Validation** — Validates that every inline citation references an existing paper in the knowledge base. Detects phantom references, mismatched metadata, and invented authors.
- **Evidence Chain Tracking** — Maps each claim in the manuscript to its supporting papers, verifying the full evidence chain from claim to source.
- **Reference List Generation** — Generates formatted APA reference lists from cited papers with proper formatting (DOI, journal, volume, pages).
- **Manuscript-Wide Citation Audit** — Runs comprehensive validation across all sections, reporting total citations, coverage, and issues.

### Self-Evaluation & Learning
- **Evidence-Grounded Self-Evaluation** — Evaluates research outputs with task-specific profiles (manuscript, hypothesis, research question, grant, reviewer, gap validation). Scores across multiple dimensions with evidence-weighted aggregation.
- **Learning Signal Extraction** — Extracts learning signals from research outputs and pipeline runs. Tracks signal strength, confidence, and decay over time.
- **Research Memory & Trends** — Maintains a research memory across runs, tracking topic trends, signal strength, and detecting emerging patterns.
- **Gap Evolution Tracking** — Records and tracks how research gaps evolve over time with confidence scores and paper counts.
- **Adaptive Configuration** — Allowlisted parameter adaptation based on learning signals with versioning and rollback capabilities.
- **Approval Governance** — Human approval workflow with states (pending, approved, rejected, revision_needed) for research outputs before they are used in downstream processes.

### Life-Science Advisory
- **Study Design Advisor** — Recommends experimental designs, controls, sample sizes, statistical tests, and validation strategies based on theme maturity.
- **Bioinformatics Mode** — Detects omics data types (genomics, transcriptomics, proteomics, metabolomics, epigenomics, metagenomics), maps to repositories (GEO, SRA, ArrayExpress, ProteomeXchange, MetaboLights), and recommends pathway tools.
- **Statistical Consultant** — Recommends statistical tests for 6 design types, estimates sample size via normal approximation, computes post-hoc power.
- **Protocol Generation** — Generates lab protocols (PCR, Western blot) and bioinformatics pipelines (RNA-seq, variant calling) with QC checklists.
- **Reproducibility Auditing** — Scores each paper across 6 dimensions: data availability, code availability, sample size, statistical rigor, validation strategy, and controls.
- **Dataset Discovery** — Discovers and catalogs relevant datasets from papers with accession numbers and repository mappings.

### Project Management & Explainability
- **Research Dashboard** — Aggregates active projects, themes, gaps, datasets, manuscripts, grants, and alerts into a single-page overview.
- **Semantic Alerts** — Compares knowledge base snapshots between runs to detect new themes, theme shifts, and confidence changes.
- **Explainability Tracing** — Every output includes evidence source, confidence score, supporting papers, alternative interpretations, and limitations.
- **Project Tracking** — Maintains a cross-run project database tracking topic progress, stages completed, and key findings.

### Pipeline & Infrastructure
- **42-Stage Pipeline** — Orchestrates all modules in dependency order with `--quick` (17 priority modules), `--life-science` (adds bioinformatics modules), `--skip`, and `--until` flags.
- **Local LLM via Ollama** — All LLM inference runs locally through Ollama. No API keys needed. MockLLM for offline/testing.
- **MCP Server** — Can be served as an MCP server (19 tools) for consumption by other agents.
- **Evidence-Grounded APA Citations** — Every citation originates from a verified bibliographic record. Citations are validated against the corpus before inclusion. Never fabricates references.

## System Requirements

- **Python 3.12+**
- **8 GB RAM minimum** (16 GB recommended)
- ~2 GB disk for the sentence-transformers model cache
- **Ollama** for local LLM inference (no API key needed)

## Installation

```bash
git clone https://github.com/dpikaArya/lfx-Insights.git
cd lfx-Insights

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Quick Start

### 1. Start Ollama

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

### 2. Run lfx Insights

```bash
python run_validation.py
```

This runs the 17 priority pipeline modules on a default biomedical query. Outputs are written to `outputs/lfx_Insights/`.

### 3. Run the Full Pipeline

```bash
python -m lfx_insights.pipeline
```

Executes all 42 stages in order. On a 21-paper corpus this completes in ~5 minutes depending on API call latency.

## Web UI (Local Browser Interface)

A minimal, self-contained browser interface (plain HTML/CSS/JS served by the API —
no React/Vue/build step) for asking the local model and inspecting evidence and
APA references. It reuses the existing `POST /api/insights` endpoint, the existing
LLM client, and the local knowledge base. Everything runs on one local service.

### Start

```bash
cd "E:\lfx-Insights\lfx-Insights"
.\start_web_ui.ps1
```

This starts the LFX Insights API (reusing it if already running) and reuses your
local Ollama. It does **not** start another Ollama service and does **not** start
the Office Add-in HTTPS server.

### Use

1. Open: **http://127.0.0.1:8000/**
2. Enter a research question (or paste text for Improve / Review / Verify).
3. Choose **Ask**, **Improve**, **Review**, **Gap**, **Evidence**, **Citations**, or **Verify**.
4. Read the result, evidence, and APA references in the page.

The connection status shows **Connected**, **Ollama unavailable**, or **API unavailable**.

## Usage

### Pipeline Modes

```bash
# Quick mode — 17 high-value modules (no paper retrieval, PDF, or full-document stages)
python src/pipeline.py --quick

# Life-science mode — enables bioinformatics, study design, statistical, and protocol modules
python src/pipeline.py --life-science

# Skip specific stages
python src/pipeline.py --skip pdf_manager citation_network_analysis

# Run up to a specific stage
python src/pipeline.py --until hypothesis_generator
```

### Individual Modules

Every module can be run independently:

```bash
python src/citation_intelligence.py
python src/contradiction_detector.py
python src/hypothesis_generator.py
python src/manuscript_copilot.py
python src/research_brief.py
```

Most modules accept `--papers`, `--consensus`, `--knowledge-base` or similar arguments to specify input files. Run any module with `--help` to see its options.


## Evidence-Grounded APA Citations

lfx Insights follows a strict citation workflow:

1. You select or enter scientific text.
2. lfx Insights retrieves relevant papers from the knowledge base.
3. Evidence is checked against the corpus.
4. Ollama generates scientific text from verified evidence.
5. Verified references are attached from knowledge base records.
6. APA in-text citations are generated deterministically (e.g., `(Smith, 2024)`, `(Smith & Kumar, 2025)`, `(Smith et al., 2024)`).
7. Citations are validated against the knowledge base.
8. You review the evidence and citations.
9. You approve the text before insertion.

Every citation must originate from an existing verified bibliographic record. The system never fabricates authors, titles, journals, years, DOIs, or references.

## Outputs

All generated files are organized under `outputs/lfx_Insights/`:

| Directory | Contents |
|-----------|----------|
| `reports/` | Executive summaries, research briefs, gap reports, hypothesis banks |
| `evidence/` | Evidence matrices, synthesis reports |
| `knowledge_base/` | Machine-readable JSON snapshots, claim graphs |
| `references/` | APA citation support files |
| `dashboard/` | Aggregated research dashboard, research memory |
| `manuscript/` | Generated manuscript drafts |
| `grants/` | Grant proposal drafts |
| `protocols/` | Lab and bioinformatics protocol checklists |
| `statistics/` | Sample size estimates, power analyses |
| `bioinformatics/` | Omics dataset reports |
| `citation_network/` | Network analysis reports |
| `pdf_library/` | PDF library indexes |
| `figures/` | Figure reference catalogs |
| `tables/` | Table reference catalogs |
| `alerts/` | Semantic change detection reports |
| `explainability/` | Evidence trace reports |
| `projects/` | Project tracking databases |
| `self_evaluation/` | Evaluation scores and learning signals |
| `approval/` | Approved research outputs |
| `adaptive_config/` | Configuration version history and rollback |

## Capabilities Matrix

| Capability | Module | Deterministic | LLM-Required | Offline-Safe |
|------------|--------|---------------|--------------|--------------|
| **Themes** | `themes/` | ✓ (embeddings) | ✓ (labeling) | ✓ (MockLLM) |
| **Evidence Strength** | `scoring/evidence_strength` | ✓ | — | ✓ |
| **Novelty** | `scoring/novelty` | ✓ | — | ✓ |
| **Opportunity** | `scoring/opportunity` | ✓ | — | ✓ |
| **Funding Alignment** | `scoring/funding_alignment` | ✓ | — | ✓ |
| **Meta-Analysis** | `scoring/meta_analysis_readiness` | ✓ | — | ✓ |
| **Gap Validation** | `scoring/gap_validation` | ✓ | — | ✓ |
| **Hypotheses** | `generation/hypotheses` | — | ✓ | ✓ (MockLLM) |
| **Questions** | `generation/questions` | — | ✓ | ✓ (MockLLM) |
| **Manuscript** | `generation/manuscript` | ✓ (citations) | ✓ (drafting) | ✓ (MockLLM) |
| **Grant** | `generation/grant` | ✓ (citations) | ✓ (drafting) | ✓ (MockLLM) |
| **Reviewer Sim** | `generation/reviewer_sim` | — | ✓ | ✓ (MockLLM) |
| **Citation Validation** | `generation/common` | ✓ | — | ✓ |
| **Study Design** | `lifescience/study_design` | ✓ | — | ✓ |
| **Bioinformatics** | `lifescience/bioinformatics` | ✓ | — | ✓ |
| **Statistics** | `lifescience/statistics` | ✓ | — | ✓ |
| **Protocols** | `lifescience/protocols` | ✓ | — | ✓ |
| **Reproducibility** | `lifescience/reproducibility` | ✓ | — | ✓ |
| **Datasets** | `lifescience/datasets` | ✓ | — | ✓ |
| **Self-Evaluation** | `projects/self_evaluation` | ✓ (scoring) | ✓ (reasoning) | ✓ (MockLLM) |
| **Learning** | `projects/learning` | ✓ | — | ✓ |
| **Gap Evolution** | `projects/gap_evolution` | ✓ | — | ✓ |
| **Approval** | `projects/approval` | ✓ | — | ✓ |
| **Adaptive Config** | `projects/adaptive_config` | ✓ | — | ✓ |
| **Dashboard** | `reporting/aggregate_report` | ✓ | — | ✓ |
| **Brief** | `reporting/aggregate_report` | ✓ | — | ✓ |
| **Explainability** | `reporting/aggregate_report` | ✓ | — | ✓ |
| **KB Snapshot** | `aggregate` | ✓ | — | ✓ |
| **Project Tracking** | `projects/project_manager` | ✓ | — | ✓ |
| **Research Memory** | `projects/research_memory` | ✓ | — | ✓ |

## Major Modules

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `search_papers.py` | Multi-source literature retrieval | Query string | `search_results.csv` |
| `cluster_themes.py` | Unsupervised theme discovery | `search_results.csv` | `consensus_themes.csv`, clustering reports |
| `generate_reports.py` | Executive summary, gaps, knowledge base | `consensus_themes.csv`, embeddings | Reports, `knowledge_base.json`, RAG chunks |
| `citation_intelligence.py` | Citation graph, PageRank, HITS | `search_results.csv` | Citation metrics, foundational papers |
| `hypothesis_generator.py` | Structured hypothesis bank | Knowledge base, gaps | `hypothesis_bank.csv` |
| `manuscript_copilot.py` | Draft manuscript sections | Evidence matrix, knowledge base | `manuscript_draft.md` |
| `contradiction_detector.py` | Cross-paper claim contradictions | `search_results.csv`, themes | `contradictory_findings.md` |
| `research_gap_validator.py` | Gap validation with confidence | Papers, evidence, existing gaps | `gap_confidence_scores.csv` |
| `study_design_advisor.py` | Design recommendations | Knowledge base, evidence strength | `study_design_report.md` |
| `bioinformatics_mode.py` | Omics data detection | `search_results.csv` | `bioinformatics_report.md` |
| `statistical_consultant.py` | Test selection, power analysis | CLI parameters | `statistical_report.md`, sample size estimates |
| `pipeline.py` | 42-stage orchestrator | All upstream outputs | Pipeline summary |

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data processing and CSV I/O |
| numpy | Numerical computing |
| scikit-learn | Clustering (NMF, hierarchical), metrics |
| sentence-transformers | MiniLM text embeddings for semantic similarity |
| networkx | Citation graph construction and analysis |
| scipy | Spatial distance computations |
| requests | HTTP API calls to Crossref, OpenAlex, Semantic Scholar |
| tqdm | Progress bars for API-heavy operations |

PDF backends (`pypdf`, `pdfplumber`, `pymupdf`) are optional and only needed for PDF figure/table extraction and PDF management.

## Use Cases

- **Literature review automation** — Search, cluster, and synthesize papers on any research topic
- **Research gap identification** — Detect and validate underexplored areas with confidence scoring
- **Manuscript preparation** — Generate drafts with inline citations and peer-review simulation
- **Grant writing** — Produce proposal components from gap and opportunity analyses
- **Bioinformatics exploration** — Identify omics datasets and recommend analysis pipelines
- **Reproducibility assessment** — Audit papers for data/code availability and statistical rigor
- **Citation validation** — Verify all citations are grounded in real papers before publication
- **Self-evaluation** — Assess research outputs with evidence-grounded scoring
- **Adaptive learning** — Track research progress and adapt configuration based on signals
- **Word document authoring** — Generate evidence-grounded scientific text with real APA citations directly in Microsoft Word

## License

MIT License

## Citation

If you use this software in your research, teaching, or publications, please cite:

Arya, D. (2026). lfx Insights (Version 2.0) [Computer software]. GitHub. https://github.com/dpikaArya/lfx-Insights

### BibTeX

@software{arya2026lfx,
  author = {Arya, D.},
  title = {lfx Insights},
  year = {2026},
  version = {2.0},
  url = {https://github.com/dpikaArya/lfx-Insights}
}
