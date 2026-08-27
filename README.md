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

## Microsoft Word Integration

lfx Insights integrates with Microsoft Word as a task pane add-in, letting you generate evidence-grounded scientific text with real APA citations directly from within your document.

### What You Can Do in Word

- Generate evidence-supported text from selected content
- Find supporting papers for any claim
- Verify citation accuracy against the knowledge base
- Show evidence for a specific claim
- Generate APA in-text citations from verified references
- Insert approved citations into your document
- Improve selected scientific text
- Run a reviewer check on selected text

### How It Works (Step by Step)

1. **Start the backend** — Open a terminal and start lfx Insights and Ollama:
   ```bash
   ollama serve
   python -m lfx_insights serve
   ```
2. **Open Word** — Launch Microsoft Word 2024 or Microsoft 365 Word.
3. **Open the task pane** — Go to **Insert > Office Add-ins > My Add-ins** and select the lfx Insights add-in. The task pane appears on the right side of your document.
4. **Select text** — Highlight the scientific text you want to work with (e.g., a claim, a paragraph, or a section).
5. **Choose an action** — In the task pane, pick an action:
   - **Generate Text** — lfx Insights retrieves relevant papers, verifies evidence, and Ollama drafts grounded text with APA citations.
   - **Find Citations** — Searches the knowledge base for papers supporting the selected claim.
   - **Verify References** — Checks that all inline citations in the selection reference real papers in the knowledge base.
   - **Show Evidence** — Displays the supporting passages and confidence scores for each claim.
6. **Review** — Read the generated text and evidence. Every citation links to a verified paper. No fabricated references.
7. **Approve & insert** — Click **Approve** to insert the text into your document, or **Revise** to regenerate.

### MS Word 2024 (Desktop)

1. Start the backend (Ollama + lfx Insights).
2. Open Word 2024.
3. Go to **Insert > Office Add-ins > My Add-ins**.
4. Click **Upload Add-in** and select the `manifest.xml` file from your lfx Insights installation.
5. The **lfx Insights** task pane opens. Select text and choose an action.

### MS Word 365 Online

1. Start the backend with HTTPS enabled (required for browser-based Word):
   ```bash
   python -m lfx_insights serve --https --certfile cert.pem --keyfile key.pem
   ```
2. Open Word 365 in your browser at [office.com](https://www.office.com).
3. Go to **Insert > Office Add-ins > My Add-ins**.
4. Click **Upload Add-in** and provide the manifest URL pointing to your HTTPS-hosted add-in.
5. The lfx Insights task pane loads. Select text and choose an action.

> **Note:** Word 365 online requires HTTPS. For local development, use a self-signed certificate trusted by your machine. Do not expose Ollama to the internet — the add-in communicates with the lfx Insights API, which handles LLM calls locally.

### MS Word 365 Desktop (Offline / Local Network)

1. Start the backend (Ollama + lfx Insights).
2. Open Microsoft 365 Word.
3. Go to **Insert > Office Add-ins > My Add-ins**.
4. Click **Upload Add-in** and select the `manifest.xml` file.
5. The task pane connects to your local lfx Insights API.

> **Tip:** For offline environments (air-gapped labs), the entire pipeline runs locally — Ollama for LLM, local embeddings, and the knowledge base stored on disk. No internet connection required after initial setup.

### Word Add-in Features

| Action | What It Does |
|--------|-------------|
| Generate Text | Retrieves papers → verifies evidence → drafts text → attaches APA citations |
| Find Citations | Searches the knowledge base for supporting papers |
| Verify References | Validates all inline citations against the knowledge base |
| Show Evidence | Displays supporting passages and confidence scores |
| Generate Citations | Creates APA in-text citations from verified references |
| Insert Citations | Inserts approved citations into the document |
| Improve Text | Revises selected text for clarity and scientific accuracy |
| Review Check | Identifies weak arguments, missing citations, unsupported claims |

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
