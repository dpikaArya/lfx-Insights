# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to SemVer.

## [0.8.0]

### Added — grounded APA `.docx` export
- **`consilium export-docx`** — render a grounded manuscript/grant section artifact to an
  APA-styled `.docx`. Fully decoupled from generation: it reads a persisted
  `<artifact>.sections.json` from a run directory and never calls an LLM or loads the
  corpus/KB. Reuses the existing `format_apa` formatter (no parallel citation logic); the
  reference list is the papers actually cited across the drafted sections (deduped,
  first-citation order), rendered alphabetically with hanging indents.
- The `manuscript` and `grant` stages now persist `<artifact>.sections.json` beside their
  markdown (a `SectionBundle`: sections + cited papers), enabling the decoupled exporter.
- `python-docx` is an **optional `[docx]` extra** (`pip install consilium[docx]`), imported
  lazily and guarded — the core install and CLI never require it; `export-docx` without the
  extra fails with a clear install hint, not a traceback.
- 413 unit tests (8 new), mypy --strict clean, ruff clean; the `.docx` tests skip cleanly
  when the extra is absent.

## [0.7.0]

### Added — cite-as-you-write + configurable generation judge (follow-up to the x4b finding)
- **Cite-as-you-write** generation self-grounding (`_selfground`): in addition to *dropping*
  misattributed citations (precision), the answerer now *attributes* a citation to any
  scorable sentence left uncited, choosing the first source that entails it (recall,
  ALCE-style). Targets the citation-recall bottleneck the oracle analysis exposed (~0.48
  recall even with perfect retrieval).
- **Configurable generation judge** (`eval.generation_judge` / `--generation-judge`,
  lexical|grounding|llm): the x4b run showed a *lexical* self-grounding gate scored by an
  *LLM* judge drops good paraphrased citations and hurts F1; matching the gate to a semantic
  (llm) judge avoids that. Default stays lexical (cheap, deterministic).
- 442 unit+e2e tests, mypy --strict clean, ruff clean, coverage 93.7%.

## [0.6.0]

### Added — eval improvements driven by the x4 breakdown
- **Oracle retrieval condition** (`OracleBackend`): perfect retrieval that feeds each case's
  own gold contexts — the ablation ceiling, isolating generation/citation quality from
  retrieval quality. The breakdown showed ExpertQA's "provided evidence" was letting `tfidf`
  masquerade as a retrieval win; `oracle` makes that explicit.
- **Entailment-gated generation** (`--ground-generation` / `eval.ground_generation`):
  `answer_question` gains an optional entailer; when set, an inline `[n]` is kept only if its
  cited passage *entails* the sentence (a cheap lexical gate, CRAG/Self-RAG-style
  self-verification), dropping misattributed citations before they are scored. Targets the
  dominant "cited-but-unentailed" failure mode (3/8 tfidf cases) found in the breakdown.
- Per-benchmark breakdown finding (11-case x4 run, gpt-4o-mini, LLM judge): on **ScholarQA**
  (open retrieval) Perspicacité beats tfidf on citation F1 (0.58 vs 0.36, precision 0.76 vs
  0.40) and is more consistent (no zero-F1 cases vs tfidf's three).
- 441 unit+e2e tests, mypy --strict clean, ruff clean, coverage 93.7%.

## [0.5.0]

### Added — more SOTA benchmarks in the eval harness (ExpertQA, LitSearch)
- **Format auto-detection** in the dataset loader: one harness now covers three benchmark
  shapes, detected per record — ScholarQABench, **ExpertQA** (Malaviya et al., NAACL 2024;
  expert multi-domain attributed QA — human-revised answer as reference, claim evidence as
  contexts), and **LitSearch** (Ajith et al., EMNLP 2024; `{query, corpusids}` scientific
  literature retrieval). Bundled fixtures `expertqa-bundled` / `litsearch-bundled` for the
  offline path.
- **Intrinsic retrieval metric** (`eval/metrics/retrieval.py`): `recall@k` and `nDCG@k`
  under binary relevance — the direct retrieval measure ScholarQABench cannot give.
  Identifier-agnostic matching (corpus id / DOI / normalised title) so it scores the
  `tfidf` and `perspicacite` conditions; wired into the runner, per-condition honest
  `Score`, report table, and CLI summary. Verified: LitSearch ablation gives `null` recall
  0.0 vs `tfidf` 1.0 on the fixture.
- Schemas verified from source (ExpertQA repo / LitSearch HF dataset). 510 unit+e2e tests,
  mypy --strict clean, ruff clean.

## [0.4.0]

### Added — ScholarQABench evaluation harness (`consilium.eval`, `consilium eval scholarqa`)
- Evaluates the **joint Perspicacité→Consilium pipeline** (Asai et al. 2024,
  arXiv:2411.14199): retrieve → synthesise a citation-grounded answer → score citation
  faithfulness + answer quality. Headline experiment is a **retrieval ablation**
  (`null` / `tfidf` / `perspicacite`) isolating Perspicacité's lift with the synthesis
  layer fixed.
- New `answer_question` entrypoint: long-form answers with inline `[n]` citations over a
  numbered, quote-grounded context (reuses the manuscript grounding gate — ungrounded
  citations are dropped and their markers stripped).
- Metrics: citation precision/recall/F1 (AutoAIS scheme, reimplemented faithfully from
  the ScholarQABench source; **pluggable** entailment judge — `lexical` deterministic /
  `grounding` via indicium `verify_quote` / `llm`); `match` + ROUGE-L correctness
  (in-repo, no new deps); a reference-guided LLM quality judge approximating Prometheus.
  Per-condition aggregates are honest `Score`s; the report emits the disclosed caveats
  (judge differs from AttrScore/TRUE-NLI; corpus ≠ peS2o; quality ≈ Prometheus).
- Bundled offline fixture + `--offline` path (deterministic, CI-safe); `EvalSettings`
  config block; `docs/eval.md`. Verified live against Perspicacité :8002.

## [0.3.1]

### Improved (live OpenAI-embedding audit)
- gap_validation: the **"related" threshold** (the Confirmed/Uncertain boundary, and what
  counts a paper as related to the gap) is now **relative to corpus coherence**
  (``max(0.30, 0.6 * coherence)``), matching the off-topic floor. Previously fixed at an
  absolute 0.30, calibrated for TF-IDF; on the OpenAI ``text-embedding-3-large`` scale
  (coherence ~0.61) it saturated — every paper scored above 0.30, so ``n_related`` pinned at
  N/N and the verdict boundary was meaningless. Now the threshold rises to ~0.37 there;
  verified live that ``n_related`` drops to 12–13/15 and verdicts discriminate on both
  embedders. TF-IDF behaviour is unchanged (low coherence → falls back to the 0.30 floor).

## [0.3.0]

### Added
- **Generalist hosted embedding tier** — set ``embedding.model`` to
  ``text-embedding-3-large`` (or any ``<provider>/<model>``) to route theme clustering and
  gap validation through LiteLLM, mirroring Perspicacité's embedding layer. Plain
  ``org/model`` Hugging Face names stay local sentence-transformers. Live-verified with a
  real OpenAI key (3072-dim).

### Improved
- gap_validation: the **off-topic floor** is now relative to corpus coherence
  (``max(0.12, 0.5 * coherence)``) and embeds gaps + papers once, so the gate adapts across
  embedders whose cosine magnitudes differ (TF-IDF vs sentence-transformers vs OpenAI).

## [0.2.2]

### Fixed (from a live full-text audit)
- datasets: recognise GEO **samples/datasets** (``GSM``/``GDS``, not just ``GSE``) plus
  **dbGaP** (``phs``) and **EGA** (``EGAS``/``EGAD``). Verified live: the full-text path now
  finds ``GSM9116755`` on a real paper that abstract-only + the old GSE-only pattern missed.
- reproducibility: data-availability detection broadened to the same accession set.
- Confirmed the pipeline `full_text` path delivers against the live Perspicacité server
  (paper_content returns real full text; accession recall 0 -> found).

## [0.2.1]

### Improved (from a live benchmark audit)
- gap_validation: verdicts are scoped to "this corpus" (no "genuinely open" over-claim);
  a new **Out of Scope** verdict + off-topic floor flags gaps with near-zero similarity to
  the whole corpus (a nonsense/off-domain gap no longer scores "Confirmed, very high" — it
  is flagged off-topic with high uncertainty).
- datasets + reproducibility: optional `full_texts` (accessions & data/code statements live
  in full text, not abstracts); pipeline `full_text` flag fetches it from the backend; the
  abstract-only path is now disclosed in the output.
- themes: k is capped for small corpora (~>=4 papers/theme) to stop over-clustering
  (a 15-paper corpus now yields ~2-3 themes, not 7); evidence-strength low-confidence caveat
  now fires for themes of <= 3 papers, not just n=1.

## [0.2.0]

### Added
- Consilium can be served over MCP (`consilium serve`, `--extra mcp`): a FastMCP
  server exposing 19 tools (themes/gaps/novelty/opportunities/hypotheses/questions/
  manuscript/grant/study-design/sample-size/protocol/…) returning structured JSON, so
  other agents consume Consilium the way it consumes Perspicacité. CI now installs the
  (public) mcp extra so the server + tests run there.


### Improved
- Generation is now genuinely QUOTE-GROUNDED: each generated citation must carry a
  verbatim supporting quote that `verify_quote` confirms is present in the cited
  paper's text; citations that are quote-less, unresolvable, or whose quote does not
  ground are dropped (was id-membership only). Hypotheses' indicium Evidence now
  carries the verified quote (ECO textual_quotation). Helpers: generation.common
  `ground_cited` / `grounded_evidence`.
- Aggregation reads only the current run's artifacts (OutputStore tracks written
  files) — no cross-run/topic contamination of dashboard/brief/capsule.
- DRY: shared `consilium.corpus_features` (max_year / theme_papers / theme_years /
  keyword_homogeneity) adopted across the scoring + life-science modules.

### Added
- Hardened Perspicacité MCP adapter: real streamable-HTTP client (initialize handshake +
  session id + SSE parsing + structuredContent unwrap). Live integration tests (`-m live`,
  `make test-live`) verified against a running Perspicacité (:8002): real `search_literature`
  → Corpus, scoring on the real corpus, and the passages call path.

### Added
- Phase 1 skeleton: package scaffold, tooling (uv, ruff, mypy --strict, pytest, pre-commit, CI).
- `config`, `errors`, `logging`, core domain `models`.
- litellm client with deterministic mock mode.
- `RetrievalBackend` Protocol + Perspicacité MCP adapter + `FakeBackend`.
- Theme discovery (seeded clustering) + LLM labeling.
- Standards layer: `verify_quote` grounding gate + indicium/ASTRA exporters.
- Output store + themes report renderer.
- In-process pipeline + `consilium` CLI (`version`, `themes`, `run`).
- Phase 2 scoring layer (deterministic, honest): shared scoring kernel
  (`scoring/common.py`) plus `evidence_strength`, `novelty`, `gap_validation`,
  `opportunity`, `funding_alignment`, `meta_analysis_readiness`. Each emits
  `Insight`s with `Score`s (components/weights/normalization/uncertainty — no bare
  magic numbers), exported to ASTRA. New CLI subcommands: `evidence-strength`,
  `novelty`, `opportunities`, `funding`, `meta-analysis`, `gaps`. Pipeline `--quick`
  now runs the full scoring set.
- Phase 5 aggregation + projects: `kb_snapshot` (unified knowledge_base.json),
  `explainability` (evidence trace), `dashboard`, `brief`, asb-schema `capsule`
  export (SciTaskCapsule), plus cross-run `project` database and `research_memory`.
  CLI: `dashboard`, `brief`. `--quick`/`--life-science` now end with aggregation; a
  full `run` executes all 17 stages.
- Phase 4 life-science advisory (correctness-critical): `statistics` (scipy/statsmodels
  power & sample-size for 5 designs, golden-value tested; no hardcoded critical values),
  `study_design` (OBI, conjunctive maturity gate), `bioinformatics` (assay-aware omics →
  repositories/EDAM; RNA-seq ≠ genomics), `protocols`, `reproducibility` (weighted 6-dim,
  adequacy not presence), `datasets` (accession discovery). CLI: `stats`, `protocol`,
  `study-design`, `bioinformatics`, `reproducibility`, `datasets`; `--life-science` set.
- Phase 3 generation layer (LLM-backed, corpus-grounded): `hypotheses` (→ indicium
  draft Claims, Bucur SuperPattern), `questions` (scored, no RNG), `manuscript`,
  `grant`, and `reviewer-sim`. Shared generation kernel (`generation/common.py`):
  APA formatting, citation verification (drops hallucinated refs), grounding, and
  output-leak detection. New models (Hypothesis, ResearchQuestion, GeneratedSection,
  ReviewComment) + indicium Claim exporter. CLI: `hypotheses`, `questions`,
  `manuscript`, `grant`, `review`. Logs now go to stderr (clean stdout JSON).
