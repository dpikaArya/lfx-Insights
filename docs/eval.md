# Evaluation harness (ScholarQABench)

Consilium ships an evaluation harness that measures the **joint
Perspicacité→Consilium pipeline** on a [ScholarQABench](https://github.com/AkariAsai/ScholarQABench)-style
task (Asai et al. 2024, [arXiv:2411.14199](https://arxiv.org/abs/2411.14199)):

> Given a scientific question, *retrieve* literature (Perspicacité) and *synthesise a
> long-form, citation-grounded answer* (Consilium), then score **both** citation
> faithfulness and answer quality.

The headline experiment is a **retrieval ablation** that isolates Perspicacité's
contribution while holding the synthesis layer fixed:

| condition | what it does |
|---|---|
| `null` | closed-book — no retrieval, the model answers from nothing (floor) |
| `tfidf` | a generic lexical baseline that retrieves from a fixed candidate pool (the dataset's own contexts) |
| `oracle` | perfect retrieval — feeds each case's own gold contexts (ceiling); isolates generation/citation quality from retrieval quality |
| `perspicacite` | open retrieval over a live Perspicacité KB |

The **lift** (`perspicacite − tfidf` on citation F1) is the quantity of interest; the
`oracle` ceiling shows how much of the remaining gap is retrieval vs generation.

Add **`--ground-generation`** to make the answerer *self-verify* its citations: each
inline `[n]` is kept only if its cited passage entails the sentence (a cheap lexical
entailment gate, CRAG/Self-RAG-style), so misattributed citations are dropped before
scoring — raising precision at some cost to recall.

## Quick start

```bash
# Offline smoke (bundled 5-case fixture, deterministic lexical judge, no network/LLM):
consilium eval scholarqa --offline --conditions null,tfidf --judge lexical

# Full ablation against a live Perspicacité (:8002) with your configured LLM:
consilium eval scholarqa --dataset path/to/scholarqa.jsonl \
  --conditions null,tfidf,perspicacite --judge llm --max-cases 50
```

Outputs land in `<output-dir>/eval/`: `eval_results.json` (the full
`AblationReport`) and `eval_report.md` (a comparison table + lift + caveats).

## Options

| flag | meaning |
|---|---|
| `--dataset` | `bundled`, or a path to a `.jsonl`/`.json` ScholarQABench file |
| `--conditions` | comma list of `null,tfidf,perspicacite` (default from config; `perspicacite` is dropped under `--offline`) |
| `--metrics` | override the per-case metric set: `citation,match,rouge,quality` |
| `--judge` | citation entailer: `lexical` (deterministic), `grounding` (indicium `verify_quote`), `llm` |
| `--max-cases` | cap the number of cases (0 = all); recorded as a caveat |
| `--offline` | in-memory fakes (no network/LLM) |

Config block (`config.yml` → `eval:`): `conditions`, `judge`, `lexical_threshold`,
`retrieval_k`, `max_cases`.

## Metrics

- **Citation precision / recall / F1** — the AutoAIS/ALCE scheme, reimplemented
  faithfully from the ScholarQABench source (sentence splitting, `<50`-char skip,
  0-indexed `[n]` markers, citation inheritance, necessary-vs-sufficient precision).
  The entailment judge is **pluggable**: `lexical` (token overlap, deterministic —
  the default for CI), `grounding` (the indicium quote-grounding gate), or `llm`.
- **Correctness** (reference-based, deterministic) — substring `match` (SciFact,
  PubMedQA short labels) and `rouge_l` (LCS F-measure; QASA / long-form references).
- **Quality** (LLM judge) — a reference-guided approximation of Prometheus's three
  aspects (organization, coverage, relevance), each scored 1–5 and normalised.
- **Retrieval** (intrinsic, deterministic) — `recall@k` and `nDCG@k` under binary
  relevance, LitSearch-style: given a query's gold relevant papers, did this
  condition's retriever fetch them? This is the direct retrieval measure ScholarQABench
  cannot give. Matching is identifier-agnostic (corpus id / DOI / normalised title), so
  it works across the `tfidf` and `perspicacite` conditions — but Perspicacité retrieves
  over its own corpus, so cross-corpus id alignment is a caveat (most meaningful for the
  `tfidf` baseline over the provided corpus).

Each per-condition aggregate is an honest `Score` (components, weights, method,
interpretation band, uncertainty) — never a bare mean.

## Datasets

The loader **auto-detects the shape per record**, so several SOTA benchmarks share one
harness. Bundled synthetic fixtures (offline/CI): `bundled` (ScholarQABench),
`expertqa-bundled`, `litsearch-bundled`. For a real run, download the benchmark and point
`--dataset` at its `.jsonl`/`.json`:

- **ScholarQABench** — multi-paper long-form (`{input, output, ctxs:[{title,text}]}`); the
  main joint task. Short-form (`{input, answer:"yes"/"no"/…}`) → `match`; QASA-style
  (`{input, answer, ctxs}`) → `rouge`.
- **ExpertQA** (Malaviya et al., NAACL 2024) — `{question, metadata.field, answers{model→
  {answer_string, revised_answer_string, claims[{claim_string, evidence}]}}}`. The
  human-revised answer becomes the reference; claim evidence passages become the contexts.
  Extends the joint eval to expert, multi-domain questions with attribution.
- **LitSearch** (Ajith et al., EMNLP 2024) — `{query, corpusids, …}`. The `corpusids`
  become the gold set for the intrinsic `retrieval` metric (recall@k/nDCG@k). For the
  offline path, ship the candidate corpus as each query's `ctxs` (with `corpusid`); full
  HF-corpus loading (64k docs) is a documented extension.

The prediction your pipeline must produce — prose with inline `[n]` citations over a
numbered context list — is generated by `consilium.eval.answer.answer_question`.

## Running an extensive evaluation (real data + real LLM, e.g. on a GPU/cloud box)

The bundled fixtures are toy demos. For real numbers, point the harness at the real
benchmark files with a real LLM and a reachable Perspicacité server.

### 1. Environment

```bash
uv sync --extra dev --extra mcp            # harness deps (standards extra not needed)

# Local mode (Ollama — no API key needed):
export CONSILIUM_LLM__MODEL=ollama/qwen2.5-coder:7b
export CONSILIUM_LLM__FALLBACK='[]'

# OR hosted mode (litellm reads the matching provider key from the environment):
# export OPENAI_API_KEY=...                   # or ANTHROPIC_API_KEY=...
# export CONSILIUM_LLM__MODEL=gpt-4o          # or claude-opus-4-8, etc.

# Perspicacité server for the `perspicacite` condition:
export CONSILIUM_PERSPICACITE__URL=http://localhost:8002/mcp
export CONSILIUM_EVAL__RETRIEVAL_K=20       # papers retrieved per question
```

Use `--judge llm` for citation scoring closest to the paper (an LLM entailment judge);
`--judge lexical` is the deterministic, free, offline-capable approximation.

### 2. Get the data

- **ScholarQABench** — `git clone https://github.com/AkariAsai/ScholarQABench`. The
  multi-paper long-form file `data/scholarqa_multi/human_answers.json` is a JSON array of
  `{input, output, ctxs}` and loads directly. Single-paper sets live under
  `data/single_paper_tasks/*.jsonl` (SciFact/PubMedQA → `match`, QASA → `rouge`).
- **ExpertQA** — `git clone https://github.com/chaitanyamalaviya/ExpertQA`. Point
  `--dataset` at `data/r2_compiled_anon.jsonl` (auto-detected as ExpertQA).
- **LitSearch** — `from datasets import load_dataset` for `princeton-nlp/LitSearch`
  (`query` + `corpus_clean` configs). Emit one JSONL line per query as
  `{"query": ..., "corpusids": [...], "ctxs": [{"corpusid": ..., "title": ..., "text": ...}, ...]}`,
  where `ctxs` is the candidate pool you want the `tfidf` condition to retrieve over (the
  query's gold paper plus distractors, or a corpus shard). `corpusids` are the gold set for
  `recall@k`/`nDCG@k`.

### 3. Run

```bash
# ScholarQABench multi-paper long-form, full ablation:
consilium eval scholarqa \
  --dataset ScholarQABench/data/scholarqa_multi/human_answers.json \
  --conditions null,tfidf,perspicacite --judge llm \
  --output-dir runs/scholarqa

# ExpertQA (expert multi-domain attribution); sample 100 to bound cost:
consilium eval scholarqa --dataset ExpertQA/data/r2_compiled_anon.jsonl \
  --conditions tfidf,perspicacite --judge llm --max-cases 100 \
  --output-dir runs/expertqa

# LitSearch (intrinsic retrieval): tfidf over the LitSearch corpus is the meaningful
# condition; perspicacite retrieves a different corpus (id-alignment caveat).
consilium eval scholarqa --dataset litsearch_combined.jsonl \
  --conditions tfidf,perspicacite --metrics retrieval \
  --output-dir runs/litsearch
```

Each run writes `runs/<name>/eval/eval_results.json` (full per-case detail) and
`eval_report.md` (the comparison table + lift + caveats). Cost/time scale with cases ×
conditions × (1 answer call + 1 quality-judge call + per-sentence judge calls); use
`--max-cases` to sample and start with `--judge lexical` for a free dry run before paying
for `--judge llm`.

## Honest scope (this is a scaffold, not a paper reproduction)

The harness deliberately does **not** claim bit-parity with the published numbers.
Disclosed deltas (also emitted as `caveats` in every report):

1. **Entailment judge differs.** The paper ships AttrScore
   (`osunlp/attrscore-flan-t5-xl`; TRUE-NLI is in the repo but commented out). We
   default to a deterministic lexical/grounding judge offline and an LLM-as-judge
   online. The judge is pluggable, but absolute citation numbers are not comparable.
2. **Corpus parity is impossible.** Perspicacité retrieves over a different corpus
   than peS2o (~45M); only the *open-retrieval* condition is paper-comparable, never
   "provided datastore".
3. **Quality judge is an approximation** of Prometheus's 8×7B checkpoints.
4. **Sentence splitting** is a regex approximation of nltk's.

Not built (documented extensions): the full ScholarQA-CS 110-case annotation-driven
rubric scorer; native AttrScore/TRUE-NLI/Prometheus checkpoints; an intrinsic
retrieval recall@k (ScholarQABench ships no such script — retrieval is judged
extrinsically through the downstream citation/answer metrics).
