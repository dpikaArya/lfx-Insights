# Eval hands-on (run on another machine)

A self-contained runbook to evaluate the **joint Perspicacité→Consilium pipeline** on
real benchmarks from a fresh machine. Pairs with the reference in [eval.md](eval.md).

> The pipeline: a scientific question → **retrieve** literature (Perspicacité) →
> **synthesise** a long-form, citation-grounded answer (Consilium) → **score** citation
> faithfulness, answer quality, and (LitSearch) retrieval. The ablation
> `null` / `tfidf` / `oracle` / `perspicacite` isolates Perspicacité's contribution.

## 0. Prerequisites

- A **running Perspicacité MCP server** (default `http://localhost:8002/mcp`). Start it on
  the box (or point at a remote one with `CONSILIUM_PERSPICACITE__URL`).
- **Python ≥3.12** and **uv** (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Local mode** (default): Ollama running locally with a model pulled (`ollama pull qwen2.5-coder:7b`). No API key needed.
- **Hosted mode** (optional): An **LLM key**: `OPENAI_API_KEY` *or* `ANTHROPIC_API_KEY` (litellm reads the matching one).

## 1. Install

```bash
git clone https://github.com/HolobiomicsLab/consilium && cd consilium
uv sync --extra dev --extra mcp          # the `standards` extra is NOT needed for eval
```

## 2. Smoke test (offline — no network, no LLM, ~5 s)

```bash
uv run consilium eval scholarqa --offline --conditions null,tfidf --judge lexical
```
Expect a JSON summary with `tfidf` retrieval working on the bundled fixture. If this prints,
the harness is healthy.

## 3. Configure the LLM + server

### Local mode (Ollama — no API key)

```bash
export CONSILIUM_LLM__MODEL=ollama/qwen2.5-coder:7b
export CONSILIUM_LLM__FALLBACK='[]'
export CONSILIUM_PERSPICACITE__URL=http://localhost:8002/mcp
export CONSILIUM_EVAL__RETRIEVAL_K=20
```

### Hosted mode (OpenAI / Anthropic)

```bash
export OPENAI_API_KEY=...                      # or ANTHROPIC_API_KEY=...
export CONSILIUM_LLM__MODEL=gpt-4o             # or claude-opus-4-8, gpt-4o-mini (cheap), ...
export CONSILIUM_LLM__FALLBACK='[]'            # don't fall back to a model you have no key for
export CONSILIUM_PERSPICACITE__URL=http://localhost:8002/mcp
export CONSILIUM_EVAL__RETRIEVAL_K=20          # papers retrieved per question
```
Confirm Perspicacité is reachable:
```bash
uv run pytest -m live -q                       # auto-skips if :8002 is down; passes if up
```

## 4. Get the data

- **ScholarQABench** — `git clone https://github.com/AkariAsai/ScholarQABench`. The
  multi-paper long-form file `data/scholarqa_multi/human_answers.json` loads directly.
- **ExpertQA** — `git clone https://github.com/chaitanyamalaviya/ExpertQA`; use
  `data/r2_compiled_anon.jsonl` (auto-detected).
- **LitSearch** — build a Consilium-loadable JSONL from the HuggingFace dataset:
  ```bash
  uv run --with datasets python scripts/litsearch_to_jsonl.py litsearch.jsonl \
    --n-queries 100 --distractors 20
  ```

## 5. Run the ablation

Use the helper (wraps env + the recommended flags) or call the CLI directly.

```bash
# scripts/run_ablation.sh <dataset> [conditions] [judge] [out_dir]
scripts/run_ablation.sh ScholarQABench/data/scholarqa_multi/human_answers.json \
    null,tfidf,oracle,perspicacite llm runs/scholarqa

scripts/run_ablation.sh ExpertQA/data/r2_compiled_anon.jsonl \
    tfidf,oracle,perspicacite llm runs/expertqa

scripts/run_ablation.sh litsearch.jsonl tfidf,perspicacite llm runs/litsearch
```

Recommended generation setting (already in the helper): **`--ground-generation
--generation-judge llm`** — the answerer self-grounds its citations (drops misattributed,
attributes uncited-but-supported = "cite-as-you-write"). Cap with `--max-cases N` while
dialling in; start with `--judge lexical` for a free dry run before paying for `--judge llm`.

## 6. The independent-judge cross-check (do not skip)

**Citation F1 is judge-dependent, and gating with the same model that scores inflates it.**
In our runs, gate≡judge (both gpt-4o-mini) gave F1 ≈ 0.91; re-scoring the *same* answers with
an independent judge gave ≈ 0.50–0.66. Always report a cross-checked number:

```bash
# Re-score the SAME (cached) answers with the deterministic, independent lexical judge:
scripts/run_ablation.sh <dataset> <conditions> lexical runs/<name>_lexcheck
```
(Generation is cached by model+prompt, so only the scoring changes.) For a *semantic* yet
independent judge, set a different model for scoring than for generation — e.g. generate with
`CONSILIUM_LLM__MODEL=gpt-4o-mini` and judge with Claude. (Note: a single judge-model override
is a documented gap — today the gate and judge share `CONSILIUM_LLM__MODEL`; until that lands,
the lexical cross-check is the independent number.)

## 7. Read the results

Each run writes `runs/<name>/eval/`:
- `eval_report.md` — per-condition table (citation F1, correctness, quality, retrieval) + the
  `perspicacite − tfidf` lift + the disclosed caveats.
- `eval_results.json` — full per-case detail (answers, cited docs, per-metric scores) for your
  own breakdowns (e.g. by `subject`/benchmark, or precision-vs-recall).

## 8. How to read the numbers (lessons from our runs)

- **ScholarQA is the fair test** (open retrieval). There, Perspicacité beat the tfidf baseline
  on citation F1 and held up best under the independent judge.
- **ExpertQA flatters `tfidf`**: its contexts *are* the gold evidence, so a lexical retriever
  matches them verbatim. Use the **`oracle`** condition to separate retrieval from generation.
- **LitSearch `perspicacite` retrieval ≈ 0 is expected** (corpus parity): the gold corpus ids
  aren't in Perspicacité's corpus. `tfidf` over the provided pool is the meaningful retrieval
  condition; for Perspicacité, align ids by DOI/title or treat retrieval as open.
- **Recall is the usual bottleneck**, not retrieval — even with the `oracle` ceiling the model
  under-cites. Cite-as-you-write targets this.
- **Cost/time** scale with cases × conditions × LLM calls (gen + per-sentence gate/attribute +
  per-sentence judge). `--max-cases` samples; conditions are independent and safe to run in
  parallel processes (only `perspicacite` touches the server).

## 9. Caveats (also emitted in every `eval_report.md`)

Scaffold, not a paper reproduction: pluggable judge (not AttrScore/TRUE-NLI), corpus ≠ peS2o
(only open-retrieval is paper-comparable), quality ≈ Prometheus, regex sentence splitter.
