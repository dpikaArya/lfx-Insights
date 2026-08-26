#!/usr/bin/env bash
# Run the lfx Insights x Perspicacité eval ablation (recommended settings).
#
# Usage:
#   scripts/run_ablation.sh <dataset> [conditions] [judge] [out_dir]
#
#   <dataset>     "bundled"/"expertqa-bundled"/"litsearch-bundled", or a path to a
#                 .jsonl/.json file (ScholarQABench / ExpertQA / LitSearch, auto-detected).
#   [conditions]  comma list of null,tfidf,oracle,perspicacite   (default: all four)
#   [judge]       citation judge: lexical | grounding | llm        (default: llm)
#   [out_dir]     output directory                                 (default: runs/<dataset>)
#
# Env (set before calling): LFX_INSIGHTS_LLM__MODEL (+ the matching provider key, if hosted),
#   LFX_INSIGHTS_PERSPICACITE__URL, LFX_INSIGHTS_EVAL__RETRIEVAL_K. See docs/eval-handson.md.
set -euo pipefail

DATASET="${1:?usage: run_ablation.sh <dataset> [conditions] [judge] [out_dir]}"
CONDITIONS="${2:-null,tfidf,oracle,perspicacite}"
JUDGE="${3:-llm}"
DEFAULT_OUT="runs/$(basename "$DATASET" | tr -c 'A-Za-z0-9' '_')"
OUT="${4:-$DEFAULT_OUT}"

: "${LFX_INSIGHTS_LLM__MODEL:=ollama/qwen2.5-coder:7b}"
: "${LFX_INSIGHTS_LLM__FALLBACK:=[]}"   # no cross-provider fallback unless caller sets one
export LFX_INSIGHTS_LLM__MODEL LFX_INSIGHTS_LLM__FALLBACK

echo "dataset=$DATASET conditions=$CONDITIONS judge=$JUDGE model=$LFX_INSIGHTS_LLM__MODEL out=$OUT"

# --ground-generation + --generation-judge <judge>: cite-as-you-write self-grounding
# (drop misattributed citations, attribute uncited-but-supported sentences).
uv run lfx-insights eval scholarqa \
  --dataset "$DATASET" \
  --conditions "$CONDITIONS" \
  --judge "$JUDGE" \
  --ground-generation --generation-judge "$JUDGE" \
  --output-dir "$OUT"

echo "report: $OUT/eval/eval_report.md"
echo "detail: $OUT/eval/eval_results.json"
