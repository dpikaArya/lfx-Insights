# ScholarQABench ablation — bundled

Perspicacité→Consilium pipeline: retrieve → synthesise grounded answer → score.

| condition | cases | citation F1 | correctness | quality | retrieval |
|---|---|---|---|---|---|
| null | 5 | 0.000 (very low) | 0.000 (very low) | 0.200 (low) | — |
| tfidf | 5 | 0.000 (very low) | 0.000 (very low) | 0.200 (low) | — |

## Caveats

- Citation entailment judge is 'lexical', not the paper's AttrScore/TRUE-NLI; absolute citation numbers are not bit-comparable to ScholarQABench.
- [null] Answer-quality is an LLM-judge approximation of Prometheus (organization/coverage/relevance), not the 8x7B checkpoints.
- [tfidf] Answer-quality is an LLM-judge approximation of Prometheus (organization/coverage/relevance), not the 8x7B checkpoints.

_Aggregates are honest Scores (components/weights/uncertainty). Model: ollama/qwen2.5-coder:7b._