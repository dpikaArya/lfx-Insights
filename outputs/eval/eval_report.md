# ScholarQABench ablation â€” bundled

PerspicacitÃ©â†’Consilium pipeline: retrieve â†’ synthesise grounded answer â†’ score.

| condition | cases | citation F1 | correctness | quality | retrieval |
|---|---|---|---|---|---|
| null | 5 | 0.000 (very low) | 0.000 (very low) | 0.200 (low) | â€” |
| tfidf | 5 | 0.000 (very low) | 0.000 (very low) | 0.200 (low) | â€” |

## Caveats

- Citation entailment judge is 'lexical', not the paper's AttrScore/TRUE-NLI; absolute citation numbers are not bit-comparable to ScholarQABench.
- [null] Answer-quality is an LLM-judge approximation of Prometheus (organization/coverage/relevance), not the 8x7B checkpoints.
- [tfidf] Answer-quality is an LLM-judge approximation of Prometheus (organization/coverage/relevance), not the 8x7B checkpoints.

_Aggregates are honest Scores (components/weights/uncertainty). Model: ollama/qwen2.5-coder:7b._