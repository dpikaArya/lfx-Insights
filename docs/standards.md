# Standards & grounding

Consilium computes with lean internal pydantic models and serializes them to the Holobiomics
LinkML standards at output boundaries (verified by round-trip instantiation against the real
models when the `standards` extra is installed).

## Mapping

| Consilium artifact | Standard class | Ontologies |
|---|---|---|
| Hypothesis | indicium `Claim` (Bucur SuperPattern, `claim_status=draft`) + `Evidence` | ECO, CiTO, Bucur |
| Corpus paper | indicium `Source` (FaBiO Expression) | FaBiO |
| Finding (gap/novelty/opportunity/…) | ASTRA `Insight` (+ `derived`, `scope`, `notes`) + `Evidence` | ASTRA, SEPIO |
| Run | asb-schema `SciTaskCapsule` (`capsule_task_id`/`capsule_card`/`capsule_artifacts`/…) | asb, p-plan, RO-Crate, PROV-O |
| Statistics / study design / bioinformatics | STATO / OBI / EDAM controlled terms | STATO, OBI, EDAM |

The indicium export is conformant: `IndiciumDocument(**claims_to_document(...))` instantiates the
real model — `claims` is a dict keyed by id, `evidences`/`sources` are lists, each `Evidence`
inlines its `of_source` and uses the ECO-label enum (`textual_quotation` /
`inference_from_background_knowledge`).

## The grounding gate

The anti-hallucination guarantee runs through indicium's `verify_quote` kernel:

- **Generation.** Each generated citation must carry a verbatim supporting quote. `ground_cited`
  keeps a citation only if its quote is verbatim-present in the cited paper's text; quote-less,
  unresolvable, or paraphrased/fabricated citations are **dropped**. Hypotheses' indicium Evidence
  carries the verified quote.
- **Fallback.** Without the `standards` extra, grounding uses a case-sensitive normalized-substring
  match that mirrors the kernel, so strictness does not depend on whether indicium is installed.

```python
from consilium.generation.common import ground_cited
# keeps W1 (quote present), drops the fabricated and quote-less ones
ground_cited([("W1", "neural networks"), ("W1", "quantum teleportation"), ("W1", "")], corpus)
# -> ["W1"]
```

## Honest scoring

A `Score` is never a bare number: it exposes `components` (each a named value + weight), the
combination `method`, an `interpretation` band, and `uncertainty`. A component-less composite is
rejected at construction.
