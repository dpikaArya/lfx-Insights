"""Build an lfx Insights-loadable LitSearch dataset from the HuggingFace release.

LitSearch (Ajith et al., EMNLP 2024) ships queries and a 64k-doc corpus as separate
configs. The lfx Insights eval loader wants each query to carry its candidate pool as
``ctxs`` (so the ``tfidf`` condition can retrieve over it) plus the gold ``corpusids``.
This script emits one JSON line per query: the gold papers + sampled distractors as
``ctxs``, bounding the pool so a full 64k-doc retrieval is not required.

Requires the ``datasets`` package (not an lfx Insights dependency):

    uv run --with datasets python scripts/litsearch_to_jsonl.py out.jsonl \
        --n-queries 100 --distractors 20
"""

from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output .jsonl path")
    ap.add_argument("--n-queries", type=int, default=100, help="number of queries (0 = all)")
    ap.add_argument("--distractors", type=int, default=20, help="random distractor docs per query")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset  # imported lazily so the repo needs no datasets dep

    rng = random.Random(args.seed)
    queries = load_dataset("princeton-nlp/LitSearch", "query")["full"]
    corpus = load_dataset("princeton-nlp/LitSearch", "corpus_clean")["full"]
    by_id = {row["corpusid"]: row for row in corpus}
    all_ids = list(by_id)

    n = len(queries) if args.n_queries == 0 else min(args.n_queries, len(queries))
    written = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(n):
            q = queries[i]
            gold = [cid for cid in q["corpusids"] if cid in by_id]
            if not gold:
                continue  # gold doc missing from corpus_clean — skip
            pool = list(dict.fromkeys([*gold, *rng.sample(all_ids, args.distractors)]))
            ctxs = [
                {
                    "corpusid": cid,
                    "title": by_id[cid]["title"],
                    "text": by_id[cid]["abstract"] or "",
                }
                for cid in pool
            ]
            fh.write(
                json.dumps(
                    {
                        "id": f"ls-{i}",
                        "query": q["query"],
                        "corpusids": gold,
                        "query_set": q.get("query_set", ""),
                        "ctxs": ctxs,
                    }
                )
                + "\n"
            )
            written += 1
    print(f"wrote {written} queries to {args.out}")


if __name__ == "__main__":
    main()
