"""Evaluation harness for the Perspicacité→lfx Insights pipeline.

Implements a ScholarQABench-style evaluation (Asai et al. 2024, arXiv:2411.14199):
given a scientific question, retrieve literature and synthesise a long-form,
citation-grounded answer, then score citation faithfulness and answer quality.

The headline experiment is a *retrieval ablation* — ``null`` (closed-book) vs
``tfidf`` (generic lexical baseline over the provided datastore) vs ``perspicacite``
(open retrieval) — holding the synthesis layer fixed, so the lift attributable to
Perspicacité is measured directly.

This is a scaffold, not a bit-reproduction of the paper: the entailment judge is
pluggable (deterministic offline, LLM-as-judge online), the corpus differs from
peS2o, and the quality judge approximates Prometheus. See ``docs/eval.md``.
"""

from __future__ import annotations
