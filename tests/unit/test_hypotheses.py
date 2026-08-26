from __future__ import annotations

import pytest

from lfx_insights.generation.hypotheses import (
    CitedRef,
    HypothesisBatch,
    HypothesisDraft,
    generate_hypotheses,
)
from lfx_insights.llm.client import MockLLM
from lfx_insights.models import Corpus, Paper

pytestmark = pytest.mark.unit


def _corpus() -> Corpus:
    return Corpus(
        kb_id="kb",
        papers=[
            Paper(
                id="W1",
                title="Graph neural networks for molecular property prediction",
                abstract=(
                    "Graph neural networks achieve strong accuracy on molecular "
                    "property prediction benchmarks."
                ),
            ),
            Paper(
                id="W2",
                title="Deep generative models for de novo drug design",
                abstract=(
                    "Deep generative models enable de novo design of novel drug-like molecules."
                ),
            ),
        ],
    )


def _batch() -> HypothesisBatch:
    return HypothesisBatch(
        items=[
            HypothesisDraft(
                subject="graph neural networks",
                qualifier="increases",  # lowercase SuperPattern term
                object="property prediction accuracy",
                statement="Graph neural networks increase property prediction accuracy.",
                rationale="W1 reports strong predictive performance.",
                independent_var="model architecture",
                dependent_var="prediction accuracy",
                methodology="benchmark on held-out molecules",
                evidence=[
                    # Grounded: quote is verbatim in W1's abstract -> survives.
                    CitedRef(
                        paper_id="W1",
                        quote="strong accuracy on molecular property prediction",
                    ),
                    # Real paper but NO quote -> dropped by the grounding gate.
                    CitedRef(paper_id="W2", quote=""),
                    # Hallucinated id with a made-up quote -> dropped.
                    CitedRef(paper_id="W404", quote="this paper does not exist"),
                ],
            ),
            HypothesisDraft(
                subject="generative models",
                qualifier="ENABLES",  # mixed case -> normalizes to "enables"
                object="de novo design",
                statement="Generative models enable de novo molecular design.",
                evidence=[
                    # Real paper but the quote is NOT in W2's abstract -> dropped.
                    CitedRef(
                        paper_id="W2",
                        quote="quantum computers solve every optimization instantly",
                    ),
                ],
            ),
            HypothesisDraft(
                subject="leaky",
                qualifier="causes",
                object="leak",
                # Leaked placeholder -> must be rejected by has_output_leak.
                statement="The {top_theme} causes downstream effects.",
                evidence=[
                    CitedRef(paper_id="W1", quote="strong accuracy"),
                ],
            ),
        ]
    )


def _responder() -> MockLLM:
    return MockLLM(responder=lambda prompt, model: _batch())


def test_builds_hypotheses_and_keeps_only_grounded_citation() -> None:
    hyps = generate_hypotheses(_corpus(), _responder())
    # Leaked draft skipped -> 2 of 3 survive.
    assert len(hyps) == 2
    first = hyps[0]
    assert first.subject == "graph neural networks"
    assert first.qualifier == "increases"
    assert first.statement == "Graph neural networks increase property prediction accuracy."
    assert first.status == "draft"
    # Only W1 survives: W2 had no quote, W404 is hallucinated.
    assert [e.paper_id for e in first.evidence] == ["W1"]
    # The surviving evidence carries the verbatim quote (real textual_quotation).
    assert first.evidence[0].quote == "strong accuracy on molecular property prediction"


def test_ungrounded_quote_is_dropped() -> None:
    hyps = generate_hypotheses(_corpus(), _responder())
    second = hyps[1]
    # W2's quote does not ground in its abstract -> no surviving evidence.
    assert second.evidence == []


def test_qualifier_normalized_to_superpattern() -> None:
    hyps = generate_hypotheses(_corpus(), _responder())
    second = hyps[1]
    # "ENABLES" -> "enables" (a valid SuperPattern term).
    assert second.qualifier == "enables"


def test_leaked_statement_rejected() -> None:
    hyps = generate_hypotheses(_corpus(), _responder())
    assert all("{top_theme}" not in h.statement for h in hyps)
    assert all(h.subject != "leaky" for h in hyps)


def test_caps_at_n() -> None:
    hyps = generate_hypotheses(_corpus(), _responder(), n=1)
    assert len(hyps) == 1


def test_provenance_carries_model_and_generated_by() -> None:
    hyps = generate_hypotheses(_corpus(), _responder())
    prov = hyps[0].provenance
    assert prov.generated_by == "lfx-insights"
    assert prov.model == "(see settings)"


def test_prompt_lists_corpus_ids_and_titles() -> None:
    llm = _responder()
    generate_hypotheses(_corpus(), llm)
    prompt = llm.calls[0]
    assert "W1: Graph neural networks for molecular property prediction" in prompt
    assert "W2: Deep generative models for de novo drug design" in prompt


def test_prompt_instructs_verbatim_quote() -> None:
    llm = _responder()
    generate_hypotheses(_corpus(), llm)
    prompt = llm.calls[0]
    assert "verbatim" in prompt.lower()
    assert "quote" in prompt.lower()


def test_empty_batch_returns_empty() -> None:
    llm = MockLLM(responder=lambda prompt, model: HypothesisBatch(items=[]))
    assert generate_hypotheses(_corpus(), llm) == []


def test_offline_minimal_response_yields_no_hypotheses() -> None:
    # MockLLM with no responder auto-builds a minimal HypothesisBatch (empty items),
    # so a grounded run simply yields zero hypotheses offline without erroring.
    llm = MockLLM()
    assert generate_hypotheses(_corpus(), llm) == []
