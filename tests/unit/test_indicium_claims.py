from __future__ import annotations

import pytest

from lfx_insights.models import Author, Corpus, EvidenceRef, Hypothesis, Paper
from lfx_insights.standards.indicium_export import (
    claims_to_document,
    hypothesis_evidence,
    hypothesis_to_claim,
)

pytestmark = pytest.mark.unit


def test_hypothesis_qualifier_normalized() -> None:
    # "up-regulates" maps via the synonym table to a Bucur term.
    h = Hypothesis(subject="A", qualifier="UP-regulates", object="B", statement="A affects B")
    assert h.qualifier == "increases"
    # A genuinely unknown relation falls back to the safe default.
    h_unknown = Hypothesis(subject="A", qualifier="zzznotareal", object="B", statement="s")
    assert h_unknown.qualifier == "is_associated_with"
    h2 = Hypothesis(subject="A", qualifier="Increases", object="B", statement="s")
    assert h2.qualifier == "increases"


def test_hypothesis_to_claim_shape() -> None:
    h = Hypothesis(subject="drug X", qualifier="treats", object="disease Y", statement="X treats Y")
    claim = hypothesis_to_claim(h, claim_id="claim-0")
    # indicium Claim required slots: id, context, subject, relation, object
    assert {"id", "context", "subject", "relation", "object"} <= set(claim)
    assert claim["claim_status"] == "draft"
    assert claim["qualifier"] == "treats"
    assert "statement" not in claim  # not an indicium Claim slot
    assert "evidence" not in claim  # evidence is separate in indicium


def test_hypothesis_evidence_uses_eco_label_and_of_source() -> None:
    h = Hypothesis(
        subject="A",
        qualifier="treats",
        object="B",
        statement="s",
        evidence=[EvidenceRef(paper_id="W1", quote="x reduced y"), EvidenceRef(paper_id="W2")],
    )
    ev = hypothesis_evidence(h, claim_id="claim-0")
    assert ev[0]["of_source"]["identifier"] == "W1"  # inlined Source object
    assert ev[0]["for_claim"] == "claim-0"
    assert ev[0]["eco_code"] == "textual_quotation"  # quoted -> ECO label enum, not a CURIE
    assert ev[1]["eco_code"] == "inference_from_background_knowledge"  # unquoted


def test_claims_to_document_round_trips_through_real_indicium() -> None:
    """Alignment: the exported document must instantiate the real indicium models."""
    im = pytest.importorskip("indicium.generated.model")
    corpus = Corpus(
        kb_id="kb", papers=[Paper(id="W1", title="T", doi="10.x/y", authors=[Author(name="A B")])]
    )
    h = Hypothesis(
        subject="drug X",
        qualifier="treats",
        object="disease Y",
        statement="X treats Y",
        evidence=[EvidenceRef(paper_id="10.x/y", quote="X reduced Y")],
    )
    doc = claims_to_document([h], corpus)
    # Should not raise: the dicts conform to Claim/Evidence/Source/IndiciumDocument.
    model = im.IndiciumDocument(**doc)
    assert len(model.claims) == 1
    assert len(model.evidences) == 1
    assert len(model.sources) == 1
