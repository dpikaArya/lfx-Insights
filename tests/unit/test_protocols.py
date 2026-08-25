from __future__ import annotations

import pytest

from consilium.lifescience.protocols import AVAILABLE_PROTOCOLS, generate_protocol
from consilium.models import Protocol

pytestmark = pytest.mark.unit

_EXPECTED_KINDS = ("rna_seq", "variant_calling", "pcr", "western_blot")


def test_available_protocols_contains_the_four_kinds() -> None:
    for kind in _EXPECTED_KINDS:
        assert kind in AVAILABLE_PROTOCOLS


def test_rna_seq_has_nonempty_steps_and_qc() -> None:
    proto = generate_protocol("rna_seq")
    assert isinstance(proto, Protocol)
    assert proto.kind == "rna_seq"
    assert proto.name
    assert proto.steps
    assert proto.qc_checklist
    assert all(isinstance(s, str) and s for s in proto.steps)
    assert all(isinstance(q, str) and q for q in proto.qc_checklist)


def test_rna_seq_qc_covers_reproducibility_and_integrity() -> None:
    qc_text = " ".join(generate_protocol("rna_seq").qc_checklist).lower()
    assert "rin" in qc_text
    assert "adapter" in qc_text
    assert "alignment" in qc_text
    assert "complexity" in qc_text
    assert "replicate" in qc_text
    assert "seed" in qc_text


def test_pcr_includes_no_template_control() -> None:
    proto = generate_protocol("pcr")
    assert proto.kind == "pcr"
    assert proto.steps
    assert proto.qc_checklist
    qc_text = " ".join(proto.qc_checklist).lower()
    assert "no-template control" in qc_text or "ntc" in qc_text


@pytest.mark.parametrize("kind", _EXPECTED_KINDS)
def test_every_kind_generates_a_complete_protocol(kind: str) -> None:
    proto = generate_protocol(kind)
    assert proto.kind == kind
    assert len(proto.steps) >= 3
    assert len(proto.qc_checklist) >= 3
    assert proto.notes == (
        "Template — adapt to your platform/organism; not a substitute for a validated SOP."
    )


def test_unknown_kind_raises_valueerror_listing_available() -> None:
    with pytest.raises(ValueError) as excinfo:
        generate_protocol("mass_spec")
    msg = str(excinfo.value)
    assert "mass_spec" in msg
    for kind in _EXPECTED_KINDS:
        assert kind in msg


def test_generate_protocol_is_deterministic() -> None:
    first = generate_protocol("western_blot")
    second = generate_protocol("western_blot")
    assert first.model_dump() == second.model_dump()


def test_returned_steps_list_is_independent_copy() -> None:
    proto = generate_protocol("variant_calling")
    original_len = len(proto.steps)
    proto.steps.append("mutated")
    assert len(generate_protocol("variant_calling").steps) == original_len
