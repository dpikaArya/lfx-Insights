from __future__ import annotations

import pytest

from lfx_insights.llm.client import MockLLM
from lfx_insights.themes.label import ThemeLabel

pytestmark = pytest.mark.unit


def test_mock_substring_match() -> None:
    llm = MockLLM(responses={"cluster": ThemeLabel(label="Matched", rationale="r")})
    out = llm.complete_structured("a cluster of papers", ThemeLabel)
    assert out.label == "Matched"


def test_mock_minimal_when_no_match() -> None:
    llm = MockLLM()
    out = llm.complete_structured("anything", ThemeLabel)
    assert out.label == "mock-label"
    assert llm.calls == ["anything"]


def test_mock_responder() -> None:
    llm = MockLLM(responder=lambda prompt, model: ThemeLabel(label="R", rationale="x"))
    out = llm.complete_structured("p", ThemeLabel)
    assert out.label == "R"


def test_mock_minimal_constructs_non_string_fields() -> None:
    from pydantic import BaseModel

    class Multi(BaseModel):
        s: str
        i: int
        f: float
        b: bool
        items: list[str]

    out = MockLLM().complete_structured("anything", Multi)
    assert out.s == "mock-s"
    assert out.i == 0
    assert out.f == 0.0
    assert out.b is False
    assert out.items == []
