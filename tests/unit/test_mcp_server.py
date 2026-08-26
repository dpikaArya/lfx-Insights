from __future__ import annotations

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from lfx_insights.mcp import build_server

pytestmark = pytest.mark.unit


async def test_expected_tools_are_registered() -> None:
    server = build_server(offline=True)
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
    expected = {
        "themes",
        "validate_gaps",
        "novelty",
        "opportunities",
        "evidence_strength",
        "funding_alignment",
        "meta_analysis_readiness",
        "hypotheses",
        "questions",
        "manuscript",
        "grant",
        "reviewer_simulation",
        "study_design",
        "bioinformatics",
        "reproducibility",
        "datasets",
        "sample_size",
        "protocol",
        "list_protocols",
    }
    assert expected <= names


async def test_themes_tool_returns_structured_themes_offline() -> None:
    server = build_server(offline=True)
    async with Client(server) as client:
        result = await client.call_tool("themes", {"topic": "drug discovery"})
    data = result.data
    assert isinstance(data, list) and len(data) >= 1
    assert "label" in data[0] and "paper_ids" in data[0]


async def test_sample_size_tool_golden_value() -> None:
    server = build_server(offline=True)
    async with Client(server) as client:
        result = await client.call_tool(
            "sample_size", {"design": "two_sample_t", "effect_size": 0.5}
        )
    assert result.data["n_per_group"] == 64


async def test_protocol_and_list_tools() -> None:
    server = build_server(offline=True)
    async with Client(server) as client:
        proto = await client.call_tool("protocol", {"kind": "rna_seq"})
        kinds = await client.call_tool("list_protocols", {})
    assert proto.data["kind"] == "rna_seq" and proto.data["steps"]
    assert "rna_seq" in kinds.data


async def test_evidence_strength_tool_offline() -> None:
    server = build_server(offline=True)
    async with Client(server) as client:
        result = await client.call_tool("evidence_strength", {"topic": "x"})
    assert isinstance(result.data, list) and result.data
    assert result.data[0]["score"]["components"]


async def test_validate_gaps_tool_runs_validation_offline() -> None:
    server = build_server(offline=True)
    async with Client(server) as client:
        result = await client.call_tool(
            "validate_gaps", {"topic": "x", "gaps": ["no work on quantum docking"]}
        )
    assert isinstance(result.data, list) and len(result.data) == 1
    assert "statement" in result.data[0] and "gap" in result.data[0]["tags"]
