from __future__ import annotations

import json

import httpx
import pytest
import respx

from lfx_insights.errors import PerspicaciteUnavailable
from lfx_insights.sources.perspicacite import (
    PerspicaciteBackend,
    _corpus_from_result,
    _parse_sse,
    _passages_from_result,
    _unwrap_tool_result,
)

pytestmark = pytest.mark.unit

URL = "http://localhost:8002/mcp"


def _sse(obj: dict) -> str:
    return f"event: message\ndata: {json.dumps(obj)}\n\n"


def _mcp_handler(
    passages: list[dict] | None = None,
    papers: list[dict] | None = None,
    content: dict | None = None,
    captured: dict | None = None,
):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get("method")
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "sess-1", "content-type": "text/event-stream"},
                text=_sse({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}),
            )
        if method == "notifications/initialized":
            return httpx.Response(202, text="")
        if method == "tools/call":
            name = body["params"]["name"]
            if captured is not None:
                captured["name"] = name
                captured["arguments"] = body["params"].get("arguments", {})
            inner: dict = {}
            if name == "get_relevant_passages":
                inner = {"passages": passages or []}
            elif name == "search_literature":
                inner = {"papers": papers or []}
            elif name == "get_paper_content":
                inner = content or {}
            data = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "structuredContent": {"result": json.dumps(inner)},
                    "content": [],
                    "isError": False,
                },
            }
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=_sse(data)
            )
        return httpx.Response(200, text=_sse({"jsonrpc": "2.0", "id": 2, "result": {}}))

    return handler


# --- pure parsers ---------------------------------------------------------


def test_parse_passages() -> None:
    out = _passages_from_result(
        {"passages": [{"text": "t", "paper_id": "W1", "location": "p1"}, {"text": ""}]}
    )
    assert len(out) == 1
    assert out[0].paper_id == "W1"


def test_parse_passages_prefers_source_doi() -> None:
    """Fix 1: ``source_doi`` is the real passage id key and wins over fallbacks."""
    out = _passages_from_result(
        {"passages": [{"text": "t", "source_doi": "10.x/real", "paper_id": "W1", "doi": "10.x/y"}]}
    )
    assert out[0].paper_id == "10.x/real"


def test_parse_corpus() -> None:
    corpus = _corpus_from_result(
        {"papers": [{"id": "W1", "title": "T", "doi": "10.x/y", "authors": ["A B"]}]}, kb_id="kb9"
    )
    assert corpus.kb_id == "kb9"
    assert corpus.papers[0].authors[0].name == "A B"


def test_parse_corpus_prefers_doi_as_id() -> None:
    """Fix 2: DOI becomes Paper.id so passage paper_id (a DOI) joins with papers."""
    corpus = _corpus_from_result(
        {"papers": [{"id": "W1", "title": "T", "doi": "10.x/y"}]}, kb_id="kb9"
    )
    assert corpus.papers[0].id == "10.x/y"
    assert corpus.by_id("10.x/y") is not None


def test_passage_paper_id_joins_corpus_paper_id() -> None:
    """Fixes 1+2 together: a passage's source_doi resolves to a corpus paper."""
    passages = _passages_from_result({"passages": [{"text": "t", "source_doi": "10.x/y"}]})
    corpus = _corpus_from_result(
        {"papers": [{"id": "W1", "title": "T", "doi": "10.x/y"}]}, kb_id="kb9"
    )
    assert corpus.by_id(passages[0].paper_id) is not None


def test_parse_sse_and_unwrap() -> None:
    msg = _parse_sse(_sse({"jsonrpc": "2.0", "id": 2, "result": {"ok": 1}}))
    assert msg["result"] == {"ok": 1}
    inner = _unwrap_tool_result({"structuredContent": {"result": json.dumps({"papers": [1]})}})
    assert inner == {"papers": [1]}


# --- MCP transport (mocked) ----------------------------------------------


@respx.mock
def test_relevant_passages_via_mcp() -> None:
    respx.post(URL).mock(side_effect=_mcp_handler(passages=[{"text": "x", "paper_id": "W1"}]))
    backend = PerspicaciteBackend(URL)
    passages = backend.relevant_passages("q", "kb")
    assert passages[0].text == "x"


@respx.mock
def test_build_kb_via_mcp() -> None:
    respx.post(URL).mock(
        side_effect=_mcp_handler(papers=[{"id": "W1", "title": "GNNs", "year": 2021}])
    )
    backend = PerspicaciteBackend(URL)
    corpus = backend.build_or_select_kb("graph neural networks")
    assert corpus.papers[0].title == "GNNs"
    assert corpus.kb_id == "graph-neural-networks"


@respx.mock
def test_relevant_passages_uses_source_doi_via_mcp() -> None:
    """Fix 1 end-to-end: source_doi from the server lands as the passage paper_id."""
    respx.post(URL).mock(side_effect=_mcp_handler(passages=[{"text": "x", "source_doi": "10.x/y"}]))
    backend = PerspicaciteBackend(URL)
    passages = backend.relevant_passages("q", "kb")
    assert passages[0].paper_id == "10.x/y"


@respx.mock
def test_paper_content_sends_doi_arg_and_reads_full_text() -> None:
    """Fix 3: get_paper_content is called with a ``doi`` arg and reads full_text first."""
    captured: dict = {}
    respx.post(URL).mock(side_effect=_mcp_handler(content={"full_text": "FULL"}, captured=captured))
    backend = PerspicaciteBackend(URL)
    text = backend.paper_content("10.x/y")
    assert text == "FULL"
    assert captured["name"] == "get_paper_content"
    assert captured["arguments"] == {"doi": "10.x/y"}


@respx.mock
def test_paper_content_falls_back_to_abstract() -> None:
    """Fix 3: when full_text is absent, abstract is the next preferred key."""
    respx.post(URL).mock(side_effect=_mcp_handler(content={"abstract": "ABS"}))
    backend = PerspicaciteBackend(URL)
    assert backend.paper_content("10.x/y") == "ABS"


@respx.mock
def test_app_level_success_false_raises_unavailable() -> None:
    """Fix 4: a payload with success=False is an app-level failure, not swallowed."""
    respx.post(URL).mock(
        side_effect=_mcp_handler(content={"success": False, "error": "kb missing"})
    )
    backend = PerspicaciteBackend(URL)
    with pytest.raises(PerspicaciteUnavailable, match="kb missing"):
        backend.paper_content("10.x/y")


@respx.mock
def test_app_level_error_field_raises_unavailable() -> None:
    """Fix 4: a payload carrying an ``error`` field raises with the error text."""
    respx.post(URL).mock(side_effect=_mcp_handler(papers=None, content={"error": "boom-detail"}))
    backend = PerspicaciteBackend(URL)
    with pytest.raises(PerspicaciteUnavailable, match="boom-detail"):
        backend.paper_content("10.x/y")


@respx.mock
def test_connect_error_raises_unavailable() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("boom"))
    backend = PerspicaciteBackend(URL)
    with pytest.raises(PerspicaciteUnavailable, match="not reachable"):
        backend.build_or_select_kb("topic")


# --- unwrap / parse edge cases -------------------------------------------


def test_unwrap_tool_result_variants() -> None:
    # structuredContent.result as a dict (not a JSON string)
    assert _unwrap_tool_result({"structuredContent": {"result": {"a": 1}}}) == {"a": 1}
    # structuredContent without a "result" key -> the structured dict itself
    assert _unwrap_tool_result({"structuredContent": {"x": 2}}) == {"x": 2}
    # content text block carrying JSON
    assert _unwrap_tool_result({"content": [{"type": "text", "text": '{"b": 2}'}]}) == {"b": 2}
    # content text block with non-JSON text
    assert _unwrap_tool_result({"content": [{"type": "text", "text": "hello"}]}) == {
        "text": "hello"
    }
    # nothing usable
    assert _unwrap_tool_result({}) == {}


def test_parse_sse_plain_json_and_notification_only() -> None:
    assert _parse_sse('{"result": {"ok": 1}}')["result"] == {"ok": 1}
    assert _parse_sse("not json at all") == {}
    # an SSE stream with only a notification (no result/error) returns the last msg
    assert _parse_sse('data: {"method": "notifications/x"}\n\n') == {"method": "notifications/x"}


# --- transport error paths ------------------------------------------------


def _init_then(handler_for_call):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get("method")
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "s", "content-type": "text/event-stream"},
                text=_sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
            )
        if method == "notifications/initialized":
            return httpx.Response(202, text="")
        return handler_for_call(body)

    return handler


@respx.mock
def test_http_500_raises_unavailable() -> None:
    respx.post(URL).mock(side_effect=_init_then(lambda body: httpx.Response(500, text="boom")))
    with pytest.raises(PerspicaciteUnavailable, match="HTTP error"):
        PerspicaciteBackend(URL).relevant_passages("q", "kb")


@respx.mock
def test_jsonrpc_error_raises_unavailable() -> None:
    respx.post(URL).mock(
        side_effect=_init_then(
            lambda body: httpx.Response(200, text=_sse({"error": {"code": -1, "message": "nope"}}))
        )
    )
    with pytest.raises(PerspicaciteUnavailable, match="failed"):
        PerspicaciteBackend(URL).relevant_passages("q", "kb")


@respx.mock
def test_paper_content_via_mcp_reads_full_text() -> None:
    respx.post(URL).mock(
        side_effect=_init_then(
            lambda body: httpx.Response(
                200,
                text=_sse(
                    {"result": {"structuredContent": {"result": json.dumps({"full_text": "FULL"})}}}
                ),
            )
        )
    )
    assert PerspicaciteBackend(URL).paper_content("10.x/y") == "FULL"
