"""PerspicacitÃ© MCP adapter â€” lfx Insights only literature backend.

Speaks MCP over streamable-HTTP (FastMCP): an ``initialize`` handshake establishes
a session, then ``tools/call`` requests are issued; responses arrive as SSE events
whose tool payload is a JSON string under ``result.structuredContent.result``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from lfx_insights import __version__
from lfx_insights.errors import PerspicaciteUnavailable
from lfx_insights.models import Author, Corpus, Paper, Passage

_PROTOCOL_VERSION = "2025-06-18"
_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:64] or "consilium"


def _passages_from_result(payload: dict[str, Any]) -> list[Passage]:
    """Parse a ``get_relevant_passages`` payload into Passages."""
    items = payload.get("passages") or payload.get("results") or []
    out: list[Passage] = []
    for it in items:
        text = it.get("text") or it.get("passage") or ""
        if not text:
            continue
        out.append(
            Passage(
                paper_id=str(
                    it.get("source_doi")
                    or it.get("paper_id")
                    or it.get("doi")
                    or it.get("id")
                    or ""
                ),
                text=text,
                location=it.get("location") or it.get("section"),
            )
        )
    return out


def _corpus_from_result(payload: dict[str, Any], kb_id: str) -> Corpus:
    """Parse a ``search_literature`` / KB payload into a Corpus."""
    items = payload.get("papers") or payload.get("results") or []
    papers: list[Paper] = []
    for it in items:
        authors = [
            Author(name=a) if isinstance(a, str) else Author(name=a.get("name", ""))
            for a in (it.get("authors") or [])
        ]
        papers.append(
            Paper(
                id=str(it.get("doi") or it.get("id") or it.get("paper_id") or it.get("title", "")),
                title=it.get("title") or "",
                doi=it.get("doi"),
                authors=authors,
                year=it.get("year"),
                abstract=it.get("abstract"),
                source=it.get("source") or "perspicacite",
                url=it.get("url"),
            )
        )
    return Corpus(kb_id=kb_id, papers=papers)


def _parse_sse(text: str) -> dict[str, Any]:
    """Return the JSON-RPC message from an SSE (or plain-JSON) response body."""
    messages: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                messages.append(json.loads(line[len("data:") :].strip()))
            except json.JSONDecodeError:
                continue
    if not messages:
        try:
            return dict(json.loads(text))
        except json.JSONDecodeError:
            return {}
    # Prefer the message that carries the response (result/error) over notifications.
    for msg in messages:
        if "result" in msg or "error" in msg:
            return msg
    return messages[-1]


def _unwrap_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a CallToolResult into the tool's structured payload dict."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        inner = structured.get("result")
        if isinstance(inner, str):
            try:
                return dict(json.loads(inner))
            except json.JSONDecodeError:
                return {"text": inner}
        if isinstance(inner, dict):
            return inner
        return structured
    for block in result.get("content", []):
        if block.get("type") == "text":
            try:
                return dict(json.loads(block["text"]))
            except json.JSONDecodeError:
                return {"text": block["text"]}
    return {}


class PerspicaciteBackend:
    """Calls a running PerspicacitÃ© MCP server over streamable-HTTP."""

    def __init__(self, url: str, timeout: int = 60) -> None:
        self.url = url
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._session_id: str | None = None
        self._initialized = False

    def close(self) -> None:
        self._client.close()

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = dict(_HEADERS)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            resp = self._client.post(self.url, json=payload, headers=headers)
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise PerspicaciteUnavailable(
                f"PerspicacitÃ© is not reachable at {self.url}. Start the MCP server "
                "and retry. Consilium does not fall back to home-grown search."
            ) from exc
        except httpx.HTTPError as exc:
            raise PerspicaciteUnavailable(f"PerspicacitÃ© HTTP error at {self.url}: {exc}") from exc
        return resp

    def _ensure_session(self) -> None:
        if self._initialized:
            return
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "consilium", "version": __version__},
                },
            }
        )
        self._session_id = resp.headers.get("mcp-session-id")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_session()
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        msg = _parse_sse(resp.text)
        if "error" in msg:
            raise PerspicaciteUnavailable(f"PerspicacitÃ© tool '{name}' failed: {msg['error']}")
        result = msg.get("result", {})
        if isinstance(result, dict) and result.get("isError"):
            raise PerspicaciteUnavailable(f"PerspicacitÃ© tool '{name}' returned an error.")
        payload = _unwrap_tool_result(result if isinstance(result, dict) else {})
        if isinstance(payload, dict) and (payload.get("success") is False or payload.get("error")):
            detail = payload.get("error") or "request was not successful"
            raise PerspicaciteUnavailable(f"PerspicacitÃ© tool '{name}' failed: {detail}")
        return payload

    def build_or_select_kb(self, topic: str, max_papers: int = 30) -> Corpus:
        payload = self._call_tool("search_literature", {"query": topic, "max_results": max_papers})
        return _corpus_from_result(payload, kb_id=_slug(topic))

    def relevant_passages(self, query: str, kb_id: str, k: int = 10) -> list[Passage]:
        payload = self._call_tool(
            "get_relevant_passages", {"query": query, "kb_name": kb_id, "k": k}
        )
        return _passages_from_result(payload)

    def paper_content(self, paper_id: str) -> str:
        payload = self._call_tool("get_paper_content", {"doi": paper_id})
        return str(
            payload.get("full_text")
            or payload.get("abstract")
            or payload.get("content")
            or payload.get("text")
            or ""
        )
