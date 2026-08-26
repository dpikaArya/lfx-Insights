"""lfx Insights as an MCP server â€” expose its capabilities as tools for other agents.

Requires the ``mcp`` extra (``uv sync --extra mcp``). See :func:`build_server`.
"""

from lfx_insights.mcp.server import build_server

__all__ = ["build_server"]
