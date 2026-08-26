"""Export lfx Insights insights to an ASTRA ``InsightCollection``-shaped dict.

Field names match the real ASTRA LinkML schema (``astra-spec``): an Insight uses
``claim`` (the finding text), ``derived`` (synthesized?), ``scope`` (conditions),
``notes`` (reasoning), ``tags``, and ``evidence`` (records with id/doi/quote/location).
An InsightCollection carries ``insights`` (+ ``notes``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lfx_insights.models import Insight


def astra_available() -> bool:
    try:
        import astra  # noqa: F401

        return True
    except ImportError:
        return False


def _insight_to_dict(insight: Insight, index: int) -> dict[str, Any]:
    return {
        "id": f"insight_{index}",
        "claim": insight.statement,
        "derived": insight.is_synthesized,
        "scope": insight.conditions,
        "notes": insight.reasoning,
        "tags": insight.tags,
        "evidence": [
            {
                "id": f"ev_{index}_{j}",
                "doi": e.paper_id,
                "quote": e.quote,
                "location": e.location,
            }
            for j, e in enumerate(insight.evidence)
        ],
    }


def insights_to_collection(
    insights: list[Insight], *, title: str = "Consilium insights"
) -> dict[str, Any]:
    """Build an ASTRA InsightCollection-shaped dict from lfx Insights insights."""
    return {
        "insights": [_insight_to_dict(ins, i) for i, ins in enumerate(insights)],
        "notes": title,
    }
