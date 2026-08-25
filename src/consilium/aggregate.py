"""Load the artifacts produced during a run, for aggregation stages."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from consilium.io.store import OutputStore


def load_run(store: OutputStore) -> dict[str, Any]:
    """Collect THIS run's artifacts into a structured dict.

    Reads only the files written during the current run (``store.written``), never
    stale files left in a shared output directory by a prior run/topic.

    Returns ``{run_dir, astra: {stage: collection}, indicium: {stage: doc}, other:
    {name: data}, markdown: [names]}``.
    """
    astra: dict[str, Any] = {}
    indicium: dict[str, Any] = {}
    other: dict[str, Any] = {}
    markdown: list[str] = []
    for name in store.written:
        path = store.path(name)
        if name.endswith(".md"):
            markdown.append(name)
            continue
        if not name.endswith(".json"):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if name.endswith(".astra.json"):
            astra[name[: -len(".astra.json")]] = data
        elif name.endswith(".indicium.json"):
            indicium[name[: -len(".indicium.json")]] = data
        else:
            other[name[: -len(".json")]] = data
    return {
        "run_dir": str(store.run_dir),
        "astra": astra,
        "indicium": indicium,
        "other": other,
        "markdown": sorted(markdown),
    }


def insight_counts(run: dict[str, Any]) -> dict[str, int]:
    """Number of insights per ASTRA artifact stage."""
    return {stage: len(coll.get("insights", [])) for stage, coll in run["astra"].items()}
