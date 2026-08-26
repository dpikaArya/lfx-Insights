"""Export a whole lfx Insights run as an asb-schema ``SciTaskCapsule``-shaped dict.

Top-level slot names match the real asb-schema ``SciTaskCapsule``
(``capsule_task_id``/``capsule_card``/``capsule_artifacts``/``capsule_created_at``/
``artifact_provenance``/``ro_crate_version``). The embedded ``capsule_card`` is a
minimal ASBTaskCard carrying the ``research_question``; a full Task Card (ports,
execution spec) is out of scope for a lfx Insights run bundle.
"""

from __future__ import annotations

import re
from typing import Any


def asb_available() -> bool:
    try:
        import asb_schema  # noqa: F401

        return True
    except ImportError:
        return False


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:64] or "consilium"


def run_to_capsule(
    topic: str, run: dict[str, Any], *, model: str | None = None, created: str | None = None
) -> dict[str, Any]:
    """Bundle a loaded run (see :func:`consilium.aggregate.load_run`) into a minimal
    SciTaskCapsule-shaped dict: research question (in the card), artifacts, provenance.
    """
    artifacts: list[dict[str, Any]] = []
    for name in run.get("markdown", []):
        artifacts.append({"artifact_id": name, "media_type": "text/markdown", "role": "report"})
    for stage in run.get("astra", {}):
        artifacts.append(
            {
                "artifact_id": f"{stage}.astra.json",
                "media_type": "application/json",
                "role": "insights",
            }
        )
    for stage in run.get("indicium", {}):
        artifacts.append(
            {
                "artifact_id": f"{stage}.indicium.json",
                "media_type": "application/json",
                "role": "claims",
            }
        )
    return {
        "capsule_task_id": _slug(topic),
        "ro_crate_version": "1.1",
        "capsule_card": {"research_question": topic, "generated_by": "consilium", "model": model},
        "capsule_artifacts": artifacts,
        "capsule_created_at": created,
        "artifact_provenance": [
            {"file_path": a["artifact_id"], "source": "consilium"} for a in artifacts
        ],
    }
