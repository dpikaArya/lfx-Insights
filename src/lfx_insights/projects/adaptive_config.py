"""Adaptive configuration — allowlisted parameters with versioning and rollback.

Self Learning may ONLY modify parameters in the allowlist below.
All changes are versioned. Rollback reverts to the previous validated version.
Configuration is stored separately from source code.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Allowlist: only these parameters may be modified by Self Learning
# ---------------------------------------------------------------------------

ALLOWED_PARAMETERS: dict[str, dict[str, Any]] = {
    "retrieval.keyword_weight": {"default": 0.5, "min": 0.0, "max": 1.0, "risk": "LOW_RISK"},
    "retrieval.semantic_weight": {"default": 0.5, "min": 0.0, "max": 1.0, "risk": "LOW_RISK"},
    "retrieval.similarity_threshold": {"default": 0.3, "min": 0.0, "max": 1.0, "risk": "LOW_RISK"},
    "retrieval.top_k": {"default": 10, "min": 1, "max": 100, "risk": "LOW_RISK"},
    "query.expansion_terms": {"default": [], "risk": "LOW_RISK"},
    "query.synonym_mappings": {"default": {}, "risk": "LOW_RISK"},
    "confidence.calibration_offset": {
        "default": 0.0, "min": -0.5, "max": 0.5, "risk": "MEDIUM_RISK",
    },
    "confidence.overconfidence_threshold": {
        "default": 0.8, "min": 0.5, "max": 1.0, "risk": "MEDIUM_RISK",
    },
    "ranking.document_ranking_weight": {"default": 0.5, "min": 0.0, "max": 1.0, "risk": "LOW_RISK"},
    "ranking.recency_weight": {"default": 0.3, "min": 0.0, "max": 1.0, "risk": "LOW_RISK"},
    "workflow.preferred_stages": {"default": [], "risk": "MEDIUM_RISK"},
}

BLOCKED_PARAMETERS: set[str] = {
    "source_code", "dependencies", "security", "credentials",
    "ollama_installation", "model_weights", "authentication",
    "database_schema", "validation_rules", "security_policies",
}


# ---------------------------------------------------------------------------
# Versioned configuration
# ---------------------------------------------------------------------------

class ConfigVersion(BaseModel):
    """A versioned snapshot of adaptive configuration."""

    version_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    validated: bool = False
    promoted: bool = False
    created_at: str


class AdaptiveConfig:
    """Versioned adaptive configuration with allowlist enforcement and rollback."""

    def __init__(self, base_dir: str | Path) -> None:
        self._path = Path(base_dir) / "projects" / "adaptive_config.json"

    def get_current(self) -> dict[str, Any]:
        """Get the current (latest promoted) configuration."""
        versions = self._load()
        promoted = [v for v in versions if v.promoted]
        if not promoted:
            return {k: v["default"] for k, v in ALLOWED_PARAMETERS.items()}
        return promoted[-1].parameters

    def get_version_history(self) -> list[ConfigVersion]:
        return self._load()

    def propose_change(
        self,
        parameter: str,
        value: Any,
        score: float | None = None,
    ) -> ConfigVersion:
        """Propose a new configuration version with a candidate change.

        Raises ValueError if parameter is not in the allowlist or value is out of range.
        """
        if parameter not in ALLOWED_PARAMETERS:
            raise ValueError(
                f"Parameter '{parameter}' is not in the allowlist. "
                f"Allowed: {', '.join(sorted(ALLOWED_PARAMETERS))}"
            )

        spec = ALLOWED_PARAMETERS[parameter]
        if (
            "min" in spec and "max" in spec
            and isinstance(value, (int, float))
            and (value < spec["min"] or value > spec["max"])
        ):
                raise ValueError(
                    f"Value {value} out of range [{spec['min']}, {spec['max']}] "
                    f"for parameter '{parameter}'"
                )

        current = self.get_current()
        current[parameter] = value

        import uuid
        version = ConfigVersion(
            version_id=uuid.uuid4().hex[:8],
            parameters=current,
            score=score,
            created_at=_now_iso(),
        )
        versions = self._load()
        versions.append(version)
        self._save(versions)
        return version

    def validate_candidate(self, version_id: str, test_score: float) -> bool:
        """Validate a candidate version against its baseline.

        Returns True if the candidate improves over the previous promoted version.
        """
        versions = self._load()
        candidate = next((v for v in versions if v.version_id == version_id), None)
        if candidate is None:
            return False

        promoted = [v for v in versions if v.promoted and v.version_id != version_id]
        baseline_score = promoted[-1].score if promoted and promoted[-1].score is not None else 0.0

        improved = test_score > baseline_score if baseline_score is not None else True

        data = candidate.model_dump()
        data["validated"] = improved
        for i, v in enumerate(versions):
            if v.version_id == version_id:
                versions[i] = ConfigVersion.model_validate(data)
                break
        self._save(versions)
        return improved

    def promote(self, version_id: str) -> bool:
        """Promote a validated version as the current active configuration."""
        versions = self._load()
        for i, v in enumerate(versions):
            if v.version_id == version_id:
                if not v.validated:
                    return False
                data = v.model_dump()
                data["promoted"] = True
                versions[i] = ConfigVersion.model_validate(data)
                self._save(versions)
                return True
        return False

    def rollback(self, to_version_id: str | None = None) -> bool:
        """Rollback to a specific version, or to the previous promoted version."""
        versions = self._load()
        promoted = [v for v in versions if v.promoted]

        if to_version_id:
            target = next((v for v in versions if v.version_id == to_version_id), None)
            if target is None or not target.promoted:
                return False
        else:
            if len(promoted) < 2:
                return False
            target = promoted[-2]

        import uuid
        rollback_version = ConfigVersion(
            version_id=uuid.uuid4().hex[:8],
            parameters=target.parameters,
            score=target.score,
            validated=True,
            promoted=True,
            created_at=_now_iso(),
        )
        versions.append(rollback_version)
        self._save(versions)
        return True

    def _load(self) -> list[ConfigVersion]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [ConfigVersion.model_validate(v) for v in data] if isinstance(data, list) else []
        except (json.JSONDecodeError, Exception):
            return []

    def _save(self, versions: list[ConfigVersion]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([v.model_dump() for v in versions], indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
