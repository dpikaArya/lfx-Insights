"""Self-learning — controlled adaptation via learning signals.

Sequence: Self Evaluation -> Human Approval -> Learning Signal ->
Self Learning -> Validation -> Promotion

Self Learning may ONLY modify parameters in the adaptive_config
allowlist. It never modifies source code, dependencies, or security.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lfx_insights.projects.approval import ApprovalManager

if TYPE_CHECKING:
    from lfx_insights.projects.self_evaluation import EvaluationResult

# ---------------------------------------------------------------------------
# Learning signal — compact, actionable observation
# ---------------------------------------------------------------------------

class LearningSignal(BaseModel):
    """A compact, actionable observation from evaluation results."""

    signal_id: str
    signal_type: str  # retrieval_adjustment | confidence_calibration | workflow_pattern | ...
    source: str  # task type that produced this signal
    observation: str
    confidence: float = Field(ge=0.0, le=1.0)
    parameter_hint: str | None = None  # which adaptive parameter this suggests changing
    suggested_value: Any = None
    risk_level: str = "LOW_RISK"
    evaluation_id: str | None = None
    created_at: str


# ---------------------------------------------------------------------------
# Signal extraction — deterministic analysis of evaluation results
# ---------------------------------------------------------------------------

def extract_signals(evaluation: EvaluationResult) -> list[LearningSignal]:
    """Extract learning signals from an evaluation result.

    Purely deterministic — no LLM calls.
    """
    signals: list[LearningSignal] = []
    now = _now_iso()

    # Signal: insufficient evidence → retrieval adjustment
    if evaluation.missing_evidence:
        signals.append(LearningSignal(
            signal_id=uuid.uuid4().hex[:8],
            signal_type="retrieval_adjustment",
            source=evaluation.task_type,
            observation=f"{len(evaluation.missing_evidence)} claims had insufficient evidence",
            confidence=0.8,
            parameter_hint="retrieval.top_k",
            risk_level="LOW_RISK",
            evaluation_id=evaluation.evaluation_id,
            created_at=now,
        ))

    # Signal: citation warnings → citation ranking
    if evaluation.citation_warnings:
        signals.append(LearningSignal(
            signal_id=uuid.uuid4().hex[:8],
            signal_type="retrieval_adjustment",
            source=evaluation.task_type,
            observation=f"{len(evaluation.citation_warnings)} citation warnings",
            confidence=0.7,
            parameter_hint="retrieval.similarity_threshold",
            risk_level="LOW_RISK",
            evaluation_id=evaluation.evaluation_id,
            created_at=now,
        ))

    # Signal: contradictions → contradiction handling
    if evaluation.contradictions:
        signals.append(LearningSignal(
            signal_id=uuid.uuid4().hex[:8],
            signal_type="retrieval_adjustment",
            source=evaluation.task_type,
            observation=f"{len(evaluation.contradictions)} contradictory evidence points",
            confidence=0.75,
            parameter_hint="retrieval.similarity_threshold",
            risk_level="LOW_RISK",
            evaluation_id=evaluation.evaluation_id,
            created_at=now,
        ))

    # Signal: overconfidence → confidence calibration
    if evaluation.confidence > 0.8 and evaluation.overall_score < 0.5:
        signals.append(LearningSignal(
            signal_id=uuid.uuid4().hex[:8],
            signal_type="confidence_calibration",
            source=evaluation.task_type,
            observation=(
                f"System confidence {evaluation.confidence:.2f}"
                f" but evaluation score {evaluation.overall_score:.2f}"
            ),
            confidence=0.85,
            parameter_hint="confidence.calibration_offset",
            suggested_value=-0.1,
            risk_level="MEDIUM_RISK",
            evaluation_id=evaluation.evaluation_id,
            created_at=now,
        ))

    # Signal: underconfidence → confidence calibration
    if evaluation.confidence < 0.3 and evaluation.overall_score > 0.7:
        signals.append(LearningSignal(
            signal_id=uuid.uuid4().hex[:8],
            signal_type="confidence_calibration",
            source=evaluation.task_type,
            observation=(
                f"System confidence {evaluation.confidence:.2f}"
                f" but evaluation score {evaluation.overall_score:.2f}"
            ),
            confidence=0.8,
            parameter_hint="confidence.calibration_offset",
            suggested_value=0.1,
            risk_level="MEDIUM_RISK",
            evaluation_id=evaluation.evaluation_id,
            created_at=now,
        ))

    # Signal: good evaluation → workflow pattern
    if evaluation.overall_score >= 0.7 and not evaluation.identified_problems:
        signals.append(LearningSignal(
            signal_id=uuid.uuid4().hex[:8],
            signal_type="workflow_pattern",
            source=evaluation.task_type,
            observation=(
                f"Successful {evaluation.task_type}"
                f" with score {evaluation.overall_score:.2f}"
            ),
            confidence=0.6,
            risk_level="LOW_RISK",
            evaluation_id=evaluation.evaluation_id,
            created_at=now,
        ))

    return signals


# ---------------------------------------------------------------------------
# Self learner — applies approved signals to adaptive config
# ---------------------------------------------------------------------------

class SelfLearner:
    """Controlled adaptation: extracts signals, validates candidates, promotes changes.

    Sequence: signals → approval request → validate candidate → promote or reject.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._signals_path = Path(base_dir) / "projects" / "learning_signals.json"
        self._approval = ApprovalManager(base_dir)
        self._config: Any = None  # lazy import to avoid circular
        self._base_dir = Path(base_dir)

    @property
    def config(self) -> Any:
        if self._config is None:
            from lfx_insights.projects.adaptive_config import AdaptiveConfig
            self._config = AdaptiveConfig(self._base_dir)
        return self._config

    def process_evaluation(self, evaluation: EvaluationResult) -> list[LearningSignal]:
        """Extract signals from evaluation and request approval for applicable changes."""
        signals = extract_signals(evaluation)
        for signal in signals:
            self._save_signal(signal)
            if signal.parameter_hint:
                self._approval.request_approval(
                    entity_type="learning_signal",
                    entity_id=signal.signal_id,
                    risk_level=signal.risk_level,
                    reason=signal.observation,
                )
        return signals

    def apply_approved_signals(self) -> list[str]:
        """Apply all approved learning signals to adaptive config.

        Returns list of version_ids created.
        """
        approved = self._approval.get_pending()
        approved = [r for r in self._approval.get_all() if r.state == "APPROVED"]

        version_ids: list[str] = []
        for record in approved:
            if record.entity_type != "learning_signal":
                continue
            signal = self._get_signal(record.entity_id)
            if signal is None or not signal.parameter_hint:
                continue

            current = self.config.get_current()
            old_value = current.get(signal.parameter_hint)
            new_value = signal.suggested_value if signal.suggested_value is not None else old_value

            if new_value == old_value:
                continue

            try:
                version = self.config.propose_change(signal.parameter_hint, new_value)
                version_ids.append(version.version_id)
            except ValueError:
                continue

        return version_ids

    def get_signals(self) -> list[LearningSignal]:
        return self._load_signals()

    def get_pending_signals(self) -> list[LearningSignal]:
        pending_approvals = self._approval.get_pending("learning_signal")
        pending_ids = {r.entity_id for r in pending_approvals}
        return [s for s in self._load_signals() if s.signal_id in pending_ids]

    def _get_signal(self, signal_id: str) -> LearningSignal | None:
        for s in self._load_signals():
            if s.signal_id == signal_id:
                return s
        return None

    def _save_signal(self, signal: LearningSignal) -> None:
        signals = self._load_signals()
        signals.append(signal)
        self._signals_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._signals_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([s.model_dump() for s in signals], indent=2), encoding="utf-8")
        os.replace(tmp, self._signals_path)

    def _load_signals(self) -> list[LearningSignal]:
        if not self._signals_path.exists():
            return []
        try:
            data = json.loads(self._signals_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [LearningSignal.model_validate(s) for s in data]
            return []
        except (json.JSONDecodeError, Exception):
            return []


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
