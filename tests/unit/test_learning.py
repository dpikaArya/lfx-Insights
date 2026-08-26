"""Tests for approval, adaptive_config, and learning modules."""

from __future__ import annotations

import pytest

from lfx_insights.projects.approval import ApprovalManager, ApprovalRecord
from lfx_insights.projects.adaptive_config import (
    AdaptiveConfig,
    ALLOWED_PARAMETERS,
    ConfigVersion,
)
from lfx_insights.projects.learning import (
    LearningSignal,
    SelfLearner,
    extract_signals,
)
from lfx_insights.projects.self_evaluation import EvaluationResult

pytestmark = pytest.mark.unit


class TestApprovalManager:
    def test_low_risk_auto_approved(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr = ApprovalManager(base)
        record = mgr.request_approval("learning_signal", "s1", "LOW_RISK")
        assert record.state == "APPROVED"

    def test_medium_risk_requires_approval(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr = ApprovalManager(base)
        record = mgr.request_approval("learning_signal", "s1", "MEDIUM_RISK")
        assert record.state == "AI_SUGGESTED"

    def test_approve(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr = ApprovalManager(base)
        record = mgr.request_approval("learning_signal", "s1", "MEDIUM_RISK")
        approved = mgr.approve(record.approval_id)
        assert approved is not None
        assert approved.state == "APPROVED"
        assert approved.reviewed_by == "researcher"

    def test_reject(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr = ApprovalManager(base)
        record = mgr.request_approval("learning_signal", "s1", "MEDIUM_RISK")
        rejected = mgr.reject(record.approval_id, reason="Not needed")
        assert rejected is not None
        assert rejected.state == "REJECTED"

    def test_is_approved(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr = ApprovalManager(base)
        r1 = mgr.request_approval("learning_signal", "s1", "LOW_RISK")
        r2 = mgr.request_approval("learning_signal", "s2", "MEDIUM_RISK")
        assert mgr.is_approved(r1.approval_id)
        assert not mgr.is_approved(r2.approval_id)

    def test_get_pending(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr = ApprovalManager(base)
        mgr.request_approval("learning_signal", "s1", "LOW_RISK")
        mgr.request_approval("learning_signal", "s2", "MEDIUM_RISK")
        pending = mgr.get_pending()
        assert len(pending) == 1
        assert pending[0].risk_level == "MEDIUM_RISK"

    def test_persistence(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        mgr1 = ApprovalManager(base)
        mgr1.request_approval("learning_signal", "s1", "LOW_RISK")
        mgr2 = ApprovalManager(base)
        assert len(mgr2.get_all()) == 1


class TestAdaptiveConfig:
    def test_get_current_defaults(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        current = cfg.get_current()
        assert "retrieval.keyword_weight" in current
        assert current["retrieval.keyword_weight"] == 0.5

    def test_propose_and_promote(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        version = cfg.propose_change("retrieval.keyword_weight", 0.7)
        assert version.parameters["retrieval.keyword_weight"] == 0.7
        validated = cfg.validate_candidate(version.version_id, test_score=0.85)
        assert validated
        promoted = cfg.promote(version.version_id)
        assert promoted
        assert cfg.get_current()["retrieval.keyword_weight"] == 0.7

    def test_propose_invalid_parameter(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        with pytest.raises(ValueError, match="not in the allowlist"):
            cfg.propose_change("source_code.malicious", True)

    def test_propose_out_of_range(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        with pytest.raises(ValueError, match="out of range"):
            cfg.propose_change("retrieval.keyword_weight", 2.0)

    def test_validation_rejects_worse(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        # Promote first version with score 0.8
        v1 = cfg.propose_change("retrieval.keyword_weight", 0.6, score=0.8)
        cfg.validate_candidate(v1.version_id, test_score=0.8)
        cfg.promote(v1.version_id)
        # Propose worse version
        v2 = cfg.propose_change("retrieval.keyword_weight", 0.4, score=0.5)
        validated = cfg.validate_candidate(v2.version_id, test_score=0.5)
        assert not validated

    def test_rollback(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        v1 = cfg.propose_change("retrieval.keyword_weight", 0.6)
        cfg.validate_candidate(v1.version_id, 0.8)
        cfg.promote(v1.version_id)
        v2 = cfg.propose_change("retrieval.keyword_weight", 0.9)
        cfg.validate_candidate(v2.version_id, 0.9)
        cfg.promote(v2.version_id)
        assert cfg.get_current()["retrieval.keyword_weight"] == 0.9
        rolled_back = cfg.rollback()
        assert rolled_back
        assert cfg.get_current()["retrieval.keyword_weight"] == 0.6

    def test_version_history(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        cfg = AdaptiveConfig(base)
        cfg.propose_change("retrieval.keyword_weight", 0.6)
        cfg.propose_change("retrieval.keyword_weight", 0.8)
        history = cfg.get_version_history()
        assert len(history) == 2

    def test_allowed_parameters_complete(self) -> None:
        assert "retrieval.keyword_weight" in ALLOWED_PARAMETERS
        assert "confidence.calibration_offset" in ALLOWED_PARAMETERS
        assert "query.expansion_terms" in ALLOWED_PARAMETERS


class TestLearningSignals:
    def test_extract_signals_from_evaluation(self) -> None:
        ev = EvaluationResult(
            evaluation_id="e1", task_id="t1", task_type="manuscript",
            overall_score=0.3, confidence=0.95,
            missing_evidence=["claim A"],
            citation_warnings=["only 1/5 verified"],
            contradictions=["contradictory point"],
            recommended_action="revise",
            created_at="2025-01-01T00:00:00Z",
        )
        signals = extract_signals(ev)
        types = {s.signal_type for s in signals}
        assert "retrieval_adjustment" in types
        assert "confidence_calibration" in types

    def test_signal_has_parameter_hint(self) -> None:
        ev = EvaluationResult(
            evaluation_id="e1", task_id="t1", task_type="claim",
            overall_score=0.3, confidence=0.95,
            recommended_action="revise",
            created_at="2025-01-01T00:00:00Z",
        )
        signals = extract_signals(ev)
        cal_signals = [s for s in signals if s.signal_type == "confidence_calibration"]
        assert len(cal_signals) == 1
        assert cal_signals[0].parameter_hint == "confidence.calibration_offset"


class TestSelfLearner:
    def test_process_evaluation(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        learner = SelfLearner(base)
        ev = EvaluationResult(
            evaluation_id="e1", task_id="t1", task_type="claim",
            overall_score=0.3, confidence=0.95,
            missing_evidence=["claim A"],
            recommended_action="revise",
            created_at="2025-01-01T00:00:00Z",
        )
        signals = learner.process_evaluation(ev)
        assert len(signals) > 0
        assert len(learner.get_signals()) > 0

    def test_persistence(self, tmp_path: object) -> None:
        import pathlib
        base = tmp_path if isinstance(tmp_path, pathlib.Path) else pathlib.Path(str(tmp_path))
        learner1 = SelfLearner(base)
        ev = EvaluationResult(
            evaluation_id="e1", task_id="t1", task_type="claim",
            overall_score=0.3, confidence=0.95,
            recommended_action="revise",
            created_at="2025-01-01T00:00:00Z",
        )
        learner1.process_evaluation(ev)
        learner2 = SelfLearner(base)
        assert len(learner2.get_signals()) > 0
