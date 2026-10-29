"""Calibration audit: metric arithmetic, replay equivalence, findings, report."""

import json
import math

import pytest

from deeptutor.learning.calibration import (
    MIN_EVENTS,
    MIN_TARGET_EVENTS,
    CalibrationReport,
    Prediction,
    audit,
    collect_predictions,
    compute_metrics,
    render_markdown,
    replay_state,
    report_to_dict,
    retention_scale,
    wilson_interval,
)
from deeptutor.learning.models import (
    KnowledgeType,
    LearningEvidence,
    LearningProgress,
    RepetitionState,
)
from deeptutor.learning.scheduler import SpacedRepetitionScheduler

from ._simulate import simulate_learner

DAY = 86400.0


def _prediction(predicted: float, success: bool, **overrides) -> Prediction:
    base = {
        "knowledge_point_id": "kp",
        "knowledge_type": KnowledgeType.MEMORY,
        "timestamp": 0.0,
        "predicted": predicted,
        "quality": 1.0 if success else 0.0,
        "success": success,
        "spacing_days": 1.0,
        "due_offset_days": 0.0,
        "at_target": True,
        "stability_days": 1.0,
        "assessment_type": "review",
        "source": "test",
    }
    base.update(overrides)
    return Prediction(**base)


def _evidence(kp_id: str, when: float, success: bool) -> LearningEvidence:
    return LearningEvidence(
        knowledge_point_id=kp_id,
        timestamp=when,
        source="test",
        assessment_type="review",
        result="correct" if success else "incorrect",
        quality=1.0 if success else 0.0,
    )


# ── metric arithmetic ──────────────────────────────────────────────────


class TestMetrics:
    def test_constant_half_calibration_is_perfect_but_blind(self):
        predictions = [_prediction(0.5, index % 2 == 0) for index in range(40)]
        metrics = compute_metrics(predictions)
        assert metrics.events == 40
        assert metrics.observed_rate == pytest.approx(0.5)
        assert metrics.bias == pytest.approx(0.0)
        assert metrics.ece == pytest.approx(0.0)
        assert metrics.brier == pytest.approx(0.25)
        assert metrics.brier_baseline == pytest.approx(0.25)
        assert metrics.skill_score == pytest.approx(0.0)
        # Every score is identical, so there is no separation to measure: AUC is
        # the coin flip, not "undefined" — the distinction matters when the
        # report says "no signal".
        assert metrics.auc == pytest.approx(0.5)

    def test_single_class_samples_have_no_auc(self):
        metrics = compute_metrics([_prediction(0.8, True) for _ in range(5)])
        assert metrics.auc is None

    def test_confident_and_correct_has_zero_loss(self):
        metrics = compute_metrics([_prediction(1.0, True) for _ in range(10)])
        assert metrics.brier == pytest.approx(0.0)
        assert metrics.log_loss == pytest.approx(0.0, abs=1e-5)
        assert metrics.observed_rate == pytest.approx(1.0)
        assert metrics.reliability[-1].count == 10
        assert metrics.reliability[-1].observed_rate == pytest.approx(1.0)

    def test_overconfidence_is_signed(self):
        # Believes 90%, delivers 50%.
        predictions = [_prediction(0.9, index < 10) for index in range(20)]
        metrics = compute_metrics(predictions)
        assert metrics.bias == pytest.approx(0.4)
        assert metrics.ece == pytest.approx(0.4)
        assert metrics.brier > metrics.brier_baseline * 0.9

    def test_auc_orders_remembered_above_forgotten(self):
        predictions = [_prediction(0.9, True), _prediction(0.8, True)]
        predictions += [_prediction(0.3, False), _prediction(0.2, False)]
        metrics = compute_metrics(predictions)
        assert metrics.auc == pytest.approx(1.0)

    def test_auc_averages_ties(self):
        predictions = [_prediction(0.5, True), _prediction(0.5, False)]
        metrics = compute_metrics(predictions)
        assert metrics.auc == pytest.approx(0.5)

    def test_reliability_bins_are_equal_width_and_complete(self):
        metrics = compute_metrics([_prediction(0.05, False), _prediction(0.95, True)])
        assert len(metrics.reliability) == 10
        assert [item.count for item in metrics.reliability][0] == 1
        assert [item.count for item in metrics.reliability][9] == 1
        assert sum(item.count for item in metrics.reliability) == 2

    def test_empty_input_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError):
            compute_metrics([])

    def test_mce_ignores_a_single_event_bin(self):
        predictions = [_prediction(0.9, index < 90) for index in range(100)]
        predictions.append(_prediction(0.25, False))  # one event, 0.25 off
        metrics = compute_metrics(predictions)
        assert metrics.ece < 0.05
        assert metrics.mce < 0.2

    def test_mce_survives_samples_below_the_bin_floor(self):
        metrics = compute_metrics([_prediction(0.05, False), _prediction(0.95, True)])
        assert math.isfinite(metrics.mce)
        assert metrics.mce == pytest.approx(0.05)


class TestWilsonInterval:
    def test_extremes_stay_inside_the_unit_interval(self):
        low, high = wilson_interval(0, 10)
        assert low == pytest.approx(0.0)
        assert 0.2 < high < 0.35
        low, high = wilson_interval(10, 10)
        assert high == pytest.approx(1.0)
        assert 0.65 < low < 0.8

    def test_widens_as_evidence_shrinks(self):
        _, wide = wilson_interval(5, 10)
        _, narrow = wilson_interval(500, 1000)
        assert wide > narrow

    def test_no_events_covers_everything(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


# ── replay ─────────────────────────────────────────────────────────────


class TestCollectPredictions:
    def test_first_answer_is_certain_and_not_yet_spaced(self):
        progress = LearningProgress(book_id="p", desired_retention=0.9)
        progress.knowledge_types["kp"] = KnowledgeType.MEMORY
        progress.learning_evidence.append(_evidence("kp", 1000.0, True))
        predictions = collect_predictions(progress)
        assert len(predictions) == 1
        assert predictions[0].predicted == pytest.approx(1.0)
        assert predictions[0].at_target is False
        assert predictions[0].spacing_days == 0.0

    def test_append_order_is_the_history_not_the_timestamp_order(self):
        """A delayed cross-surface write keeps its source timestamp.

        The scheduler replays the list as stored, so the audit must too: sorting
        by timestamp here would grade the learner against a history they never
        lived.
        """

        progress = LearningProgress(book_id="p", desired_retention=0.9)
        progress.knowledge_types["kp"] = KnowledgeType.MEMORY
        progress.learning_evidence.append(_evidence("kp", 3000.0, True))
        progress.learning_evidence.append(_evidence("kp", 2000.0, False))
        predictions = collect_predictions(progress)
        assert [item.timestamp for item in predictions] == [3000.0, 2000.0]
        # The late-arriving event cannot move the clock backwards.
        assert predictions[1].spacing_days == 0.0

    def test_same_session_practice_is_not_counted_as_spaced(self):
        progress = LearningProgress(book_id="p", desired_retention=0.9)
        progress.knowledge_types["kp"] = KnowledgeType.MEMORY
        progress.learning_evidence.append(_evidence("kp", 1000.0, True))
        progress.learning_evidence.append(_evidence("kp", 1000.0 + 600.0, True))
        predictions = collect_predictions(progress)
        assert predictions[0].at_target is False
        assert predictions[1].at_target is False
        assert predictions[1].spacing_days == pytest.approx(600.0 / DAY)

    def test_spaced_review_after_days_is_counted(self):
        progress = LearningProgress(book_id="p", desired_retention=0.9)
        progress.knowledge_types["kp"] = KnowledgeType.CONCEPT
        progress.learning_evidence.append(_evidence("kp", 1000.0, True))
        progress.learning_evidence.append(_evidence("kp", 1000.0 + 20 * DAY, True))
        predictions = collect_predictions(progress)
        assert predictions[1].at_target is True
        assert predictions[1].due_offset_days > 0

    def test_audit_does_not_mutate_the_snapshot(self):
        scenario = simulate_learner(1, knowledge_points=3, events_per_point=4)
        before = scenario.progress.model_dump()
        audit(scenario.progress)
        assert scenario.progress.model_dump() == before

    def test_replay_matches_the_state_the_app_stored(self):
        scenario = simulate_learner(2, knowledge_points=4, events_per_point=6)
        report = audit(scenario.progress)
        assert report.divergences == ()
        for kp_id, stored in scenario.progress.repetition_states.items():
            replayed = replay_state(scenario.progress, kp_id)
            assert replayed is not None
            assert replayed.review_count == stored.review_count
            assert replayed.lapse_count == stored.lapse_count
            assert replayed.stability == pytest.approx(stored.stability, rel=1e-9)


# ── stability ratio ────────────────────────────────────────────────────


class TestRetentionScale:
    def test_needs_enough_spaced_reviews(self):
        predictions = [_prediction(0.9, True) for _ in range(MIN_TARGET_EVENTS - 1)]
        assert retention_scale(predictions, target=0.9) is None

    def test_recovers_a_calibrated_learner(self):
        scenario = simulate_learner(
            0, stability_ratio=1.0, knowledge_points=14, events_per_point=16
        )
        estimate = audit(scenario.progress).retention_scale
        assert estimate is not None
        assert estimate.low < 1.0 < estimate.high
        assert 0.7 < estimate.scale < 1.4

    def test_detects_a_learner_who_forgets_faster(self):
        scenario = simulate_learner(
            0, stability_ratio=0.5, knowledge_points=14, events_per_point=16
        )
        estimate = audit(scenario.progress).retention_scale
        assert estimate is not None
        assert estimate.scale < 0.7
        assert estimate.recommended_target > 0.9  # shorter intervals, higher target

    def test_detects_a_learner_who_retains_longer(self):
        scenario = simulate_learner(
            0, stability_ratio=2.0, knowledge_points=20, events_per_point=25
        )
        estimate = audit(scenario.progress).retention_scale
        assert estimate is not None
        assert estimate.scale > 1.3
        assert estimate.recommended_target < 0.9  # longer intervals, lower target

    def test_lateness_alone_is_not_miscalibration(self):
        """A learner who answers three days late recalls less, but the model
        predicted less too. The estimate must stay at parity while the observed
        rate drops well below the target.
        """

        scenario = simulate_learner(
            3,
            stability_ratio=1.0,
            knowledge_points=16,
            events_per_point=20,
            lateness_days=3.0,
        )
        report = audit(scenario.progress)
        estimate = report.retention_scale
        assert estimate is not None
        assert estimate.low < 1.0 < estimate.high
        assert estimate.mean_lateness_days > 1.0
        assert estimate.observed_rate < 0.9

    def test_is_deterministic(self):
        scenario = simulate_learner(5, knowledge_points=8, events_per_point=10)
        first = audit(scenario.progress).retention_scale
        second = audit(scenario.progress).retention_scale
        assert first == second


# ── findings ───────────────────────────────────────────────────────────


class TestFindings:
    def test_small_snapshots_refuse_to_judge(self):
        scenario = simulate_learner(0, knowledge_points=2, events_per_point=5)
        report = audit(scenario.progress)
        assert [item.code for item in report.findings] == ["calibration.insufficient_evidence"]
        assert report.findings[0].severity == "P3"
        # The numbers are still reported; only the verdict is withheld.
        assert report.metrics is not None
        assert report.metrics.events == 10
        assert report.metrics.events < MIN_EVENTS
        assert report.worst_severity == "P3"

    def test_calibrated_learner_gets_no_high_severity_finding(self):
        for seed in range(4):
            scenario = simulate_learner(
                seed, stability_ratio=1.0, knowledge_points=20, events_per_point=25
            )
            report = audit(scenario.progress)
            severe = [item.code for item in report.findings if item.severity in ("P0", "P1")]
            assert severe == [], f"seed {seed}: {severe}"
            assert any(item.code == "calibration.stability_matches" for item in report.findings)

    def test_forgetting_faster_is_reported_as_optimistic(self):
        scenario = simulate_learner(
            0, stability_ratio=0.4, knowledge_points=20, events_per_point=25
        )
        report = audit(scenario.progress)
        mismatch = [
            item for item in report.findings if item.code == "calibration.stability_mismatch"
        ]
        assert len(mismatch) == 1
        assert mismatch[0].severity == "P1"
        assert "optimistic" in mismatch[0].title
        assert mismatch[0].evidence["recommended_target"] > 0.9

    def test_retaining_longer_is_reported_as_pessimistic(self):
        scenario = simulate_learner(
            0, stability_ratio=3.0, knowledge_points=20, events_per_point=25
        )
        report = audit(scenario.progress)
        mismatch = [
            item for item in report.findings if item.code == "calibration.stability_mismatch"
        ]
        assert len(mismatch) == 1
        assert "pessimistic" in mismatch[0].title

    def test_stored_counter_drift_is_surfaced(self):
        scenario = simulate_learner(4, knowledge_points=4, events_per_point=5)
        first = next(iter(scenario.progress.repetition_states.values()))
        first.review_count += 3
        report = audit(scenario.progress)
        codes = [item.code for item in report.findings]
        assert "calibration.state_divergence" in codes
        divergence = next(
            item for item in report.findings if item.code == "calibration.state_divergence"
        )
        assert divergence.evidence["count"] >= 1
        assert divergence.evidence["sample"][0]["fields"] == ["review_count"]

    def test_bias_is_reported_when_it_is_large(self):
        progress = LearningProgress(book_id="bias", desired_retention=0.9)
        scheduler = SpacedRepetitionScheduler(desired_retention=0.9)
        now = 1_000_000.0
        for index in range(30):
            kp_id = f"kp{index:02d}"
            progress.knowledge_types[kp_id] = KnowledgeType.CONCEPT
            state = scheduler.get_initial_state(KnowledgeType.CONCEPT, now=now)
            progress.repetition_states[kp_id] = state
            # Every answer is a failure, the model keeps believing otherwise.
            event = _evidence(kp_id, now, False)
            progress.learning_evidence.append(event)
        report = audit(progress)
        codes = {item.code for item in report.findings}
        assert "calibration.overconfident" in codes

    def test_auc_warning_needs_substance(self):
        scenario = simulate_learner(6, knowledge_points=4, events_per_point=4)
        report = audit(scenario.progress)
        assert "calibration.non_discriminating" not in {item.code for item in report.findings}


# ── report ─────────────────────────────────────────────────────────────


class TestReport:
    def test_dict_is_json_serialisable_and_ordered(self):
        scenario = simulate_learner(7, knowledge_points=6, events_per_point=8)
        payload = report_to_dict(audit(scenario.progress))
        text = json.dumps(payload, allow_nan=False)
        assert '"path_id": "simulated-path"' in text
        assert payload["events"] == 48
        assert set(payload) == {
            "path_id",
            "generated_at",
            "desired_retention",
            "events",
            "spaced_events",
            "metrics",
            "at_target",
            "per_type",
            "worst_knowledge_points",
            "retention_scale",
            "divergences",
            "findings",
        }

    def test_markdown_carries_the_number_that_matters(self):
        scenario = simulate_learner(
            0, stability_ratio=0.4, knowledge_points=20, events_per_point=25
        )
        report = audit(scenario.progress)
        text = render_markdown(report)
        assert report.retention_scale is not None
        assert "## Findings" in text
        assert "## Reliability" in text
        assert "## By knowledge type" in text
        assert "## Stability ratio and target feasibility" in text
        assert f"{report.retention_scale.scale:.2f}x" in text
        assert "deeptutor/learning/calibration.py" in text

    def test_empty_path_renders_without_crashing(self):
        report = audit(LearningProgress(book_id="empty"))
        assert report.metrics is None
        text = render_markdown(report)
        assert "Retention calibration — empty" in text
        assert "insufficient_evidence" in text
        json.dumps(report_to_dict(report), allow_nan=False)

    def test_worst_severity_tracks_findings(self):
        scenario = simulate_learner(
            0, stability_ratio=0.4, knowledge_points=20, events_per_point=25
        )
        report = audit(scenario.progress)
        assert isinstance(report, CalibrationReport)
        assert report.worst_severity == "P1"
