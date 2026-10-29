"""Retention calibration audit for the adaptive spaced-repetition scheduler.

``scheduler.py`` predicts, for every knowledge point, the probability that the
learner can still retrieve it (``SpacedRepetitionScheduler.retrievability``) and
then schedules the next review from that belief.  Nothing in the package ever
asks the opposite question: *was the belief right?*  Its own docstring says the
baseline is "not a calibrated FSRS fit", and ``services/practice/analytics.py``
only counts questions, mistakes and reviews per calendar day.

This module measures the model against the learner.  It replays the durable
evidence log exactly the way the app replayed it (an append-only
:class:`~deeptutor.learning.models.LearningEvidence` list per knowledge point,
stepped through the real ``schedule_review`` transition), and for every event
records what the scheduler believed *before* the answer arrived next to what
actually happened.  From those pairs it reports:

* **Calibration** — reliability bins, expected/max calibration error, Brier
  score and its skill against the base rate, log loss.
* **Discrimination** — AUC, so a well-calibrated-but-constant predictor cannot
  hide behind a small bias.
* **Target feasibility** — the stability ratio ``true / model`` by maximum
  likelihood over the paired events, the recall actually observed on those
  retrievals, how late learners answer, and the ``desired_retention`` that would
  deliver the promise the current target cannot keep.
* **Integrity** — whether the persisted repetition state still equals the state
  the evidence log implies (a target change or a cross-surface write can make
  them diverge without any evidence saying so).

No LLM, no network, no I/O: the audit is a pure function of a
:class:`~deeptutor.learning.models.LearningProgress`.  Everything is
deterministic, including the bootstrap (fixed seed), so the same snapshot always
renders the same report.

The prediction itself is never reimplemented here: if upstream changes the
forgetting model, the audit measures the new one the next time it runs.  ``retrievability`` and
``schedule_review`` are called on the scheduler instance you pass in, so an
upstream model change flows into the audit instead of drifting away from it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Any, Iterable, Sequence

from deeptutor.learning.models import (
    KnowledgeType,
    LearningEvidence,
    LearningProgress,
    RepetitionState,
)
from deeptutor.learning.scheduler import (
    _SAME_SESSION_DAYS,
    DEFAULT_DESIRED_RETENTION,
    SpacedRepetitionScheduler,
    _resolved_quality,
)

#: A retrieval counts as successful at the same threshold the scheduler uses to
#: decide that an outcome is a failure (``_FAIL_QUALITY``): partial credit below
#: it is, by the model's own definition, a lapse.
SUCCESS_THRESHOLD = 0.5

#: Below this many paired events nothing below is trustworthy.  A learner can
#: answer twenty questions in one evening; calibration needs spaced retrievals
#: more than it needs volume, but these gates stop the report from shouting.
MIN_EVENTS = 20
MIN_TYPE_EVENTS = 15
MIN_TARGET_EVENTS = 8

#: A bias smaller than this is inside the noise a tutor should ignore.
BIAS_TOLERANCE = 0.10
TYPE_BIAS_TOLERANCE = 0.12
WORST_KP_EVENTS = 4
AUC_FLOOR = 0.6

#: The stability-ratio estimate is reported only when its interval is clear of
#: parity by this much; inside it, the model tracks the learner.
SCALE_TOLERANCE = 0.15
#: Search range for the stability ratio.  A learner 5x faster or slower than
#: the model is already outside what a single interval multiplier should fix.
SCALE_RANGE = (0.10, 5.0)

BINS = 10
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 20261029
_SECONDS_PER_DAY = 86400.0
_EPS = 1e-9

_SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


@dataclass(frozen=True)
class Prediction:
    """One scheduler belief paired with the outcome that followed it."""

    knowledge_point_id: str
    knowledge_type: KnowledgeType
    timestamp: float
    #: Retrievability the scheduler believed *before* this event was applied.
    predicted: float
    #: Normalised review strength the scheduler would read from the evidence.
    quality: float
    success: bool
    #: Days since the previous review of this knowledge point (0 for the first).
    spacing_days: float
    #: Days past ``next_review_at`` at the moment of the answer (negative = early).
    due_offset_days: float
    #: True when the scheduler itself would treat the event as a fresh spaced
    #: retrieval rather than same-session practice.
    at_target: bool
    stability_days: float
    assessment_type: str
    source: str


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


@dataclass(frozen=True)
class Metrics:
    events: int
    successes: int
    mean_predicted: float
    observed_rate: float
    #: ``mean_predicted - observed_rate``; positive means overconfident.
    bias: float
    brier: float
    brier_baseline: float
    skill_score: float
    log_loss: float
    ece: float
    mce: float
    auc: float | None
    reliability: tuple[ReliabilityBin, ...]


@dataclass(frozen=True)
class ScaleEstimate:
    """How far the model's stability estimate is from the learner's reality."""

    target: float
    observed_rate: float
    mean_predicted: float
    #: Mean days between the scheduled due time and the answer; the reason the
    #: observed rate is reported next to the ratio instead of instead of it.
    mean_lateness_days: float
    #: ``true_stability / model_stability``: 1.0 = the model is right, 0.5 = the
    #: learner forgets twice as fast, 2.0 = twice as slow.
    scale: float
    low: float
    high: float
    events: int
    #: ``target ** scale``, clamped to the range ``desired_retention`` accepts.
    recommended_target: float
    clamped: bool


@dataclass(frozen=True)
class KpCalibration:
    knowledge_point_id: str
    knowledge_type: KnowledgeType
    events: int
    mean_predicted: float
    observed_rate: float
    bias: float
    brier: float


@dataclass(frozen=True)
class StateDivergence:
    knowledge_point_id: str
    knowledge_type: KnowledgeType
    fields: tuple[str, ...]
    replayed: dict[str, float]
    stored: dict[str, float]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    title: str
    detail: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class CalibrationReport:
    path_id: str
    generated_at: float
    desired_retention: float
    predictions: tuple[Prediction, ...]
    metrics: Metrics | None
    at_target: Metrics | None
    per_type: dict[str, Metrics]
    worst_knowledge_points: tuple[KpCalibration, ...]
    retention_scale: ScaleEstimate | None
    divergences: tuple[StateDivergence, ...]
    findings: tuple[Finding, ...]

    @property
    def worst_severity(self) -> str | None:
        if not self.findings:
            return None
        return min(self.findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 9)).severity


# ── prediction collection ────────────────────────────────────────────────


def _span_days(seconds: float) -> float:
    return float(seconds) / _SECONDS_PER_DAY


def _scheduler_for(progress: LearningProgress, scheduler: Any | None) -> Any:
    if scheduler is not None:
        return scheduler
    return SpacedRepetitionScheduler(desired_retention=progress.desired_retention)


def _evidence_by_point(
    evidence: Iterable[LearningEvidence],
) -> dict[str, list[LearningEvidence]]:
    """Group evidence per knowledge point in append order.

    Order is the aggregate's own order, not a re-sort by timestamp: the
    scheduler's ``replay`` walks the list as stored, and a delayed cross-surface
    write keeps its source timestamp while landing later in the list.  Sorting
    here would produce a *different* history than the one the learner lived.
    """

    grouped: dict[str, list[LearningEvidence]] = {}
    for event in evidence:
        grouped.setdefault(event.knowledge_point_id, []).append(event)
    return grouped


def _same_session_window_days(scheduler: Any, state: RepetitionState) -> float:
    """Mirror ``SpacedRepetitionScheduler.schedule_review``'s replay guard.

    Delegated, not duplicated: the window is
    ``min(_SAME_SESSION_DAYS, interval_for(stability, DEFAULT) * 0.5)``.
    """

    stability = max(state.stability, _EPS)
    interval = scheduler._interval_from_stability(stability, DEFAULT_DESIRED_RETENTION)
    return min(_SAME_SESSION_DAYS, interval * 0.5)


def collect_predictions(
    progress: LearningProgress,
    *,
    scheduler: Any | None = None,
) -> list[Prediction]:
    """Replay ``progress`` and pair every belief with the answer that followed."""

    engine = _scheduler_for(progress, scheduler)
    desired = float(progress.desired_retention or DEFAULT_DESIRED_RETENTION)
    unit = engine._seconds_per_unit()
    predictions: list[Prediction] = []

    for kp_id, events in _evidence_by_point(progress.learning_evidence).items():
        if not events:
            continue
        kp_type = progress.knowledge_types.get(kp_id, KnowledgeType.MEMORY)
        state = engine.get_initial_state(
            kp_type, now=events[0].timestamp, desired_retention=desired
        )
        for event in events:
            # Exactly the anchor ``schedule_review`` uses, so an early-arriving
            # event cannot be scored against a deadline it never moved.
            moment = (
                max(event.timestamp, state.last_review_at)
                if state.last_review_at is not None
                else event.timestamp
            )
            predicted = float(engine.retrievability(state, now=moment))
            quality = float(_resolved_quality(event))
            spacing_days = (
                _span_days(moment - state.last_review_at)
                if state.last_review_at is not None
                else 0.0
            )
            window = _same_session_window_days(engine, state)
            predictions.append(
                Prediction(
                    knowledge_point_id=kp_id,
                    knowledge_type=kp_type,
                    timestamp=float(event.timestamp),
                    predicted=min(1.0, max(0.0, predicted)),
                    quality=min(1.0, max(0.0, quality)),
                    success=quality >= SUCCESS_THRESHOLD,
                    spacing_days=max(0.0, spacing_days),
                    due_offset_days=_span_days(moment - state.next_review_at),
                    at_target=state.last_review_at is not None and spacing_days >= window,
                    # ``stability`` is already in days; only the timestamps need
                    # converting.
                    stability_days=max(0.0, float(state.stability)),
                    assessment_type=event.assessment_type,
                    source=event.source,
                )
            )
            engine.schedule_review(state, kp_type, event, now=moment)
    return predictions


# ── metrics ──────────────────────────────────────────────────────────────


def _auc(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """Rank AUC with ties averaged (Mann–Whitney U over the predicted scores)."""

    positives = sum(1 for _, success in pairs if success)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda item: item[0])
    ranks: list[float] = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    positive_rank_sum = sum(
        rank for rank, (_, success) in zip(ranks, ordered, strict=True) if success
    )
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _reliability(pairs: Sequence[tuple[float, bool]]) -> tuple[ReliabilityBin, ...]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(BINS)]
    for predicted, success in pairs:
        buckets[min(int(predicted * BINS), BINS - 1)].append((predicted, success))
    bins: list[ReliabilityBin] = []
    for position, bucket in enumerate(buckets):
        lower = position / BINS
        upper = (position + 1) / BINS
        if not bucket:
            bins.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0))
            continue
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_predicted=sum(predicted for predicted, _ in bucket) / len(bucket),
                observed_rate=sum(1 for _, success in bucket if success) / len(bucket),
            )
        )
    return tuple(bins)


def compute_metrics(predictions: Sequence[Prediction]) -> Metrics:
    """Calibration and discrimination of a batch of paired beliefs."""

    if not predictions:
        raise ValueError("compute_metrics needs at least one prediction")
    pairs = [(item.predicted, item.success) for item in predictions]
    events = len(pairs)
    successes = sum(1 for _, success in pairs if success)
    observed = successes / events
    mean_predicted = sum(predicted for predicted, _ in pairs) / events
    brier = (
        sum((predicted - (1.0 if success else 0.0)) ** 2 for predicted, success in pairs) / events
    )
    baseline = observed * (1.0 - observed)
    log_loss = (
        -sum(
            math.log(max(predicted if success else 1.0 - predicted, 1e-6))
            for predicted, success in pairs
        )
        / events
    )
    reliability = _reliability(pairs)
    weighted = sum(
        item.count * abs(item.observed_rate - item.mean_predicted) for item in reliability
    )
    ece = weighted / events
    # A single two-event bin must not set the worst-case number: MCE only reads
    # bins holding at least 5% of the events (and at least five of them).
    floor = max(5, int(0.05 * events))
    populated = [item for item in reliability if item.count >= floor]
    mce = max(abs(item.observed_rate - item.mean_predicted) for item in populated)
    return Metrics(
        events=events,
        successes=successes,
        mean_predicted=mean_predicted,
        observed_rate=observed,
        bias=mean_predicted - observed,
        brier=brier,
        brier_baseline=baseline,
        skill_score=(1.0 - brier / baseline) if baseline > 0 else 0.0,
        log_loss=log_loss,
        ece=ece,
        mce=mce,
        auc=_auc(pairs),
        reliability=reliability,
    )


def wilson_interval(successes: int, events: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — sane at small n and at 0/1 successes."""

    if events <= 0:
        return (0.0, 1.0)
    phat = successes / events
    denominator = 1.0 + z * z / events
    centre = (phat + z * z / (2 * events)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / events + z * z / (4 * events * events)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _log_loss_at_ratio(
    log_predictions: Sequence[float], outcomes: Sequence[bool], ratio: float
) -> float:
    """Log loss if the learner's stability were ``ratio`` times the model's.

    ``retrievability`` is exponential, so re-scaling stability is exact:
    ``exp(-t / (ratio * S)) == exp(-t / S) ** (1 / ratio)``.  That is why the
    estimator needs no reconstruction of the elapsed time it cannot see, and
    why a learner who simply answers late is not blamed on the model.
    """

    inverse = 1.0 / ratio
    total = 0.0
    for log_p, outcome in zip(log_predictions, outcomes, strict=True):
        value = math.exp(log_p * inverse)
        if value > 1.0 - 1e-9:
            value = 1.0 - 1e-9
        elif value < 1e-9:
            value = 1e-9
        total -= math.log(value) if outcome else math.log1p(-value)
    return total


def _estimate_ratio(
    log_predictions: Sequence[float], outcomes: Sequence[bool], *, iterations: int = 18
) -> float:
    """Maximum-likelihood stability ratio by ternary search on ``log(ratio)``."""

    low, high = math.log(SCALE_RANGE[0]), math.log(SCALE_RANGE[1])
    for _ in range(iterations):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        if _log_loss_at_ratio(log_predictions, outcomes, math.exp(left)) <= _log_loss_at_ratio(
            log_predictions, outcomes, math.exp(right)
        ):
            high = right
        else:
            low = left
    return math.exp((low + high) / 2.0)


def retention_scale(
    predictions: Sequence[Prediction],
    *,
    target: float,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> ScaleEstimate | None:
    """Estimate how far the model's stability is from this learner's reality.

    Maximum likelihood over the paired events: the ratio ``k`` for which
    ``prediction ** (1 / k)`` best explains the outcomes.  ``k < 1`` means the
    learner forgets faster than the model believes (intervals run long);
    ``k > 1`` means the model is needlessly pessimistic.  Interval by a
    fixed-seed bootstrap, so the same snapshot always yields the same numbers.

    The observed success rate is reported next to it but never substituted for
    it: a learner who answers three days late succeeds less often than the
    target without the model being wrong about anything.
    """

    spaced = [item for item in predictions if item.at_target]
    if len(spaced) < MIN_TARGET_EVENTS:
        return None
    log_predictions = [math.log(min(1.0 - 1e-9, max(1e-9, item.predicted))) for item in spaced]
    outcomes = [item.success for item in spaced]
    scale = _estimate_ratio(log_predictions, outcomes)

    # Cluster bootstrap: events of one knowledge point share its stability and
    # its learner, so resampling events independently would pretend there is
    # more evidence than there is. Groups are resampled whole.
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(spaced):
        groups.setdefault(item.knowledge_point_id, []).append(index)
    members = list(groups.values())
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        picked: list[int] = []
        for _ in range(len(members)):
            picked.extend(members[rng.randrange(len(members))])
        draws.append(
            _estimate_ratio(
                [log_predictions[index] for index in picked],
                [outcomes[index] for index in picked],
            )
        )
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]

    rate = sum(1 for item in spaced if item.success) / len(spaced)
    mean_predicted = sum(item.predicted for item in spaced) / len(spaced)
    lateness = sum(item.due_offset_days for item in spaced) / len(spaced)
    recommended = target**scale
    clamped = not (0.7 <= recommended <= 0.99) or scale in SCALE_RANGE
    return ScaleEstimate(
        target=target,
        observed_rate=rate,
        mean_predicted=mean_predicted,
        mean_lateness_days=lateness,
        scale=scale,
        low=low,
        high=high,
        events=len(spaced),
        recommended_target=min(0.99, max(0.7, recommended)),
        clamped=clamped,
    )


def _worst_points(
    predictions: Sequence[Prediction], *, limit: int = WORST_KP_EVENTS
) -> tuple[KpCalibration, ...]:
    grouped: dict[str, list[Prediction]] = {}
    for item in predictions:
        grouped.setdefault(item.knowledge_point_id, []).append(item)
    rows: list[KpCalibration] = []
    for kp_id, items in grouped.items():
        if len(items) < WORST_KP_EVENTS:
            continue
        successes = sum(1 for item in items if item.success)
        mean_predicted = sum(item.predicted for item in items) / len(items)
        observed = successes / len(items)
        rows.append(
            KpCalibration(
                knowledge_point_id=kp_id,
                knowledge_type=items[0].knowledge_type,
                events=len(items),
                mean_predicted=mean_predicted,
                observed_rate=observed,
                bias=mean_predicted - observed,
                brier=sum((item.predicted - (1.0 if item.success else 0.0)) ** 2 for item in items)
                / len(items),
            )
        )
    rows.sort(key=lambda row: (-abs(row.bias), row.knowledge_point_id))
    return tuple(rows[:limit])


# ── integrity ────────────────────────────────────────────────────────────


def replay_state(
    progress: LearningProgress,
    knowledge_point_id: str,
    *,
    scheduler: Any | None = None,
) -> RepetitionState | None:
    """The state the evidence log implies for one knowledge point."""

    events = _evidence_by_point(progress.learning_evidence).get(knowledge_point_id)
    if not events:
        return None
    engine = _scheduler_for(progress, scheduler)
    kp_type = progress.knowledge_types.get(knowledge_point_id, KnowledgeType.MEMORY)
    desired = float(progress.desired_retention or DEFAULT_DESIRED_RETENTION)
    return engine.replay(
        kp_type,
        events,
        now=events[0].timestamp,
        desired_retention=desired,
    )


#: Only the counters are compared. ``stability``, ``difficulty`` and
#: ``interval_index`` are anchor-dependent: a replay starts at the first
#: evidence timestamp, while the app may have created the state minutes earlier,
#: so their difference is not by itself evidence of a lost write. The counters
#: are not: they can only come from the evidence that was actually applied.
_COMPARED_FIELDS = ("review_count", "lapse_count")


def _divergences(
    progress: LearningProgress, *, scheduler: Any | None = None
) -> tuple[StateDivergence, ...]:
    rows: list[StateDivergence] = []
    for kp_id, stored in progress.repetition_states.items():
        if stored.stability <= 0:
            # Never hydrated: the scheduler still has to fill it from
            # ``interval_index``, so there is nothing to compare yet.
            continue
        replayed = replay_state(progress, kp_id, scheduler=scheduler)
        if replayed is None:
            continue
        changed: list[str] = []
        left: dict[str, float] = {}
        right: dict[str, float] = {}
        for name in _COMPARED_FIELDS:
            replay_value = float(getattr(replayed, name))
            stored_value = float(getattr(stored, name))
            if abs(replay_value - stored_value) > 1e-6:
                changed.append(name)
                left[name] = round(replay_value, 6)
                right[name] = round(stored_value, 6)
        if changed:
            rows.append(
                StateDivergence(
                    knowledge_point_id=kp_id,
                    knowledge_type=progress.knowledge_types.get(kp_id, KnowledgeType.MEMORY),
                    fields=tuple(changed),
                    replayed=left,
                    stored=right,
                )
            )
    rows.sort(key=lambda row: row.knowledge_point_id)
    return tuple(rows)


# ── findings ─────────────────────────────────────────────────────────────


def _bias_finding(metrics: Metrics, *, tolerance: float, scope: str) -> Finding | None:
    low, high = wilson_interval(metrics.successes, metrics.events)
    # Interval of ``mean_predicted - observed`` given the Wilson bounds.
    bias_low = metrics.mean_predicted - high
    bias_high = metrics.mean_predicted - low
    if bias_low > tolerance:
        return Finding(
            code="calibration.overconfident",
            severity="P1",
            title=f"{scope} is overconfident by {metrics.bias:+.0%}",
            detail=(
                f"Predicted recall {metrics.mean_predicted:.0%} vs observed "
                f"{metrics.observed_rate:.0%} on {metrics.events} events "
                f"(bias {metrics.bias:+.0%}, interval {bias_low:+.0%}..{bias_high:+.0%}). "
                "The learner forgets faster than the model believes, so scheduled "
                "intervals run long."
            ),
            evidence={
                "scope": scope,
                "events": metrics.events,
                "mean_predicted": round(metrics.mean_predicted, 4),
                "observed_rate": round(metrics.observed_rate, 4),
                "bias": round(metrics.bias, 4),
                "bias_interval": [round(bias_low, 4), round(bias_high, 4)],
            },
        )
    if bias_high < -tolerance:
        return Finding(
            code="calibration.underconfident",
            severity="P1",
            title=f"{scope} is underconfident by {-metrics.bias:+.0%}",
            detail=(
                f"Predicted recall {metrics.mean_predicted:.0%} vs observed "
                f"{metrics.observed_rate:.0%} on {metrics.events} events "
                f"(bias {metrics.bias:+.0%}, interval {bias_low:+.0%}..{bias_high:+.0%}). "
                "The learner retains more than the model believes, so reviews come "
                "back earlier than they need to."
            ),
            evidence={
                "scope": scope,
                "events": metrics.events,
                "mean_predicted": round(metrics.mean_predicted, 4),
                "observed_rate": round(metrics.observed_rate, 4),
                "bias": round(metrics.bias, 4),
                "bias_interval": [round(bias_low, 4), round(bias_high, 4)],
            },
        )
    return None


def _scale_finding(scale: ScaleEstimate, desired: float) -> Finding | None:
    """Fire only when the ratio's interval clears parity entirely.

    The burden of proof is on the audit: a wide interval that contains 1.0 is
    "not established", never "miscalibrated".
    """

    band = 1.0 - SCALE_TOLERANCE
    ceiling = 1.0 + SCALE_TOLERANCE
    if scale.high >= band and scale.low <= ceiling:
        return None
    optimist = scale.high < band
    direction = "shorten" if optimist else "lengthen"
    return Finding(
        code="calibration.stability_mismatch",
        severity="P1",
        title=(
            f"Model stability is {scale.scale:.2f}x this learner "
            f"({'optimistic' if optimist else 'pessimistic'})"
        ),
        detail=(
            f"Across {scale.events} genuinely spaced retrievals, maximum likelihood puts "
            f"true stability at {scale.scale:.2f}x the model's estimate "
            f"(95% {scale.low:.2f}..{scale.high:.2f}; 1.00 is correct). Predicted recall "
            f"at those retrievals averaged {scale.mean_predicted:.0%} and "
            f"{scale.observed_rate:.0%} succeeded, with answers arriving "
            f"{scale.mean_lateness_days:+.1f} days from their due time. Intervals should "
            f"{direction} by {abs(1 - scale.scale):.0%}, or the target move from "
            f"{desired:.2f} to {scale.recommended_target:.2f}."
        ),
        evidence={
            "events": scale.events,
            "scale": round(scale.scale, 4),
            "scale_interval": [round(scale.low, 4), round(scale.high, 4)],
            "desired_retention": desired,
            "recommended_target": round(scale.recommended_target, 4),
            "observed_rate": round(scale.observed_rate, 4),
            "mean_predicted": round(scale.mean_predicted, 4),
            "mean_lateness_days": round(scale.mean_lateness_days, 3),
        },
    )


def _findings(
    predictions: Sequence[Prediction],
    metrics: Metrics | None,
    at_target: Metrics | None,
    per_type: dict[str, Metrics],
    scale: ScaleEstimate | None,
    divergences: tuple[StateDivergence, ...],
    desired: float,
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    if metrics is None or metrics.events < MIN_EVENTS:
        return (
            Finding(
                code="calibration.insufficient_evidence",
                severity="P3",
                title=f"Only {0 if metrics is None else metrics.events} review events with a prediction",
                detail=(
                    f"Calibration needs at least {MIN_EVENTS} paired events, and this "
                    "snapshot has fewer. Nothing about the model is claimed here: "
                    "absence of a finding is not evidence of calibration."
                ),
                evidence={
                    "events": 0 if metrics is None else metrics.events,
                    "required": MIN_EVENTS,
                },
            ),
        )

    bias = _bias_finding(metrics, tolerance=BIAS_TOLERANCE, scope="The path")
    if bias is not None:
        findings.append(bias)

    for name in sorted(per_type):
        type_metrics = per_type[name]
        if type_metrics.events < MIN_TYPE_EVENTS:
            continue
        found = _bias_finding(
            type_metrics, tolerance=TYPE_BIAS_TOLERANCE, scope=f"{name} objectives"
        )
        if found is not None:
            findings.append(
                Finding(
                    code=f"calibration.type_bias.{name}",
                    severity=found.severity,
                    title=found.title,
                    detail=found.detail,
                    evidence=found.evidence,
                )
            )

    if scale is not None:
        mismatch = _scale_finding(scale, desired)
        if mismatch is not None:
            findings.append(mismatch)
        else:
            findings.append(
                Finding(
                    code="calibration.stability_matches",
                    severity="P3",
                    title=f"Model stability matches this learner ({scale.scale:.2f}x)",
                    detail=(
                        f"Across {scale.events} genuinely spaced retrievals the ratio is "
                        f"{scale.scale:.2f} (95% {scale.low:.2f}..{scale.high:.2f}), inside the "
                        f"±{SCALE_TOLERANCE:.0%} band around parity. Observed recall "
                        f"{scale.observed_rate:.0%} against a {desired:.0%} target, answers "
                        f"{scale.mean_lateness_days:+.1f} days from due."
                    ),
                    evidence={
                        "events": scale.events,
                        "scale": round(scale.scale, 4),
                        "scale_interval": [round(scale.low, 4), round(scale.high, 4)],
                        "observed_rate": round(scale.observed_rate, 4),
                    },
                )
            )

    if metrics.events >= 50 and metrics.auc is not None and metrics.auc < AUC_FLOOR:
        findings.append(
            Finding(
                code="calibration.non_discriminating",
                severity="P2",
                title=f"Predicted recall barely separates remembered from forgotten ({metrics.auc:.2f} AUC)",
                detail=(
                    f"AUC {metrics.auc:.2f} over {metrics.events} events. A predictor can be "
                    "unbiased on average and still be useless per item; keep the bias numbers "
                    "but do not trust per-knowledge-point rankings built on it."
                ),
                evidence={
                    "events": metrics.events,
                    "auc": round(metrics.auc, 4),
                    "floor": AUC_FLOOR,
                },
            )
        )

    if divergences:
        findings.append(
            Finding(
                code="calibration.state_divergence",
                severity="P2",
                title=f"{len(divergences)} knowledge points store a state the evidence log does not imply",
                detail=(
                    "Stored repetition state differs from a clean replay of the same evidence. "
                    "A deliberate target change, a migrated snapshot, or a cross-surface write "
                    "can do this; so can a lost update. Predictions above are replayed from the "
                    "evidence log, so they remain internally consistent either way."
                ),
                evidence={
                    "count": len(divergences),
                    "sample": [
                        {
                            "knowledge_point_id": row.knowledge_point_id,
                            "fields": list(row.fields),
                        }
                        for row in divergences[:5]
                    ],
                },
            )
        )

    if not findings:
        findings.append(
            Finding(
                code="calibration.consistent",
                severity="P3",
                title="No miscalibration found",
                detail=(
                    f"{metrics.events} events, bias {metrics.bias:+.0%}, ECE {metrics.ece:.0%}, "
                    f"Brier {metrics.brier:.3f} against a {metrics.brier_baseline:.3f} base rate. "
                    "Nothing crossed the reporting thresholds; thresholds are fixed and listed "
                    "in docs/CALIBRATION.md."
                ),
                evidence={
                    "events": metrics.events,
                    "bias": round(metrics.bias, 4),
                    "ece": round(metrics.ece, 4),
                    "brier": round(metrics.brier, 4),
                },
            )
        )
    findings.sort(key=lambda item: (_SEVERITY_ORDER.get(item.severity, 9), item.code))
    return tuple(findings)


# ── report ───────────────────────────────────────────────────────────────


def audit(
    progress: LearningProgress,
    *,
    scheduler: Any | None = None,
    now: float | None = None,
) -> CalibrationReport:
    """Full calibration audit of one learning path. Pure, deterministic."""

    engine = _scheduler_for(progress, scheduler)
    predictions = collect_predictions(progress, scheduler=engine)
    desired = float(progress.desired_retention or DEFAULT_DESIRED_RETENTION)
    metrics = compute_metrics(predictions) if predictions else None

    spaced = [item for item in predictions if item.at_target]
    at_target = compute_metrics(spaced) if spaced else None

    grouped: dict[str, list[Prediction]] = {}
    for item in predictions:
        grouped.setdefault(item.knowledge_type.value, []).append(item)
    per_type = {name: compute_metrics(items) for name, items in sorted(grouped.items())}

    scale = retention_scale(predictions, target=desired) if spaced else None
    divergences = _divergences(progress, scheduler=engine)
    findings = _findings(predictions, metrics, at_target, per_type, scale, divergences, desired)
    return CalibrationReport(
        path_id=progress.book_id,
        generated_at=time.time() if now is None else float(now),
        desired_retention=desired,
        predictions=tuple(predictions),
        metrics=metrics,
        at_target=at_target,
        per_type=per_type,
        worst_knowledge_points=_worst_points(predictions),
        retention_scale=scale,
        divergences=divergences,
        findings=findings,
    )


def report_to_dict(report: CalibrationReport) -> dict[str, Any]:
    """JSON-ready form. Stable key order, no floats-as-strings."""

    def metrics_dict(metrics: Metrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "events": metrics.events,
            "successes": metrics.successes,
            "mean_predicted": round(metrics.mean_predicted, 4),
            "observed_rate": round(metrics.observed_rate, 4),
            "bias": round(metrics.bias, 4),
            "brier": round(metrics.brier, 4),
            "brier_baseline": round(metrics.brier_baseline, 4),
            "skill_score": round(metrics.skill_score, 4),
            "log_loss": round(metrics.log_loss, 4),
            "ece": round(metrics.ece, 4),
            "mce": round(metrics.mce, 4),
            "auc": None if metrics.auc is None else round(metrics.auc, 4),
            "reliability": [
                {
                    "lower": round(item.lower, 2),
                    "upper": round(item.upper, 2),
                    "count": item.count,
                    "mean_predicted": round(item.mean_predicted, 4),
                    "observed_rate": round(item.observed_rate, 4),
                }
                for item in metrics.reliability
            ],
        }

    scale = report.retention_scale
    return {
        "path_id": report.path_id,
        "generated_at": report.generated_at,
        "desired_retention": report.desired_retention,
        "events": len(report.predictions),
        "spaced_events": sum(1 for item in report.predictions if item.at_target),
        "metrics": metrics_dict(report.metrics),
        "at_target": metrics_dict(report.at_target),
        "per_type": {name: metrics_dict(value) for name, value in report.per_type.items()},
        "worst_knowledge_points": [
            {
                "knowledge_point_id": row.knowledge_point_id,
                "knowledge_type": row.knowledge_type.value,
                "events": row.events,
                "mean_predicted": round(row.mean_predicted, 4),
                "observed_rate": round(row.observed_rate, 4),
                "bias": round(row.bias, 4),
                "brier": round(row.brier, 4),
            }
            for row in report.worst_knowledge_points
        ],
        "retention_scale": None
        if scale is None
        else {
            "target": round(scale.target, 4),
            "observed_rate": round(scale.observed_rate, 4),
            "mean_predicted": round(scale.mean_predicted, 4),
            "mean_lateness_days": round(scale.mean_lateness_days, 3),
            "scale": round(scale.scale, 4),
            "interval": [round(scale.low, 4), round(scale.high, 4)],
            "events": scale.events,
            "recommended_target": round(scale.recommended_target, 4),
            "clamped": scale.clamped,
        },
        "divergences": [
            {
                "knowledge_point_id": row.knowledge_point_id,
                "knowledge_type": row.knowledge_type.value,
                "fields": list(row.fields),
                "replayed": row.replayed,
                "stored": row.stored,
            }
            for row in report.divergences
        ],
        "findings": [
            {
                "code": item.code,
                "severity": item.severity,
                "title": item.title,
                "detail": item.detail,
                "evidence": item.evidence,
            }
            for item in report.findings
        ],
    }


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_markdown(report: CalibrationReport) -> str:
    """Human report: verdict first, then the numbers that support it."""

    lines: list[str] = []
    lines.append(f"# Retention calibration — {report.path_id}")
    lines.append("")
    severity = report.worst_severity or "—"
    lines.append(
        f"Events: {len(report.predictions)} · "
        f"spaced: {sum(1 for item in report.predictions if item.at_target)} · "
        f"target: {report.desired_retention:.2f} · worst finding: {severity}"
    )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    for item in report.findings:
        lines.append(f"- **{item.severity} · `{item.code}`** — {item.title}")
        lines.append(f"  {item.detail}")
    if not report.findings:
        lines.append("- none")
    lines.append("")

    metrics = report.metrics
    if metrics is not None:
        lines.append("## Belief vs outcome")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| events | {metrics.events} |")
        lines.append(f"| mean predicted | {_pct(metrics.mean_predicted)} |")
        lines.append(f"| observed | {_pct(metrics.observed_rate)} |")
        lines.append(f"| bias | {metrics.bias:+.1%} |")
        lines.append(f"| Brier | {metrics.brier:.4f} |")
        lines.append(f"| Brier (base rate) | {metrics.brier_baseline:.4f} |")
        lines.append(f"| skill score | {metrics.skill_score:+.2f} |")
        lines.append(f"| log loss | {metrics.log_loss:.4f} |")
        lines.append(f"| ECE / MCE | {metrics.ece:.1%} / {metrics.mce:.1%} |")
        auc = "n/a" if metrics.auc is None else f"{metrics.auc:.3f}"
        lines.append(f"| AUC | {auc} |")
        lines.append("")
        lines.append("## Reliability")
        lines.append("")
        lines.append("| predicted band | n | mean predicted | observed | gap |")
        lines.append("|---|---|---|---|---|")
        for item in metrics.reliability:
            if not item.count:
                continue
            lines.append(
                f"| {item.lower:.1f}–{item.upper:.1f} | {item.count} | "
                f"{_pct(item.mean_predicted)} | {_pct(item.observed_rate)} | "
                f"{item.observed_rate - item.mean_predicted:+.0%} |"
            )
        lines.append("")

    if report.per_type:
        lines.append("## By knowledge type")
        lines.append("")
        lines.append("| type | n | predicted | observed | bias | Brier |")
        lines.append("|---|---|---|---|---|---|")
        for name in sorted(report.per_type):
            row = report.per_type[name]
            lines.append(
                f"| {name} | {row.events} | {_pct(row.mean_predicted)} | "
                f"{_pct(row.observed_rate)} | {row.bias:+.0%} | {row.brier:.3f} |"
            )
        lines.append("")

    if report.worst_knowledge_points:
        lines.append("## Largest per-objective gaps")
        lines.append("")
        lines.append("| objective | type | n | predicted | observed | bias |")
        lines.append("|---|---|---|---|---|---|")
        for row in report.worst_knowledge_points:
            lines.append(
                f"| `{row.knowledge_point_id}` | {row.knowledge_type.value} | {row.events} | "
                f"{_pct(row.mean_predicted)} | {_pct(row.observed_rate)} | {row.bias:+.0%} |"
            )
        lines.append("")

    scale = report.retention_scale
    if scale is not None:
        lines.append("## Stability ratio and target feasibility")
        lines.append("")
        lines.append(
            f"- True stability vs model stability: **{scale.scale:.2f}x** "
            f"(95% {scale.low:.2f}..{scale.high:.2f}); 1.00 means the model is right, "
            f"0.50 means the learner forgets twice as fast."
        )
        lines.append(
            f"- At those {scale.events} spaced retrievals the model predicted "
            f"{_pct(scale.mean_predicted)} and {_pct(scale.observed_rate)} succeeded, "
            f"with answers {scale.mean_lateness_days:+.1f} days from their due time."
        )
        lines.append(
            f"- Target that would deliver the {scale.target:.0%} it promises: "
            f"**{scale.recommended_target:.2f}**"
            + (" (clamped to the supported range)" if scale.clamped else "")
            + "."
        )
        lines.append("")

    if report.divergences:
        lines.append("## Stored state vs replayed state")
        lines.append("")
        lines.append("| objective | fields | replayed | stored |")
        lines.append("|---|---|---|---|")
        for row in report.divergences[:10]:
            replay_text = ", ".join(f"{k}={v:g}" for k, v in row.replayed.items())
            stored_text = ", ".join(f"{k}={v:g}" for k, v in row.stored.items())
            lines.append(
                f"| `{row.knowledge_point_id}` | {', '.join(row.fields)} | "
                f"{replay_text} | {stored_text} |"
            )
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "Each event is scored with the scheduler's own `retrievability()` on the state "
        "that preceded it, then applied through the real `schedule_review()` transition. "
        "Intervals use the Wilson score interval; the multiplier interval is a fixed-seed "
        "bootstrap (" + str(BOOTSTRAP_SAMPLES) + " resamples). Thresholds are constants in "
        "`deeptutor/learning/calibration.py` and documented in `docs/CALIBRATION.md`."
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "BIAS_TOLERANCE",
    "BOOTSTRAP_SAMPLES",
    "CalibrationReport",
    "Finding",
    "KpCalibration",
    "Metrics",
    "MIN_EVENTS",
    "MIN_TARGET_EVENTS",
    "Prediction",
    "ReliabilityBin",
    "ScaleEstimate",
    "StateDivergence",
    "SUCCESS_THRESHOLD",
    "audit",
    "collect_predictions",
    "compute_metrics",
    "render_markdown",
    "replay_state",
    "report_to_dict",
    "retention_scale",
    "wilson_interval",
]
