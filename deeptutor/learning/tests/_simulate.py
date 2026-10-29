"""Synthetic learners whose forgetting truth is known, for calibration tests.

The generator drives the *real* scheduler: state is created with
``get_initial_state`` and every answer is applied through ``schedule_review``,
exactly as ``LearningService`` does.  What it makes up is the learner — the
probability that an answer is still retrievable:

    P(correct) = exp(-elapsed_days / (stability_ratio * model_stability))

``stability_ratio`` is therefore the ground truth the audit is asked to recover:
1.0 means the model is exactly right, 0.5 a learner who forgets twice as fast,
2.0 one who retains twice as long.  Answers arrive ``lateness_days`` after the
scheduled due time (uniform), because real learners do, and an auditor that
confuses lateness with miscalibration will show it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from deeptutor.learning.models import (
    KnowledgeType,
    LearningEvidence,
    LearningProgress,
    RepetitionState,
)
from deeptutor.learning.scheduler import SpacedRepetitionScheduler

SECONDS_PER_DAY = 86400.0
DEFAULT_TYPES = (
    KnowledgeType.MEMORY,
    KnowledgeType.CONCEPT,
    KnowledgeType.PROCEDURE,
    KnowledgeType.DESIGN,
)


@dataclass
class TruthEvent:
    knowledge_point_id: str
    knowledge_type: KnowledgeType
    timestamp: float
    #: Probability the simulator used to draw this answer.
    p_true: float
    #: The same truth evaluated at the scheduled due time instead of at the
    #: moment the learner answered: what the retention target actually promises,
    #: free of the lateness the learner brought with them.
    p_true_at_due: float
    success: bool
    #: Stability the model believed for this learner at that moment, in days.
    model_stability_days: float
    #: Ground-truth stability, in days.
    true_stability_days: float


@dataclass
class Scenario:
    progress: LearningProgress
    truth: list[TruthEvent] = field(default_factory=list)

    @property
    def mean_ratio(self) -> float:
        """Average ``true / model`` stability over the events that were generated."""
        if not self.truth:
            return 0.0
        return sum(
            item.true_stability_days / item.model_stability_days for item in self.truth
        ) / len(self.truth)

    @property
    def mean_p_true(self) -> float:
        if not self.truth:
            return 0.0
        return sum(item.p_true for item in self.truth) / len(self.truth)


def simulate_learner(
    seed: int,
    *,
    stability_ratio: float = 1.0,
    knowledge_points: int = 10,
    events_per_point: int = 12,
    target: float = 0.9,
    lateness_days: float = 1.5,
    types: tuple[KnowledgeType, ...] = DEFAULT_TYPES,
    path_id: str = "simulated-path",
) -> Scenario:
    """Run one synthetic learner through the real scheduler."""

    rng = random.Random(seed)
    progress = LearningProgress(book_id=path_id, desired_retention=target)
    scheduler = SpacedRepetitionScheduler(desired_retention=target)
    origin = 1_760_000_000.0
    truth: list[TruthEvent] = []

    for index in range(knowledge_points):
        kp_id = f"kp{index:02d}"
        kp_type = types[index % len(types)]
        progress.knowledge_types[kp_id] = kp_type
        moment = origin + index * 3600.0
        state: RepetitionState = scheduler.get_initial_state(
            kp_type, now=moment, desired_retention=target
        )
        progress.repetition_states[kp_id] = state
        for _ in range(events_per_point):
            if state.last_review_at is None:
                # The first answer lands when the state is created, exactly like
                # ``LearningService.record_*``: create state, then grade.
                moment = max(moment, state.next_review_at)
            else:
                due = state.next_review_at
                moment = max(
                    state.last_review_at + 60.0,
                    due + rng.uniform(0.0, lateness_days) * SECONDS_PER_DAY,
                )
            elapsed_days = (
                max(0.0, moment - state.last_review_at) / SECONDS_PER_DAY
                if state.last_review_at is not None
                else 0.0
            )
            due_elapsed_days = (
                max(0.0, state.next_review_at - state.last_review_at) / SECONDS_PER_DAY
                if state.last_review_at is not None
                else max(0.0, state.next_review_at - moment) / SECONDS_PER_DAY
            )
            model_stability = max(state.stability, 0.02)
            true_stability = max(0.02, model_stability * stability_ratio)
            p_true = math.exp(-elapsed_days / true_stability)
            p_true_at_due = math.exp(-due_elapsed_days / true_stability)
            success = rng.random() < p_true
            evidence = LearningEvidence(
                knowledge_point_id=kp_id,
                timestamp=moment,
                source="simulated",
                assessment_type="review",
                result="correct" if success else "incorrect",
                quality=1.0 if success else 0.0,
                attempt_count=1,
            )
            progress.learning_evidence.append(evidence)
            truth.append(
                TruthEvent(
                    knowledge_point_id=kp_id,
                    knowledge_type=kp_type,
                    timestamp=moment,
                    p_true=p_true,
                    p_true_at_due=p_true_at_due,
                    success=success,
                    model_stability_days=model_stability,
                    true_stability_days=true_stability,
                )
            )
            scheduler.schedule_review(state, kp_type, evidence, now=moment)

    progress.updated_at = max(item.timestamp for item in truth) if truth else origin
    return Scenario(progress=progress, truth=truth)
