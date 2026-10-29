"""Measure the calibration audit against learners whose forgetting is known.

Every number in ``docs-for-dev/CALIBRATION.md`` and in the pull request body is
produced here.  Nothing is hand-typed, nothing calls an LLM, and every scenario
is seeded, so a reviewer can re-run this file and get the same table.

    python tools/calib-bench/simulate.py            # table + JSON artifact
    python tools/calib-bench/simulate.py --quick    # fewer seeds

The ground truth is defined by ``deeptutor.learning.tests._simulate``: an answer
is remembered with probability ``exp(-elapsed / (ratio * model_stability))``, so
``ratio`` is exactly what the audit's stability estimator is asked to recover.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deeptutor.learning.calibration import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    audit,
)
from deeptutor.learning.tests._simulate import simulate_learner  # noqa: E402

RATIOS = (0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
SNAPSHOTS = ((14, 16), (20, 25))
ARTIFACT = ROOT / "tools" / "calib-bench" / "last-run.json"


def _summarise(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.mean(ordered), 3),
        "p05": round(ordered[max(0, int(0.05 * (len(ordered) - 1)))], 3),
        "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
    }


def run_scenario(ratio: float, seeds: int, points: int, events: int) -> dict:
    scales: list[float] = []
    covered = 0
    flagged = 0
    durations: list[float] = []
    breakdown: dict[str, int] = {}
    for seed in range(seeds):
        scenario = simulate_learner(
            seed,
            stability_ratio=ratio,
            knowledge_points=points,
            events_per_point=events,
        )
        started = time.perf_counter()
        report = audit(scenario.progress)
        durations.append(time.perf_counter() - started)
        estimate = report.retention_scale
        if estimate is not None:
            scales.append(estimate.scale)
            # The interval must cover the ratio the simulator actually used.
            if estimate.low <= ratio <= estimate.high:
                covered += 1
        codes = [item.code for item in report.findings if item.severity in ("P0", "P1")]
        for code in codes:
            breakdown[code] = breakdown.get(code, 0) + 1
        if codes:
            flagged += 1
    return {
        "ratio": ratio,
        "knowledge_points": points,
        "events_per_point": events,
        "events": points * events,
        "seeds": seeds,
        "scale": _summarise(scales),
        "ratio_covered": round(covered / max(1, len(scales)), 3),
        "flagged_share": round(flagged / seeds, 3),
        "findings": breakdown,
        "audit_ms": round(1000 * statistics.median(durations), 1),
    }


def closed_loop(ratio: float, seeds: int, points: int, events: int, target: float) -> dict:
    """Apply the recommendation and re-measure the recall the schedule promises.

    Recall is read at the *due* time, not at the moment the learner answered: a
    learner who answers three days late recalls less without the schedule being
    wrong, and the target is a promise about the deadline.
    """

    before_due: list[float] = []
    after_due: list[float] = []
    before_observed: list[float] = []
    after_observed: list[float] = []
    recommended: list[float] = []
    for seed in range(seeds):
        base = simulate_learner(
            seed,
            stability_ratio=ratio,
            knowledge_points=points,
            events_per_point=events,
            target=target,
        )
        baseline = audit(base.progress)
        before_due.append(_mean_due_recall(baseline, base))
        before_observed.append(
            baseline.retention_scale.observed_rate if baseline.retention_scale else 0.0
        )
        if baseline.retention_scale is None:
            continue
        new_target = baseline.retention_scale.recommended_target
        recommended.append(new_target)
        retuned = simulate_learner(
            seed,
            stability_ratio=ratio,
            knowledge_points=points,
            events_per_point=events,
            target=new_target,
        )
        audited = audit(retuned.progress)
        after_due.append(_mean_due_recall(audited, retuned))
        after_observed.append(
            audited.retention_scale.observed_rate if audited.retention_scale else 0.0
        )
    return {
        "ratio": ratio,
        "target": target,
        "seeds": seeds,
        "due_recall_before": round(statistics.mean(before_due), 3) if before_due else None,
        "observed_before": round(statistics.mean(before_observed), 3) if before_observed else None,
        "recommended_target": round(statistics.mean(recommended), 3) if recommended else None,
        "due_recall_after": round(statistics.mean(after_due), 3) if after_due else None,
        "observed_after": round(statistics.mean(after_observed), 3) if after_observed else None,
    }


def _mean_due_recall(report, scenario) -> float:
    values = [
        truth.p_true_at_due
        for prediction, truth in zip(report.predictions, scenario.truth, strict=True)
        if prediction.at_target
    ]
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="8 seeds per cell")
    parser.add_argument(
        "--json", action="store_true", help="print the artifact instead of the table"
    )
    args = parser.parse_args()
    seeds = 8 if args.quick else args.seeds
    target = 0.9

    started = time.time()
    scenarios = [
        run_scenario(ratio, seeds, points, events)
        for points, events in SNAPSHOTS
        for ratio in RATIOS
    ]
    loops = [
        closed_loop(ratio, min(seeds, 10), 20, 25, target) for ratio in (0.3, 0.5, 1.0, 2.0, 3.0)
    ]

    clean = [row for row in scenarios if row["ratio"] == 1.0]
    false_alarm = statistics.mean(row["flagged_share"] for row in clean) if clean else 0.0
    smallest = [row for row in scenarios if row["events"] == SNAPSHOTS[0][0] * SNAPSHOTS[0][1]]
    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seeds_per_cell": seeds,
        "target": target,
        "scenarios": scenarios,
        "closed_loop": loops,
        "summary": {
            "false_alarm_share_at_parity": round(false_alarm, 3),
            "detection_at_parity_small_snapshot": round(
                statistics.mean(row["flagged_share"] for row in smallest if row["ratio"] != 1.0),
                3,
            ),
            "elapsed_s": round(time.time() - started, 1),
        },
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(artifact, indent=1, ensure_ascii=False))
        return 0

    print(
        f"# calibration bench · {seeds} seeds/cell · bootstrap {BOOTSTRAP_SAMPLES} · target {target}"
    )
    print()
    print(
        "| events | true ratio | median estimate | mean | ratio covered | flagged P1 | audit ms |"
    )
    print("|---|---|---|---|---|---|---|")
    for row in scenarios:
        print(
            "| {events} | {ratio:.1f} | {median:.2f} | {mean:.2f} | {cover:.0%} | {flag:.0%} | {ms} |".format(
                events=row["events"],
                ratio=row["ratio"],
                median=row["scale"]["median"],
                mean=row["scale"]["mean"],
                cover=row["ratio_covered"],
                flag=row["flagged_share"],
                ms=row["audit_ms"],
            )
        )
    print()
    print(
        "| true ratio | recall at due before | recommended target | recall at due after | answered recall |"
    )
    print("|---|---|---|---|---|")
    for row in loops:
        print(
            "| {ratio:.1f} | {before:.0%} | {target:.2f} | {after:.0%} | {obs:.0%} |".format(
                ratio=row["ratio"],
                before=row["due_recall_before"] or 0.0,
                target=row["recommended_target"] or 0.0,
                after=row["due_recall_after"] or 0.0,
                obs=row["observed_after"] or 0.0,
            )
        )
    print()
    print(f"false alarm at parity: {false_alarm:.0%}")
    print(f"artifact: {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
