"""CLI contract for ``deeptutor learning calibration``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deeptutor.learning.tests._simulate import simulate_learner
from deeptutor_cli.main import app

runner = CliRunner()


def _write(tmp_path: Path, ratio: float, name: str = "progress.json", points: int = 20) -> Path:
    scenario = simulate_learner(
        0, stability_ratio=ratio, knowledge_points=points, events_per_point=25
    )
    target = tmp_path / name
    target.write_text(
        json.dumps(scenario.progress.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    return target


@pytest.fixture(scope="module")
def miscalibrated(tmp_path_factory) -> Path:
    return _write(tmp_path_factory.mktemp("miscalibrated"), 0.4)


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory) -> Path:
    return _write(tmp_path_factory.mktemp("calibrated"), 1.0)


def test_miscalibrated_path_exits_one_and_names_the_finding(miscalibrated):
    result = runner.invoke(app, ["learning", "calibration", str(miscalibrated)])
    assert result.exit_code == 1, result.stdout
    assert "calibration.stability_mismatch" in result.stdout
    assert "Retention calibration" in result.stdout


def test_calibrated_path_exits_zero(calibrated):
    result = runner.invoke(app, ["learning", "calibration", str(calibrated)])
    assert result.exit_code == 0, result.stdout
    assert "calibration.stability_matches" in result.stdout


def test_json_flag_returns_machine_readable_report(miscalibrated):
    result = runner.invoke(app, ["learning", "calibration", str(miscalibrated), "--json"])
    payload = json.loads(result.stdout)
    assert payload["events"] == 500
    assert payload["retention_scale"]["scale"] < 1.0
    assert payload["findings"][0]["severity"] == "P1"


def test_accepts_a_wrapped_progress_payload(tmp_path):
    scenario = simulate_learner(0, stability_ratio=1.0, knowledge_points=8, events_per_point=8)
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"progress": scenario.progress.model_dump(mode="json")}), encoding="utf-8"
    )
    result = runner.invoke(app, ["learning", "calibration", str(wrapped), "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["events"] == 64


def test_writes_the_report_to_a_file(tmp_path, calibrated):
    output = tmp_path / "nested" / "report.md"
    result = runner.invoke(app, ["learning", "calibration", str(calibrated), "-o", str(output)])
    assert result.exit_code == 0, result.stdout
    assert output.is_file()
    assert "Retention calibration" in output.read_text(encoding="utf-8")


def test_missing_source_is_a_usage_error():
    result = runner.invoke(app, ["learning", "calibration"])
    assert result.exit_code == 2
    message = (result.stdout or "") + (result.stderr or "")
    assert "--store-root" in message


def test_unknown_store_path_is_a_usage_error(tmp_path):
    result = runner.invoke(
        app,
        ["learning", "calibration", "--store-root", str(tmp_path), "--path", "does-not-exist"],
    )
    assert result.exit_code == 2
    assert (
        "no learning path" in (result.stderr or "").lower()
        or "no learning path" in (result.stdout or "").lower()
    )


def test_command_is_registered_on_the_real_cli():
    assert "learning" in {group.name for group in app.registered_groups}
