"""Inspect the retention model that drives Mastery Path reviews."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from deeptutor.learning.calibration import audit, render_markdown, report_to_dict
from deeptutor.learning.models import LearningProgress

EXIT_CLEAN = 0
EXIT_FINDING = 1
EXIT_USAGE = 2


def _load_snapshot(path: Path) -> LearningProgress:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "progress" in payload:
        payload = payload["progress"]
    return LearningProgress.model_validate(payload)


def _load_from_store(store_root: Path, path_id: str) -> LearningProgress:
    from deeptutor.learning.storage import LearningStore

    progress = LearningStore(store_root).load(path_id)
    if progress is None:
        raise ValueError(f"no learning path {path_id!r} under {store_root}")
    return progress


def load_progress(
    snapshot: Path | None,
    store_root: Path | None,
    path_id: str | None,
) -> LearningProgress:
    """Read one path, from an exported snapshot or from a learning store."""

    if snapshot is not None and store_root is not None:
        raise ValueError("pass either a snapshot file or --store-root, not both")
    if snapshot is not None:
        return _load_snapshot(snapshot)
    if store_root is None or path_id is None:
        raise ValueError("give a snapshot file, or --store-root together with --path")
    return _load_from_store(store_root, path_id)


def register(app: typer.Typer) -> None:
    @app.command("calibration")
    def calibration(
        snapshot: Path | None = typer.Argument(
            None, help="Exported LearningProgress JSON (as stored in mastery_paths.state_json)."
        ),
        store_root: Path | None = typer.Option(
            None, "--store-root", help="Directory holding mastery.sqlite3."
        ),
        path_id: str | None = typer.Option(
            None, "--path", help="Path id to read with --store-root."
        ),
        as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
        output: Path | None = typer.Option(
            None, "--output", "-o", help="Write the report to a file."
        ),
    ) -> None:
        """Compare scheduled recall with the recall the learner actually showed.

        Exit codes: 0 no blocking finding, 1 a P0/P1 finding, 2 usage or read
        error. The report never needs an LLM, a network, or a running server.
        """
        try:
            progress = load_progress(snapshot, store_root, path_id)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(EXIT_USAGE) from exc

        report = audit(progress)
        text = (
            json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)
            if as_json
            else render_markdown(report)
        )
        if output is None:
            typer.echo(text)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            typer.echo(f"wrote {output}")

        severity = report.worst_severity
        raise typer.Exit(EXIT_FINDING if severity in ("P0", "P1") else EXIT_CLEAN)
