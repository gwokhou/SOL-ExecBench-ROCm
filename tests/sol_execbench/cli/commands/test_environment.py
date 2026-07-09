from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli.commands import environment as cli_environment
from sol_execbench.core.platform.environment import (
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
)


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        generated_at="2026-05-25T00:00:00+00:00",
        collection_status=EnvironmentEvidenceStatus.AVAILABLE,
    )


def test_environment_snapshot_sidecar_disabled_by_default(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "trace.jsonl"
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, raising=False)
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        output,
        collector=lambda: _snapshot(),
    )

    assert written is None
    assert not output.with_name("trace.jsonl.environment.json").exists()


def test_environment_snapshot_sidecar_uses_explicit_path(tmp_path: Path, monkeypatch):
    sidecar = tmp_path / "run" / "env.json"
    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, str(sidecar))
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        tmp_path / "trace.jsonl",
        collector=lambda: _snapshot(),
    )

    assert written == sidecar
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == "sol_execbench.environment_snapshot.v2"
    assert payload["collection_status"] == "available"


def test_environment_snapshot_sidecar_can_be_derived_from_trace_output(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "trace.jsonl"
    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, "1")
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        output,
        collector=lambda: _snapshot(),
    )

    assert written == tmp_path / "trace.jsonl.environment.json"
    assert written is not None
    assert json.loads(written.read_text())["collection_status"] == "available"


def test_environment_snapshot_request_without_output_path_is_nonfatal(monkeypatch):
    calls = 0

    def collector() -> EnvironmentSnapshot:
        nonlocal calls
        calls += 1
        return _snapshot()

    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_ENABLE_ENV, "1")
    monkeypatch.delenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, raising=False)

    written = cli_environment._write_environment_snapshot_sidecar(
        None, collector=collector
    )

    assert written is None
    assert calls == 0


def test_environment_snapshot_collection_failure_is_nonfatal(
    tmp_path: Path, monkeypatch
):
    sidecar = tmp_path / "env.json"
    monkeypatch.setenv(cli_environment.ENV_SNAPSHOT_PATH_ENV, str(sidecar))

    def collector() -> EnvironmentSnapshot:
        raise RuntimeError("probe failed")

    written = cli_environment._write_environment_snapshot_sidecar(
        tmp_path / "trace.jsonl",
        collector=collector,
    )

    assert written is None
    assert not sidecar.exists()
