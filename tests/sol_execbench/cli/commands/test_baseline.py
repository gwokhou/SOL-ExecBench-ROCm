from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.commands import baseline as cli_baseline
from sol_execbench.cli.main import cli


def test_baseline_export_writes_registry_and_prints_message(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "baseline_registry.json"
    trace_path.write_text("{}\n")

    def fake_export(
        *,
        trace_path: Path,
        output_path: Path,
        target_id: str,
        sol_version: str,
        timing_policy: str,
    ) -> dict[str, object]:
        assert trace_path == trace_path_arg
        assert output_path == output_path_arg
        assert target_id == "gemm"
        assert sol_version == "rev-a"
        assert timing_policy == "latency_ms"
        registry: dict[str, object] = {
            "target_id": target_id,
            "sol_version": sol_version,
        }
        output_path.write_text(json.dumps(registry))
        return registry

    trace_path_arg = trace_path
    output_path_arg = output_path
    monkeypatch.setattr(cli_baseline, "export_hip_baseline_registry", fake_export)

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "export",
            "--trace",
            str(trace_path),
            "--output",
            str(output_path),
            "--target-id",
            "gemm",
            "--sol-version",
            "rev-a",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output_path.read_text()) == {
        "target_id": "gemm",
        "sol_version": "rev-a",
    }
    normalized_output = result.output.replace("\n", "")
    assert "Wrote measured baseline registry to" in normalized_output
    assert str(output_path) in normalized_output


def test_baseline_export_json_prints_sorted_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    output_path = tmp_path / "baseline_registry.json"
    trace_path.write_text("{}\n")

    def fake_export(
        *,
        trace_path: Path,
        output_path: Path,
        target_id: str,
        sol_version: str,
        timing_policy: str,
    ) -> dict[str, object]:
        return {"z": 1, "a": {"b": 2}}

    monkeypatch.setattr(cli_baseline, "export_hip_baseline_registry", fake_export)

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "export",
            "--trace",
            str(trace_path),
            "--output",
            str(output_path),
            "--target-id",
            "gemm",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == '{"a": {"b": 2}, "z": 1}\n'
