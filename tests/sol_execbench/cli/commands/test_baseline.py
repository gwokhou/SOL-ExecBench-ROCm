from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.commands import baseline as cli_baseline
from sol_execbench.cli.main import cli
from sol_execbench.core.scoring.release_baseline import (
    build_release_baseline_bundle as real_build_release_baseline_bundle,
    load_release_baseline_bundle,
    write_release_baseline_outputs,
)


def _release_inputs(tmp_path: Path) -> tuple[Path, Path]:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps([{"definition": "gemm", "workload_uuid": "w1"}]),
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "definition": "gemm",
                "workload": {"uuid": "w1"},
                "evaluation": {
                    "status": "PASSED",
                    "performance": {"latency_ms": 1.25},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return suite_path, trace_path


def test_release_build_writes_compact_baseline_and_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    suite_path, trace_path = _release_inputs(tmp_path)
    captured: dict[str, object] = {}

    def fake_build(**kwargs: object) -> tuple[object, object]:
        captured.update(kwargs)
        return real_build_release_baseline_bundle(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_baseline, "build_release_baseline_bundle", fake_build)

    baseline_path = tmp_path / "baseline.json"
    bundle_path = tmp_path / "bundle.json"
    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "release-build",
            "--suite-manifest",
            str(suite_path),
            "--trace",
            str(trace_path),
            "--release",
            "v2.14",
            "--baseline-output",
            str(baseline_path),
            "--bundle-output",
            str(bundle_path),
            "--solution",
            "hipblaslt",
            "--solution-sha256",
            "a" * 64,
            "--environment-fingerprint",
            "gfx1200-rocm7.1",
            "--clock-policy",
            "locked",
            "--timing-policy",
            "median-100",
            "--compiler-build-id",
            "rocm-7.1",
            "--latency-tolerance-rel",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    assert baseline_path.exists()
    assert bundle_path.exists()
    provenance = captured["provenance"]
    assert provenance.clock_policy == "locked"
    assert provenance.suite_manifest_sha256 is not None


def test_release_build_requires_positive_tolerance(tmp_path: Path) -> None:
    suite_path, trace_path = _release_inputs(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "release-build",
            "--suite-manifest",
            str(suite_path),
            "--trace",
            str(trace_path),
            "--release",
            "v2.14",
            "--baseline-output",
            str(tmp_path / "baseline.json"),
            "--bundle-output",
            str(tmp_path / "bundle.json"),
            "--solution",
            "hipblaslt",
            "--solution-sha256",
            "a" * 64,
            "--environment-fingerprint",
            "gfx1200-rocm7.1",
            "--clock-policy",
            "locked",
            "--timing-policy",
            "median-100",
            "--compiler-build-id",
            "rocm-7.1",
            "--latency-tolerance-rel",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_release_verify_writes_report(monkeypatch, tmp_path: Path) -> None:
    suite_path, trace_path = _release_inputs(tmp_path)
    baseline, bundle = real_build_release_baseline_bundle(
        suite_workloads=json.loads(suite_path.read_text(encoding="utf-8")),
        trace_path=trace_path,
        release="v2.14",
        provenance=cli_baseline.ReleaseProvenance(
            solution="hipblaslt",
            solution_sha256="a" * 64,
            environment_fingerprint="gfx1200-rocm7.1",
            clock_policy="locked",
            compiler_build_id="rocm-7.1",
            timing_policy="median-100",
            suite_manifest_sha256=cli_baseline.sha256_file(suite_path),
        ),
        authority_by_key={},
        latency_tolerance_rel=0.05,
    )
    baseline_path = tmp_path / "baseline.json"
    bundle_path = tmp_path / "bundle.json"
    write_release_baseline_outputs(
        baseline=baseline,
        bundle=bundle,
        baseline_path=baseline_path,
        bundle_path=bundle_path,
    )
    rerun_path = tmp_path / "rerun.jsonl"
    rerun_path.write_text(trace_path.read_text(encoding="utf-8"), encoding="utf-8")

    @dataclass
    class FakeReport:
        release: str = "v2.14"
        summary: dict[str, int] | None = None

        def to_dict(self) -> dict[str, object]:
            return {"schema_version": "sol_execbench.release_baseline_verification.v1"}

    def fake_verify(**kwargs: object) -> FakeReport:
        assert kwargs["bundle"] == load_release_baseline_bundle(bundle_path)
        assert kwargs["rerun_provenance"].clock_policy == "locked"
        return FakeReport()

    monkeypatch.setattr(cli_baseline, "verify_release_baseline_rerun", fake_verify)
    output_path = tmp_path / "verification.json"
    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "release-verify",
            "--bundle",
            str(bundle_path),
            "--rerun-trace",
            str(rerun_path),
            "--output",
            str(output_path),
            "--solution-sha256",
            "a" * 64,
            "--environment-fingerprint",
            "gfx1200-rocm7.1",
            "--clock-policy",
            "locked",
            "--timing-policy",
            "median-100",
            "--compiler-build-id",
            "rocm-7.1",
            "--suite-manifest-sha256",
            cli_baseline.sha256_file(suite_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


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
