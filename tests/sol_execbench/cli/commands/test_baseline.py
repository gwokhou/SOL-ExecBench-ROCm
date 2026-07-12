from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from click.testing import CliRunner

from sol_execbench.cli.commands import baseline as cli_baseline
from sol_execbench.cli.commands.baseline import export as cli_baseline_export
from sol_execbench.cli.commands.baseline import release as cli_baseline_release
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


def test_publication_verify_rejects_incomplete_contract(tmp_path: Path) -> None:
    solution = tmp_path / "candidate.json"
    trace = tmp_path / "candidate.trace.jsonl"
    solution.write_text('{"name":"candidate"}\n', encoding="utf-8")
    trace.write_text('{"trace":true}\n', encoding="utf-8")
    manifest_path = tmp_path / "publication.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.evidence_publication_manifest.v1",
                "release": "v-test",
                "scope": "authority-slice:gfx1200:gemm:1-workload",
                "source_repository": "https://github.com/example/repo",
                "source_revision": "a" * 40,
                "container_image_digest": "sha256:" + "b" * 64,
                "artifact_base_uri": "https://example.invalid/releases/v-test/",
                "candidate": {
                    "solution_ref": "candidate.json",
                    "solution_sha256": cli_baseline.sha256_file(solution),
                    "trace_relative_path": "candidate.trace.jsonl",
                    "trace_sha256": cli_baseline.sha256_file(trace),
                    "timing_relative_path": "candidate.trace.jsonl",
                    "timing_sha256": cli_baseline.sha256_file(trace),
                },
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "publication",
            "verify",
            "--manifest",
            str(manifest_path),
            "--artifact-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "artifacts must be non-empty" in result.output


def test_release_build_writes_compact_baseline_and_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    suite_path, trace_path = _release_inputs(tmp_path)
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> tuple[object, object]:
        captured.update(kwargs)
        return real_build_release_baseline_bundle(**kwargs)

    monkeypatch.setattr(
        cli_baseline_release, "build_release_baseline_bundle", fake_build
    )

    baseline_path = tmp_path / "baseline.json"
    bundle_path = tmp_path / "bundle.json"
    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "release",
            "build",
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
            "--scope",
            "test-authority-slice:gemm:1-workload",
            "--latency-tolerance-rel",
            "0.05",
        ],
    )

    assert result.exit_code == 0, result.output
    assert baseline_path.exists()
    assert bundle_path.exists()
    provenance = cast(cli_baseline.ReleaseProvenance, captured["provenance"])
    assert provenance.clock_policy == "locked"
    assert provenance.suite_manifest_sha256 is not None


def test_release_build_requires_positive_tolerance(tmp_path: Path) -> None:
    suite_path, trace_path = _release_inputs(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "release",
            "build",
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
            "--scope",
            "test-authority-slice:gemm:1-workload",
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
        suite_manifest_ref=str(suite_path),
        suite_manifest_sha256=cli_baseline.sha256_file(suite_path),
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
        summary: dict[str, int] = field(
            default_factory=lambda: {"passed": 0, "total": 0}
        )

        def to_dict(self) -> dict[str, object]:
            return {"schema_version": "sol_execbench.release_baseline_verification.v1"}

    def fake_verify(**kwargs: Any) -> FakeReport:
        assert kwargs["bundle"] == load_release_baseline_bundle(bundle_path)
        assert kwargs["rerun_provenance"].clock_policy == "locked"
        return FakeReport()

    monkeypatch.setattr(
        cli_baseline_release, "verify_release_baseline_rerun", fake_verify
    )
    output_path = tmp_path / "verification.json"
    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "release",
            "verify",
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


def test_release_build_and_verify_share_the_manifest_file_digest(
    tmp_path: Path,
) -> None:
    suite_path, trace_path = _release_inputs(tmp_path)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(
        json.dumps(
            [
                {
                    "definition": "gemm",
                    "workload_uuid": "w1",
                    "bound_ref": "bound.json",
                    "bound_sha256": "b" * 64,
                    "hardware_model_ref": "model.json",
                    "hardware_model_sha256": "c" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    bundle_path = tmp_path / "bundle.json"
    runner = CliRunner()
    build = runner.invoke(
        cli,
        [
            "baseline",
            "release",
            "build",
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
            "--authority-json",
            str(authority_path),
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
            "--scope",
            "test-authority-slice:gemm:1-workload",
            "--latency-tolerance-rel",
            "0.05",
        ],
    )
    assert build.exit_code == 0, build.output

    rerun_path = tmp_path / "rerun.jsonl"
    rerun_path.write_text(
        json.dumps(
            {
                "definition": "gemm",
                "workload": {"uuid": "w1"},
                "evaluation": {
                    "status": "PASSED",
                    "performance": {"latency_ms": 1.25},
                    "release_baseline": {
                        "bound_sha256": "b" * 64,
                        "hardware_model_sha256": "c" * 64,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    verification_path = tmp_path / "verification.json"
    verify = runner.invoke(
        cli,
        [
            "baseline",
            "release",
            "verify",
            "--bundle",
            str(bundle_path),
            "--rerun-trace",
            str(rerun_path),
            "--output",
            str(verification_path),
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

    assert verify.exit_code == 0, verify.output
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = json.loads(verification_path.read_text(encoding="utf-8"))
    assert bundle["suite_manifest_ref"] == str(suite_path)
    assert bundle["suite_manifest_sha256"] == cli_baseline.sha256_file(suite_path)
    assert report["bundle_ref"] == str(bundle_path)
    assert report["bundle_sha256"] == cli_baseline.sha256_file(bundle_path)
    assert report["workloads"][0]["classification"] == "official"
    assert (
        "suite_manifest_checksum_mismatch"
        not in report["workloads"][0]["blocker_reason_codes"]
    )


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
    monkeypatch.setattr(
        cli_baseline_export, "export_hip_baseline_registry", fake_export
    )

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

    monkeypatch.setattr(
        cli_baseline_export, "export_hip_baseline_registry", fake_export
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "baseline",
            "export",
            "--trace",
            str(trace_path),
            "--output",
            str(output_path),
            "--target-id",
            "gemm",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["data"]["registry"] == {
        "a": {"b": 2},
        "z": 1,
    }


def test_selection_build_requires_and_freezes_every_suite_workload(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "suite.json"
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "selection.json"
    suite_path.write_text(
        json.dumps({"workloads": [{"definition": "gemm", "workload_uuid": "w1"}]}),
        encoding="utf-8",
    )
    candidate = {
        "definition": "gemm",
        "workload_uuid": "w1",
        "candidate": "hipblas",
        "solution_sha256": "a" * 64,
        "backend": "hipblas",
        "backend_version": "7.1",
        "build_id": "build-1",
        "dependencies": ["hipblas"],
        "timings_ms": [1.0, 1.0, 1.0],
        "median_ms": 1.0,
        "spread_rel": 0.0,
        "correctness_passed": True,
    }
    candidates_path.write_text(json.dumps([candidate]), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "selection",
            "build",
            "--suite-manifest",
            str(suite_path),
            "--candidates",
            str(candidates_path),
            "--scope",
            "gfx1200:test",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output_path.read_text())["selections"][0]["winner"] == "hipblas"
