from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli.sidecars import profile as cli_profile_sidecars
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
)


def test_profile_sidecar_is_disabled_when_no_profile_result(tmp_path: Path):
    output = tmp_path / "trace.jsonl"

    written = cli_profile_sidecars._write_profile_sidecar(output, None)

    assert written is None
    assert not (tmp_path / "trace.jsonl.profile.json").exists()


def test_profile_sidecar_records_diagnostic_metadata(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    result = Rocprofv3ProfileResult(
        status="unavailable",
        command=("rocprofv3", "--", "python", "eval_driver.py"),
        output_directory=tmp_path / "trace.jsonl.rocprofv3",
        output_file="profile",
        skipped_reason="rocprofv3 is not available on PATH",
        profiler_available=False,
    )

    written = cli_profile_sidecars._write_profile_sidecar(output, result)

    assert written == tmp_path / "trace.jsonl.profile.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.rocprofv3_profile.v1"
    assert payload["status"] == "unavailable"
    assert payload["diagnostic_only"] is True
    assert payload["score_authority"] is False
    assert payload["skipped_reason"] == "rocprofv3 is not available on PATH"


def test_profile_summary_sidecar_tracks_trace_output(tmp_path: Path):
    output = tmp_path / "trace.jsonl"

    assert cli_profile_sidecars._profile_summary_sidecar_path(output) == (
        tmp_path / "trace.jsonl.profile-summary.json"
    )
    assert cli_profile_sidecars._profile_summary_sidecar_path(None) is None


def test_profile_summary_sidecar_records_bounded_metadata(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    profile_metadata = tmp_path / "trace.jsonl.profile.json"
    profile_metadata.write_text(
        '{"schema_version":"sol_execbench.rocprofv3_profile.v1"}\n'
    )
    profile_artifact_dir = tmp_path / "trace.jsonl.rocprofv3" / "trace"
    profile_artifact_dir.mkdir(parents=True)
    profile_artifact = profile_artifact_dir / "trace.rocpd"
    profile_artifact.write_text("profile artifact\n")
    counter_artifact = profile_artifact_dir / "trace_counters.csv"
    counter_artifact.write_text("Metric,Value,Unit\nSQ_INSTS_VALU,12,count\n")
    result = Rocprofv3ProfileResult(
        status="success",
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path / "trace.jsonl.rocprofv3",
        output_file="trace",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=profile_artifact,
                kind="rocpd",
                size_bytes=profile_artifact.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=counter_artifact,
                kind="counter_csv",
                size_bytes=counter_artifact.stat().st_size,
            ),
        ),
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="complete",
        reason_codes=("rocprof_artifacts_registered",),
    )

    written = cli_profile_sidecars._write_profile_summary_sidecar(
        output,
        result,
        profile_sidecar_path=profile_metadata,
        run_id="run-001",
        sol_version="v1.43",
    )

    assert written == tmp_path / "trace.jsonl.profile-summary.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.profile_summary.v2"
    assert payload["status"] == "available"
    assert payload["authority"] == "diagnostic"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["run_id"] == "run-001"
    assert payload["identity"]["sol_version"] == "v1.43"
    assert "sol_contract_version" not in payload["identity"]
    assert payload["summary"]["profiler_status"] == "success"
    assert payload["summary"]["artifact_count"] == 2
    assert payload["summary"]["artifact_coverage_status"] == "complete"
    assert payload["summary"]["reason_codes"] == ["rocprof_artifacts_registered"]
    assert payload["summary"]["kernel_metrics"] == [
        {
            "kernel_name": "trace_counters",
            "name": "SQ_INSTS_VALU",
            "value": 12,
            "unit": "count",
            "source": "trace_counters.csv",
            "artifact": "trace_counters.csv",
            "parse_status": "available",
        }
    ]
    assert payload["summary"]["bottleneck_hints"][0]["category"] == "compute_bound"
    citation_kinds = {citation["kind"] for citation in payload["artifact_citations"]}
    assert citation_kinds == {"trace", "profile_metadata", "profiler_artifact"}
    profiler_citations = [
        citation
        for citation in payload["artifact_citations"]
        if citation["kind"] == "profiler_artifact"
    ]
    assert {citation["path"] for citation in profiler_citations} == {
        "trace.rocpd",
        "trace_counters.csv",
    }
    assert {citation["size_bytes"] for citation in profiler_citations} == {
        profile_artifact.stat().st_size,
        counter_artifact.stat().st_size,
    }
    for citation in payload["artifact_citations"]:
        assert "/" not in citation["path"]
        assert citation["sha256"] is not None
        assert len(citation["sha256"]) == 64
    assert "profile artifact" not in json.dumps(payload)


def test_profile_output_directory_tracks_trace_output(tmp_path: Path):
    output = tmp_path / "run" / "trace.jsonl"

    assert cli_profile_sidecars._profile_output_directory(output, tmp_path) == (
        tmp_path / "run" / "trace.jsonl.rocprofv3"
    )


def test_profile_output_directory_is_absolute_for_relative_trace_output(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    assert cli_profile_sidecars._profile_output_directory(
        Path("out/trace.jsonl"), tmp_path
    ) == (tmp_path / "out" / "trace.jsonl.rocprofv3")
