from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sol_execbench.core.bench.rocm_profiler import counter_collection
from sol_execbench.core.bench.rocm_profiler.counter_collection import (
    COUNTER_REASON_ARTIFACT_INCOMPLETE,
    COUNTER_REASON_COLLECTED,
    COUNTER_REASON_PMC_CHECK_FAILED,
    COUNTER_REASON_UNSUPPORTED,
    collect_rocprofv3_counters,
)
from sol_execbench.core.bench.rocm_profiler.counter_provenance import (
    Rocprofv3CounterProvenance,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileStatus,
)
from sol_execbench.core.data.json_utils import load_json_file

_REQUIRED_COUNTERS = """
Name:gfx1200
Counter_Name : SQ_WAVES_sum
Counter_Name : FETCH_SIZE
Counter_Name : GL2C_EA_WRREQ_64B_sum
Counter_Name : GL2C_MISS_sum
Counter_Name : GL2C_HIT_sum
Counter_Name : LDSBankConflict
"""


def _request(tmp_path: Path) -> Rocprofv3ProfileRequest:
    return Rocprofv3ProfileRequest(
        application_command=("python", "evaluate.py"),
        output_directory=tmp_path / "profile",
        output_file="profile",
        working_directory=tmp_path,
        timeout_seconds=30,
    )


def test_counter_collection_discovers_availability_and_hashes_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = tmp_path / "rocprofv3"
    tool.write_text("fake executable", encoding="utf-8")
    monkeypatch.setattr(
        counter_collection,
        "resolve_rocm_tool",
        lambda _name: tool,
    )
    request = _request(tmp_path)

    def runner(command, _cwd, _timeout):
        if "info" in command:
            assert command[-4:] == ["-d", "0", "info", "--pmc"]
            return subprocess.CompletedProcess(
                command, 0, _REQUIRED_COUNTERS, ""
            )
        if "pmc-check" in command:
            assert command[1:4] == ["-d", "0", "pmc-check"]
            return subprocess.CompletedProcess(
                command,
                0,
                "Following input counters can be collected together",
                "",
            )
        job = yaml.safe_load(Path(command[2]).read_text(encoding="utf-8"))
        assert len(job["jobs"]) == 1
        entry = job["jobs"][0]
        pass_dir = Path(entry["output_directory"])
        pass_dir.mkdir(parents=True)
        rows = "".join(
            f"1,kernel,1,1,{counter},1\n" for counter in entry["pmc"]
        )
        pass_index = int(pass_dir.name.removeprefix("pass_"))
        (pass_dir / f"{pass_index}_counter_collection.csv").write_text(
            "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,"
            "Counter_Name,Counter_Value\n" + rows,
            encoding="utf-8",
        )
        (pass_dir / f"{pass_index}_results.rocpd").write_bytes(b"audit")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = collect_rocprofv3_counters(request, runner=runner)

    assert result.status is Rocprofv3ProfileStatus.SUCCESS
    assert COUNTER_REASON_COLLECTED in result.reason_codes
    assert set(result.provenance) == {
        "application_command_sha256",
        "application_executable_sha256",
        "availability_sha256",
        "configuration_sha256",
        "counter_definition_sha256",
        "pmc_check_sha256",
        "profiler_sha256",
    }
    assert {artifact.kind for artifact in result.artifacts} >= {
        Rocprofv3ArtifactKind.COUNTER_CSV,
        Rocprofv3ArtifactKind.ROCPD,
    }
    provenance = load_json_file(
        Rocprofv3CounterProvenance,
        request.output_directory / "profile.counter-metadata.json",
    )
    assert provenance.pmc_check_sha256 == result.provenance["pmc_check_sha256"]


def test_unsupported_counter_provenance_schema_is_rejected() -> None:
    with pytest.raises((ValidationError, ValueError), match="schema_version"):
        Rocprofv3CounterProvenance.model_validate(
            {"schema_version": "unsupported"}
        )


def test_counter_collection_fails_closed_for_unsupported_counter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = tmp_path / "rocprofv3-avail"
    tool.write_text("fake executable", encoding="utf-8")
    monkeypatch.setattr(
        counter_collection,
        "resolve_rocm_tool",
        lambda _name: tool,
    )

    def runner(command, _cwd, _timeout):
        return subprocess.CompletedProcess(
            command,
            0,
            "Name:gfx1200\nCounter_Name : SQ_WAVES_sum\n",
            "",
        )

    result = collect_rocprofv3_counters(_request(tmp_path), runner=runner)

    assert result.status is Rocprofv3ProfileStatus.UNAVAILABLE
    assert result.reason_codes == (COUNTER_REASON_UNSUPPORTED,)


def test_counter_collection_rejects_non_gfx1200_device(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = tmp_path / "rocprofv3-avail"
    tool.write_text("fake executable", encoding="utf-8")
    monkeypatch.setattr(
        counter_collection, "resolve_rocm_tool", lambda _name: tool
    )

    result = collect_rocprofv3_counters(
        _request(tmp_path),
        runner=lambda command, _cwd, _timeout: subprocess.CompletedProcess(
            command,
            0,
            _REQUIRED_COUNTERS.replace("gfx1200", "gfx1100"),
            "",
        ),
    )

    assert result.status is Rocprofv3ProfileStatus.UNAVAILABLE
    assert (
        result.skipped_reason
        == "required counters are unsupported: architecture:gfx1200"
    )


def test_counter_collection_rejects_incomplete_successful_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = tmp_path / "rocprofv3"
    tool.write_text("fake executable", encoding="utf-8")
    monkeypatch.setattr(
        counter_collection,
        "resolve_rocm_tool",
        lambda _name: tool,
    )

    def runner(command, _cwd, _timeout):
        if "info" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                _REQUIRED_COUNTERS,
                "",
            )
        if "pmc-check" in command:
            return subprocess.CompletedProcess(command, 0, "supported", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = collect_rocprofv3_counters(_request(tmp_path), runner=runner)

    assert result.status is Rocprofv3ProfileStatus.FAILED
    assert COUNTER_REASON_ARTIFACT_INCOMPLETE in result.reason_codes
    assert not any(
        artifact.kind is Rocprofv3ArtifactKind.COUNTER_CSV
        for artifact in result.artifacts
    )


def test_counter_collection_fails_closed_when_pmc_check_rejects_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = tmp_path / "rocprofv3-avail"
    tool.write_text("fake executable", encoding="utf-8")
    monkeypatch.setattr(
        counter_collection,
        "resolve_rocm_tool",
        lambda _name: tool,
    )

    def runner(command, _cwd, _timeout):
        if "info" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                _REQUIRED_COUNTERS,
                "",
            )
        return subprocess.CompletedProcess(command, 1, "", "not supported")

    result = collect_rocprofv3_counters(_request(tmp_path), runner=runner)

    assert result.status is Rocprofv3ProfileStatus.FAILED
    assert result.reason_codes == (COUNTER_REASON_PMC_CHECK_FAILED,)
