from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from sol_execbench.core.bench.rocm_profiler import counter_collection
from sol_execbench.core.bench.rocm_profiler.counter_collection import (
    COUNTER_REASON_COLLECTED,
    COUNTER_REASON_UNSUPPORTED,
    collect_rocprofv3_counters,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileStatus,
)

_REQUIRED_COUNTERS = """
Name:gfx1200
Counter_Name : SQ_WAVES
Counter_Name : SQ_INSTS_VALU
Counter_Name : FETCH_SIZE
Counter_Name : GL2C_EA_WRREQ_64B_sum
Counter_Name : SQ_INSTS_LDS
Counter_Name : SQ_WAVE_CYCLES
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
        job = yaml.safe_load(Path(command[2]).read_text(encoding="utf-8"))
        assert all(len(entry["pmc"]) == 1 for entry in job["jobs"])
        pass_dir = request.output_directory / "pass_1"
        pass_dir.mkdir(parents=True)
        (pass_dir / "1_counter_collection.csv").write_text(
            "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,"
            "Counter_Name,Counter_Value\n"
            "1,kernel,1,1,SQ_WAVES,1\n",
            encoding="utf-8",
        )
        (pass_dir / "1_results.rocpd").write_bytes(b"audit")
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
        "profiler_sha256",
    }
    assert {artifact.kind for artifact in result.artifacts} >= {
        Rocprofv3ArtifactKind.COUNTER_CSV,
        Rocprofv3ArtifactKind.ROCPD,
    }


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
            "Name:gfx1200\nCounter_Name : SQ_WAVES\n",
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
