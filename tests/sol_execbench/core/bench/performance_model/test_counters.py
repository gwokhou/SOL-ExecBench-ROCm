from __future__ import annotations

from pathlib import Path

import yaml

from sol_execbench.core.bench.rocm_profiler.counters import (
    CounterPassCSV,
    build_rocprofv3_counter_command,
    parse_and_align_counter_passes,
    parse_available_architectures,
    parse_available_counters,
    write_counter_job,
)

_HEADER = (
    "Correlation_Id,Dispatch_Id,Queue_Id,Grid_Size,Kernel_Name,Workgroup_Size,"
    "Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp,Duration\n"
)


def _write_pass(path: Path, dispatch: int, counter: str, value: str) -> None:
    path.write_text(
        _HEADER
        + f"{dispatch},{dispatch},0,1024,kernel,64,{counter},{value},100,200,100\n",
        encoding="utf-8",
    )


def test_counter_parser_aligns_passes_without_exposing_duration(
    tmp_path: Path,
) -> None:
    first = tmp_path / "pass_1.csv"
    second = tmp_path / "pass_2.csv"
    _write_pass(first, 7, "SQ_WAVES", "32")
    _write_pass(second, 99, "FETCH_SIZE", "2")

    dispatches = parse_and_align_counter_passes(
        [CounterPassCSV(1, first), CounterPassCSV(2, second)],
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        required_counters={"SQ_WAVES", "FETCH_SIZE"},
    )

    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert dispatch.valid is True
    assert dispatch.counter_passes == [1, 2]
    assert dispatch.counters == {"FETCH_SIZE": 2048.0, "SQ_WAVES": 32.0}
    assert "duration" not in dispatch.model_dump()


def test_counter_parser_invalidates_cross_pass_misalignment(
    tmp_path: Path,
) -> None:
    first = tmp_path / "pass_1.csv"
    second = tmp_path / "pass_2.csv"
    _write_pass(first, 1, "SQ_WAVES", "32")
    second.write_text(
        _HEADER + "2,2,0,2048,other_kernel,64,FETCH_SIZE,1024,100,200,100\n",
        encoding="utf-8",
    )

    dispatches = parse_and_align_counter_passes(
        [CounterPassCSV(1, first), CounterPassCSV(2, second)],
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
    )

    assert len(dispatches) == 2
    assert all(not dispatch.valid for dispatch in dispatches)
    assert all(
        "cross_pass_alignment_mismatch" in dispatch.reason_codes
        for dispatch in dispatches
    )


def test_availability_and_controlled_job_formats(tmp_path: Path) -> None:
    output = """
    GPU:0
    Name:gfx1200
    Counter_Name : SQ_WAVES
    Counter_Name : FETCH_SIZE
    """
    assert parse_available_counters(output) == {"SQ_WAVES", "FETCH_SIZE"}
    assert parse_available_architectures(output) == {"gfx1200"}

    job = tmp_path / "counters.yaml"
    write_counter_job(
        job,
        [["SQ_WAVES"], ["FETCH_SIZE"]],
        output_directory=str(tmp_path / "out"),
    )
    payload = yaml.safe_load(job.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["output_format"] == ["csv", "rocpd"]
    assert build_rocprofv3_counter_command(
        ["python", "evaluate.py"],
        input_path=job,
    ) == [
        "rocprofv3",
        "--input",
        str(job),
        "--",
        "python",
        "evaluate.py",
    ]
