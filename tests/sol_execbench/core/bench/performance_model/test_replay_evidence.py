from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.replay_evidence import (
    REPLAY_EVIDENCE_ITERATIONS,
    RawPerformanceReplayRecord,
    build_performance_replay_evidence,
)
from sol_execbench.core.bench.performance_model.timing_evidence import (
    PerformanceTimingEvidenceSidecar,
    WorkloadTimingEvidence,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
)
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.integrity import stable_json_checksum
from sol_execbench.core.platform.runtime import (
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)

_HEADER = (
    "Dispatch_Id,Kernel_Name,Grid_Size,Workgroup_Size,"
    "Counter_Name,Counter_Value\n"
)
_EMPTY_CACHE_SHA256 = stable_json_checksum(
    {
        "detected_l2_bytes": None,
        "clear_buffer_bytes": None,
        "source": None,
        "fallback_reason": None,
    }
)


def _topology() -> PCIeTopologyIdentity:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=8,
        max_width=16,
    )
    return PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )


def _write_timing(path: Path) -> None:
    timing = PerformanceTimingEvidenceSidecar(
        run_id="a" * 64,
        trace_sha256="a" * 64,
        solution_sha256="b" * 64,
        workloads=[
            WorkloadTimingEvidence(
                workload_uuid="w0",
                input_sha256="c" * 64,
                latency_ms=1.0,
                lower_ms=0.9,
                upper_ms=1.1,
                trial_samples_ms=[[1.0]],
                warmup_runs=3,
                timing_protocol="device_event_v1",
            )
        ],
    )
    atomic_write_json_value(path, timing.model_dump(mode="json"))


def _environment() -> list[RuntimeGPUTelemetry]:
    return [
        RuntimeGPUTelemetry(
            phase=phase,
            gpu_id="gpu-0",
            gpu_bdf="0000:03:00.0",
            pcie_topology=_topology(),
            performance_level="AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK",
            temperature_c=50.0,
            foreign_process_count=0,
        )
        for phase in ("pre", "post")
    ]


def test_replay_evidence_binds_input_process_and_cross_pass_dispatches(
    tmp_path: Path,
) -> None:
    timing_path = tmp_path / "timing.json"
    _write_timing(timing_path)
    csv_paths: list[Path] = []
    for pass_index, counter in ((1, "SQ_WAVES"), (2, "FETCH_SIZE")):
        pass_dir = tmp_path / f"pass_{pass_index}"
        pass_dir.mkdir()
        csv_path = pass_dir / "counter.csv"
        csv_path.write_text(
            _HEADER + f"1,kernel,1,1,{counter},1\n",
            encoding="utf-8",
        )
        csv_paths.append(csv_path)
        record = RawPerformanceReplayRecord(
            pid=100 + pass_index,
            parent_pid=10,
            process_executable_sha256="d" * 64,
            pass_index=pass_index,
            workload_uuid="w0",
            input_sha256="c" * 64,
            cache_identity_sha256=_EMPTY_CACHE_SHA256,
            marker_ranges=[
                f"sol_execbench/w0/iteration/{index}"
                for index in range(REPLAY_EVIDENCE_ITERATIONS)
            ],
        )
        atomic_write_jsonl_values(
            tmp_path / f"performance-replay-raw-{pass_index}.jsonl",
            [record.model_dump(mode="json")],
        )

    replay = build_performance_replay_evidence(
        staging_dir=tmp_path,
        run_id="a" * 64,
        candidate_sha256="f" * 64,
        canonical_timing_path=timing_path,
        artifact_paths=csv_paths,
        expected_gpu_id="gpu-0",
        expected_gpu_bdf="0000:03:00.0",
        environment=_environment(),
    )

    assert replay.status is DiagnosticSidecarStatus.AVAILABLE
    assert replay.alignment_digest is not None
    assert len(replay.processes) == 2


def test_replay_evidence_fails_closed_on_input_drift(tmp_path: Path) -> None:
    timing_path = tmp_path / "timing.json"
    _write_timing(timing_path)
    record = RawPerformanceReplayRecord(
        pid=101,
        parent_pid=10,
        process_executable_sha256="d" * 64,
        pass_index=1,
        workload_uuid="w0",
        input_sha256="0" * 64,
        cache_identity_sha256=_EMPTY_CACHE_SHA256,
        marker_ranges=[
            f"sol_execbench/w0/iteration/{index}"
            for index in range(REPLAY_EVIDENCE_ITERATIONS)
        ],
    )
    atomic_write_jsonl_values(
        tmp_path / "performance-replay-raw-1.jsonl",
        [record.model_dump(mode="json")],
    )
    pass_dir = tmp_path / "pass_1"
    pass_dir.mkdir()
    csv_path = pass_dir / "counter.csv"
    csv_path.write_text(
        _HEADER + "1,kernel,1,1,SQ_WAVES,1\n",
        encoding="utf-8",
    )

    replay = build_performance_replay_evidence(
        staging_dir=tmp_path,
        run_id="a" * 64,
        candidate_sha256="f" * 64,
        canonical_timing_path=timing_path,
        artifact_paths=[csv_path],
        expected_gpu_id="gpu-0",
        expected_gpu_bdf="0000:03:00.0",
        environment=_environment(),
    )

    assert replay.status is DiagnosticSidecarStatus.PARTIAL
    assert "replay_input_sha256_mismatch" in replay.reason_codes


def test_replay_evidence_parses_only_registered_counter_csvs(
    tmp_path: Path,
) -> None:
    timing_path = tmp_path / "timing.json"
    _write_timing(timing_path)
    record = RawPerformanceReplayRecord(
        pid=101,
        parent_pid=10,
        process_executable_sha256="d" * 64,
        pass_index=1,
        workload_uuid="w0",
        input_sha256="c" * 64,
        cache_identity_sha256=_EMPTY_CACHE_SHA256,
        marker_ranges=[
            f"sol_execbench/w0/iteration/{index}"
            for index in range(REPLAY_EVIDENCE_ITERATIONS)
        ],
    )
    atomic_write_jsonl_values(
        tmp_path / "performance-replay-raw-1.jsonl",
        [record.model_dump(mode="json")],
    )
    pass_dir = tmp_path / "pass_1"
    pass_dir.mkdir()
    counter_path = pass_dir / "counter.csv"
    counter_path.write_text(
        _HEADER + "1,kernel,1,1,SQ_WAVES,1\n",
        encoding="utf-8",
    )
    marker_path = pass_dir / "marker_api_trace.csv"
    marker_path.write_text(
        "Domain,Function\nMARKER_CORE_RANGE_API,iteration/0\n",
        encoding="utf-8",
    )

    replay = build_performance_replay_evidence(
        staging_dir=tmp_path,
        run_id="a" * 64,
        candidate_sha256="f" * 64,
        canonical_timing_path=timing_path,
        artifact_paths=[counter_path, marker_path],
        counter_paths=[counter_path],
        expected_gpu_id="gpu-0",
        expected_gpu_bdf="0000:03:00.0",
        environment=_environment(),
    )

    assert replay.status is DiagnosticSidecarStatus.AVAILABLE
    assert marker_path.name in replay.artifact_sha256
