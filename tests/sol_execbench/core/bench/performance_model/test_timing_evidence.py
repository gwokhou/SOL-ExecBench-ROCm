from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.timing_evidence import (
    RawPerformanceTimingRecord,
    build_performance_timing_evidence,
    hierarchical_bootstrap_interval,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_jsonl_values,
)
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput, Workload


def _trace() -> Trace:
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
            checks=[{"type": "numeric", "output": "output"}],
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="gfx1200"),
            timestamp="2026-07-29T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(latency_ms=1.0),
        ),
    )


def test_hierarchical_bootstrap_is_deterministic() -> None:
    trials = [[0.9, 1.0, 1.1], [0.95, 1.0, 1.05]]

    assert hierarchical_bootstrap_interval(
        trials
    ) == hierarchical_bootstrap_interval(trials)


def test_timing_evidence_binds_raw_samples_to_trace(tmp_path: Path) -> None:
    trace = _trace()
    trace_path = tmp_path / "trace.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    atomic_write_jsonl_values(trace_path, [trace])
    atomic_write_jsonl_values(
        raw_path,
        [
            RawPerformanceTimingRecord(
                workload_uuid="w0",
                latency_ms=1.0,
                trial_samples_ms=[[0.9, 1.0, 1.1]],
                warmup_runs=3,
                timing_protocol="device_event_v1",
            )
        ],
    )

    sidecar = build_performance_timing_evidence(
        raw_path=raw_path,
        trace_path=trace_path,
        traces=[trace],
        solution_sha256="a" * 64,
    )

    assert sidecar.workloads[0].lower_ms <= 1.0
    assert sidecar.workloads[0].upper_ms >= 1.0

    atomic_write_jsonl_values(
        raw_path,
        [
            RawPerformanceTimingRecord(
                workload_uuid="w0",
                latency_ms=2.0,
                trial_samples_ms=[[2.0]],
                warmup_runs=3,
                timing_protocol="device_event_v1",
            )
        ],
    )
    with pytest.raises(ValueError, match="does not match canonical"):
        build_performance_timing_evidence(
            raw_path=raw_path,
            trace_path=trace_path,
            traces=[trace],
            solution_sha256="a" * 64,
        )
