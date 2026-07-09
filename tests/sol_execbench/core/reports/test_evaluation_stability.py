from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3TimingEvidence,
    Rocprofv3TimingRow,
)
from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
)
from sol_execbench.core.reports.evaluation_stability import (
    EVALUATION_STABILITY_SCHEMA_VERSION,
    EvaluationStabilityReport,
    build_evaluation_stability_report,
    render_evaluation_stability_markdown,
)


def test_evaluation_stability_classifies_core_statuses():
    report = build_evaluation_stability_report(
        timing_evidence=[
            {
                "schema_version": "sol_execbench.rocprofv3_timing.v1",
                "workload_uuid": "stable",
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "warmup_runs": 4,
                "clock_locked": True,
                "runtime_ms_distribution": [10.0, 10.1, 9.9, 10.0],
            },
            {
                "workload_uuid": "noisy",
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "clock_locked": True,
                "runtime_ms_distribution": [10.0, 20.0, 35.0],
            },
            {
                "workload_uuid": "few",
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "runtime_ms_distribution": [10.0],
            },
            {
                "workload_uuid": "missing",
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
            },
            {
                "workload_uuid": "clock",
                "backend": "rocprofv3",
                "activity_domain": "kernel_activity",
                "clock_locked": False,
                "runtime_ms_distribution": [10.0, 10.1, 10.0],
            },
            {
                "workload_uuid": "fallback",
                "backend": "device_events",
                "fallback_applied": True,
                "runtime_ms_distribution": [10.0, 10.0, 10.0],
            },
            {
                "workload_uuid": "unsupported",
                "backend": "unsupported",
                "activity_domain": "unsupported",
            },
        ],
        created_at="2026-05-31T00:00:00Z",
    )

    assert report.schema_version == EVALUATION_STABILITY_SCHEMA_VERSION
    assert report.report_checksum is not None
    statuses = {
        workload.workload_ref: workload.stability_status
        for workload in report.workloads
    }
    assert statuses == {
        "stable": "stable",
        "noisy": "noisy",
        "few": "insufficient_samples",
        "missing": "missing_timing",
        "clock": "clock_unlocked",
        "fallback": "profiler_overhead_risk",
        "unsupported": "backend_unsupported",
    }
    assert report.status_totals.stable == 1
    assert report.status_totals.noisy == 1
    assert report.status_totals.backend_unsupported == 1


def test_evaluation_stability_accepts_rocm_timing_evidence_shape():
    evidence = Rocprofv3TimingEvidence(
        tool_version="rocprofv3 7.0",
        gpu_architecture="gfx1201",
        activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
        aggregation_rule="aggregate kernel rows",
        backend=TimingBackend.ROCPROFV3,
        interpretation="kernel activity duration",
        warmup_runs=2,
        iterations=3,
        trial_count=3,
        clock_locked=True,
        parsed_rows=(
            Rocprofv3TimingRow(
                name="kernel_a", domain="Kernel", duration_ns=10_000_000
            ),
            Rocprofv3TimingRow(
                name="kernel_b", domain="Kernel", duration_ns=10_100_000
            ),
            Rocprofv3TimingRow(name="kernel_c", domain="Kernel", duration_ns=9_900_000),
        ),
    )

    report = build_evaluation_stability_report(
        timing_evidence=[evidence.to_dict() | {"workload_uuid": "w-rocm"}],
        created_at="2026-05-31T00:00:00Z",
    )

    workload = report.workloads[0]
    assert workload.workload_ref == "w-rocm"
    assert workload.backend == "rocprofv3"
    assert workload.activity_domain == "kernel_activity"
    assert workload.warmup_runs == 2
    assert workload.measured_repeat_count == 3
    assert workload.stability_status == "stable"


def test_evaluation_stability_report_is_strict_and_deterministic():
    report = build_evaluation_stability_report(
        timing_evidence=[
            {
                "workload_uuid": "w1",
                "backend": "rocprofv3",
                "runtime_ms_distribution": [1.0, 1.0, 1.0],
            }
        ],
        created_at="2026-05-31T00:00:00Z",
    )
    payload = report.model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        EvaluationStabilityReport.model_validate(payload)

    repeat = build_evaluation_stability_report(
        timing_evidence=[
            {
                "workload_uuid": "w1",
                "backend": "rocprofv3",
                "runtime_ms_distribution": [1.0, 1.0, 1.0],
            }
        ],
        created_at="2026-05-31T00:00:00Z",
    )
    assert json.loads(report.to_json()) == json.loads(repeat.to_json())


def test_evaluation_stability_markdown_keeps_claim_boundaries_visible():
    report = build_evaluation_stability_report(
        timing_evidence=[],
        created_at="2026-05-31T00:00:00Z",
    )
    markdown = render_evaluation_stability_markdown(report)

    for expected in (
        "timing-quality interpretation only",
        "not correctness authority",
        "not score authority",
        "not paper parity",
        "not leaderboard authority",
        "not native-host validation",
        "not new-hardware validation",
        "`timing_quality_interpretation`: true",
        "`score_authority`: false",
    ):
        assert expected in markdown
