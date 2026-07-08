from __future__ import annotations

from sol_execbench.core.bench.profile_summary_hints import derive_bottleneck_hints
from sol_execbench.core.bench.profile_summary_models import (
    ProfileSummaryKernelMetric,
)


def test_derive_bottleneck_hints_reports_insufficient_counters() -> None:
    hints = derive_bottleneck_hints(
        [
            ProfileSummaryKernelMetric(
                kernel_name="kernel",
                name="kernel_duration_ms",
                value=0.1,
                unit="ms",
                source="trace.csv",
                artifact="trace.csv",
            )
        ]
    )

    assert [hint.model_dump(mode="json") for hint in hints] == [
        {
            "category": "insufficient_counters",
            "severity": "low",
            "confidence": "high",
            "message": "No bounded counter artifact was available for bottleneck classification.",
            "source_metrics": [],
            "evidence_artifacts": [],
        }
    ]


def test_derive_bottleneck_hints_uses_percent_and_fraction_l2_thresholds() -> None:
    percent_hints = derive_bottleneck_hints(
        [
            ProfileSummaryKernelMetric(
                kernel_name="kernel",
                name="L2_CACHE_HIT_RATE",
                value=42,
                unit="percent",
                source="counters.csv",
                artifact="counters.csv",
            )
        ]
    )
    fraction_hints = derive_bottleneck_hints(
        [
            ProfileSummaryKernelMetric(
                kernel_name="kernel",
                name="L2_CACHE_HIT_RATE",
                value=0.95,
                unit="ratio",
                source="counters.csv",
                artifact="counters.csv",
            )
        ]
    )

    assert [hint.category for hint in percent_hints] == ["memory_l2_bound"]
    assert [hint.category for hint in fraction_hints] == ["unknown"]


def test_derive_bottleneck_hints_reports_lds_and_valu_signals() -> None:
    hints = derive_bottleneck_hints(
        [
            ProfileSummaryKernelMetric(
                kernel_name="kernel",
                name="SQ_INSTS_VALU",
                value=120000,
                unit="count",
                source="counters.csv",
                artifact="counters.csv",
            ),
            ProfileSummaryKernelMetric(
                kernel_name="kernel",
                name="LDS_BANK_CONFLICT",
                value=8,
                unit="count",
                source="counters.csv",
                artifact="counters.csv",
            ),
        ]
    )

    assert [hint.category for hint in hints] == ["lds_bound", "compute_bound"]
    assert [hint.evidence_artifacts for hint in hints] == [
        ["counters.csv"],
        ["counters.csv"],
    ]
