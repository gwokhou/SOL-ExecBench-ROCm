from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryBottleneckHint as FacadeBottleneckHint,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryContent as FacadeContent,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryKernelMetric as FacadeKernelMetric,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryMetric as FacadeMetric,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryStructuredMetric as FacadeStructuredMetric,
)
from sol_execbench.core.bench.profile_summary_models import (
    ProfileSummaryBottleneckHint,
    ProfileSummaryContent,
    ProfileSummaryKernelMetric,
    ProfileSummaryMetric,
    ProfileSummaryStructuredMetric,
)


def test_profile_summary_model_names_remain_reexported_from_facade() -> None:
    assert FacadeMetric is ProfileSummaryMetric
    assert FacadeStructuredMetric is ProfileSummaryStructuredMetric
    assert FacadeKernelMetric is ProfileSummaryKernelMetric
    assert FacadeBottleneckHint is ProfileSummaryBottleneckHint
    assert FacadeContent is ProfileSummaryContent


def test_profile_summary_content_models_remain_strict_and_frozen() -> None:
    metric = ProfileSummaryKernelMetric(
        kernel_name="kernel",
        name="kernel_duration_ms",
        value=1.25,
        unit="ms",
        source="trace.csv",
    )

    with pytest.raises(ValidationError):
        ProfileSummaryKernelMetric.model_validate(
            {
                **metric.model_dump(mode="json"),
                "unexpected": "field",
            }
        )

    with pytest.raises(ValidationError):
        metric.name = "changed"  # type: ignore[misc]


def test_profile_summary_content_accepts_empty_artifact_summary() -> None:
    content = ProfileSummaryContent(artifact_count=0)

    assert content.artifact_count == 0
    assert content.metrics == []
    assert content.workload_metrics == []
    assert content.kernel_metrics == []
    assert content.bottleneck_hints == []
    assert content.parse_warnings == []
