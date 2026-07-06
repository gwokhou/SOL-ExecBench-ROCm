from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryArtifactCitation as FacadeArtifactCitation,
)
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
    ProfileSummaryArtifactCitation,
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
    assert FacadeArtifactCitation is ProfileSummaryArtifactCitation
    assert FacadeContent is ProfileSummaryContent


@pytest.mark.parametrize(
    ("model_type", "payload", "mutation_field"),
    [
        (
            ProfileSummaryMetric,
            {
                "name": "artifact_count",
                "value": 1,
                "unit": "count",
                "source": "rocprofv3_profile_metadata",
            },
            "name",
        ),
        (
            ProfileSummaryStructuredMetric,
            {
                "name": "artifact_coverage_status",
                "value": "complete",
                "source": "rocprofv3_profile_metadata",
            },
            "name",
        ),
        (
            ProfileSummaryKernelMetric,
            {
                "kernel_name": "kernel",
                "name": "kernel_duration_ms",
                "value": 1.25,
                "unit": "ms",
                "source": "trace.csv",
            },
            "name",
        ),
        (
            ProfileSummaryBottleneckHint,
            {
                "category": "compute_bound",
                "severity": "low",
                "confidence": "low",
                "message": "Compute-bound signal detected.",
            },
            "message",
        ),
        (
            ProfileSummaryArtifactCitation,
            {
                "kind": "trace",
                "label": "trace.csv",
                "path": "trace.csv",
                "size_bytes": 12,
            },
            "label",
        ),
        (
            ProfileSummaryContent,
            {
                "artifact_count": 0,
            },
            "artifact_count",
        ),
    ],
)
def test_profile_summary_content_models_remain_strict_and_frozen(
    model_type: type,
    payload: dict[str, object],
    mutation_field: str,
) -> None:
    instance = model_type.model_validate(payload)

    with pytest.raises(ValidationError):
        model_type.model_validate(
            {
                **payload,
                "unexpected": "field",
            }
        )

    with pytest.raises(ValidationError):
        setattr(instance, mutation_field, "changed")


def test_profile_summary_content_accepts_empty_artifact_summary() -> None:
    content = ProfileSummaryContent(artifact_count=0)

    assert content.artifact_count == 0
    assert content.metrics == []
    assert content.workload_metrics == []
    assert content.kernel_metrics == []
    assert content.bottleneck_hints == []
    assert content.parse_warnings == []
