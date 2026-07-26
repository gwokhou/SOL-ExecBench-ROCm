# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime-profile routing into Decision precedence."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.decision.decision_models import (
    DecisionBottleneckClass,
)
from sol_execbench.core.bench.decision.runtime import runtime_decision_precedence
from sol_execbench.core.bench.profile_summary.models import (
    ProfileSummaryHintCategory,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)


def _result(
    tmp_path: Path,
    *,
    status: Rocprofv3ProfileStatus,
    metric: str | None = None,
) -> Rocprofv3ProfileResult:
    artifacts = ()
    if metric is not None:
        counter = tmp_path / f"{metric}.csv"
        counter.write_text(f"Metric,Value,Unit\n{metric},1,count\n")
        artifacts = (
            Rocprofv3ProfileArtifact(
                path=counter,
                kind="counter_csv",
                size_bytes=counter.stat().st_size,
            ),
        )
    return Rocprofv3ProfileResult(
        status=status,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=artifacts,
        skipped_reason="missing"
        if status is Rocprofv3ProfileStatus.UNAVAILABLE
        else None,
        failed_reason="failed" if status is Rocprofv3ProfileStatus.FAILED else None,
    )


def test_missing_and_unavailable_profiles_have_no_precedence(tmp_path: Path) -> None:
    assert runtime_decision_precedence(None).available is False
    unavailable = runtime_decision_precedence(
        _result(tmp_path, status=Rocprofv3ProfileStatus.UNAVAILABLE)
    )
    assert unavailable.available is False


def test_data_without_classification_has_no_precedence(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text("Domain,Name,DurationNs\nKERNEL_DISPATCH,kernel,10\n")
    profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3",),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=trace,
                kind="trace_csv",
                size_bytes=trace.stat().st_size,
            ),
        ),
    )

    assert runtime_decision_precedence(profile).available is False


def test_lds_runtime_classification_supersedes_only_static_lds(
    tmp_path: Path,
) -> None:
    precedence = runtime_decision_precedence(
        _result(
            tmp_path,
            status=Rocprofv3ProfileStatus.SUCCESS,
            metric="LDS_BANK_CONFLICT",
        )
    )

    assert precedence.available is True
    assert precedence.categories == (ProfileSummaryHintCategory.LDS_BOUND,)
    assert precedence.demoted_classes == frozenset(
        {DecisionBottleneckClass.LDS_PRESSURE_HIGH}
    )


def test_compute_classification_is_runtime_context_not_a_static_conflict(
    tmp_path: Path,
) -> None:
    precedence = runtime_decision_precedence(
        _result(
            tmp_path,
            status=Rocprofv3ProfileStatus.SUCCESS,
            metric="SQ_INSTS_VALU",
        )
    )

    assert precedence.available is True
    assert precedence.categories == (ProfileSummaryHintCategory.COMPUTE_BOUND,)
    assert precedence.demoted_classes == frozenset()
