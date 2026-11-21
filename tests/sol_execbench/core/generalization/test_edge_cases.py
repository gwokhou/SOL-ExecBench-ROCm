# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError
from sol_execbench_type_helpers import make_trace, make_workload

from sol_execbench.cli.commands import generalization as cli_generalization
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.trace import Performance
from sol_execbench.core.dataset.aka_compatibility import (
    AKACorpusSelection,
    AKAProblemSelection,
)
from sol_execbench.core.dataset.aka_contract import (
    AKAFusionDepth,
    AKAOperation,
    AKAPassKind,
    AKASourceFamily,
    AKASuite,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusEntry,
    AKACorpusManifest,
    _selection_coverage,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusOperationFamily,
    CorpusProfile,
    WorkloadRegime,
    WorkloadRole,
)
from sol_execbench.core.generalization.models import (
    CellResultStatus,
    CellWorkloadResult,
)
from sol_execbench.core.generalization.workflow import (
    PlannedStudy,
    _trace_outcome,
)


def test_trace_outcome_maps_zero_speedup_to_none() -> None:
    trace = make_trace(
        definition="toy",
        solution="sol",
        workload=make_workload(
            uuid="w0",
            axes={"N": 4},
            inputs={"x": {"type": "random"}},
        ),
        evaluation={
            "status": "PASSED",
            "environment": {"hardware": "gfx1200", "libs": {}},
            "timestamp": "2026-08-17T00:00:00Z",
            "correctness": {},
            "performance": {
                "latency_ms": 1.25,
                "reference_latency_ms": 0.0,
                "speedup_factor": 0.0,
            },
        },
    )

    status, compiled, correct, speedup = _trace_outcome(trace)

    assert status is CellResultStatus.PASSED
    assert compiled is True
    assert correct is True
    assert speedup is None


@pytest.mark.parametrize("bad_float", [float("nan"), float("inf")])
def test_performance_and_cell_result_reject_non_finite_floats(
    bad_float: float,
) -> None:
    with pytest.raises(ValidationError):
        Performance(latency_ms=bad_float)
    with pytest.raises(ValidationError):
        Performance(reference_latency_ms=bad_float)
    with pytest.raises(ValidationError):
        Performance(speedup_factor=bad_float)
    with pytest.raises(ValidationError):
        CellWorkloadResult(
            semantic_id="s0",
            workload_uuid="w0",
            slot_id="slot-0",
            role=WorkloadRole.DEVELOPMENT,
            regime=next(iter(WorkloadRegime)),
            operation_family=next(iter(CorpusOperationFamily)),
            profiles=(CorpusProfile.CORE,),
            status=CellResultStatus.PASSED,
            compiled=True,
            correct=True,
            speedup=bad_float,
        )


def test_write_planned_study_cleans_up_staging_on_validation_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "study"
    invalid_study = cast(
        PlannedStudy,
        SimpleNamespace(
            plan=SimpleNamespace(model_dump=lambda mode="json": {}),
            agent_views={},
        ),
    )

    with pytest.raises(ValueError, match="requires schema_version"):
        cli_generalization._write_planned_study(output, invalid_study)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_aka_selection_coverage_counts_materialized_workloads(
    tmp_path: Path,
) -> None:
    entry = AKACorpusEntry(
        slot="slot-1",
        task_path="tasks/test",
        problem_name="prob",
        operation=AKAOperation.MATMUL,
        input_dtypes=(DType.FLOAT32,),
        output_dtypes=(DType.FLOAT32,),
        capabilities=(),
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        suite=AKASuite.TORCH2HIP,
        workload_uuids=("w0", "w1", "w2", "w3"),
    )
    manifest = AKACorpusManifest(
        path=tmp_path / "manifest.yaml",
        source={},
        execution_targets={},
        formal_analysis={},
        entries=(entry,),
        materialized_problem_sha256={},
        formal_coverage_requirements={
            "combinations": [{"operation": "matmul", "min_workloads": 3}],
        },
        official_scoring={},
    )
    # Only 2 of 4 authored workloads materialized on target.
    selected_w0 = make_workload(
        uuid="w0",
        axes={},
        inputs={"x": {"type": "random"}},
    )
    selected_w1 = make_workload(
        uuid="w1",
        axes={},
        inputs={"x": {"type": "random"}},
    )
    selection = AKACorpusSelection(
        problems=(
            AKAProblemSelection(
                entry=entry,
                workloads=(selected_w0, selected_w1),
            ),
        ),
        decisions=(),
    )

    report = _selection_coverage(manifest, selection)

    assert report["workload_count"] == 2
    assert len(report["formal_coverage_gaps"]) == 1
    assert report["formal_coverage_gaps"][0]["matched_workloads"] == 2
