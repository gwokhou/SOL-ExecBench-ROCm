from __future__ import annotations

from pathlib import Path

import torch

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.access_evidence import (
    WorkloadAccessEvidence,
    build_performance_access_evidence,
    summarize_integer_inputs,
)
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.integrity import sha256_file


def test_access_summary_reuses_dtype_and_omits_raw_indices() -> None:
    summaries = summarize_integer_inputs(
        {
            "indices": torch.tensor([1, 1, 2, 4, 4, 4], dtype=torch.int64),
            "payload": torch.ones(6),
        }
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.dtype is DType.INT64
    assert summary.unique_index_count == 3
    assert summary.maximum_multiplicity == 3
    assert summary.duplicate_fraction == 0.5
    assert not {
        "raw_indices",
        "values",
        "sampled_indices",
    } & set(summary.model_dump(mode="json"))


def test_access_sidecar_binds_trace_and_canonical_input(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    pattern = summarize_integer_inputs(
        {"target": torch.tensor([0, 2, 1], dtype=torch.int32)}
    )[0]

    sidecar = build_performance_access_evidence(
        trace_path=trace,
        workloads=[
            WorkloadAccessEvidence(
                workload_uuid="workload-1",
                canonical_input_sha256="a" * 64,
                patterns=[pattern],
            )
        ],
    )

    assert sidecar.status is DiagnosticSidecarStatus.AVAILABLE
    assert sidecar.run_id == sha256_file(trace)
    assert sidecar.trace_sha256 == sha256_file(trace)
    assert sidecar.workloads[0].canonical_input_sha256 == "a" * 64
