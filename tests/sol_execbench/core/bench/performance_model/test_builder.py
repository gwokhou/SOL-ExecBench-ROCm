from __future__ import annotations

from sol_execbench.core.bench.performance_model import builder
from sol_execbench.core.bench.performance_model.models import (
    CompiledCharacterization,
    DispatchEvidence,
    EvidenceReference,
    ResourceFootprint,
)


def _source() -> EvidenceReference:
    return EvidenceReference(kind="test", sha256="a" * 64)


def _dispatch(
    kernel: str,
    scratch_bytes: int = 0,
    *,
    iteration: int = 0,
    value: float = 1.0,
) -> DispatchEvidence:
    return DispatchEvidence(
        workload_uuid="workload-1",
        candidate_sha256="c" * 64,
        dispatch_id="1",
        kernel_symbol=kernel,
        grid=(1, 1, 1),
        workgroup=(1, 1, 1),
        iteration_ordinal=iteration,
        counters={"SQ_WAVES": value},
        runtime_footprint=ResourceFootprint(scratch_bytes=scratch_bytes),
    )


def test_runtime_footprint_wins_and_preserves_static_conflict() -> None:
    compiled = CompiledCharacterization(
        candidate_sha256="c" * 64,
        code_object_sha256="d" * 64,
        gpu_architecture="gfx1200",
        kernel_symbol="kernel",
        footprint=ResourceFootprint(scratch_bytes=256),
        source=_source(),
    )

    [result] = builder._record_static_runtime_conflicts(
        [_dispatch("kernel")],
        [compiled],
    )

    assert result.runtime_footprint == ResourceFootprint(scratch_bytes=0)
    assert result.evidence_conflicts == [
        "static_runtime_scratch_bytes_conflict"
    ]


def test_dispatch_without_static_kernel_identity_is_invalid() -> None:
    [result] = builder._record_static_runtime_conflicts(
        [_dispatch("unexpected")],
        [],
    )

    assert result.valid is False
    assert result.reason_codes == ["dispatch_static_kernel_identity_mismatch"]


def test_replayed_dispatches_collapse_to_one_representative_invocation() -> (
    None
):
    dispatches = [
        *[
            _dispatch("a", iteration=index, value=index + 1)
            for index in range(2)
        ],
        *[
            _dispatch("b", iteration=index, value=index + 1)
            for index in range(4)
        ],
    ]

    result = builder._collapse_replayed_dispatches(dispatches)

    assert [
        (item.kernel_symbol, item.iteration_ordinal) for item in result
    ] == [
        ("a", 0),
        ("b", 0),
        ("b", 1),
    ]
    assert [item.counters["SQ_WAVES"] for item in result] == [1.5, 2.0, 3.0]
    assert all(item.start_timestamp_ns is None for item in result)
