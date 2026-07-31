from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticIdentity,
    DiagnosticSidecarStatus,
    SizedDiagnosticArtifactCitation,
)
from sol_execbench.core.bench.performance_model import builder
from sol_execbench.core.bench.performance_model.models import (
    CompiledCharacterization,
    DispatchEvidence,
    EvidenceReference,
    ResourceFootprint,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryContent,
    ProfileSummaryReasonCode,
    ProfileSummarySidecar,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticISAAnalysis,
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceKernel,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
)
from sol_execbench.core.data.json_utils import atomic_write_jsonl_values
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput, Workload
from sol_execbench.core.integrity import sha256_file


def _source() -> EvidenceReference:
    return EvidenceReference(kind="test", sha256="a" * 64)


def _trace(
    *,
    hardware: str = "gfx1200",
    rocm: str | None = "7.2",
    clocks_locked: bool | None = True,
    timing_protocol: str | None = "device_event_v1",
) -> Trace:
    libraries = {"rocm": rocm} if rocm is not None else {}
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="workload-1",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
            checks=[{"type": "numeric", "output": "output"}],
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(
                hardware=hardware,
                libs=libraries,
                clocks_locked=clocks_locked,
                timing_protocol=timing_protocol,
            ),
            timestamp="2026-07-29T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(latency_ms=1.0),
        ),
    )


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
        start_timestamp_ns=iteration * 10,
        end_timestamp_ns=iteration * 10 + 1,
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


def test_runtime_demangled_symbol_matches_static_itanium_symbol() -> None:
    compiled = CompiledCharacterization(
        candidate_sha256="c" * 64,
        code_object_sha256="d" * 64,
        gpu_architecture="gfx1200",
        kernel_symbol="_Z10toy_kernelPf",
        footprint=ResourceFootprint(scratch_bytes=0),
        source=_source(),
    )

    [result] = builder._record_static_runtime_conflicts(
        [_dispatch("toy_kernel(float*)")],
        [compiled],
    )

    assert result.valid is True
    assert result.reason_codes == []


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
    assert [item.start_timestamp_ns for item in result] == [0, 0, 10]


def test_replayed_dispatches_preserve_concurrent_lanes() -> None:
    dispatches = [
        _dispatch("stage", iteration=index).model_copy(
            update={"queue_id": lane},
        )
        for index in range(2)
        for lane in ("left", "right")
    ]

    result = builder._collapse_replayed_dispatches(dispatches)

    assert [(item.queue_id, item.start_timestamp_ns) for item in result] == [
        ("left", 0),
        ("right", 0),
    ]


def test_counter_artifacts_require_cited_relative_path_and_hash(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "trace.jsonl.profile-summary.json"
    artifact = (
        tmp_path
        / "trace.jsonl.rocprofv3"
        / "pass_1"
        / "1_counter_collection.csv"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("counter evidence", encoding="utf-8")
    profile = ProfileSummarySidecar(
        status=DiagnosticSidecarStatus.PARTIAL,
        reason_code=ProfileSummaryReasonCode.PROFILE_PARTIAL,
        identity=DiagnosticIdentity(
            generated_at="2026-07-29T00:00:00Z",
            sol_version="4.0.0",
            trace_path="trace.jsonl",
            run_id="run-1",
        ),
        summary=ProfileSummaryContent(artifact_count=1),
        artifact_citations=[
            SizedDiagnosticArtifactCitation(
                kind="profiler_artifact",
                label="counter_csv",
                path="pass_1/1_counter_collection.csv",
                sha256=sha256_file(artifact),
                size_bytes=artifact.stat().st_size,
            ),
        ],
    )

    assert builder._counter_artifact_paths(profile, profile_path) == [
        artifact.resolve(),
    ]

    artifact.write_text("counter evidencf", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        builder._counter_artifact_paths(profile, profile_path)


def test_multiple_kernels_do_not_reuse_aggregate_isa_analysis(
    tmp_path: Path,
) -> None:
    static_path = tmp_path / "static.json"
    static_path.write_text("static evidence", encoding="utf-8")
    code_hash = "d" * 64
    static = StaticKernelEvidenceSidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="code-object",
                artifact_type="rocm_binary",
                status=StaticKernelEvidenceStatus.COLLECTED,
                sha256=code_hash,
            ),
        ],
        kernels=[
            StaticKernelEvidenceKernel(name="kernel_a"),
            StaticKernelEvidenceKernel(name="kernel_b"),
        ],
        isa_analyses=[
            StaticISAAnalysis(
                artifact_id="code-object",
                architecture="gfx1200",
                status=StaticKernelEvidenceStatus.COLLECTED,
                functional_group_counts={"VALU": 20},
                code_object_sha256=code_hash,
            ),
        ],
    )

    compiled, _ = builder._compiled_characterizations(
        static,
        static_path,
        "gfx1200",
    )

    assert all(not item.functional_group_counts for item in compiled)
    assert all(
        item.reason_codes == ["static_isa_kernel_mapping_ambiguous"]
        for item in compiled
    )
    assert builder._compiled_reason_codes(compiled) == []


def test_frontier_requires_matching_runtime_identity() -> None:
    canonical = _trace()

    assert builder._frontier_identity_reasons(canonical, _trace()) == []
    assert builder._frontier_identity_reasons(
        canonical,
        _trace(
            hardware="gfx1100",
            rocm="7.1",
            clocks_locked=False,
            timing_protocol=None,
        ),
    ) == [
        "frontier_gpu_architecture_mismatch",
        "frontier_rocm_version_mismatch",
        "frontier_clock_state_mismatch",
        "frontier_timing_protocol_unverified",
    ]


def test_frontier_time_fails_closed_on_identity_mismatch(
    tmp_path: Path,
) -> None:
    canonical = _trace()
    frontier_path = tmp_path / "frontier.jsonl"
    atomic_write_jsonl_values(
        frontier_path,
        [_trace(hardware="gfx1100")],
    )

    assert builder._frontier_time(
        frontier_path,
        canonical.workload.uuid,
        canonical,
    ) == (None, ["frontier_gpu_architecture_mismatch"])
