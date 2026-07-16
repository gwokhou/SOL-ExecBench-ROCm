from pathlib import Path

from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
)
from sol_execbench.core.bench.static_kernel.isa_analysis import (
    collect_static_isa_analyses,
)
from sol_execbench.core.platform.amdgpu_code_object import ExtractedCodeObject
from sol_execbench.core.platform.isa_validation import (
    IsaDisassemblyAnalysis,
    IsaSpecProvenance,
)


def _artifact(path: Path) -> StaticKernelEvidenceArtifact:
    return StaticKernelEvidenceArtifact(
        artifact_id="kernel",
        artifact_type="rocm_binary",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        persisted_path=path.name,
        target_architecture="gfx1200",
        inspectable=True,
    )


def test_collect_static_isa_analysis_emits_structured_diagnostics(
    tmp_path, monkeypatch
) -> None:
    binary = tmp_path / "kernel.bin"
    binary.write_bytes(b"binary")
    code_object = tmp_path / "kernel.hsaco"
    code_object.write_bytes(b"hsaco")
    extracted = ExtractedCodeObject(
        architecture="gfx1200",
        path=code_object,
        sha256="a" * 64,
        disassembly="v_wmma_f32_16x16x16_bf16",
        disassembly_sha256="b" * 64,
    )
    provenance = IsaSpecProvenance(
        architecture="gfx1200",
        family="rdna4",
        release="test",
        spec_sha256="c" * 64,
        decoder_version="test",
        specification_architecture="gfx1200",
    )
    monkeypatch.setattr(
        "sol_execbench.core.bench.static_kernel.isa_analysis.extract_code_object",
        lambda *_, **__: extracted,
    )
    monkeypatch.setattr(
        "sol_execbench.core.bench.static_kernel.isa_analysis.analyze_isa_disassembly",
        lambda *_, **__: IsaDisassemblyAnalysis(
            architecture="gfx1200",
            decoded_instruction_count=1,
            functional_group_counts={"VectorALU": 1},
            functional_subgroup_counts={"WMMA": 1},
            observed_matrix_units=("wmma",),
            matched_instruction_counts={},
            provenance=provenance,
        ),
    )

    analyses, tool_runs, generated = collect_static_isa_analyses(
        artifacts=[_artifact(binary)],
        evidence_root=tmp_path / "evidence",
        sidecar_base=tmp_path,
        timeout_seconds=30.0,
    )

    assert analyses[0].decoded_instruction_count == 1
    assert analyses[0].observed_matrix_units == ["wmma"]
    assert tool_runs[0].status == StaticKernelEvidenceStatus.COLLECTED
    assert {item.artifact_type for item in generated} == {
        "code_object",
        "isa_disassembly",
    }


def test_collect_static_isa_analysis_is_soft_when_tooling_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    binary = tmp_path / "kernel.bin"
    binary.write_bytes(b"binary")

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("clang-offload-bundler")

    monkeypatch.setattr(
        "sol_execbench.core.bench.static_kernel.isa_analysis.extract_code_object",
        unavailable,
    )

    analyses, tool_runs, generated = collect_static_isa_analyses(
        artifacts=[_artifact(binary)],
        evidence_root=tmp_path / "evidence",
        sidecar_base=tmp_path,
        timeout_seconds=30.0,
    )

    assert analyses[0].status == StaticKernelEvidenceStatus.UNAVAILABLE
    assert analyses[0].reason_code == "isa_artifact_tool_unavailable"
    assert tool_runs[0].status == StaticKernelEvidenceStatus.UNAVAILABLE
    assert generated == []


def test_collect_static_isa_analysis_rejects_unsafe_target_path(tmp_path) -> None:
    binary = tmp_path / "kernel.bin"
    binary.write_bytes(b"binary")
    artifact = _artifact(binary).model_copy(
        update={"target_architecture": "../gfx1200"}
    )

    analyses, tool_runs, generated = collect_static_isa_analyses(
        artifacts=[artifact],
        evidence_root=tmp_path / "evidence",
        sidecar_base=tmp_path,
        timeout_seconds=30.0,
    )

    assert analyses[0].status == StaticKernelEvidenceStatus.UNAVAILABLE
    assert tool_runs[0].status == StaticKernelEvidenceStatus.UNAVAILABLE
    assert generated == []
