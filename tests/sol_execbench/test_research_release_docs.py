from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_doc(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def test_claims_doc_defines_allowed_and_unsupported_claims():
    text = _read_doc("docs/CLAIMS.md")

    for allowed in (
        "ROCm-port evidence",
        "Runtime evidence",
        "Profiling evidence",
        "AMD-native-derived evidence",
        "Research-preview evidence",
    ):
        assert allowed in text

    for unsupported in (
        "NVIDIA B200",
        "official leaderboard parity",
        "Upstream NVlabs/SOLAR equivalence",
        "full 235-problem paper validation",
        "Curated-slice results as paper-level benchmark results",
    ):
        assert unsupported in text


def test_claims_doc_requires_evidence_before_claim_upgrades():
    text = _read_doc("docs/CLAIMS.md")

    for required in (
        "Complete 235-problem denominator accounting",
        "Full adapted suite on real `gfx94*` hardware",
        "side-by-side comparison against upstream SOLAR",
        "Stable submission format",
        "RGA/code-object/GPUOpen ISA artifacts",
    ):
        assert required in text


def test_curated_slice_defines_scope_commands_and_artifacts():
    text = _read_doc("docs/curated_rocm_slice.md")

    for required in (
        "tests/sol_execbench/samples/rmsnorm",
        "examples/triton/rmsnorm",
        "examples/hip_cpp/rmsnorm",
        "examples/hipblas/gemm",
        "uv run sol-execbench",
        "--profile rocprofv3",
        "unscored",
        "unavailable",
    ):
        assert required in text


def test_researcher_guide_and_cookbook_cover_core_roles_and_recipes():
    guide = _read_doc("docs/RESEARCHER-GUIDE.md")
    cookbook = _read_doc("docs/COOKBOOK.md")

    for required in (
        "GPU kernel author",
        "compiler/backend researcher",
        "agent kernel-optimization researcher",
        "benchmark/reproducibility researcher",
        "canonical trace",
    ):
        assert required in guide

    for required in (
        "single-kernel evaluation",
        "HIP/C++ solution",
        "curated slice",
        "AMD-native score evidence",
        "rocprofv3",
    ):
        assert required in cookbook


def test_v1_15_release_closure_preserves_research_preview_boundary():
    text = _read_doc("docs/v1_15_release_closure.md")

    for required in (
        "research-grade benchmark preview",
        "Full 235-problem paper validation",
        "official leaderboard parity",
        "Trace JSONL",
        "Environment JSON",
        "Profile sidecars",
        "AMD score reports",
        "pass",
        "fail",
        "skip",
        "unavailable",
        "unscored",
        "PAPER-02",
        "STATIC-01",
    ):
        assert required in text


def test_rocm_toolchain_routing_docs_define_matrix_and_claim_boundaries():
    text = _read_doc("docs/rocm_toolchain_routing.md")
    claims = _read_doc("docs/CLAIMS.md")

    for required in (
        "Evidence Levels",
        "Tool Lifecycle",
        "Status Vocabulary",
        "rocprofv3",
        "rocprofiler-systems",
        "ROCm Systems",
        "RGA",
        "llvm-objdump",
        "Static Kernel Evidence remains a v1.17 milestone",
    ):
        assert required in text

    for boundary in (
        "Toolchain routing evidence",
        "Toolchain routing as correctness, performance, static-kernel",
        "Static Kernel Evidence in v1.16",
    ):
        assert boundary in claims
