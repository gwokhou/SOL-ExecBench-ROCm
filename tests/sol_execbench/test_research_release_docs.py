from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_doc(path: str) -> str:
    return (REPO_ROOT / path).read_text()


V1_19_GUIDE = "docs/v1_19_evidence_guide.md"
V1_19_ENTRY_DOCS = (
    "docs/CLAIMS.md",
    "docs/TESTING.md",
    "docs/RESEARCHER-GUIDE.md",
)
V1_19_NEGATIVE_CLAIMS = (
    "no full 235-problem paper validation",
    "no upstream SOLAR parity",
    "no score authority",
    "no leaderboard readiness",
    "no CDNA 3/MI300X/CDNA4 validation",
    "no native-host ROCm Matrix validation",
    "no new-hardware validation",
)


def test_claims_doc_defines_allowed_and_unsupported_claims():
    text = _read_doc("docs/CLAIMS.md")

    for allowed in (
        "ROCm-port evidence",
        "Runtime evidence",
        "Profiling evidence",
        "Static Kernel Evidence",
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
        "Trace-adjacent static sidecars",
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
        "readelf",
        "--static-evidence auto",
    ):
        assert required in text

    for boundary in (
        "Toolchain routing evidence",
        "Toolchain routing as correctness, performance, static-kernel",
        "Static Kernel Evidence as correctness, performance, timing, score",
    ):
        assert boundary in claims


def test_static_kernel_evidence_docs_define_usage_and_boundaries():
    text = _read_doc("docs/static_kernel_evidence.md")
    claims = _read_doc("docs/CLAIMS.md")
    guide = _read_doc("docs/RESEARCHER-GUIDE.md")

    for required in (
        "--static-evidence auto",
        "--static-evidence none",
        "sol_execbench.static_kernel_evidence.v1",
        "collected",
        "partial",
        "unavailable",
        "unsupported",
        "failed",
        "skipped",
        "llvm-objdump",
        "readelf",
        "diagnostic-only",
    ):
        assert required in text

    for boundary in (
        "correctness authority",
        "performance authority",
        "timing authority",
        "score authority",
        "paper-parity authority",
        "leaderboard authority",
    ):
        assert boundary in text
        assert boundary in claims or boundary.replace("-", " ") in claims

    assert "docs/static_kernel_evidence.md" in guide


def test_static_kernel_evidence_docs_keep_deferred_scope_explicit():
    combined = "\n".join(
        [
            _read_doc("docs/static_kernel_evidence.md"),
            _read_doc("docs/CLAIMS.md"),
            _read_doc("docs/internal/v1_17_static_kernel_evidence_validation.md"),
        ]
    )

    for deferred in (
        "CDNA 3",
        "CDNA 4",
        "Triton ROCm cache capture",
        "RGA-rich resource parsing",
        "paper-scale static coverage",
    ):
        assert deferred in combined

    forbidden_positive_claims = (
        "CDNA 3 static evidence validation passed",
        "CDNA 4 static evidence validation passed",
        "Triton cache capture is supported",
        "RGA-rich resource parsing is complete",
        "paper-scale static coverage is complete",
    )
    for claim in forbidden_positive_claims:
        assert claim not in combined


def test_v1_17_static_evidence_validation_artifact_records_live_boundary():
    text = _read_doc("docs/internal/v1_17_static_kernel_evidence_validation.md")

    for required in (
        "RDNA 4 `gfx1200`",
        "static evidence collected",
        "benchmark correctness did not pass",
        "rmsnorm.trace.jsonl.static-evidence.json",
        "artifact count",
        "tool-run count",
        "llvm-objdump",
        "readelf",
        "does not claim benchmark correctness",
        "does not validate CDNA 3",
    ):
        assert required in text


def test_v1_19_guide_is_linked_from_public_entry_docs():
    for path in V1_19_ENTRY_DOCS:
        text = _read_doc(path)
        assert "docs/v1_19_evidence_guide.md" in text


def test_v1_19_guide_names_evidence_surfaces_and_scripts():
    text = _read_doc(V1_19_GUIDE)

    for required in (
        "execution closure",
        "paper denominator report",
        "Matrix schema export",
        "Matrix semantic diff",
        "AMD bound sanity",
        "scripts/run_dataset.py",
        "scripts/report_paper_denominator.py",
        "scripts/export_matrix_schema.py",
        "scripts/diff_matrix_reports.py",
        "scripts/report_amd_bound_sanity.py",
    ):
        assert required in text


def test_v1_19_docs_keep_required_negative_claim_boundaries_visible():
    guide = _read_doc(V1_19_GUIDE)
    linked_docs = "\n".join(_read_doc(path) for path in V1_19_ENTRY_DOCS)

    for boundary in V1_19_NEGATIVE_CLAIMS:
        assert boundary in guide
        assert boundary in linked_docs


def test_v1_19_guide_keeps_canonical_contract_semantics_unchanged():
    text = _read_doc(V1_19_GUIDE)

    for required in (
        "canonical Trace, Definition, Workload, Solution",
        "correctness, timing, score, and evaluator contract semantics are unchanged",
        "sidecars/reports only",
    ):
        assert required in text


def test_v1_19_guide_uses_unconditional_paper_validation_boundaries():
    text = _read_doc(V1_19_GUIDE)

    assert "no full 235-problem paper validation by this sidecar alone" in text
    assert "no full 235-problem paper validation by this report alone" in text
    assert "unless all 235 scoped paper problems" not in text
    assert "when any denominator records or required evidence are absent" not in text


def test_v1_19_guide_uses_relative_demo_paths_and_real_script_options():
    text = _read_doc(V1_19_GUIDE)

    for forbidden in ("/home/", "/tmp/", "/var/"):
        assert forbidden not in text
    assert "UV_CACHE_DIR=out/v1_19_demo/uv-cache" in text
    assert "--compatibility-matrix out/v1_19_demo/matrix.json" in text
    assert "--amd-sol-artifact out/v1_19_demo/amd-sol/demo.amd-sol-v2.json" in text
    assert "--solar-artifact out/v1_19_demo/solar/demo.solar-derivation.json" in text
    assert "--paper-denominator" not in text
    assert "--matrix-report" not in text
