from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def _read_doc(path: str) -> str:
    return (REPO_ROOT / path).read_text()


V1_19_GUIDE = "docs/internal/v1_19_evidence_guide.md"
V1_19_ENTRY_DOCS = (
    "docs/user/CLAIMS.md",
    "docs/user/TESTING.md",
    "docs/user/RESEARCHER-GUIDE.md",
)
V1_19_NEGATIVE_CLAIMS = (
    "no full 235-problem paper validation",
    "no upstream SOLAR parity",
    "no score authority",
    "no leaderboard readiness",
    "no CDNA3-family validation, including MI300X, and no CDNA4 validation",
    "no native-host ROCm Matrix validation",
    "no new-hardware validation",
)


def test_claims_doc_defines_allowed_and_unsupported_claims():
    text = _read_doc("docs/user/CLAIMS.md")

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
    text = _read_doc("docs/user/CLAIMS.md")

    for required in (
        "Complete 235-problem denominator accounting",
        "Full adapted suite on real `gfx94*` hardware",
        "side-by-side comparison against upstream SOLAR",
        "Stable submission format",
        "Trace-adjacent static sidecars",
    ):
        assert required in text


def test_engineering_prerelease_support_matrix_boundaries():
    rocm = _read_doc("docs/user/rocm.md")
    claims = _read_doc("docs/user/CLAIMS.md")
    release_validation = _read_doc("docs/internal/release_candidate_validation.md")

    for required in (
        "Engineering Prerelease Support Matrix",
        "RDNA 4",
        "engineering-prerelease evidence",
        "Docker/container ROCm user-space",
        "not native-host validation",
        "MI300X and MI308X are sibling GPU products",
        "MI308X (`gfx942`)",
        "gfx942",
        "CDNA4 validation is unavailable",
        "suitable hardware is not currently accessible",
    ):
        assert required in rocm

    for required in (
        "container ROCm user-space",
        "not native-host validation",
        "MI300X and MI308X are sibling GPU products",
        "recorded on MI308X",
        "CDNA4 validation is unavailable",
    ):
        assert required in claims

    for required in (
        "support matrix",
        "Docker/container user-space evidence",
        "full MI300X validation under CDNA3",
        "CDNA4 validation",
    ):
        assert required in release_validation


def test_v1_25_release_notes_keep_claim_boundaries_visible():
    release_notes = _read_doc("docs/internal/v1_25_release_notes.md")
    claims = _read_doc("docs/user/CLAIMS.md")
    release_validation = _read_doc("docs/internal/release_candidate_validation.md")

    for required in (
        "engineering prerelease",
        "Trace JSONL",
        "canonical",
        "diagnostic-only",
        "provisional prerelease evidence",
        "deferred",
        "unavailable",
        "paper parity",
        "upstream SOLAR parity",
        "leaderboard readiness",
        "hard-sandbox authority",
        "Full MI300X validation under CDNA3",
        "CDNA4 validation is unavailable",
        "Docker/container ROCm user-space evidence is not native-host validation",
    ):
        assert required in release_notes

    for linked_doc in (claims, release_validation):
        assert "docs/internal/v1_25_release_notes.md" in linked_doc
        assert "canonical run artifact" in linked_doc
        assert "diagnostic-only sidecar" in linked_doc


def test_first_run_guide_covers_minimal_trace_and_troubleshooting():
    getting_started = _read_doc("docs/user/GETTING-STARTED.md")

    for required in (
        "First-Run Checklist",
        "uv sync --all-groups",
        "uv run sol-execbench --format json environment doctor",
        "out/first-run.trace.jsonl",
        "--trace-output out/first-run.trace.jsonl",
        "canonical Trace JSONL",
        "`status`",
        "`correctness`",
        "`latency_ms`",
        "`speedup_factor`",
        "`environment`",
        "no-trace diagnostics sidecar",
        "known limitations",
        "PyTorch ROCm compatibility namespace",
        "not NVIDIA CUDA runtime support",
    ):
        assert required in getting_started


def test_v1_25_release_materials_cover_checklist_and_public_navigation():
    checklist = _read_doc("docs/internal/v1_25_prerelease_checklist.md")
    release_notes = _read_doc("docs/internal/v1_25_release_notes.md")
    readme = _read_doc("README.md")

    for required in (
        "git status --short",
        "UV_CACHE_DIR=/tmp/uv-cache uv run pytest",
        "scripts/internal/release/release_candidate_validation.py",
        "Review claim boundaries",
        "git tag -a v1.25.0-rc1",
        "git push origin main --tags",
    ):
        assert required in checklist

    for required in (
        "Shipped Capability",
        "Validation Evidence",
        "Known Limitations",
        "Deferred Claims",
        "docs/internal/v1_25_prerelease_checklist.md",
        "docs/user/rocm_timing.md",
        "docs/user/GETTING-STARTED.md",
    ):
        assert required in release_notes

    for required in (
        "v1.25 Engineering Prerelease",
        "docs/internal/v1_25_release_notes.md",
        "docs/internal/v1_25_prerelease_checklist.md",
        "docs/user/rocm.md",
        "docs/user/CLAIMS.md",
        "docs/user/RESEARCHER-GUIDE.md",
        "docs/user/rocm_timing.md",
        "docs/user/GETTING-STARTED.md",
        "troubleshooting",
    ):
        assert required in readme


def test_v1_21_docs_keep_debt_reduction_separate_from_external_claims():
    claims = _read_doc("docs/user/CLAIMS.md")
    development = _read_doc("docs/user/DEVELOPMENT.md")
    concerns = _read_doc(".planning/codebase/CONCERNS.md")

    for required in (
        "v1.21 reduces codebase debt",
        "does not add hard sandboxing",
        "multi-tenant safety",
        "paper-scale SOLAR parity",
        "hosted leaderboard authority",
    ):
        assert required in claims

    for helper in (
        "core.dataset.run_state",
        "core.bench.eval_runtime",
        "core.scoring.amd_bound_classification",
        "core.scoring.solar_derivation_status",
        "core.bench.static_kernel_status",
    ):
        assert helper in development

    for status in ("bounded", "deferred"):
        assert status in concerns


def test_curated_slice_defines_scope_commands_and_artifacts():
    text = _read_doc("docs/internal/curated_rocm_slice.md")

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
    guide = _read_doc("docs/user/RESEARCHER-GUIDE.md")
    cookbook = _read_doc("docs/user/COOKBOOK.md")

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
    text = _read_doc("docs/internal/v1_15_release_closure.md")

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
    text = _read_doc("docs/user/rocm_toolchain_routing.md")
    claims = _read_doc("docs/user/CLAIMS.md")

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
    text = _read_doc("docs/user/static_kernel_evidence.md")
    claims = _read_doc("docs/user/CLAIMS.md")
    guide = _read_doc("docs/user/RESEARCHER-GUIDE.md")

    for required in (
        "--static-evidence auto",
        "--static-evidence none",
        "sol_execbench.static_kernel_evidence.v2",
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

    assert "docs/user/static_kernel_evidence.md" in guide


def test_static_kernel_evidence_docs_keep_deferred_scope_explicit():
    combined = "\n".join(
        [
            _read_doc("docs/user/static_kernel_evidence.md"),
            _read_doc("docs/user/CLAIMS.md"),
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
        assert "docs/internal/v1_19_evidence_guide.md" in text


def test_v1_19_guide_names_evidence_surfaces_and_scripts():
    text = _read_doc(V1_19_GUIDE)

    for required in (
        "execution closure",
        "paper denominator report",
        "Matrix schema export",
        "Matrix semantic diff",
        "AMD bound sanity",
        "scripts/run_dataset.py",
        "scripts/internal/reports/report_paper_denominator.py",
        "scripts/internal/reports/export_matrix_schema.py",
        "scripts/internal/reports/diff_matrix_reports.py",
        "scripts/internal/reports/report_amd_bound_sanity.py",
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
    assert "--output out/v1_19_demo/run-dataset" in text
    assert "--output-dir out/v1_19_demo/run-dataset" not in text
    assert "--model all" in text
    assert "--output-dir out/v1_19_demo/matrix-schema" in text
    assert "--json-out out/v1_19_demo/matrix_diff.json" in text
    assert "--markdown-out out/v1_19_demo/matrix_diff.md" in text
    assert "--compatibility-matrix out/v1_19_demo/matrix.json" in text
    assert "--amd-sol-artifact out/v1_19_demo/amd-sol/demo.amd-sol-v5.json" in text
    assert "--solar-artifact out/v1_19_demo/solar/demo.solar-derivation.json" in text
    assert "--before" not in text
    assert "--after" not in text
    assert "--json-output out/v1_19_demo/matrix_diff.json" not in text
    assert "--markdown-output out/v1_19_demo/matrix_diff.md" not in text
    assert "--paper-denominator" not in text
    assert "--matrix-report" not in text
