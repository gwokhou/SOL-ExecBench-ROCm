from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[4]
CLAIMS = REPO_ROOT / "docs" / "user" / "CLAIMS.md"
TESTING = REPO_ROOT / "docs" / "user" / "TESTING.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_claims_document_docker_matrix_native_host_boundary() -> None:
    text = _text(CLAIMS)

    assert re.search(
        r"container ROCm user-space\s+on recorded host\s+driver/devices",
        text,
    )
    assert "They do not prove native host ROCm validation" in text
    assert "Docker Matrix Entries as native host ROCm validation" in text


def test_claims_document_target_requested_vs_observed_evidence() -> None:
    text = _text(CLAIMS)

    assert "Target/requested values" in text
    assert "Observed evidence" in text
    assert "Target identity is required" in text
    assert (
        "host, container, Python dependency, dependency policy, toolchain, and GPU"
        in text
    )


def test_claims_document_mixed_version_debug_authority_boundary() -> None:
    text = _text(CLAIMS)

    assert "Illegal mixed-version Targets are blocked by default" in text
    assert re.search(r"bounded probes or\s+smoke diagnostics only", text)
    assert re.search(
        r"score authority,\s+paper-parity authority, or\s+leaderboard authority",
        text,
    )
    assert re.search(
        r"cannot create `container_validated` or\s+`host_validated`",
        text,
    )


def test_claims_document_recorded_container_validation_scope() -> None:
    text = _text(CLAIMS)

    assert "2026-05-29 live checks" in text
    assert "sol-execbench:rocm-7.0.2-complete" in text
    assert "sol-execbench:rocm-7.2-complete" in text
    assert "--record-container-validation" in text
    assert "ROCm 7.0.2 remains unlocked performance evidence" in text
    assert "not native-host ROCm hardware validation" in text


def test_testing_docs_list_cpu_safe_matrix_guardrail_commands() -> None:
    text = _text(TESTING)

    assert "ROCm Matrix Guardrails" in text
    assert "status classification" in text
    assert "reason-code classification" in text
    assert "schema serialization" in text
    assert "mixed-version blocking" in text
    assert "unknown Target rejection" in text
    assert "test_rocm_matrix_docs.py" in text
    assert "bash -n scripts/run_docker.sh" in text


def test_testing_docs_document_marker_gated_live_validation() -> None:
    text = _text(TESTING)

    assert "Live ROCm validation is marker-gated" in text
    assert "requires_rocm" in text
    assert "requires_rdna4" in text
    assert "requires_cdna3" in text
    assert "tests/sol_execbench/core/platform/test_cdna3_hardware_marker.py" in text
    assert "not full MI300X hardware-validation evidence" in text
    assert "not a `gfx94*` validation target" in text
    assert "current host ROCm 7.2.x environment" in text
    assert "default validation does not require host reinstall" in text
    assert re.search(
        r"ROCm 7\.0\.x or\s+ROCm 7\.1\.x native-host validation requires a matching host",
        text,
    )


def test_testing_docs_include_compatibility_matrix_summary_table() -> None:
    text = _text(TESTING)

    assert "Compatibility Matrix Summary" in text
    assert "Target id | Local image tag | Requested ROCm user-space" in text
    assert "rocm-7.0.2-ubuntu-24.04-container" in text
    assert "rocm-7.1.1-ubuntu-24.04-container" in text
    assert "rocm-7.2.0-ubuntu-24.04-container" in text
    assert "container_validated" in text
    assert "CLOCKS_LOCKED=0" in text
    assert "CLOCKS_LOCKED=1" in text
    assert "linear_backward` passed 3/3 workloads" in text
    assert "--record-container-validation" in text
    assert "--allow-untested-target-smoke" in text
    assert "--allow-mixed-version-dependencies" in text
    assert "benchmark_allowed=false" in text
    assert "status=mixed_version" in text
    assert "rocm-7.0.2-linear-wrapper-official.jsonl" in text
    assert "rocm-7.0.2-linear-wrapper-official.compatibility.json" in text
    assert "rocm-7.2-linear-wrapper-official.jsonl" in text
    assert "rocm-7.2-linear-wrapper-official.compatibility.json" in text
    assert "target-specific PyTorch ROCm" in text
    assert "torch==2.10.0+rocm7.0" in text
    assert "torch==2.11.0+rocm7.2" in text
    assert "rocm-7.2-linear-wrapper-smoke.jsonl" in text
    assert "rocm-7.2-linear-wrapper-smoke.compatibility.json" in text
    assert "sol-execbench:rocm-7.0.2-complete" in text
    assert "sol-execbench:rocm-7.2-complete" in text
