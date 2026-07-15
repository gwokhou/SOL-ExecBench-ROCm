from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sol_execbench.core.data.trace import Trace
from sol_execbench.core.platform.docker_matrix import load_docker_target_manifest
from sol_execbench.core.evidence import runtime_evidence
from sol_execbench.core.evidence.runtime_evidence import (
    RuntimeFailureEvidence,
    build_aggregate_report,
    build_host_evidence,
    build_dependency_observation,
    build_runtime_matrix_entry,
    collect_gpu_evidence,
    load_matrix_entry,
    write_aggregate_report,
    write_matrix_entry,
)
from sol_execbench_type_helpers import make_trace


REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"


def _target():
    return load_docker_target_manifest(MANIFEST_PATH).targets_by_id[
        "rocm-7.2.0-ubuntu-24.04-container"
    ]


def _matching_observation():
    return build_dependency_observation(
        torch_distribution_version="2.11.0+rocm7.2",
        torch_version="2.11.0+rocm7.2",
        torch_local_version="rocm7.2",
        torch_rocm_target="rocm7.2",
        torch_hip_version="7.2.0",
        torch_cuda_version=None,
        torch_device_available=True,
        torchvision_distribution_version="0.26.0+rocm7.2",
        triton_rocm_distribution_version="3.6.0",
        triton_rocm_status="installed",
        container_rocm_user_space_version="7.2.0",
        hipcc_version="HIP version: 7.2.0",
        toolchain_rocm_version="7.2.0",
    )


def test_dependency_observation_preserves_auto_collected_container_versions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_evidence,
        "collect_pytorch_dependency_observation",
        lambda: build_dependency_observation(
            torch_distribution_version="2.11.0+rocm7.2",
            container_rocm_user_space_version="7.2.0",
            hipcc_version="HIP version: 7.2.0",
            toolchain_rocm_version="7.2.0",
        ),
    )

    observation = build_dependency_observation()

    assert observation.container_rocm_user_space_version == "7.2.0"
    assert observation.hipcc_version == "HIP version: 7.2.0"
    assert observation.toolchain_rocm_version == "7.2.0"


def test_runtime_entry_keeps_observed_evidence_scopes_separate() -> None:
    entry = build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
        host=build_host_evidence(
            rocm_version="7.2.0",
            driver_version="6.14.0",
            dev_kfd_present=True,
            dev_kfd_accessible=True,
            dev_dri_present=True,
            dev_dri_accessible=True,
        ),
        gpu=collect_gpu_evidence(
            device_count=1,
            device_name="AMD Radeon",
            gfx_architecture="gfx1200",
            visible_device_environment={"HIP_VISIBLE_DEVICES": "0"},
        ),
    )
    observed = entry.model_dump(mode="json")["observed"]

    assert set(observed) == {
        "host",
        "container",
        "python_dependency",
        "dependency_policy",
        "toolchain",
        "gpu",
    }
    assert observed["host"]["rocm_version"] == "7.2.0"
    assert observed["container"]["rocm_user_space_version"] == "7.2.0"
    assert observed["python_dependency"]["torch_version"] == "2.11.0+rocm7.2"
    assert observed["python_dependency"]["torch_hip_version"] == "7.2.0"
    assert observed["python_dependency"]["torch_cuda_version"] is None
    assert observed["python_dependency"]["torch_device_available"] is True
    assert observed["python_dependency"]["triton_rocm_status"] == "installed"
    assert observed["dependency_policy"]["expected_local_version"] == "rocm7.2"
    assert observed["toolchain"]["hipcc_version"] == "HIP version: 7.2.0"
    assert observed["gpu"]["device_count"] == 1
    assert observed["gpu"]["device_name"] == "AMD Radeon"
    assert observed["gpu"]["gfx_architecture"] == "gfx1200"
    assert observed["gpu"]["visible_device_environment"] == {"HIP_VISIBLE_DEVICES": "0"}


def test_runtime_entry_can_record_container_validation_after_success() -> None:
    entry = build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
        container_validated=True,
    )
    payload = entry.model_dump(mode="json")

    assert payload["status"] == "container_validated"
    assert payload["reason_code"] == "container_user_space_validated"
    assert payload["claim_boundary"]["container_user_space_validated"] is True
    assert payload["claim_boundary"]["native_host_validated"] is False
    assert payload["claim_boundary"]["score_authority"] is False


def test_writes_per_target_sidecar_and_aggregate_status_counts(tmp_path: Path) -> None:
    not_tested = build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
    )
    unavailable = build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
        runtime_unavailable_reason="/dev/kfd is missing.",
        failure_evidence=[
            RuntimeFailureEvidence(
                category="setup_runtime",
                status="blocked",
                message="host preflight blocked",
            )
        ],
    )
    entry_path = tmp_path / "target.compatibility.json"
    report_path = tmp_path / "matrix.json"

    write_matrix_entry(entry_path, unavailable)
    write_aggregate_report(report_path, [not_tested, load_matrix_entry(entry_path)])

    entry_payload = json.loads(entry_path.read_text())
    report_payload = json.loads(report_path.read_text())
    assert (
        entry_payload["schema_version"] == "sol_execbench.rocm_compatibility_matrix.v1"
    )
    assert entry_payload["status"] == "runtime_unavailable"
    assert entry_payload["artifacts"][0]["kind"] == "runtime_evidence_setup_runtime"
    assert (
        report_payload["schema_version"] == "sol_execbench.rocm_compatibility_matrix.v1"
    )
    assert report_payload["status_counts"] == {
        "not_tested": 1,
        "runtime_unavailable": 1,
    }


def test_aggregate_report_validates_counts() -> None:
    entry = build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
    )
    report = build_aggregate_report([entry], generated_at="2026-05-28T00:00:00+00:00")

    assert report.model_dump(mode="json")["status_counts"] == {"not_tested": 1}


def test_failure_categories_do_not_mutate_canonical_trace_payload() -> None:
    trace = make_trace(
        definition="demo", workload={"uuid": "w0", "axes": {}, "inputs": {}}
    )
    before = trace.model_dump(mode="json")

    build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
        failure_evidence=[
            RuntimeFailureEvidence(category="dependency", status="blocked"),
            RuntimeFailureEvidence(category="benchmark_correctness", status="failed"),
            RuntimeFailureEvidence(category="benchmark_performance", status="recorded"),
        ],
    )

    assert trace.model_dump(mode="json") == before
    assert "compatibility" not in Trace.model_fields


def test_runtime_evidence_cli_collects_and_aggregates(tmp_path: Path) -> None:
    entry_path = tmp_path / "entry.json"
    report_path = tmp_path / "report.json"
    collect = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.evidence.runtime_evidence",
            "collect-target",
            "--manifest",
            str(MANIFEST_PATH),
            "--output",
            str(entry_path),
            "--host-rocm-version",
            "7.2.0",
            "--host-driver-version",
            "6.14.0",
            "--dev-kfd-present",
            "true",
            "--dev-kfd-accessible",
            "true",
            "--dev-dri-present",
            "true",
            "--dev-dri-accessible",
            "true",
            "--torch-distribution-version",
            "2.11.0+rocm7.2",
            "--torch-version",
            "2.11.0+rocm7.2",
            "--torch-local-version",
            "rocm7.2",
            "--torch-rocm-target",
            "rocm7.2",
            "--torch-hip-version",
            "7.2.0",
            "--torch-cuda-version",
            "none",
            "--torch-device-available",
            "true",
            "--torchvision-distribution-version",
            "0.26.0+rocm7.2",
            "--triton-rocm-distribution-version",
            "3.6.0",
            "--triton-rocm-status",
            "installed",
            "--container-rocm-user-space-version",
            "7.2.0",
            "--hipcc-version",
            "HIP version: 7.2.0",
            "--toolchain-rocm-version",
            "7.2.0",
            "--container-validated",
            "--device-count",
            "1",
            "--device-name",
            "AMD Radeon",
            "--gfx-architecture",
            "gfx1200",
            "--visible-device-env",
            "HIP_VISIBLE_DEVICES=0",
            "--failure-category",
            "dependency",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(collect.stdout)
    assert entry_path.is_file()
    assert payload["status"] == "container_validated"
    assert payload["observed"]["gpu"]["gfx_architecture"] == "gfx1200"
    assert payload["artifacts"][0]["kind"] == "runtime_evidence_dependency"

    aggregate = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.evidence.runtime_evidence",
            "aggregate",
            "--output",
            str(report_path),
            str(entry_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(aggregate.stdout)
    assert report_path.is_file()
    assert report["status_counts"] == {"container_validated": 1}


def test_runtime_evidence_cli_rejects_invalid_boolean_without_traceback(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sol_execbench.core.evidence.runtime_evidence",
            "collect-target",
            "--manifest",
            str(MANIFEST_PATH),
            "--output",
            str(tmp_path / "entry.json"),
            "--dev-kfd-present",
            "maybe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "expected boolean value" in completed.stderr
    assert "Traceback" not in completed.stderr
