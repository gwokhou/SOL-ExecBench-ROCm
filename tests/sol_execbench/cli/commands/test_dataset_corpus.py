from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from click.testing import CliRunner

from sol_execbench.cli.commands import dataset as cli_dataset
from sol_execbench.cli.main import cli
from sol_execbench.core.platform.memory_quota import (
    GPUMemoryQuotaEvidence,
    capacity_probe_digest,
    derive_usable_budget,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml"
TARGET = ROOT / "problems/LLM_CORE/targets/gfx1200.yaml"
GIB = 1024**3


def _capacity() -> GPUMemoryQuotaEvidence:
    usable = derive_usable_budget(
        runtime_free_bytes=12 * GIB,
        environment_quota_bytes=10 * GIB,
        stable_allocatable_bytes=8 * GIB,
        harness_reserve_bytes=0,
    )
    payload: dict[str, Any] = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": "test gfx1200",
        "gfx_target": "gfx1200",
        "torch_version": "test",
        "hip_version": "test",
        "collected_at": datetime(2026, 8, 15, tzinfo=UTC),
        "runtime_free_bytes": 12 * GIB,
        "runtime_total_bytes": 16 * GIB,
        "environment_quota_bytes": 10 * GIB,
        "stable_allocatable_bytes": 8 * GIB,
        "harness_reserve_bytes": 0,
        "safety_percent": 85,
        "usable_budget_bytes": usable,
        "capacity_probe_digest": "0" * 64,
    }
    provisional = GPUMemoryQuotaEvidence.model_construct(**payload)
    payload["capacity_probe_digest"] = capacity_probe_digest(provisional)
    return GPUMemoryQuotaEvidence.model_validate(payload)


def test_corpus_validate_cli_reports_frozen_release() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "validate",
            "--manifest",
            str(MANIFEST),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["data"]["release_id"] == "LLM_CORE_V2"
    assert response["data"]["definitions"] == 36
    assert response["data"]["generation_rules"] == 36
    assert response["data"]["concrete_workloads"] == 0


def test_corpus_generate_cli_uses_isolated_measured_quota(
    tmp_path: Path,
    monkeypatch,
) -> None:
    capacity = _capacity()
    monkeypatch.setattr(
        cli_dataset,
        "collect_gpu_memory_quota_isolated",
        lambda *_args, **_kwargs: capacity,
    )
    output = tmp_path / "generation"
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "generate",
            "--manifest",
            str(MANIFEST),
            "--target-descriptor",
            str(TARGET),
            "--device",
            "cuda:0",
            "--environment-quota",
            str(10 * GIB),
            "--profile",
            "core",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["data"]["qualification_status"] == "hardware_qualified"
    assert (
        response["data"]["usable_budget_bytes"] == capacity.usable_budget_bytes
    )
    record = yaml.safe_load(
        (output / "target-view-manifest.yaml").read_text(encoding="utf-8"),
    )
    assert record["capacity_evidence"]["capacity_probe_digest"] == (
        capacity.capacity_probe_digest
    )
    assert record["coverage"]["workloads"] > 0


def test_corpus_generate_cli_requires_one_target_source(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "generate",
            "--manifest",
            str(MANIFEST),
            "--output",
            str(tmp_path / "selection"),
        ],
    )

    assert result.exit_code == 2
    response = json.loads(result.output)
    assert response["error"]["code"] == "invalid_static_target"


def test_corpus_generate_cli_reports_capacity_probe_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_probe(*_args, **_kwargs):
        raise RuntimeError("no visible ROCm capacity")

    monkeypatch.setattr(
        cli_dataset,
        "collect_gpu_memory_quota_isolated",
        fail_probe,
    )
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "dataset",
            "corpus",
            "generate",
            "--manifest",
            str(MANIFEST),
            "--target-descriptor",
            str(TARGET),
            "--output",
            str(tmp_path / "selection"),
        ],
    )

    assert result.exit_code == 3
    response = json.loads(result.output)
    assert response["error"]["code"] == "corpus_capacity_probe_unavailable"
