# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.dataset.aka_compatibility import (
    AkaWorkloadDecision,
    materialization_target,
    select_corpus_for_target,
)
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.platform.runtime import RocmDeviceInfo

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"


def _device(gfx_target: str) -> RocmDeviceInfo:
    return RocmDeviceInfo(
        device="cuda:0",
        index=0,
        name=f"test {gfx_target}",
        gfx_target=gfx_target,
        total_memory_bytes=16 * 1024**3,
        l2_cache_bytes=4 * 1024**2,
        torch_version="test",
        hip_version="test",
    )


def test_unknown_gfx_target_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported AKA execution target"):
        materialization_target(_device("gfx9999"))


def test_gfx942_static_filter_excludes_fp8_without_live_probe() -> None:
    manifest = AkaCorpusManifest.load(MANIFEST)
    entry = next(item for item in manifest.entries if item.dtype == "float8_e4m3fn")

    def unexpected_probe(*_args):
        raise AssertionError("static-incompatible workload reached the live probe")

    selection = select_corpus_for_target(
        authored_root=manifest.authored_root,
        entries=[entry],
        execution_target=manifest.execution_targets["gfx942"],
        target=materialization_target(_device("gfx942")),
        probe=unexpected_probe,
    )

    assert selection.problems == ()
    assert selection.decisions
    assert {item.reason_code for item in selection.decisions} == {
        "unsupported_target_dtype"
    }
    assert all(item.stage == "static" for item in selection.decisions)


def test_static_storage_filter_excludes_oversized_reference_without_probe() -> None:
    manifest = AkaCorpusManifest.load(MANIFEST)
    entry = next(
        item
        for item in manifest.entries
        if item.problem_name == "l2n55_matmul_maxpool_sum_scale"
    )

    def unexpected_probe(*_args):
        raise AssertionError("static-incompatible workload reached the live probe")

    selection = select_corpus_for_target(
        authored_root=manifest.authored_root,
        entries=[entry],
        execution_target=manifest.execution_targets["gfx1200"],
        target=materialization_target(_device("gfx1200")),
        probe=unexpected_probe,
    )

    assert selection.problems == ()
    assert len(selection.decisions) == 3
    assert {item.reason_code for item in selection.decisions} == {
        "reference_ipc_payload_limit"
    }
    assert all(item.stage == "static" for item in selection.decisions)
    assert all(
        item.metrics["reference_case_bytes"] > item.metrics["ipc_limit_bytes"]
        for item in selection.decisions
    )
    assert selection.decisions[0].metrics["input_storage_bytes"] == 4_299_292_672
    assert selection.decisions[0].metrics["reference_case_bytes"] == 4_299_292_800


def test_live_probe_decisions_partition_workloads() -> None:
    manifest = AkaCorpusManifest.load(MANIFEST)
    entry = manifest.entries[0]
    seen: list[str] = []

    def alternating_probe(problem_dir, _row_index, workload, _target, _timeout):
        seen.append(workload.uuid)
        included = len(seen) % 2 == 1
        return AkaWorkloadDecision(
            problem_path=f"{problem_dir.parent.name}/{problem_dir.name}",
            workload_uuid=workload.uuid,
            included=included,
            stage="live_probe",
            reason_code="probe_passed" if included else "probe_oom",
        )

    selection = select_corpus_for_target(
        authored_root=manifest.authored_root,
        entries=[entry],
        execution_target=manifest.execution_targets["gfx1200"],
        target=materialization_target(_device("gfx1200")),
        probe=alternating_probe,
    )

    assert len(selection.decisions) == len(seen)
    assert {item.workload_uuid for item in selection.decisions} == set(seen)
    assert [item.workload_uuid for item in selection.excluded] == seen[1::2]
    assert [item.uuid for item in selection.problems[0].workloads] == seen[::2]
