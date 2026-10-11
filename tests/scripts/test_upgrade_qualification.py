# SPDX-License-Identifier: Apache-2.0
"""Upgrade measurements must not bypass incomplete or stale qualification."""

from pathlib import Path

import pytest

from scripts.internal.upgrade.freeze_workloads import freeze
from scripts.internal.upgrade.qualification import (
    qualify,
    require_gate,
    selected_items,
)
from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def experiment(tmp_path: Path, frozen_tensor_writer) -> tuple:
    root = tmp_path / "inputs"
    manifest = freeze(REPO_ROOT, root, tensor_writer=frozen_tensor_writer)
    plan = tmp_path / "plan.json"
    atomic_write_json_value(
        plan,
        {
            "scope_id": "unit-test",
            "source_revision": "a" * 40,
            "images": {
                "old": "sha256:" + "a" * 64,
                "new": "sha256:" + "b" * 64,
            },
            "measurements_root": str(tmp_path / "measurements"),
        },
    )
    return root, plan, tmp_path / "qualification", manifest


def _execute(protocol, root, item, stack, output, *, timed):
    row = {
        "id": item["id"],
        "stack": stack,
        "correct": True,
        "timed": timed,
        "image": protocol["images"][stack],
        "input_sha256": {"x": "c" * 64},
    }
    atomic_write_json_value(output, row)
    return row


def test_complete_chain_and_tampered_artifact(experiment: tuple) -> None:
    root, plan, qualification, _ = experiment
    for stage in BatchGPUQualificationStage:
        qualify(root, plan, qualification, stage, _execute)
    gate = require_gate(
        root, plan, qualification, BatchGPUQualificationStage.FULL
    )
    assert len(gate.item_ids) == 48
    assert gate.performance_authority is False
    (qualification / gate.receipts[0].artifacts[0].path).write_text("{}")
    with pytest.raises(ValueError, match="drift"):
        require_gate(root, plan, qualification, BatchGPUQualificationStage.FULL)


def test_full_requires_parent_and_failure_never_issues_gate(
    experiment: tuple,
) -> None:
    root, plan, qualification, _ = experiment
    with pytest.raises(FileNotFoundError):
        qualify(
            root, plan, qualification, BatchGPUQualificationStage.FULL, _execute
        )
    qualify(
        root, plan, qualification, BatchGPUQualificationStage.STATIC, _execute
    )

    def fail(*args, **kwargs):
        raise RuntimeError("numerical failure")

    with pytest.raises(RuntimeError, match="numerical failure"):
        qualify(
            root, plan, qualification, BatchGPUQualificationStage.CANARY, fail
        )
    assert not (qualification / "canary/gate.json").exists()


def test_canary_covers_each_family_axis_extrema(experiment: tuple) -> None:
    _, _, _, manifest = experiment
    selected = selected_items(manifest, BatchGPUQualificationStage.CANARY)
    for family in {i["family"] for i in manifest["items"]}:
        full = [i for i in manifest["items"] if i["family"] == family]
        subset = [i for i in selected if i["family"] == family]
        for axis in full[0]["axes"]:
            values = [i["axes"][axis] for i in subset]
            assert min(values) == min(i["axes"][axis] for i in full)
            assert max(values) == max(i["axes"][axis] for i in full)
