"""Focused tests for gfx1200 source-transition control-plane behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    inventory_regular_tree,
)
from sol_execbench.core.bench.performance_model.source_transition import (
    DevelopmentCaseRebind,
    SourcePathStageImpact,
    SourceTransitionDisposition,
    SourceTransitionStage,
)
from sol_execbench.core.bench.performance_model.vram_policy import (
    select_vram_working_set_policy,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value


def _digest(value: int) -> str:
    return f"{value:064x}"


def test_source_diff_requires_exact_name_status_coverage(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    monkeypatch.setattr(transition, "_exact_revision", lambda value: value)
    monkeypatch.setattr(
        transition,
        "_run_git",
        lambda _arguments, text=False: "M\tactual.py\n" if text else b"patch",
    )
    review = (
        SourcePathStageImpact(
            path="declared.py",
            change="modified",
            affected_stages=(SourceTransitionStage.QUALIFICATION_GPU,),
            rationale="declared wrong path",
        ),
    )

    with pytest.raises(ValueError, match="exactly cover"):
        transition._verify_source_diff("1" * 40, "2" * 40, review)


def test_policy_projection_ignores_only_provenance(
    load_script,
    tmp_path: Path,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    base = select_vram_working_set_policy(
        gpu_architecture="gfx1200",
        gpu_id="gpu-1",
        total_memory_bytes=8 * (1 << 30),
        source_revision="1" * 40,
        created_at="2026-08-10T00:00:00+00:00",
    )
    provenance_only = base.model_copy(
        update={
            "source_revision": "2" * 40,
            "created_at": "2026-08-11T00:00:00+00:00",
        }
    )
    capacity_change = select_vram_working_set_policy(
        gpu_architecture="gfx1200",
        gpu_id="gpu-1",
        total_memory_bytes=16 * (1 << 30),
        source_revision="2" * 40,
        created_at="2026-08-11T00:00:00+00:00",
    )
    paths = tuple(tmp_path / f"policy-{index}.json" for index in range(3))
    for path, policy in zip(
        paths, (base, provenance_only, capacity_change), strict=True
    ):
        atomic_write_json_value(path, policy.model_dump(mode="json"))

    base_sha, _ = transition._policy_projection(paths[0])
    provenance_sha, _ = transition._policy_projection(paths[1])
    capacity_sha, _ = transition._policy_projection(paths[2])

    assert base_sha == provenance_sha
    assert base_sha != capacity_sha


def test_design_projection_ignores_policy_link_but_detects_case_change(
    load_script,
    tmp_path: Path,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    base = transition.collector._design_payload(520, _digest(1))
    link_only = transition.collector._design_payload(520, _digest(2))
    changed = transition.collector._design_payload(520, _digest(2))
    first_axis = next(iter(changed["cases"][0]["axes"]))
    changed["cases"][0]["axes"][first_axis] += 1
    paths = tuple(tmp_path / f"design-{index}.json" for index in range(3))
    for path, design in zip(paths, (base, link_only, changed), strict=True):
        atomic_write_json_value(path, design)

    base_sha, _ = transition._design_projection(paths[0])
    link_sha, _ = transition._design_projection(paths[1])
    changed_sha, _ = transition._design_projection(paths[2])

    assert base_sha == link_sha
    assert base_sha != changed_sha


def test_problem_tree_projection_detects_any_file_change(
    load_script,
    tmp_path: Path,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    root = tmp_path / "problems"
    root.mkdir()
    definition = root / "definition.json"
    definition.write_text("one\n", encoding="utf-8")
    before, _ = transition._inventory_projection(root)

    definition.write_text("two\n", encoding="utf-8")
    after, _ = transition._inventory_projection(root)

    assert before != after


def test_raw_collection_projection_ignores_qualification_only_nodes(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    nodes = """class CaseSpec: pass
def _case_dir(): return 1
def _run_logged(): return 2
def _expected_case_workload(): return 3
def _verify_resumable_evidence(): return 4
def _collect_case(): return 5
def _remove_trace_artifacts(): return 6"""
    source = {
        "base": f"QUALIFICATION_TIMEOUT = 300\n{nodes}\n",
        "target": f"QUALIFICATION_TIMEOUT = 900\n{nodes}\n",
    }
    monkeypatch.setattr(
        transition,
        "_run_git",
        lambda arguments, text=False: source[arguments[1].split(":", 1)[0]],
    )

    assert transition._raw_collection_projection(
        "base"
    ) == transition._raw_collection_projection("target")

    source["target"] = source["target"].replace(
        "def _collect_case(): return 5", "def _collect_case(): return 99"
    )
    assert transition._raw_collection_projection(
        "base"
    ) != transition._raw_collection_projection("target")


def test_prepare_rebind_refuses_existing_target_before_verification(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_case = source_root / "case"
    target_case = target_root / "case"
    source_case.mkdir(parents=True)
    target_case.mkdir(parents=True)
    monkeypatch.setattr(transition, "_case_spec", lambda _case: object())
    monkeypatch.setattr(
        transition.collector,
        "_case_dir",
        lambda root, _spec: source_case if root == source_root else target_case,
    )

    with pytest.raises(ValueError, match="refusing to overwrite"):
        transition._prepare_rebind_case(
            source_root,
            target_root,
            SimpleNamespace(case_id="point_fit-elementwise-00"),
        )


def test_rebind_requires_current_target_qualification_gates(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    qualification_root = tmp_path / "qualification"
    source_root.mkdir()
    target_root.mkdir()
    qualification_root.mkdir()
    monkeypatch.setattr(
        transition,
        "_verify_attestation",
        lambda _path: SimpleNamespace(
            base_source_revision="1" * 40,
            target_source_revision="2" * 40,
            stage_decisions=tuple(
                SimpleNamespace(
                    stage=stage,
                    disposition=SourceTransitionDisposition.UNCHANGED,
                )
                for stage in SourceTransitionStage
            ),
        ),
    )
    monkeypatch.setattr(
        transition,
        "load_json_file",
        lambda _model, _path: SimpleNamespace(cases=[]),
    )
    monkeypatch.setattr(
        transition.collector,
        "_require_qualification_root",
        lambda _root, provided: provided,
    )
    monkeypatch.setattr(
        transition.collector,
        "_require_collection_qualification",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("qualification gate identity drift")
        ),
    )
    arguments = SimpleNamespace(
        attestation=tmp_path / "transition.json",
        source_root=source_root,
        target_root=target_root,
        qualification_root=qualification_root,
        output=tmp_path / "receipt.json",
        case_id=["point_fit-elementwise-00"],
    )

    with pytest.raises(ValueError, match="qualification gate identity drift"):
        transition._rebind(arguments)


def test_rebind_refuses_transition_that_changes_raw_collection(
    load_script,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    decisions = tuple(
        SimpleNamespace(
            stage=stage,
            disposition=(
                SourceTransitionDisposition.CHANGED
                if stage is SourceTransitionStage.RAW_COLLECTION
                else SourceTransitionDisposition.UNCHANGED
            ),
        )
        for stage in SourceTransitionStage
    )

    with pytest.raises(ValueError, match="raw_collection"):
        transition._require_rebindable_transition(
            SimpleNamespace(stage_decisions=decisions)
        )


def test_staged_rebind_rolls_back_when_target_verification_fails(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transition = load_script(
        "scripts/internal/rdna4/manage_rdna4_source_transition.py"
    )
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "trace.jsonl.performance-evidence.json").write_text(
        "{}\n", encoding="utf-8"
    )
    inventory = inventory_regular_tree(staged)
    record = DevelopmentCaseRebind(
        case_id="point_fit-elementwise-00",
        workload_kind="elementwise",
        phase="point_fit",
        workload_uuid="diagnostic-elementwise-point_fit-1x1",
        evidence_manifest_sha256=_digest(70),
        inventory=inventory,
    )
    target = tmp_path / "target" / record.case_id
    case = SimpleNamespace(case_id=record.case_id)
    monkeypatch.setattr(transition, "_case_spec", lambda _case: object())
    monkeypatch.setattr(
        transition.collector,
        "_verify_resumable_evidence",
        lambda *_args: (_ for _ in ()).throw(ValueError("candidate mismatch")),
    )

    with pytest.raises(ValueError, match="candidate mismatch"):
        transition._commit_staged_cases_with_design(
            ((record, tmp_path / "source", target),),
            {record.case_id: staged},
            tmp_path / "target",
            (case,),
            tmp_path / "receipt.json",
            SimpleNamespace(model_dump=lambda **_kwargs: {}),
        )

    assert staged.is_dir()
    assert not target.exists()
    assert not (tmp_path / "receipt.json").exists()
