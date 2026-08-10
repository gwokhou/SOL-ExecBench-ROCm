"""Mandatory pre-collection qualification gates for RDNA4 diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    NumericCheckResult,
    Performance,
    Trace,
)


def _prepared_corpus(corpus: Any, root: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "SOL_EXECBENCH_DIAGNOSTIC_STORE",
        str(root.parent / "store"),
    )
    corpus._preregister(root, 160, "1" * 40)
    corpus._prepare(root)


def _successful_trace(corpus: Any, definition: str, workload: Any) -> Trace:
    check = workload.checks[0]
    return Trace(
        definition=definition,
        workload=workload,
        solution="packaged_candidate",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="gfx1200"),
            timestamp="2026-08-09T00:00:00+00:00",
            correctness=Correctness(
                check_results=[
                    NumericCheckResult(
                        output=check.output,
                        round_index=0,
                        passed=True,
                        max_relative_error=0.0,
                        max_absolute_error=0.0,
                        matched_ratio=1.0,
                    )
                ]
            ),
            performance=Performance(),
        ),
    )


def _install_successful_evaluator(corpus: Any, monkeypatch) -> list[list[str]]:
    commands: list[list[str]] = []
    monkeypatch.setattr(corpus, "_container_output_path", lambda path: path)

    def run(command: list[str], log_path: Path) -> None:
        commands.append(command)
        workload_path = Path(command[command.index("--workload") + 1])
        trace_path = Path(command[command.index("--trace-output") + 1])
        definition_path = Path(command[command.index("--definition") + 1])
        definition = corpus.load_json_value(definition_path)["name"]
        workloads = corpus.load_jsonl_file(corpus.Workload, workload_path)
        corpus.atomic_write_jsonl_values(
            trace_path,
            [
                _successful_trace(corpus, definition, workload).model_dump(
                    mode="json"
                )
                for workload in workloads
            ],
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("qualification passed\n", encoding="utf-8")

    monkeypatch.setattr(corpus, "_run_logged", run)
    return commands


def test_static_gate_is_zero_gpu_and_covers_the_full_design(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    root = tmp_path / "corpus"
    qualification = tmp_path / "qualification"
    _prepared_corpus(corpus, root, monkeypatch)
    monkeypatch.setattr(
        corpus,
        "_run_logged",
        lambda *_args, **_kwargs: pytest.fail("static gate used GPU evaluator"),
    )

    corpus._run_qualification_stage(
        root,
        qualification,
        corpus.BatchGPUQualificationStage.STATIC,
        None,
    )

    gate = corpus._verify_qualification_gate(
        root,
        qualification,
        corpus.BatchGPUQualificationStage.STATIC,
    )
    assert gate.role == "all"
    assert len(gate.case_ids) == 660
    assert gate.performance_authority is False
    assert gate.receipts == ()


def test_canary_is_risk_first_and_selects_each_axis_extreme(
    load_script,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    canaries = corpus._canary_cases("held_out", 160)

    assert canaries[0].family is corpus.WorkloadKind.TRANSFORMER
    role_cases = corpus._cases("held_out", 160)
    for family in corpus.FAMILIES:
        family_cases = [case for case in role_cases if case.family is family]
        selected = [case for case in canaries if case.family is family]
        for axis in family_cases[0].axes:
            assert min(case.axes[axis] for case in selected) == min(
                case.axes[axis] for case in family_cases
            )
            assert max(case.axes[axis] for case in selected) == max(
                case.axes[axis] for case in family_cases
            )


def test_collect_requires_verified_three_gate_chain_before_first_case(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    root = tmp_path / "corpus"
    qualification = tmp_path / "qualification"
    _prepared_corpus(corpus, root, monkeypatch)
    commands = _install_successful_evaluator(corpus, monkeypatch)
    for stage, role in (
        (corpus.BatchGPUQualificationStage.STATIC, None),
        (corpus.BatchGPUQualificationStage.CANARY, "held_out"),
        (corpus.BatchGPUQualificationStage.FULL, "held_out"),
    ):
        corpus._run_qualification_stage(root, qualification, stage, role)
    assert commands
    assert all(
        command[0].endswith("scripts/run_docker.sh") for command in commands
    )
    assert all(
        "--allow-untested-target-smoke" in command for command in commands
    )
    assert all(
        "--unsafe-local-execution" not in command for command in commands
    )
    assert all("rocprofv3-counters" not in command for command in commands)
    assert all(
        command[command.index("--timeout") + 1]
        == str(corpus.QUALIFICATION_BATCH_TIMEOUT_SECONDS)
        for command in commands
    )
    assert all(command[-1] == "none" for command in commands)

    collected: list[str] = []
    monkeypatch.setattr(
        corpus,
        "_collect_case",
        lambda _root, case, *, force: collected.append(case.case_id),
    )
    arguments = SimpleNamespace(
        root=root,
        stage="collect",
        role="held_out",
        family=None,
        case_id=None,
        limit=1,
        force=False,
        source_corpus=[],
        qualification_root=qualification,
    )
    corpus._execute_cases(arguments)
    assert collected[0] == "held_out-elementwise-00"
    assert len(collected) == len(corpus.FAMILIES)

    solution = root / "problems" / "transformer_block" / "solution.json"
    solution.write_text(
        solution.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    collected.clear()
    with pytest.raises(ValueError, match="qualification gate identity drift"):
        corpus._execute_cases(arguments)
    assert collected == []


def test_collect_refuses_missing_gates_before_invoking_case(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    root = tmp_path / "corpus"
    _prepared_corpus(corpus, root, monkeypatch)
    monkeypatch.setattr(
        corpus,
        "_collect_case",
        lambda *_args, **_kwargs: pytest.fail(
            "collection started without gates"
        ),
    )
    arguments = SimpleNamespace(
        root=root,
        stage="collect",
        role="held_out",
        family=None,
        case_id=None,
        limit=1,
        force=False,
        source_corpus=[],
        qualification_root=tmp_path / "missing-qualification",
    )

    with pytest.raises(FileNotFoundError):
        corpus._execute_cases(arguments)


def test_partial_trace_artifacts_are_never_resumed(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    case = corpus._cases("held_out", 160)[0]
    trace = corpus._case_dir(tmp_path, case) / "trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("partial\n", encoding="utf-8")
    monkeypatch.setattr(
        corpus,
        "_run_logged",
        lambda *_args, **_kwargs: pytest.fail("partial evidence was resumed"),
    )

    with pytest.raises(ValueError, match="partial collection artifacts"):
        corpus._collect_case(tmp_path, case, force=False)


def test_incomplete_manifest_is_not_treated_as_already_collected(
    load_script,
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    case = corpus._cases("held_out", 160)[0]
    trace = corpus._case_dir(tmp_path, case) / "trace.jsonl"
    evidence = trace.with_name(f"{trace.name}.performance-evidence.json")
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        corpus,
        "load_and_verify_performance_evidence_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("performance evidence is incomplete")
        ),
    )
    monkeypatch.setattr(
        corpus,
        "_run_logged",
        lambda *_args, **_kwargs: pytest.fail(
            "incomplete evidence was resumed"
        ),
    )

    with pytest.raises(ValueError, match="performance evidence is incomplete"):
        corpus._collect_case(tmp_path, case, force=False)


def test_qualification_root_must_be_isolated(
    load_script, tmp_path: Path
) -> None:
    corpus = load_script(
        "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
    )
    root = tmp_path / "corpus"

    with pytest.raises(ValueError, match="outside the collection root"):
        corpus._require_qualification_root(root, root / "qualification")
