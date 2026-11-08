"""Bounded CPU development execution; no holdout, mapper, or GPU entry."""

import platform
import time
from pathlib import Path

import torch

from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.bench.judge.binary_oracle import exact_dyadic_output
from sol_execbench.core.bench.judge.candidate_catalog import (
    candidate_callable,
    candidate_solution,
)
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeStrategy,
    JudgeTask,
    development_cases,
    load_case_problem,
)
from sol_execbench.core.bench.judge.diagnostics import (
    certify_witness,
    classify,
    original_numeric_check,
    stronger_check,
)
from sol_execbench.core.bench.judge.inputs import (
    JudgeInputs,
    generate_development_inputs,
)
from sol_execbench.core.bench.judge.interval_oracle import (
    OracleUncertainError,
    layernorm_oracle,
)
from sol_execbench.core.bench.judge.oracle_checks import certify_oracle
from sol_execbench.core.bench.judge.report import (
    DiagnosticStatus,
    JudgeDevelopmentReport,
    JudgeResult,
)
from sol_execbench.core.bench.judge.storage import (
    input_scale,
    save_case_arrays,
    source_identity,
    tensor_identity,
)
from sol_execbench.core.bench.reward_hack.static_review import (
    review_solution_sources,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_bytes, sha256_file


def run_development(
    repository: Path,
    output: Path,
    *,
    cases: tuple[JudgeCase, ...] | None = None,
    run_nonce: str,
) -> JudgeDevelopmentReport:
    """Execute owned development controls and write a CPU-only evidence envelope."""
    selected = development_cases() if cases is None else cases
    if not selected or len(selected) > len(development_cases()):
        raise ValueError("development case count exceeds the frozen budget")
    for case in selected:
        case.require_development()
    if len({case.logical_id for case in selected}) != len(selected):
        raise ValueError("duplicate development case")
    output.mkdir(parents=True, exist_ok=False)
    results: list[JudgeResult] = []
    artifacts: dict[str, str] = {}
    costs: dict[str, dict[str, float]] = {}
    points: dict[str, int] = {}
    for index, case in enumerate(selected):
        start = time.monotonic()
        definition, workload = load_case_problem(repository, case)
        inputs = generate_development_inputs(
            case, definition, workload, run_nonce=run_nonce
        )
        case_dir = output / f"case-{index:03d}"
        case_results, paths, checked = _run_case(
            case, definition, workload, inputs, case_dir
        )
        points[case.logical_id] = checked
        costs[case.logical_id] = {
            "total_cpu_wall_seconds": time.monotonic() - start
        }
        results.extend(case_results)
        artifacts.update(
            {str(path.relative_to(output)): sha256_file(path) for path in paths}
        )
    report = JudgeDevelopmentReport(
        complete_development_denominator=set(selected)
        == set(development_cases()),
        case_count=len(selected),
        runtime={
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_threads": str(torch.get_num_threads()),
        },
        costs=costs,
        independent_oracle_points=points,
        execution_complete=not any(
            row.diagnostic_status is DiagnosticStatus.ORACLE_UNCERTAIN
            for row in results
        ),
        source_sha256=source_identity(),
        artifacts=artifacts,
        results=results,
    )
    atomic_write_json_value(
        output / "development.json", report.model_dump(mode="json")
    )
    return report


def _run_case(
    case: JudgeCase,
    definition: Definition,
    workload: Workload,
    inputs: JudgeInputs,
    output: Path,
) -> tuple[list[JudgeResult], tuple[Path, ...], int]:
    _, reference_fn = load_reference_function(definition.reference)
    values: list[torch.Tensor | float] = list(inputs.values)
    if case.task is JudgeTask.LAYERNORM:
        values[-1] = float(values[-1])
    reference = reference_fn(*values)
    message = None
    try:
        if case.task is JudgeTask.LAYERNORM:
            expected, exact, _ = layernorm_oracle(inputs.values)
        else:
            expected, exact = exact_dyadic_output(case.task, inputs.values)
    except OracleUncertainError as exc:
        expected, exact, message = None, (), str(exc)
    checked = (
        0
        if expected is None
        else certify_oracle(case.task, inputs, expected, exact)
    )
    candidates: dict[str, torch.Tensor] = {}
    results = [
        _run_candidate(
            case,
            strategy,
            definition,
            workload,
            inputs,
            reference,
            expected,
            message,
            candidates,
        )
        for strategy in JudgeStrategy
    ]
    paths = save_case_arrays(
        output, inputs, reference, expected, exact, candidates
    )
    return results, paths, checked


def _run_candidate(
    case: JudgeCase,
    strategy: JudgeStrategy,
    definition: Definition,
    workload: Workload,
    inputs: JudgeInputs,
    reference: torch.Tensor,
    expected: torch.Tensor | None,
    oracle_message: str | None,
    candidates: dict[str, torch.Tensor],
) -> JudgeResult:
    solution = candidate_solution(case.task, strategy, definition.name)
    review = review_solution_sources(
        solution, output_dtypes={"output": reference.dtype}
    )
    result = JudgeResult(
        case_id=case.logical_id,
        strategy=strategy,
        input_sha256=tensor_identity(inputs.values),
        candidate_sha256=sha256_bytes(solution.sources[0].content.encode()),
        static_blocked=review.blocked,
        static_review=review.to_dict(),
        original_checks=[
            check.model_dump(mode="json") for check in workload.checks
        ],
        diagnostic_status=DiagnosticStatus.NOT_REACHED,
    )
    if review.blocked:
        return result
    if expected is None:
        result.diagnostic_status = DiagnosticStatus.ORACLE_UNCERTAIN
        result.message = oracle_message
        return result
    actual = candidate_callable(case.task, strategy)(*inputs.values)
    candidates[strategy] = actual
    numeric, margin = original_numeric_check(
        definition, workload, inputs, reference, actual
    )
    stronger = stronger_check(case, inputs, actual, expected)
    result.original_numeric_passed = numeric
    result.stronger_guarantee_passed = stronger
    result.diagnostic_threshold = (
        "4 ULP of correctly rounded FP32"
        if case.task is JudgeTask.LAYERNORM
        else "frozen structural/small-contribution witness; no extra threshold"
    )
    result.minimum_tolerance_margin = margin
    result.diagnostic_status = classify(numeric, stronger)
    result.witness_certified = certify_witness(
        case, strategy, numeric, stronger
    )
    result.oracle_reference_max_error = float(
        (reference.double() - expected.double()).abs().max()
    )
    scale = input_scale(inputs.values)
    if scale is not None and bool(torch.isfinite(actual).all()):
        error = (actual.double() - expected.double()).abs()
        normalized = torch.where(
            scale > 0,
            error / scale.clamp_min(torch.finfo(torch.float64).tiny),
            error,
        )
        result.input_scale_normalized_max_error = float(normalized.max())
    if bool(torch.isfinite(actual).all()):
        result.candidate_oracle_max_error = float(
            (actual.double() - expected.double()).abs().max()
        )
    return result
