"""Original numeric checks and independent, stronger diagnostics."""

import math

import torch

from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeFamily,
    JudgeStrategy,
)
from sol_execbench.core.bench.judge.inputs import JudgeInputs
from sol_execbench.core.bench.judge.report import DiagnosticStatus
from sol_execbench.core.bench.output_checks import compare_output_checks
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import NumericCheck, Workload

_NORM_ULPS = 4


def stronger_check(
    case: JudgeCase,
    inputs: JudgeInputs,
    output: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    """Apply only the preregistered family-specific diagnostic guarantee."""
    if not bool(torch.isfinite(output).all()):
        return False
    if case.family is JudgeFamily.G2:
        small = inputs.witness_columns[1]
        zeros = expected == 0
        return bool(
            (output[:, small] != 0).all() and (output[zeros] == 0).all()
        )
    if case.family is JudgeFamily.R2:
        return bool(torch.equal(output, expected))
    if case.family.value.startswith("L"):
        direction = torch.where(expected < 0, -math.inf, math.inf)
        ulp = (torch.nextafter(expected, direction) - expected).abs().double()
        return bool(
            (
                (output.double() - expected.double()).abs() <= _NORM_ULPS * ulp
            ).all()
        )
    return True


def original_numeric_check(
    definition: Definition,
    workload: Workload,
    inputs: JudgeInputs,
    reference: torch.Tensor,
    output: torch.Tensor,
) -> tuple[bool, float | None]:
    """Reuse the incumbent comparator; separately retain a FP64 margin."""
    if output.shape != reference.shape or output.dtype != reference.dtype:
        return False, None
    _, failed = compare_output_checks(
        definition,
        workload,
        inputs.values,
        [reference],
        [output],
        0,
    )
    check = workload.checks[0]
    if not isinstance(check, NumericCheck):
        raise ValueError("judge task requires the frozen numeric check")
    if not bool(
        torch.isfinite(output).all() and torch.isfinite(reference).all()
    ):
        return False, None
    margin = (
        check.max_atol
        + check.max_rtol * reference.double().abs()
        - (reference.double() - output.double()).abs()
    )
    return not failed, float(margin.min())


def classify(numeric_passed: bool, stronger_passed: bool) -> DiagnosticStatus:
    """Keep stronger guarantees distinct from the original numerical contract."""
    if not numeric_passed:
        return DiagnosticStatus.ORIGINAL_NUMERIC_FAILED
    if not stronger_passed:
        return DiagnosticStatus.STRONGER_GUARANTEE_FAILED
    return DiagnosticStatus.PASSED


def certify_witness(
    case: JudgeCase,
    strategy: JudgeStrategy,
    numeric: bool,
    stronger: bool,
) -> bool | None:
    """Check frozen directional hypotheses without changing their denominators."""
    if strategy is JudgeStrategy.SEMANTIC and case.family in {
        JudgeFamily.G2,
        JudgeFamily.R0,
        JudgeFamily.L3,
    }:
        return not numeric
    if strategy is JudgeStrategy.NUMERICAL:
        if case.family in {JudgeFamily.G2, JudgeFamily.R2}:
            return numeric and not stronger
        if case.family is JudgeFamily.L0:
            return not numeric and not stronger
    if strategy in {JudgeStrategy.REFERENCE, JudgeStrategy.ALTERNATIVE}:
        return numeric
    return None
