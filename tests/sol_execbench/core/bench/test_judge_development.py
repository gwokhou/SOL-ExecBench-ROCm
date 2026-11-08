"""CPU-only tests of frozen development semantics, never h0 or GPU execution."""

from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from sol_execbench.core.bench.judge import development
from sol_execbench.core.bench.judge.binary_oracle import (
    exact_dyadic_output,
    round_fraction,
)
from sol_execbench.core.bench.judge.candidate_catalog import candidate_callable
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeFamily,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
    development_cases,
)
from sol_execbench.core.bench.judge.executor import execute_development
from sol_execbench.core.bench.judge.inputs import generate_seeded_inputs
from sol_execbench.core.bench.judge.interval_oracle import (
    IntervalArithmetic,
    layernorm_oracle,
)
from sol_execbench.core.bench.judge.staging import stage_development_case
from sol_execbench.core.bench.reference_service import ReferenceService

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU access is forbidden in E1 CPU checks")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize(
    "value,dtype,expected",
    [
        (Fraction(1) + Fraction(1, 2**30), torch.float64, 1 + 2**-30),
        (Fraction(1) + Fraction(1, 256), torch.bfloat16, 1),
        (
            Fraction(1) + Fraction(1, 256) + Fraction(1, 2**54),
            torch.bfloat16,
            1 + 2**-7,
        ),
        (Fraction(1, 2**150), torch.float32, 0),
        (Fraction(3, 2**150), torch.float32, 2**-148),
    ],
    ids=[
        "fp64-difference",
        "bf16-tie-even",
        "avoid-double-rounding",
        "subnormal-tie-zero",
        "subnormal-tie-even",
    ],
)
def test_direct_target_rounding(value, dtype, expected):
    assert round_fraction(value, dtype) == expected


def test_integer_overflow_falls_back_to_exact_arithmetic():
    x = torch.tensor([[2**60, 1, -(2**60)]], dtype=torch.float64)
    rounded, exact = exact_dyadic_output(JudgeTask.SUM, (x,))
    assert exact == (Fraction(1),)
    assert rounded.item() == 1


def test_oracle_bounds_real_work_before_copying():
    x = torch.empty((4096, 4096), device="meta")
    with pytest.raises(ValueError, match="budget"):
        exact_dyadic_output(JudgeTask.SUM, (x,))
    with pytest.raises(ValueError, match="nonempty"):
        exact_dyadic_output(JudgeTask.SUM, ())


def test_directed_interval_encloses_independent_sqrt():
    bounds = IntervalArithmetic(precision=80).inverse_sqrt(Fraction(7, 13))
    with localcontext() as context:
        context.prec = 120
        independent = (Decimal(13) / Decimal(7)).sqrt()
    assert bounds.lower <= independent <= bounds.upper


@pytest.mark.parametrize(
    "family", [JudgeFamily.L0, JudgeFamily.L1, JudgeFamily.L2], ids=str
)
def test_layernorm_oracle_matches_symmetric_analytic_solution(family):
    values = generate_seeded_inputs(
        JudgeTask.LAYERNORM, family, {"B": 2, "N": 8}, 1701
    ).values
    rounded, bounds, precision = layernorm_oracle(values)
    x, _, _, eps = values
    with localcontext() as context:
        context.prec = 120
        mean = sum((Decimal(float(v)) for v in x[0]), Decimal(0)) / 8
        variance = (
            sum(((Decimal(float(v)) - mean) ** 2 for v in x[0]), Decimal(0)) / 8
        )
        expected = [
            float(
                (Decimal(float(v)) - mean)
                / (variance + Decimal(float(eps))).sqrt()
            )
            for v in x[0]
        ]
    assert torch.equal(rounded[0], torch.tensor(expected))
    assert len(bounds) == x.numel()
    assert precision == 80


def test_holdout_rejected_before_any_output_or_allocation(
    tmp_path, monkeypatch
):
    case = JudgeCase(
        task=JudgeTask.GEMM,
        workload_index=0,
        family=JudgeFamily.G2,
        instance=JudgeInstance.HOLDOUT,
    )
    monkeypatch.setattr(
        development,
        "generate_development_inputs",
        lambda *a, **kw: pytest.fail("allocated holdout"),
    )
    with pytest.raises(ValueError, match="holdout"):
        execute_development(
            ROOT, tmp_path / "out", cases=(case,), run_nonce="a" * 64
        )
    with pytest.raises(ValueError, match="holdout"):
        stage_development_case(ROOT, tmp_path / "stage", case)
    assert not list(tmp_path.iterdir())


def test_g2_development_witness_retains_all_outputs(tmp_path):
    case = next(c for c in development_cases() if c.family is JudgeFamily.G2)
    report = development.run_development(
        ROOT, tmp_path / "out", cases=(case,), run_nonce="b" * 64
    )
    by_strategy = {row.strategy: row for row in report.results}
    assert by_strategy[JudgeStrategy.SEMANTIC].original_numeric_passed is False
    weakened = by_strategy[JudgeStrategy.NUMERICAL]
    assert weakened.original_numeric_passed is True
    assert weakened.stronger_guarantee_passed is False
    assert weakened.witness_certified is True
    assert weakened.minimum_tolerance_margin is not None
    assert weakened.minimum_tolerance_margin > 0
    assert not report.performance_authority
    assert not report.holdout_generated
    with np.load(
        tmp_path / "out/case-000/arrays.npz", allow_pickle=False
    ) as arrays:
        assert all(
            f"candidate_{strategy}" in arrays for strategy in JudgeStrategy
        )
        assert "oracle_exact" in arrays
        assert "witness_columns" in arrays


@pytest.mark.parametrize("task", list(JudgeTask), ids=str)
def test_staged_case_uses_existing_trusted_reference_service(tmp_path, task):
    case = next(c for c in development_cases() if c.task is task)
    stage_development_case(ROOT, tmp_path / "problem", case)
    service = ReferenceService(
        staging_dir=tmp_path / "problem",
        device="cpu",
        input_nonce="c" * 64,
        definition_path=tmp_path / "problem/definition.json",
    )
    workload = service.workloads[0]
    inputs = service.prepare_inputs(workload, 0, 0)
    result = candidate_callable(task, JudgeStrategy.REFERENCE)(*inputs)
    assert result.device.type == "cpu"
    assert (
        tuple(result.shape)
        == service.definition.get_output_shapes(workload.axes)["output"]
    )


def test_cpu_subprocess_executor(tmp_path):
    case = next(c for c in development_cases() if c.family is JudgeFamily.R0)
    report = execute_development(
        ROOT, tmp_path / "isolated", cases=(case,), run_nonce="d" * 64
    )
    assert report.case_count == 1
    assert len(report.results) == 4
    assert all(
        row.evaluator_status == "not_run_cpu_diagnostic"
        for row in report.results
    )


def test_worker_failure_preserves_selected_denominator(tmp_path, monkeypatch):
    from sol_execbench.core.bench.judge import executor

    def timeout(*args, **kwargs):
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr(executor, "_run_worker", timeout)
    report = execute_development(
        ROOT,
        tmp_path / "incomplete",
        cases=development_cases()[:2],
        run_nonce="e" * 64,
    )
    assert not report.execution_complete
    assert report.failure_reason == "synthetic timeout"
    assert len(report.results) == 8
    assert all(row.input_sha256 is None for row in report.results)


def test_g2_has_explicit_structural_zero_witness():
    values = generate_seeded_inputs(
        JudgeTask.GEMM, JudgeFamily.G2, {"M": 2, "K": 8, "N": 4}, 1701
    )
    rounded, _ = exact_dyadic_output(JudgeTask.GEMM, values.values)
    assert int(torch.count_nonzero(rounded)) == 4
    assert bool((rounded[:, values.witness_columns[0]] == 1).all())
    assert bool((rounded[:, values.witness_columns[1]] == 2**-20).all())
