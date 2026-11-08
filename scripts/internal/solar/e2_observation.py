"""Bounded actual-call evidence and frozen E1 oracle, without candidate replay."""

import io
import signal
import time
from dataclasses import dataclass

import numpy as np
import torch

from scripts.internal.solar.e2_contracts import ORACLE_TIMEOUT_SECONDS
from sol_execbench.core.bench.judge.binary_oracle import exact_dyadic_output
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeStrategy,
    JudgeTask,
)
from sol_execbench.core.bench.judge.diagnostics import (
    certify_witness,
    classify,
    original_numeric_check,
    stronger_check,
)
from sol_execbench.core.bench.judge.inputs import JudgeInputs
from sol_execbench.core.bench.judge.interval_oracle import layernorm_oracle
from sol_execbench.core.bench.judge.oracle_checks import certify_oracle
from sol_execbench.core.bench.judge.storage import input_scale, tensor_identity
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation:
    """Private CPU copies of the one actual reference and candidate invocation."""

    inputs: JudgeInputs
    reference: torch.Tensor
    actual: torch.Tensor
    coordinate: dict[str, object]


def _oracle_timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError("single oracle exceeded the frozen 120-second limit")


def diagnose(
    case: JudgeCase,
    strategy: JudgeStrategy,
    observation: Observation,
    definition: Definition,
    workload: Workload,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Apply the unchanged whole-output oracle with a hard per-call deadline."""
    previous = signal.signal(signal.SIGALRM, _oracle_timeout)
    signal.setitimer(signal.ITIMER_REAL, ORACLE_TIMEOUT_SECONDS)
    start = time.monotonic()
    cpu_start = time.process_time()
    try:
        result, arrays = _diagnose(
            case, strategy, observation, definition, workload
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result["oracle_wall_seconds"] = time.monotonic() - start
    result["oracle_cpu_seconds"] = time.process_time() - cpu_start
    return result, arrays


def _diagnose(
    case: JudgeCase,
    strategy: JudgeStrategy,
    observation: Observation,
    definition: Definition,
    workload: Workload,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    inputs, reference, actual = (
        observation.inputs,
        observation.reference,
        observation.actual,
    )
    if case.task is JudgeTask.LAYERNORM:
        expected, exact, precision = layernorm_oracle(inputs.values)
    else:
        expected, exact = exact_dyadic_output(case.task, inputs.values)
        precision = None
    checked = certify_oracle(case.task, inputs, expected, exact)
    numeric, margin = original_numeric_check(
        definition, workload, inputs, reference, actual
    )
    stronger = stronger_check(case, inputs, actual, expected)
    reference_numeric, reference_margin = original_numeric_check(
        definition,
        workload,
        inputs,
        reference,
        expected,
    )
    finite = bool(torch.isfinite(actual).all())
    result = {
        "diagnostic_status": classify(numeric, stronger).value,
        "original_numeric_passed": numeric,
        "minimum_tolerance_margin": margin,
        "stronger_guarantee_passed": stronger,
        "witness_certified": certify_witness(case, strategy, numeric, stronger),
        "oracle_reference_max_error": float(
            (reference.double() - expected.double()).abs().max()
        ),
        "oracle_reference_original_numeric_passed": reference_numeric,
        "oracle_reference_tolerance_margin": reference_margin,
        "candidate_oracle_max_error": float(
            (actual.double() - expected.double()).abs().max()
        )
        if finite
        else None,
        "independent_oracle_points": checked,
        "interval_precision": precision,
        "false_rejection_label": "unassigned_requires_reference_uncertainty_and_strict_margin",
    }
    arrays = {
        "oracle_rounded": expected.double().numpy(),
        "oracle_exact": np.array([str(item) for item in exact]),
    }
    scale = input_scale(inputs.values)
    if scale is not None:
        arrays["sum_absolute_contributions"] = scale.numpy()
        error = (actual.double() - expected.double()).abs()
        result["input_scale_normalized_max_error"] = (
            float(
                torch.where(
                    scale > 0,
                    error / scale.clamp_min(torch.finfo(torch.float64).tiny),
                    error,
                ).max()
            )
            if finite
            else None
        )
    return result, arrays


def encode_observation(
    observation: Observation,
    extra_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], bytes]:
    """Compress raw finite-precision tensors losslessly while still worker-private."""
    inputs = observation.inputs
    arrays = {
        f"input_{index}": value.double().numpy()
        for index, value in enumerate(inputs.values)
    }
    arrays.update(
        {
            "input_dtypes": np.array(
                [str(value.dtype) for value in inputs.values]
            ),
            "witness_columns": np.array(inputs.witness_columns, dtype=np.int64),
            "reference": observation.reference.double().numpy(),
            "actual": observation.actual.double().numpy(),
            "output_dtype": np.array(str(observation.reference.dtype)),
            **extra_arrays,
        }
    )
    arrays.update(
        {
            f"construction_{key}": np.array(value, dtype=np.int64)
            for key, value in inputs.construction.items()
        }
    )
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
    return {
        **observation.coordinate,
        "input_sha256": tensor_identity(inputs.values),
        "reference_sha256": tensor_identity((observation.reference,)),
        "candidate_output_sha256": tensor_identity((observation.actual,)),
        "witness_columns": list(inputs.witness_columns),
    }, payload
