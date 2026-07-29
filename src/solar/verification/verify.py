# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed execution and numerical verification of SOLAR IR graphs."""

# The executor intentionally mirrors semantic operation classifications.
# Its intentionally self-contained replay routines also import optional torch
# dependencies lazily so non-PyTorch tooling can load the module.
# pylint: disable=duplicate-code,too-many-statements,import-outside-toplevel,consider-using-from-import,too-many-locals,use-maxsplit-arg,too-few-public-methods,too-many-arguments,unspecified-encoding,too-many-branches,too-many-return-statements,not-callable

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from solar.ir.contracts import IRLifecycle, layer_operation
from solar.schema_versions import IR_VERIFICATION_SCHEMA_VERSION
from solar.verification.contracts import TolerancePolicy, VerificationPolicy
from solar.verification.errors import IRExecutionError, VerificationError
from solar.verification.executor import (
    IRGraphExecutor,
)
from solar.verification.numerics import (
    alias_relation,
    assert_close,
    clone,
    pattern_inputs,
    torch_equation,
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_TOKEN = re.compile(r"[A-Za-z][0-9]*")


def _load_module(path: Path) -> Any:
    name = f"_solar_verify_{_sha256(path)[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationError(f"cannot import reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PreparedCase = tuple[
    dict[str, Any],
    int,
    str,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
]


@dataclass(frozen=True)
class _CallableInputFactory:
    factory: Callable[[int], Sequence[Any]]

    def __call__(
        self,
        parameters: Mapping[str, Any],
        device: str,
    ) -> Sequence[Any]:
        del device
        return self.factory(int(parameters["seed"]))


def _prepare_case(
    input_factory: Callable[..., Any],
    graph: Mapping[str, Any],
    case: Mapping[str, Any],
    device: str,
    preserved_input_indices: Sequence[int],
) -> _PreparedCase:
    import torch

    parameters = dict(case.get("parameters") or {})
    seed, pattern = int(case["seed"]), str(case["pattern"])
    generated = input_factory({**parameters, "seed": seed}, device)
    inputs = (
        tuple(generated)
        if isinstance(generated, (tuple, list))
        else (generated,)
    )
    if any(index >= len(inputs) for index in preserved_input_indices):
        raise VerificationError(
            "preserved_input_indices select unavailable reference arguments",
        )
    reference_inputs = clone(
        pattern_inputs(
            inputs,
            pattern,
            preserved_input_indices=preserved_input_indices,
        ),
    )
    source_indices = graph.get("source_input_indices")
    if source_indices is None:
        reference_tensor_inputs = tuple(
            value
            for value in reference_inputs
            if isinstance(value, torch.Tensor)
        )
    else:
        try:
            reference_tensor_inputs = tuple(
                reference_inputs[int(index)] for index in source_indices
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise VerificationError(
                "graph has invalid source_input_indices",
            ) from exc
        if not all(
            isinstance(value, torch.Tensor) for value in reference_tensor_inputs
        ):
            raise VerificationError(
                "graph source_input_indices must select tensor arguments",
            )
    return (
        parameters,
        seed,
        pattern,
        reference_inputs,
        reference_tensor_inputs,
        clone(reference_tensor_inputs),
    )


def _verify_case(
    reference: Callable[..., Any],
    executor: IRGraphExecutor,
    graph: Mapping[str, Any],
    prepared: _PreparedCase,
    policy: TolerancePolicy,
) -> dict[str, float]:
    import torch

    (
        _,
        _,
        pattern,
        reference_inputs,
        reference_tensor_inputs,
        executor_inputs,
    ) = prepared
    atol = policy.atol
    rtol = policy.rtol
    required_ratio = policy.required_matched_ratio
    error_cap = policy.max_error_cap
    allow_negative_inf = policy.allow_negative_inf
    with torch.enable_grad():
        expected = reference(*reference_inputs)
    with torch.inference_mode():
        actual = executor(*executor_inputs)
        try:
            stats = assert_close(
                actual,
                expected,
                atol,
                rtol,
                required_matched_ratio=required_ratio,
                max_error_cap=error_cap,
                allow_negative_inf=allow_negative_inf,
                allow_matching_nan=pattern in {"zeros", "boundary"},
            )
        except VerificationError as error:
            if (
                error_cap is not None
                or not str(error).startswith("numerical mismatch:")
                or not _einsum_roundoff_equivalent(
                    graph,
                    executor_inputs,
                    actual,
                    expected,
                )
            ):
                raise
            stats = assert_close(
                actual,
                expected,
                atol,
                rtol,
                required_matched_ratio=0.0,
                allow_negative_inf=allow_negative_inf,
                allow_matching_nan=pattern in {"zeros", "boundary"},
            )
            stats["roundoff_bound_verified"] = 1.0
        assert_close(executor_inputs, reference_tensor_inputs, atol, rtol)
        if alias_relation(actual, executor_inputs) != alias_relation(
            expected,
            reference_tensor_inputs,
        ):
            raise VerificationError(
                "output/input alias relationships differ from the reference",
            )
    return stats


def _run_cases(
    reference: Callable[..., Any],
    input_factory: Callable[..., Any],
    graph: Mapping[str, Any],
    lifecycle: IRLifecycle,
    cases: Sequence[Mapping[str, Any]],
    *,
    tolerance: TolerancePolicy,
    device: str,
    check_shapes: bool,
) -> list[dict[str, Any]]:
    import torch

    executor = IRGraphExecutor(
        graph,
        lifecycle,
        check_shapes=check_shapes,
    )
    results: list[dict[str, Any]] = []
    for case in cases:
        prepared = _prepare_case(
            input_factory,
            graph,
            case,
            device,
            tuple(getattr(tolerance, "preserved_input_indices", ())),
        )
        parameters, seed, pattern, *_ = prepared
        stats = _verify_case(reference, executor, graph, prepared, tolerance)
        results.append(
            {
                "seed": seed,
                "pattern": pattern,
                "parameters_sha256": _canonical_hash(parameters),
                **stats,
            },
        )
        del prepared
        if torch.cuda.is_available() and str(device).startswith(
            ("cuda", "rocm"),
        ):
            torch.cuda.empty_cache()
    return results


def _einsum_roundoff_equivalent(
    graph: Mapping[str, Any],
    operands: Sequence[Any],
    actual: Any,
    expected: Any,
) -> bool:
    """Prove a pure float32 contraction against the standard γₙ error bound."""
    import torch

    operations = [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() != "start"
    ]
    if (
        len(operations) != 1
        or not isinstance(actual, torch.Tensor)
        or not isinstance(expected, torch.Tensor)
        or actual.dtype != torch.float32
        or expected.dtype != torch.float32
        or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            for value in operands
        )
    ):
        return False
    semantic = layer_operation(operations[0])
    if semantic.get("kind") != "einsum":
        return False
    equation = str(semantic.get("equation", ""))
    reduction_size = _einsum_reduction_size(equation, operands)
    unit_roundoff = 2.0**-24
    if reduction_size is None or reduction_size * unit_roundoff >= 1.0:
        return False
    try:
        normalized_equation = torch_equation(equation)
        precise = [value.double() for value in operands]
        oracle = torch.einsum(normalized_equation, *precise)
        absolute_sum = torch.einsum(
            normalized_equation,
            *(value.abs() for value in precise),
        )
    except (RuntimeError, IRExecutionError):
        return False
    gamma = (
        reduction_size * unit_roundoff / (1.0 - reduction_size * unit_roundoff)
    )
    bound = gamma * absolute_sum + unit_roundoff * oracle.abs()
    slack = torch.finfo(torch.float64).eps * torch.maximum(
        torch.ones_like(bound),
        oracle.abs(),
    )
    limit = bound + slack
    return bool(
        ((actual.double() - oracle).abs() <= limit).all()
        and ((expected.double() - oracle).abs() <= limit).all(),
    )


def _einsum_reduction_size(
    equation: str,
    operands: Sequence[Any],
) -> int | None:
    if "->" not in equation:
        return None
    inputs, output = equation.split("->", maxsplit=1)
    terms = inputs.split(",")
    if len(terms) != len(operands):
        return None
    dimensions: dict[str, int] = {}
    input_tokens: set[str] = set()
    for term, operand in zip(terms, operands, strict=True):
        tokens = _TOKEN.findall(term)
        if len(tokens) != operand.ndim:
            return None
        for token, size in zip(tokens, operand.shape, strict=True):
            if token in dimensions and dimensions[token] != int(size):
                return None
            dimensions[token] = int(size)
            input_tokens.add(token)
    reduced = input_tokens - set(_TOKEN.findall(output))
    if not reduced:
        return None
    return math.prod(dimensions[token] for token in reduced)


def _verification_cases(
    parameters: Mapping[str, Any],
    policy: VerificationPolicy,
) -> list[dict[str, Any]]:
    if len({int(seed) for seed in policy.seeds}) < 3:
        raise VerificationError(
            "trusted verification requires at least three seeds",
        )
    if not {"random", "zeros", "boundary"}.issubset(policy.patterns):
        raise VerificationError(
            "trusted verification requires random, zeros, and boundary patterns",
        )
    return [
        {
            "parameters": dict(parameters),
            "seed": int(seed),
            "pattern": str(pattern),
        }
        for seed in policy.seeds
        for pattern in policy.patterns
    ]


def _tolerance_record(policy: TolerancePolicy) -> dict[str, Any]:
    return {
        "atol": float(policy.atol),
        "rtol": float(policy.rtol),
        "required_matched_ratio": float(policy.required_matched_ratio),
        "max_error_cap": policy.max_error_cap,
        "allow_negative_inf": bool(policy.allow_negative_inf),
    }


def _verification_policy_record(
    policy: VerificationPolicy,
) -> dict[str, Any]:
    return {
        **_tolerance_record(policy),
        "preserved_input_indices": list(policy.preserved_input_indices),
    }


def create_verification_artifact(
    *,
    reference_path: str | Path,
    reference_entry_point: str,
    input_factory_name: str,
    graph_path: str | Path,
    workload_name: str,
    workload_parameters: Mapping[str, Any],
    output_path: str | Path,
    policy: VerificationPolicy,
    lifecycle: IRLifecycle,
) -> dict[str, Any]:
    """Verify and write a deterministic, hash-bound ``verification.yaml``."""
    reference_path = Path(reference_path).resolve()
    graph_path = Path(graph_path).resolve()
    module = _load_module(reference_path)
    reference = getattr(module, reference_entry_point)
    input_factory = getattr(module, input_factory_name)
    graph = yaml.safe_load(graph_path.read_text()) or {}
    cases = _verification_cases(workload_parameters, policy)
    execution = _execution_identity(policy.device)
    results = _run_cases(
        reference,
        input_factory,
        graph,
        lifecycle,
        cases,
        tolerance=policy,
        device=policy.device,
        check_shapes=True,
    )
    artifact = _file_attestation(
        reference_path=reference_path,
        reference_entry_point=reference_entry_point,
        input_factory_name=input_factory_name,
        graph_path=graph_path,
        workload_name=workload_name,
        workload_parameters=workload_parameters,
        tolerance=_verification_policy_record(policy),
        execution=execution,
        cases=cases,
        results=results,
    )
    Path(output_path).write_text(yaml.safe_dump(artifact, sort_keys=False))
    return artifact


def _file_attestation(
    *,
    reference_path: Path,
    reference_entry_point: str,
    input_factory_name: str,
    graph_path: Path,
    workload_name: str,
    workload_parameters: Mapping[str, Any],
    tolerance: Mapping[str, Any],
    execution: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {
                "name": reference_path.name,
                "digest": {"sha256": _sha256(reference_path)},
            },
            {
                "name": graph_path.name,
                "digest": {"sha256": _sha256(graph_path)},
            },
        ],
        "predicateType": (
            "https://solar-rocm.dev/attestations/source-to-einsum/v2"
        ),
        "predicate": {
            "status": "passed",
            "verifier": IR_VERIFICATION_SCHEMA_VERSION,
            "reference": {
                "entry_point": reference_entry_point,
                "input_factory": input_factory_name,
            },
            "workload": {
                "name": workload_name,
                "parameters_sha256": _canonical_hash(workload_parameters),
            },
            "tolerance": dict(tolerance),
            "execution": dict(execution),
            "cases": list(cases),
            "results": list(results),
        },
    }


def verify_callable_conversion(
    *,
    reference: Callable[..., Any],
    input_factory: Callable[[int], Sequence[Any]],
    reference_name: str,
    reference_sha256: str,
    graph_path: str | Path,
    output_path: str | Path,
    policy: VerificationPolicy,
    lifecycle: IRLifecycle,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a callable reference and write a hash-bound attestation.

    This is the boundary-safe variant used by :mod:`solar.api`: it knows
    nothing about benchmark definitions or workload schemas.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", reference_sha256):
        raise VerificationError("reference_sha256 must be a lowercase SHA-256")
    graph_path = Path(graph_path).resolve()
    if graph is None:
        graph = yaml.safe_load(graph_path.read_text()) or {}
    cases = _verification_cases({}, policy)
    results = _run_cases(
        reference,
        _CallableInputFactory(input_factory),
        graph,
        lifecycle,
        cases,
        tolerance=policy,
        device=policy.device,
        check_shapes=True,
    )
    artifact = _callable_attestation(
        reference_name=reference_name,
        reference_sha256=reference_sha256,
        graph_path=graph_path,
        tolerance=_verification_policy_record(policy),
        execution=_execution_identity(policy.device),
        cases=cases,
        results=results,
    )
    Path(output_path).write_text(yaml.safe_dump(artifact, sort_keys=False))
    return artifact


def _callable_attestation(
    *,
    reference_name: str,
    reference_sha256: str,
    graph_path: Path,
    tolerance: Mapping[str, Any],
    execution: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {"name": reference_name, "digest": {"sha256": reference_sha256}},
            {
                "name": graph_path.name,
                "digest": {"sha256": _sha256(graph_path)},
            },
        ],
        "predicateType": "https://solar-rocm.dev/attestations/callable-to-einsum/v1",
        "predicate": {
            "status": "passed",
            "verifier": IR_VERIFICATION_SCHEMA_VERSION,
            "tolerance": dict(tolerance),
            "execution": dict(execution),
            "cases": list(cases),
            "results": list(results),
        },
    }


def replay_verification_artifact(
    artifact: Mapping[str, Any],
    *,
    reference_path: str | Path,
    graph_path: str | Path,
    workload_name: str,
    workload_parameters: Mapping[str, Any],
    required_tolerance: TolerancePolicy,
    lifecycle: IRLifecycle,
    device: str | None = None,
) -> None:
    """Validate every binding and numerically replay a verification artifact."""
    reference_path = Path(reference_path).resolve()
    graph_path = Path(graph_path).resolve()
    predicate = _validated_replay_predicate(
        artifact,
        reference_path,
        graph_path,
        workload_name,
        workload_parameters,
    )
    recorded_tolerance = _validated_recorded_tolerance(
        predicate.get("tolerance") or {},
        required_tolerance,
    )
    cases, results = _validated_replay_cases(
        predicate,
        workload_parameters,
    )
    replay_device = _validated_replay_device(
        predicate.get("execution") or {},
        device,
    )
    reference_data = predicate.get("reference") or {}
    module = _load_module(reference_path)
    graph = yaml.safe_load(graph_path.read_text()) or {}
    replay = _run_cases(
        getattr(module, str(reference_data["entry_point"])),
        getattr(module, str(reference_data["input_factory"])),
        graph,
        lifecycle,
        cases,
        tolerance=recorded_tolerance,
        device=replay_device,
        check_shapes=True,
    )
    identity = ("seed", "pattern", "parameters_sha256")
    for expected, actual in zip(results, replay, strict=True):
        if any(expected.get(key) != actual.get(key) for key in identity):
            raise VerificationError("verification replay identity mismatch")


def _validated_replay_predicate(
    artifact: Mapping[str, Any],
    reference_path: Path,
    graph_path: Path,
    workload_name: str,
    workload_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    if artifact.get("_type") != "https://in-toto.io/Statement/v1":
        raise VerificationError(
            "verification artifact must be an in-toto Statement v1",
        )
    if artifact.get("predicateType") != (
        "https://solar-rocm.dev/attestations/source-to-einsum/v2"
    ):
        raise VerificationError("unsupported verification predicate type")
    predicate = artifact.get("predicate") or {}
    if (
        predicate.get("status") != "passed"
        or predicate.get("verifier") != IR_VERIFICATION_SCHEMA_VERSION
    ):
        raise VerificationError(
            "verification artifact is not a trusted passing result",
        )
    digests = {
        str(subject.get("name")): (subject.get("digest") or {}).get("sha256")
        for subject in artifact.get("subject") or []
    }
    if digests.get(reference_path.name) != _sha256(reference_path):
        raise VerificationError("verification reference SHA-256 mismatch")
    if digests.get(graph_path.name) != _sha256(graph_path):
        raise VerificationError("verification graph SHA-256 mismatch")
    workload_data = predicate.get("workload") or {}
    if workload_data.get("name") != workload_name:
        raise VerificationError("verification workload name mismatch")
    if workload_data.get("parameters_sha256") != _canonical_hash(
        workload_parameters,
    ):
        raise VerificationError("verification workload parameters mismatch")
    return predicate


def _validated_recorded_tolerance(
    tolerance: Mapping[str, Any],
    required: TolerancePolicy,
) -> TolerancePolicy:
    recorded = VerificationPolicy(
        atol=float(tolerance.get("atol", math.inf)),
        rtol=float(tolerance.get("rtol", math.inf)),
        required_matched_ratio=float(
            tolerance.get("required_matched_ratio", -1.0),
        ),
        max_error_cap=(
            float(tolerance["max_error_cap"])
            if tolerance.get("max_error_cap") is not None
            else None
        ),
        allow_negative_inf=bool(
            tolerance.get("allow_negative_inf", False),
        ),
        preserved_input_indices=tuple(
            int(index)
            for index in tolerance.get("preserved_input_indices") or ()
        ),
    )
    cap_is_weaker = required.max_error_cap is not None and (
        recorded.max_error_cap is None
        or recorded.max_error_cap > required.max_error_cap
    )
    if (
        not math.isfinite(recorded.atol)
        or not math.isfinite(recorded.rtol)
        or not math.isfinite(recorded.required_matched_ratio)
        or recorded.atol > required.atol
        or recorded.rtol > required.rtol
        or recorded.required_matched_ratio < required.required_matched_ratio
        or cap_is_weaker
        or recorded.allow_negative_inf
        and not required.allow_negative_inf
        or isinstance(required, VerificationPolicy)
        and recorded.preserved_input_indices
        != tuple(required.preserved_input_indices)
    ):
        raise VerificationError(
            "verification tolerance is weaker than benchmark tolerance",
        )
    return recorded


def _validated_replay_cases(
    predicate: Mapping[str, Any],
    workload_parameters: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    cases = list(predicate.get("cases") or [])
    results = list(predicate.get("results") or [])
    if len(cases) != len(results) or len(cases) < 9:
        raise VerificationError(
            "verification artifact lacks the required cases",
        )
    if len({int(case["seed"]) for case in cases}) < 3:
        raise VerificationError(
            "verification artifact lacks three independent seeds",
        )
    if not {"random", "zeros", "boundary"}.issubset(
        str(case["pattern"]) for case in cases
    ):
        raise VerificationError("verification artifact lacks boundary patterns")
    if any(
        dict(case.get("parameters") or {}) != dict(workload_parameters)
        for case in cases
    ):
        raise VerificationError(
            "verification cases are not bound to workload parameters",
        )
    return cases, results


def _validated_replay_device(
    execution: Mapping[str, Any],
    requested_device: str | None,
) -> str:
    recorded_device = str(execution.get("device_type", ""))
    if recorded_device not in {"cpu", "cuda"}:
        raise VerificationError(
            "verification artifact has no supported replay device",
        )
    replay_device = requested_device or recorded_device
    expected_backend = str(execution.get("backend", ""))
    actual_execution = _execution_identity(replay_device)
    if expected_backend not in {"cpu", "cuda", "rocm"}:
        raise VerificationError(
            "verification artifact has no execution backend identity",
        )
    if actual_execution.get("backend") != expected_backend:
        raise VerificationError(
            "verification replay backend differs from recorded backend",
        )
    if expected_backend == "rocm":
        for field in ("hip_version", "gfx_target"):
            if execution.get(field) != actual_execution.get(field):
                raise VerificationError(
                    f"verification replay {field} differs from recorded ROCm device",
                )
    return replay_device


def _execution_identity(device: str) -> dict[str, Any]:
    """Identify the actual PyTorch backend selected by a device string."""
    import torch

    if not str(device).startswith("cuda"):
        return {"device_type": "cpu", "backend": "cpu", "device": str(device)}
    if not torch.cuda.is_available():
        raise VerificationError(
            f"requested CUDA/HIP device is unavailable: {device}",
        )
    selected = torch.device(device)
    index = (
        selected.index
        if selected.index is not None
        else torch.cuda.current_device()
    )
    if index < 0 or index >= torch.cuda.device_count():
        raise VerificationError(
            f"requested CUDA/HIP device index is invalid: {device}",
        )
    properties = torch.cuda.get_device_properties(index)
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        gfx_target = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        if not gfx_target.startswith("gfx"):
            raise VerificationError(
                "HIP runtime selected a device without an AMD gfx target",
            )
        return {
            "device_type": "cuda",
            "backend": "rocm",
            "device": f"cuda:{index}",
            "hip_version": str(hip_version),
            "device_name": str(properties.name),
            "gfx_target": gfx_target,
        }
    return {
        "device_type": "cuda",
        "backend": "cuda",
        "device": f"cuda:{index}",
        "device_name": str(properties.name),
    }
