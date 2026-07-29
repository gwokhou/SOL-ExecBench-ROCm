# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed semantic equivalence checks for the authored AKA corpus."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import torch

from sol_execbench.core.bench.input_generation import gen_inputs
from sol_execbench.core.bench.output_checks import compare_output_checks
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import NumericCheck, Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)
from sol_execbench.core.dataset.aka_contract import (
    AKAArtifactRole,
    AKACorpusRole,
    AKASuite,
)
from sol_execbench.core.dataset.aka_corpus import AKACorpusEntry
from sol_execbench.core.dataset.aka_task import function_arg_names

Oracle = Callable[[Mapping[str, object]], object]


class _MoEModel(Protocol):
    correction_bias: torch.Tensor


class CrosscheckStatus(StrEnum):
    """Source-equivalence outcome for one authored problem."""

    PASSED = "passed"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass(frozen=True)
class AKAEquivalenceReport:
    """Result for one authored problem."""

    problem_name: str
    passed: bool
    crosscheck: CrosscheckStatus
    workloads_checked: int
    outputs_checked: int
    detail: str


def load_problem(problem_dir: Path) -> tuple[Definition, tuple[Workload, ...]]:
    """Load one committed problem and its complete workload inventory."""
    definition = Definition.model_validate_json(
        (problem_dir / "definition.json").read_text(encoding="utf-8"),
    )
    workloads = tuple(
        Workload.model_validate_json(line)
        for line in (problem_dir / "workload.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    validate_problem_contract(definition, list(workloads))
    return definition, workloads


def materialize_inputs(
    definition: Definition,
    workload: Workload,
    *,
    seed: int,
    device: torch.device,
    custom_inputs_fn: Callable[..., object] | None = None,
) -> tuple[list[object], dict[str, object]]:
    """Create deterministic inputs matching a Definition/Workload contract."""
    ordered = gen_inputs(
        definition,
        workload,
        str(device),
        custom_inputs_fn=custom_inputs_fn,
        seed=seed,
    )
    named = dict(zip(definition.inputs, ordered, strict=True))
    return ordered, named


def execute_reference(reference_source: str) -> Callable[..., object]:
    """Compile a trusted authored reference and return its ``run`` callable."""
    namespace: dict[str, object] = {"torch": torch}
    exec(  # noqa: S102 -- committed first-party benchmark source
        compile(reference_source, "<aka-authored-reference>", "exec"),
        namespace,
    )
    run = namespace.get("run")
    if not callable(run):
        raise ValueError("authored reference does not define callable run")
    return run


def execute_reference_entrypoints(
    definition: Definition,
) -> tuple[Callable[..., object], Callable[..., object] | None]:
    """Compile trusted authored reference and custom-input entrypoints."""
    namespace: dict[str, object] = {"torch": torch}
    exec(  # noqa: S102 -- committed first-party benchmark source
        compile(definition.reference, "<aka-authored-reference>", "exec"),
        namespace,
    )
    run = namespace.get("run")
    if not callable(run):
        raise ValueError("authored reference does not define callable run")
    custom = (
        namespace.get(definition.custom_inputs_entrypoint)
        if definition.custom_inputs_entrypoint
        else None
    )
    if definition.custom_inputs_entrypoint and not callable(custom):
        raise ValueError("authored custom input entrypoint is not callable")
    return run, cast(Callable[..., object] | None, custom)


def normalize_outputs(
    value: object,
    definition: Definition,
    workload: Workload,
    *,
    source: str,
) -> tuple[torch.Tensor, ...]:
    """Normalize and validate every declared output without coercing dtype."""
    names = tuple(definition.outputs)
    outputs = _ordered_outputs(value, names, source=source)
    shapes = definition.get_output_shapes(workload.axes)
    dtypes = definition.torch_output_dtypes
    for name, output, dtype in zip(names, outputs, dtypes, strict=True):
        _validate_output_tensor(
            output,
            expected_shape=shapes[name],
            expected_dtype=dtype,
            allow_negative_inf=any(
                isinstance(check, NumericCheck)
                and check.output == name
                and check.allow_negative_inf
                for check in workload.checks
            ),
            label=f"{source}.{name}",
        )
    return outputs


def _ordered_outputs(
    value: object,
    names: Sequence[str],
    *,
    source: str,
) -> tuple[torch.Tensor, ...]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if set(mapping) != set(names):
            raise ValueError(
                f"{source} output keys {sorted(map(str, mapping))} "
                f"do not match {sorted(names)}",
            )
        values = tuple(mapping[name] for name in names)
    elif isinstance(value, (tuple, list)):
        if len(value) != len(names):
            raise ValueError(
                f"{source} returned {len(value)} outputs; expected {len(names)}",
            )
        values = tuple(value)
    else:
        if len(names) != 1:
            raise ValueError(
                f"{source} returned one output; expected {len(names)}",
            )
        values = (value,)
    if not all(torch.is_tensor(output) for output in values):
        kinds = [type(output).__name__ for output in values]
        raise TypeError(f"{source} returned non-tensor output(s): {kinds}")
    return cast(tuple[torch.Tensor, ...], values)


def _validate_output_tensor(
    output: torch.Tensor,
    *,
    expected_shape: tuple[int, ...] | None,
    expected_dtype: torch.dtype,
    allow_negative_inf: bool,
    label: str,
) -> None:
    if expected_shape is None or tuple(output.shape) != expected_shape:
        raise ValueError(
            f"{label} shape {tuple(output.shape)} != declared {expected_shape}",
        )
    if output.dtype != expected_dtype:
        raise ValueError(
            f"{label} dtype {output.dtype} != declared {expected_dtype}",
        )
    finite = torch.isfinite(output.float())
    if allow_negative_inf:
        finite |= output.float() == float("-inf")
    if not bool(finite.all()):
        raise ValueError(f"{label} contains disallowed non-finite values")


def load_aka_oracle(entry: AKACorpusEntry, aka_root: Path) -> Oracle | None:
    """Load a provenance-bound AKA semantic oracle for an entry."""
    if entry.role is AKACorpusRole.COMPATIBILITY_SENTINEL:
        return None
    source = _artifact_path(
        entry,
        aka_root,
        AKAArtifactRole.SEMANTIC_REFERENCE,
    )
    if entry.suite is AKASuite.TORCH2HIP:
        return _load_torch2hip_oracle(entry, source)
    if entry.problem_name in {
        "fused_add_rmsnorm_bf16",
        "per_token_i8_quant",
        "rope_thd_fwd_bf16",
        "dynamic_mxfp8_quant",
        "moe_topk_softmax",
    }:
        return _load_torch2flydsl_model_oracle(entry, source)
    if entry.problem_name == "rmsnorm_bwd":
        return _load_rmsnorm_backward_oracle(source)
    if entry.problem_name == "silu_and_mul_bf16":
        return _load_silu_and_mul_oracle(source)
    raise ValueError(f"no AKA oracle loader for {entry.problem_name}")


def _artifact_path(
    entry: AKACorpusEntry,
    aka_root: Path,
    role: AKAArtifactRole,
) -> Path:
    artifact = next(
        (item for item in entry.aka_artifacts if item.role == role),
        None,
    )
    if artifact is None:
        raise ValueError(
            f"{entry.problem_name} has no {role} provenance binding",
        )
    return aka_root / entry.task_path / artifact.path


def _load_torch2hip_oracle(entry: AKACorpusEntry, source: Path) -> Oracle:
    text = source.read_text(encoding="utf-8")
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(text, str(source), "exec"), namespace)  # noqa: S102
    function_name = (
        "kd_loss_fn" if entry.problem_name == "14007_kd_loss" else "module_fn"
    )
    module_fn = namespace.get(function_name)
    if not callable(module_fn):
        raise ValueError(f"AKA semantic reference has no module_fn: {source}")
    callable_module = cast(Callable[..., object], module_fn)
    adapter = _TORCH2HIP_ADAPTERS.get(entry.problem_name)
    if adapter is not None:
        return lambda named: adapter(callable_module, named)
    arg_names = function_arg_names(text, "module_fn")

    def direct(named: Mapping[str, object]) -> object:
        missing = [name for name in arg_names if name not in named]
        if missing:
            raise ValueError(
                f"unadapted AKA signature mismatch for {entry.problem_name}: {missing}",
            )
        return callable_module(*(named[name] for name in arg_names))

    return direct


def _load_torch2flydsl_model_oracle(
    entry: AKACorpusEntry,
    source: Path,
) -> Oracle:
    namespace: dict[str, object] = {"torch": torch}
    exec(  # noqa: S102 -- committed first-party benchmark source
        compile(source.read_text(encoding="utf-8"), str(source), "exec"),
        namespace,
    )
    model_type = namespace.get("Model")
    if not callable(model_type):
        raise ValueError(f"AKA semantic reference has no Model: {source}")
    if entry.problem_name == "fused_add_rmsnorm_bf16":
        constructor = cast(
            Callable[..., Callable[..., object]],
            model_type,
        )
        return lambda values: constructor(values["eps"])(
            values["input"],
            values["weight"],
            values["residual"],
        )
    if entry.problem_name == "per_token_i8_quant":
        model = cast(Callable[..., object], model_type)()
        return lambda values: cast(Callable[..., object], model)(
            values["input"]
        )
    if entry.problem_name == "rope_thd_fwd_bf16":
        model = cast(Callable[..., object], model_type)()
        return lambda values: cast(Callable[..., object], model)(
            values["input"],
            values["cu_seqlens"],
            values["freqs"],
        )
    if entry.problem_name == "dynamic_mxfp8_quant":
        model = cast(Callable[..., object], model_type)(32)
        return lambda values: cast(Callable[..., object], model)(
            values["input"]
        )
    return _moe_model_oracle(cast(Callable[..., object], model_type))


def _moe_model_oracle(model_type: Callable[..., object]) -> Oracle:
    def oracle(values: Mapping[str, object]) -> object:
        gating = values["gating"]
        bias = values["bias"]
        if not isinstance(gating, torch.Tensor) or not isinstance(
            bias, torch.Tensor
        ):
            raise TypeError("MoE gating and bias inputs must be tensors")
        use_bias = bool(torch.count_nonzero(bias).item())
        model = model_type(
            gating.shape[-1],
            values["topk"],
            values["route_scale"],
            use_bias,
        )
        model = cast(torch.nn.Module, model).to(gating.device)
        if use_bias:
            parameter = cast(_MoEModel, model).correction_bias
            parameter.data.copy_(bias)
        return model(gating)

    return oracle


def _layernorm_adapter(
    fn: Callable[..., object],
    values: Mapping[str, object],
) -> object:
    x = values["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("layernorm input x must be a tensor")
    return fn(
        x,
        values["weight"],
        values["bias"],
        (x.shape[-1],),
        values["eps"],
    )


def _sum_adapter(
    fn: Callable[..., object],
    values: Mapping[str, object],
) -> object:
    return fn(values["x"], -1)


def _maxpool_adapter(
    fn: Callable[..., object],
    values: Mapping[str, object],
) -> object:
    return fn(values["x"], 2, 2, 0, 1)


def _conv_adapter(
    fn: Callable[..., object],
    values: Mapping[str, object],
) -> object:
    return fn(values["x"], values["weight"], values["bias"], 1, 0, 1, 1)


def _depthwise_adapter(
    fn: Callable[..., object],
    values: Mapping[str, object],
) -> object:
    x = values["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("depthwise convolution input x must be a tensor")
    return fn(x, values["weight"], values["bias"], 1, 0, x.shape[1])


def _kd_loss_adapter(
    fn: Callable[..., object],
    values: Mapping[str, object],
) -> object:
    return fn(values["input"], values["target"], values["temperature"])


_TORCH2HIP_ADAPTERS: dict[
    str,
    Callable[[Callable[..., object], Mapping[str, object]], object],
] = {
    "l1n40_layernorm": _layernorm_adapter,
    "l1n47_sum_reduction": _sum_adapter,
    "l1n42_maxpool2d": _maxpool_adapter,
    "l1n63_conv2d": _conv_adapter,
    "l1n82_conv_depthwise": _depthwise_adapter,
    "14007_kd_loss": _kd_loss_adapter,
}


def _load_rmsnorm_backward_oracle(source: Path) -> Oracle:
    text = source.read_text(encoding="utf-8")
    function_source = _top_level_function_source(text, "torch_rmsnorm_fwd")
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(function_source, str(source), "exec"), namespace)  # noqa: S102
    forward = cast(Callable[..., object], namespace["torch_rmsnorm_fwd"])

    def oracle(values: Mapping[str, object]) -> object:
        x = _grad_clone(values["x"], "x")
        g = _grad_clone(values["g"], "g")
        grad_output = values["grad_output"]
        if not isinstance(grad_output, torch.Tensor):
            raise TypeError("RMSNorm grad_output must be a tensor")
        result = forward(x, g, False, x.dtype)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "AKA RMSNorm forward oracle returned an invalid result",
            )
        output = result[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("AKA RMSNorm forward oracle returned a non-tensor")
        output.backward(grad_output)
        if x.grad is None or g.grad is None:
            raise RuntimeError("AKA RMSNorm oracle did not produce gradients")
        return x.grad.to(x.dtype), g.grad.to(g.dtype)

    return oracle


def _grad_clone(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value.clone().detach().requires_grad_()


def _load_silu_and_mul_oracle(source: Path) -> Oracle:
    namespace: dict[str, object] = {"torch": torch}
    text = source.read_text(encoding="utf-8")
    exec(compile(text, str(source), "exec"), namespace)  # noqa: S102
    model_type = namespace.get("Model")
    if not callable(model_type):
        raise ValueError(f"AKA semantic reference has no Model: {source}")
    model = cast(Callable[..., object], model_type)(0.0)
    if not callable(model):
        raise TypeError(f"AKA Model instance is not callable: {source}")
    callable_model = cast(Callable[..., object], model)
    return lambda values: callable_model(values["input"])


def _top_level_function_source(text: str, name: str) -> str:
    module = ast.parse(text)
    for node in module.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            source = ast.get_source_segment(text, node)
            if source:
                return source
    raise KeyError(f"top-level function {name!r} not found")


def check_problem_equivalence(
    entry: AKACorpusEntry,
    problem_dir: Path,
    aka_root: Path,
    *,
    device: torch.device,
    seed: int = 2000,
    max_workloads: int | None = None,
) -> AKAEquivalenceReport:
    """Validate every selected workload/output against the bound AKA oracle."""
    definition, workloads = load_problem(problem_dir)
    if entry.role is AKACorpusRole.TARGET_INCOMPATIBLE:
        return AKAEquivalenceReport(
            entry.problem_name,
            True,
            CrosscheckStatus.NOT_APPLICABLE,
            0,
            0,
            "execution excluded by the manifest's target-incompatible role",
        )
    selected = workloads if max_workloads is None else workloads[:max_workloads]
    if not selected:
        raise ValueError(f"{entry.problem_name} has no workloads to validate")
    try:
        return _check_executable_problem(
            entry,
            definition,
            selected,
            aka_root,
            device=device,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001 -- convert to a complete corpus report
        return AKAEquivalenceReport(
            entry.problem_name,
            False,
            CrosscheckStatus.FAILED,
            0,
            0,
            str(exc),
        )


def _check_executable_problem(
    entry: AKACorpusEntry,
    definition: Definition,
    workloads: Sequence[Workload],
    aka_root: Path,
    *,
    device: torch.device,
    seed: int,
) -> AKAEquivalenceReport:
    authored, custom_inputs_fn = execute_reference_entrypoints(definition)
    oracle = load_aka_oracle(entry, aka_root)
    output_count = 0
    for index, workload in enumerate(workloads):
        ordered, named = materialize_inputs(
            definition,
            workload,
            seed=seed + index,
            device=device,
            custom_inputs_fn=custom_inputs_fn,
        )
        authored_outputs = normalize_outputs(
            authored(*ordered),
            definition,
            workload,
            source="authored",
        )
        output_count += len(authored_outputs)
        if oracle is not None:
            oracle_outputs = normalize_outputs(
                oracle(named),
                definition,
                workload,
                source="AKA",
            )
            _compare_outputs(
                definition,
                ordered,
                authored_outputs,
                oracle_outputs,
                workload,
            )
    crosscheck = (
        CrosscheckStatus.PASSED
        if oracle is not None
        else CrosscheckStatus.NOT_APPLICABLE
    )
    return AKAEquivalenceReport(
        entry.problem_name,
        True,
        crosscheck,
        len(workloads),
        output_count,
        "all declared workloads and outputs satisfy the authored/AKA contract",
    )


def _compare_outputs(
    definition: Definition,
    inputs: Sequence[object],
    authored: Sequence[torch.Tensor],
    oracle: Sequence[torch.Tensor],
    workload: Workload,
) -> None:
    stats, exceeds = compare_output_checks(
        definition,
        workload,
        inputs,
        oracle,
        authored,
        0,
    )
    if exceeds:
        raise ValueError(
            "authored outputs diverge from AKA oracle: "
            f"max_abs={stats.max_absolute_error:.3e}, "
            f"max_rel={stats.max_relative_error:.3e}",
        )


__all__ = [
    "AKAEquivalenceReport",
    "CrosscheckStatus",
    "check_problem_equivalence",
    "execute_reference",
    "execute_reference_entrypoints",
    "load_aka_oracle",
    "load_problem",
    "materialize_inputs",
    "normalize_outputs",
]
