# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed execution and numerical verification of SOLAR einsum graphs."""

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
import string
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from solar.einsum.semantics import (
    SemanticGraphError,
    annotate_semantics,
    validate_semantic_graph,
)
from solar.verification.errors import EinsumExecutionError, VerificationError
from solar.verification.numerics import (
    alias_relation as _alias_relation,
    assert_close as _assert_close,
    clone as _clone,
    pattern_inputs as _pattern_inputs,
)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_TOKEN = re.compile(r"[A-Za-z][0-9]*")


def _torch_equation(equation: str) -> str:
    """Map SOLAR rank tokens (including A0) to torch's one-letter ranks."""
    ranks_only = equation.replace("->", "")
    if not equation or "->" not in equation or any(c in ranks_only for c in "()+-"):
        raise EinsumExecutionError(
            f"unsupported extended einsum equation: {equation!r}"
        )
    tokens: list[str] = []
    for token in _TOKEN.findall(equation):
        if token not in tokens:
            tokens.append(token)
    alphabet = string.ascii_letters
    if len(tokens) > len(alphabet):
        raise EinsumExecutionError("einsum uses more ranks than torch can represent")
    mapping = dict(zip(tokens, alphabet))
    return _TOKEN.sub(lambda match: mapping[match.group(0)], equation)


def _shapes(layer: Mapping[str, Any]) -> list[tuple[int, ...]]:
    outputs = (layer.get("tensor_shapes") or {}).get("outputs") or []
    return [tuple(int(dim) for dim in shape) for shape in outputs]


def _decode_semantic_argument(
    argument: Any, operands: Sequence[Any], layer_id: str
) -> Any:
    import torch

    if argument == "preserve_format":
        return torch.preserve_format
    if argument == "contiguous_format":
        return torch.contiguous_format
    if isinstance(argument, list):
        return [
            _decode_semantic_argument(item, operands, layer_id) for item in argument
        ]
    if isinstance(argument, tuple):
        return tuple(
            _decode_semantic_argument(item, operands, layer_id) for item in argument
        )
    if not isinstance(argument, Mapping):
        return argument
    if "tensor" in argument:
        index = int(argument["tensor"])
        if index < 0 or index >= len(operands):
            raise EinsumExecutionError(
                f"layer {layer_id} references missing tensor argument {index}"
            )
        return operands[index]
    if "dtype" in argument:
        dtype = getattr(torch, str(argument["dtype"]), None)
        if not isinstance(dtype, torch.dtype):
            raise EinsumExecutionError(
                f"layer {layer_id} references invalid dtype {argument['dtype']!r}"
            )
        return dtype
    if "device" in argument:
        return torch.device(str(argument["device"]))
    if "layout" in argument:
        layout = getattr(torch, str(argument["layout"]), None)
        if not isinstance(layout, torch.layout):
            raise EinsumExecutionError(
                f"layer {layer_id} references invalid layout {argument['layout']!r}"
            )
        return layout
    if "value" in argument:
        value = argument["value"]
        if value == "__ellipsis__":
            return Ellipsis
        if value == "preserve_format":
            return torch.preserve_format
        if value == "contiguous_format":
            return torch.contiguous_format
        return value
    if "slice" in argument:
        values = [
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument["slice"]
        ]
        return slice(*values)
    raise EinsumExecutionError(f"layer {layer_id} has an invalid semantic argument")


class EinsumGraphExecutor:
    """Execute the exact subset of extended einsum understood by SOLAR."""

    def __init__(self, graph: Mapping[str, Any], *, check_shapes: bool = True):
        try:
            validate_semantic_graph(graph)
        except SemanticGraphError as exc:
            raise EinsumExecutionError(str(exc)) from exc
        layers = graph.get("layers") or {}
        if not isinstance(layers, Mapping) or not layers:
            raise EinsumExecutionError("einsum graph has no layers")
        self.layers = dict(layers)
        declared_outputs = graph.get("outputs")
        if declared_outputs is None:
            declared_outputs = (graph.get("graph_signature") or {}).get("joint_outputs")
        self.declared_outputs = (
            [str(name) for name in declared_outputs]
            if isinstance(declared_outputs, list)
            else None
        )
        self.check_shapes = check_shapes
        self._validate_layers()

    def _validate_layers(self) -> None:
        for layer_id, layer in self.layers.items():
            if not isinstance(layer, Mapping):
                raise EinsumExecutionError(f"layer {layer_id} is not a mapping")
            if str(layer.get("type", "")).lower() == "start":
                continue
            dtypes = layer.get("tensor_dtypes") or {}
            shapes = layer.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                if len(dtypes.get(side) or []) != len(shapes.get(side) or []):
                    raise EinsumExecutionError(
                        f"layer {layer_id} lacks explicit per-tensor dtype metadata"
                    )

    def __call__(self, *inputs: Any) -> Any:
        import torch

        values: dict[str, Any] = {}
        input_index = 0
        produced = {
            name
            for layer in self.layers.values()
            for name in ((layer.get("tensor_names") or {}).get("outputs") or [])
        }
        start_ids = [
            layer_id
            for layer_id, layer in self.layers.items()
            if str(layer.get("type", "")).lower() == "start"
        ]
        for layer_id in start_ids:
            names = (self.layers[layer_id].get("tensor_names") or {}).get(
                "outputs"
            ) or []
            for name in names:
                if input_index >= len(inputs):
                    raise EinsumExecutionError(
                        "not enough inputs for graph start tensors"
                    )
                values[str(name)] = inputs[input_index]
                input_index += 1

        external_names: list[str] = []
        for layer_id, layer in self.layers.items():
            if layer_id in start_ids:
                continue
            for name in (layer.get("tensor_names") or {}).get("inputs") or []:
                if (
                    name not in produced
                    and name not in values
                    and name not in external_names
                ):
                    external_names.append(str(name))
        for name in external_names:
            if input_index >= len(inputs):
                raise EinsumExecutionError(f"missing external tensor {name}")
            values[name] = inputs[input_index]
            input_index += 1
        if input_index != len(inputs):
            raise EinsumExecutionError(
                f"graph consumes {input_index} inputs but reference supplied {len(inputs)}"
            )

        pending = {
            key: value for key, value in self.layers.items() if key not in start_ids
        }
        consumed: set[str] = set()
        while pending:
            progressed = False
            for layer_id, layer in list(pending.items()):
                names = [
                    str(name)
                    for name in ((layer.get("tensor_names") or {}).get("inputs") or [])
                ]
                if not all(name in values for name in names):
                    continue
                operands = [values[name] for name in names]
                result = self._execute_layer(layer_id, layer, operands)
                output_names = [
                    str(name)
                    for name in ((layer.get("tensor_names") or {}).get("outputs") or [])
                ]
                results = (
                    list(result) if isinstance(result, (tuple, list)) else [result]
                )
                if len(output_names) != len(results):
                    raise EinsumExecutionError(
                        f"layer {layer_id} returned {len(results)} outputs, "
                        f"expected {len(output_names)}"
                    )
                expected_shapes = _shapes(layer)
                for index, (output_name, output) in enumerate(
                    zip(output_names, results)
                ):
                    if not isinstance(output, torch.Tensor):
                        raise EinsumExecutionError(
                            f"layer {layer_id} output {index} is not a tensor"
                        )
                    if (
                        self.check_shapes
                        and tuple(output.shape) != expected_shapes[index]
                    ):
                        raise EinsumExecutionError(
                            f"layer {layer_id} output {index} produced {tuple(output.shape)}, "
                            f"expected {expected_shapes[index]}"
                        )
                    values[output_name] = output
                consumed.update(names)
                del pending[layer_id]
                progressed = True
            if not progressed:
                missing = {
                    layer_id: [
                        name
                        for name in (
                            (layer.get("tensor_names") or {}).get("inputs") or []
                        )
                        if name not in values
                    ]
                    for layer_id, layer in pending.items()
                }
                raise EinsumExecutionError(
                    f"unresolvable graph dependencies: {missing}"
                )

        if self.declared_outputs is not None:
            missing_outputs = [
                name for name in self.declared_outputs if name not in values
            ]
            if missing_outputs:
                raise EinsumExecutionError(
                    f"graph declares unavailable outputs: {missing_outputs}"
                )
            ordered_terminal = self.declared_outputs
        else:
            terminal = [
                name for name in produced if name not in consumed and name in values
            ]
            ordered_terminal = [
                str(name)
                for layer in self.layers.values()
                for name in ((layer.get("tensor_names") or {}).get("outputs") or [])
                if name in terminal
            ]
        if not ordered_terminal:
            raise EinsumExecutionError("einsum graph has no terminal output")
        outputs = tuple(values[name] for name in ordered_terminal)
        return outputs[0] if len(outputs) == 1 else outputs

    def _execute_layer(
        self, layer_id: str, layer: Mapping[str, Any], operands: Sequence[Any]
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        semantic = layer["semantic_op"]
        if semantic["kind"] == "einsum":
            return torch.einsum(_torch_equation(str(semantic["equation"])), *operands)

        arguments = [
            _decode_semantic_argument(item, operands, layer_id)
            for item in semantic.get("arguments") or []
        ]
        kwargs = {
            str(key): _decode_semantic_argument(value, operands, layer_id)
            for key, value in (semantic.get("kwargs") or {}).items()
        }
        target = str(semantic.get("target", ""))
        output_shapes = _shapes(layer)

        effects = semantic.get("effects") or {}
        if effects.get("mutates"):
            if not arguments:
                raise EinsumExecutionError(
                    f"mutating operation {target!r} at {layer_id} has no receiver"
                )
            method = getattr(arguments[0], f"{target}_", None)
            if method is None:
                raise EinsumExecutionError(
                    f"mutating operation {target!r} at {layer_id} is unavailable"
                )
            return method(*arguments[1:], **kwargs)

        binary = {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "div": torch.div,
            "eq": torch.eq,
            "ge": torch.ge,
            "gt": torch.gt,
            "le": torch.le,
            "lt": torch.lt,
            "ne": torch.ne,
            "pow": torch.pow,
            "maximum": torch.maximum,
            "minimum": torch.minimum,
            "bitwise_and": torch.bitwise_and,
        }
        unary = {
            "abs": torch.abs,
            "bitwise_not": torch.bitwise_not,
            "cos": torch.cos,
            "elu": functional.elu,
            "exp": torch.exp,
            "gelu": functional.gelu,
            "hardsigmoid": functional.hardsigmoid,
            "hardswish": functional.hardswish,
            "leaky_relu": functional.leaky_relu,
            "log": torch.log,
            "mish": functional.mish,
            "neg": torch.neg,
            "relu": functional.relu,
            "rsqrt": torch.rsqrt,
            "sigmoid": torch.sigmoid,
            "silu": functional.silu,
            "sin": torch.sin,
            "sqrt": torch.sqrt,
            "square": torch.square,
            "tanh": torch.tanh,
        }
        if target in binary:
            return binary[target](*arguments, **kwargs)
        if target in {"mm", "bmm", "matmul", "addmm", "where"}:
            return getattr(torch, target)(*arguments, **kwargs)
        if target == "masked_fill":
            return arguments[0].masked_fill(*arguments[1:], **kwargs)
        if target == "cumsum":
            return torch.cumsum(*arguments, **kwargs)
        if target in unary:
            return unary[target](*arguments, **kwargs)
        if target == "identity":
            return arguments[0]
        if target == "to":
            return arguments[0].to(*arguments[1:], **kwargs)
        if target in {"bfloat16", "float", "half", "int", "long"}:
            return getattr(arguments[0], target)()
        if target == "type_as":
            return arguments[0].type_as(*arguments[1:], **kwargs)
        if target == "clone":
            return arguments[0].clone(**kwargs)
        if target == "detach":
            return arguments[0].detach()
        if target in {"softmax", "log_softmax"}:
            return getattr(functional, target)(*arguments, **kwargs)
        if target in {
            "sum",
            "mean",
            "prod",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "logsumexp",
        }:
            return getattr(torch, target)(*arguments, **kwargs)
        if target in {"view", "reshape"}:
            if len(arguments) > 1:
                return getattr(arguments[0], target)(*arguments[1:], **kwargs)
            shape = kwargs.pop("shape", output_shapes[0])
            return getattr(arguments[0], target)(tuple(shape))
        if target == "flatten":
            return torch.flatten(*arguments, **kwargs)
        if target == "contiguous":
            return arguments[0].contiguous(**kwargs)
        if target in {
            "squeeze",
            "unsqueeze",
            "permute",
            "repeat",
            "repeat_interleave",
            "expand",
        }:
            return getattr(arguments[0], target)(*arguments[1:], **kwargs)
        if target == "transpose":
            if len(arguments) == 1 and not kwargs:
                if arguments[0].ndim != 2:
                    raise EinsumExecutionError(
                        f"layer {layer_id} requires explicit transpose dimensions"
                    )
                return arguments[0].t()
            return torch.transpose(*arguments, **kwargs)
        if target in {"cat", "stack"}:
            if arguments and isinstance(arguments[0], (list, tuple)):
                return getattr(torch, target)(*arguments, **kwargs)
            return getattr(torch, target)(arguments, **kwargs)
        if target == "vstack":
            return torch.vstack(*arguments, **kwargs)
        if target in {"chunk", "split"}:
            return getattr(torch, target)(*arguments, **kwargs)
        if target in {"gather", "scatter", "index_select", "select", "narrow"}:
            return getattr(torch, target)(*arguments, **kwargs)
        if target == "getitem":
            index = arguments[1]
            if isinstance(index, list) and any(
                isinstance(item, slice) or item is None or item is Ellipsis
                for item in index
            ):
                index = tuple(index)
            return arguments[0][index]
        if target == "slice":
            dim = int(kwargs.get("dim", 0))
            start = kwargs.get("start")
            end = kwargs.get("end")
            step = kwargs.get("step")
            slices = [slice(None)] * arguments[0].ndim
            slices[dim] = slice(start, end, step)
            return arguments[0][tuple(slices)]
        if target == "linear":
            return functional.linear(*arguments, **kwargs)
        if target.startswith("conv_transpose"):
            return getattr(functional, target)(*arguments, **kwargs)
        if target in {"conv1d", "conv2d", "conv3d"}:
            return getattr(functional, target)(*arguments, **kwargs)
        if target in {
            "batch_norm",
            "group_norm",
            "instance_norm",
            "layer_norm",
            "embedding",
            "embedding_bag",
            "dropout",
            "max_pool2d",
        }:
            return getattr(functional, target)(*arguments, **kwargs)
        if target == "scaled_dot_product_attention":
            return functional.scaled_dot_product_attention(*arguments, **kwargs)
        if target in {
            "quantize_per_tensor",
            "quantize_per_channel",
            "fake_quantize_per_tensor_affine",
            "fake_quantize_per_channel_affine",
        }:
            return getattr(torch, target)(*arguments, **kwargs)
        if target == "dequantize":
            return arguments[0].dequantize()
        if target in {"ones_like", "zeros_like"}:
            return getattr(torch, target)(*arguments, **kwargs)
        if target == "clamp":
            return torch.clamp(*arguments, **kwargs)
        if target.isidentifier() and hasattr(torch.ops.aten, target):
            packet = getattr(torch.ops.aten, target)
            overload_name = str(semantic.get("overload", "default"))
            overload = getattr(packet, overload_name, None)
            if overload is None:
                raise EinsumExecutionError(
                    f"ATen operation {target}.{overload_name} is unavailable"
                )
            return overload(*arguments, **kwargs)
        raise EinsumExecutionError(
            f"operation {target!r} at {layer_id} is not executable exactly"
        )


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
_ComparisonPolicy = tuple[float, float, float, float | None, bool]


def _prepare_case(
    input_factory: Callable[..., Any],
    graph: Mapping[str, Any],
    case: Mapping[str, Any],
    device: str,
) -> _PreparedCase:
    import torch

    parameters = dict(case.get("parameters") or {})
    seed, pattern = int(case["seed"]), str(case["pattern"])
    generated = input_factory({**parameters, "seed": seed}, device)
    inputs = tuple(generated) if isinstance(generated, (tuple, list)) else (generated,)
    reference_inputs = _clone(_pattern_inputs(inputs, pattern))
    source_indices = graph.get("source_input_indices")
    if source_indices is None:
        reference_tensor_inputs = tuple(
            value for value in reference_inputs if isinstance(value, torch.Tensor)
        )
    else:
        try:
            reference_tensor_inputs = tuple(
                reference_inputs[int(index)] for index in source_indices
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise VerificationError("graph has invalid source_input_indices") from exc
        if not all(
            isinstance(value, torch.Tensor) for value in reference_tensor_inputs
        ):
            raise VerificationError(
                "graph source_input_indices must select tensor arguments"
            )
    return (
        parameters,
        seed,
        pattern,
        reference_inputs,
        reference_tensor_inputs,
        _clone(reference_tensor_inputs),
    )


def _verify_case(
    reference: Callable[..., Any],
    executor: EinsumGraphExecutor,
    graph: Mapping[str, Any],
    prepared: _PreparedCase,
    policy: _ComparisonPolicy,
) -> dict[str, float]:
    import torch

    _, _, pattern, reference_inputs, reference_tensor_inputs, executor_inputs = prepared
    atol, rtol, required_ratio, error_cap, allow_negative_inf = policy
    with torch.enable_grad():
        expected = reference(*reference_inputs)
    with torch.inference_mode():
        actual = executor(*executor_inputs)
        try:
            stats = _assert_close(
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
                    graph, executor_inputs, actual, expected
                )
            ):
                raise
            stats = _assert_close(
                actual,
                expected,
                atol,
                rtol,
                required_matched_ratio=0.0,
                allow_negative_inf=allow_negative_inf,
                allow_matching_nan=pattern in {"zeros", "boundary"},
            )
            stats["roundoff_bound_verified"] = 1.0
        _assert_close(executor_inputs, reference_tensor_inputs, atol, rtol)
        if _alias_relation(actual, executor_inputs) != _alias_relation(
            expected, reference_tensor_inputs
        ):
            raise VerificationError(
                "output/input alias relationships differ from the reference"
            )
    return stats


def _run_cases(
    reference: Callable[..., Any],
    input_factory: Callable[..., Any],
    graph: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
    max_error_cap: float | None,
    allow_negative_inf: bool,
    device: str,
    check_shapes: bool,
) -> list[dict[str, Any]]:
    import torch

    executor = EinsumGraphExecutor(graph, check_shapes=check_shapes)
    policy = (atol, rtol, required_matched_ratio, max_error_cap, allow_negative_inf)
    results: list[dict[str, Any]] = []
    for case in cases:
        prepared = _prepare_case(input_factory, graph, case, device)
        parameters, seed, pattern, *_ = prepared
        stats = _verify_case(reference, executor, graph, prepared, policy)
        results.append(
            {
                "seed": seed,
                "pattern": pattern,
                "parameters_sha256": _canonical_hash(parameters),
                **stats,
            }
        )
        del prepared
        if torch.cuda.is_available() and str(device).startswith(("cuda", "rocm")):
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
    semantic = operations[0].get("semantic_op") or {}
    if semantic.get("kind") != "einsum":
        return False
    equation = str(semantic.get("equation", ""))
    reduction_size = _einsum_reduction_size(equation, operands)
    unit_roundoff = 2.0**-24
    if reduction_size is None or reduction_size * unit_roundoff >= 1.0:
        return False
    try:
        torch_equation = _torch_equation(equation)
        precise = [value.double() for value in operands]
        oracle = torch.einsum(torch_equation, *precise)
        absolute_sum = torch.einsum(torch_equation, *(value.abs() for value in precise))
    except (RuntimeError, EinsumExecutionError):
        return False
    gamma = reduction_size * unit_roundoff / (1.0 - reduction_size * unit_roundoff)
    bound = gamma * absolute_sum + unit_roundoff * oracle.abs()
    slack = torch.finfo(torch.float64).eps * torch.maximum(
        torch.ones_like(bound), oracle.abs()
    )
    limit = bound + slack
    return bool(
        ((actual.double() - oracle).abs() <= limit).all()
        and ((expected.double() - oracle).abs() <= limit).all()
    )


def _einsum_reduction_size(equation: str, operands: Sequence[Any]) -> int | None:
    if "->" not in equation:
        return None
    inputs, output = equation.split("->", maxsplit=1)
    terms = inputs.split(",")
    if len(terms) != len(operands):
        return None
    dimensions: dict[str, int] = {}
    input_tokens: set[str] = set()
    for term, operand in zip(terms, operands):
        tokens = _TOKEN.findall(term)
        if len(tokens) != operand.ndim:
            return None
        for token, size in zip(tokens, operand.shape):
            if token in dimensions and dimensions[token] != int(size):
                return None
            dimensions[token] = int(size)
            input_tokens.add(token)
    reduced = input_tokens - set(_TOKEN.findall(output))
    if not reduced:
        return None
    return math.prod(dimensions[token] for token in reduced)


def create_verification_artifact(
    *,
    reference_path: str | Path,
    reference_entry_point: str,
    input_factory_name: str,
    graph_path: str | Path,
    workload_name: str,
    workload_parameters: Mapping[str, Any],
    output_path: str | Path,
    atol: float,
    rtol: float,
    required_matched_ratio: float = 1.0,
    max_error_cap: float | None = None,
    allow_negative_inf: bool = False,
    seeds: Sequence[int] = (11, 29, 47),
    patterns: Sequence[str] = ("random", "zeros", "boundary"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Verify and write a deterministic, hash-bound ``verification.yaml``."""
    reference_path = Path(reference_path).resolve()
    graph_path = Path(graph_path).resolve()
    module = _load_module(reference_path)
    reference = getattr(module, reference_entry_point)
    input_factory = getattr(module, input_factory_name)
    graph = yaml.safe_load(graph_path.read_text()) or {}
    cases = [
        {"parameters": dict(workload_parameters), "seed": int(seed), "pattern": pattern}
        for seed in seeds
        for pattern in patterns
    ]
    if len(set(int(seed) for seed in seeds)) < 3:
        raise VerificationError("trusted verification requires at least three seeds")
    if not {"random", "zeros", "boundary"}.issubset(patterns):
        raise VerificationError(
            "trusted verification requires random, zeros, and boundary patterns"
        )
    execution = _execution_identity(device)
    results = _run_cases(
        reference,
        input_factory,
        graph,
        cases,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
        max_error_cap=max_error_cap,
        allow_negative_inf=allow_negative_inf,
        device=device,
        check_shapes=True,
    )
    artifact = {
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
        "predicateType": ("https://solar-rocm.dev/attestations/source-to-einsum/v2"),
        "predicate": {
            "status": "passed",
            "verifier": "solar.verification.einsum.v2",
            "reference": {
                "entry_point": reference_entry_point,
                "input_factory": input_factory_name,
            },
            "workload": {
                "name": workload_name,
                "parameters_sha256": _canonical_hash(workload_parameters),
            },
            "tolerance": {
                "atol": float(atol),
                "rtol": float(rtol),
                "required_matched_ratio": float(required_matched_ratio),
                "max_error_cap": max_error_cap,
                "allow_negative_inf": bool(allow_negative_inf),
            },
            "execution": execution,
            "cases": cases,
            "results": results,
        },
    }
    Path(output_path).write_text(yaml.safe_dump(artifact, sort_keys=False))
    return artifact


def verify_callable_conversion(
    *,
    reference: Callable[..., Any],
    input_factory: Callable[[int], Sequence[Any]],
    reference_name: str,
    reference_sha256: str,
    graph_path: str | Path,
    output_path: str | Path,
    atol: float,
    rtol: float,
    required_matched_ratio: float = 1.0,
    max_error_cap: float | None = None,
    allow_negative_inf: bool = False,
    seeds: Sequence[int] = (11, 29, 47),
    patterns: Sequence[str] = ("random", "zeros", "boundary"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Verify a callable reference and write a hash-bound attestation.

    This is the boundary-safe variant used by :mod:`solar.api`: it knows
    nothing about benchmark definitions or workload schemas.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", reference_sha256):
        raise VerificationError("reference_sha256 must be a lowercase SHA-256")
    if len(set(int(seed) for seed in seeds)) < 3:
        raise VerificationError("trusted verification requires at least three seeds")
    if not {"random", "zeros", "boundary"}.issubset(patterns):
        raise VerificationError(
            "trusted verification requires random, zeros, and boundary patterns"
        )
    graph_path = Path(graph_path).resolve()
    graph = yaml.safe_load(graph_path.read_text()) or {}
    cases = [
        {"parameters": {}, "seed": int(seed), "pattern": str(pattern)}
        for seed in seeds
        for pattern in patterns
    ]

    def compatible_factory(
        parameters: Mapping[str, Any], _device: str
    ) -> Sequence[Any]:
        return input_factory(int(parameters["seed"]))

    results = _run_cases(
        reference,
        compatible_factory,
        graph,
        cases,
        atol=atol,
        rtol=rtol,
        required_matched_ratio=required_matched_ratio,
        max_error_cap=max_error_cap,
        allow_negative_inf=allow_negative_inf,
        device=device,
        check_shapes=True,
    )
    artifact = _callable_attestation(
        reference_name=reference_name,
        reference_sha256=reference_sha256,
        graph_path=graph_path,
        tolerance={
            "atol": float(atol),
            "rtol": float(rtol),
            "required_matched_ratio": float(required_matched_ratio),
            "max_error_cap": max_error_cap,
            "allow_negative_inf": bool(allow_negative_inf),
        },
        execution=_execution_identity(device),
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
            {"name": graph_path.name, "digest": {"sha256": _sha256(graph_path)}},
        ],
        "predicateType": "https://solar-rocm.dev/attestations/callable-to-einsum/v1",
        "predicate": {
            "status": "passed",
            "verifier": "solar.verification.einsum.v2",
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
    atol: float,
    rtol: float,
    required_matched_ratio: float = 1.0,
    max_error_cap: float | None = None,
    allow_negative_inf: bool = False,
    device: str | None = None,
) -> None:
    """Validate every binding and numerically replay a verification artifact."""
    reference_path = Path(reference_path).resolve()
    graph_path = Path(graph_path).resolve()
    if artifact.get("_type") != "https://in-toto.io/Statement/v1":
        raise VerificationError("verification artifact must be an in-toto Statement v1")
    if artifact.get("predicateType") != (
        "https://solar-rocm.dev/attestations/source-to-einsum/v2"
    ):
        raise VerificationError("unsupported verification predicate type")
    predicate = artifact.get("predicate") or {}
    if (
        predicate.get("status") != "passed"
        or predicate.get("verifier") != "solar.verification.einsum.v2"
    ):
        raise VerificationError("verification artifact is not a trusted passing result")
    subjects = artifact.get("subject") or []
    digests = {
        str(subject.get("name")): (subject.get("digest") or {}).get("sha256")
        for subject in subjects
    }
    reference_data = predicate.get("reference") or {}
    workload_data = predicate.get("workload") or {}
    tolerance = predicate.get("tolerance") or {}
    execution = predicate.get("execution") or {}
    if digests.get(reference_path.name) != _sha256(reference_path):
        raise VerificationError("verification reference SHA-256 mismatch")
    if digests.get(graph_path.name) != _sha256(graph_path):
        raise VerificationError("verification graph SHA-256 mismatch")
    if workload_data.get("name") != workload_name:
        raise VerificationError("verification workload name mismatch")
    if workload_data.get("parameters_sha256") != _canonical_hash(workload_parameters):
        raise VerificationError("verification workload parameters mismatch")
    recorded_atol = float(tolerance.get("atol", math.inf))
    recorded_rtol = float(tolerance.get("rtol", math.inf))
    recorded_ratio = float(tolerance.get("required_matched_ratio", -1.0))
    recorded_cap_raw = tolerance.get("max_error_cap")
    recorded_cap = float(recorded_cap_raw) if recorded_cap_raw is not None else None
    cap_is_weaker = max_error_cap is not None and (
        recorded_cap is None or recorded_cap > max_error_cap
    )
    negative_inf_is_weaker = (
        bool(tolerance.get("allow_negative_inf", False)) and not allow_negative_inf
    )
    if (
        not all(math.isfinite(value) for value in (recorded_atol, recorded_rtol))
        or not math.isfinite(recorded_ratio)
        or recorded_atol > atol
        or recorded_rtol > rtol
        or recorded_ratio < required_matched_ratio
        or cap_is_weaker
        or negative_inf_is_weaker
    ):
        raise VerificationError(
            "verification tolerance is weaker than benchmark tolerance"
        )
    cases = predicate.get("cases") or []
    results = predicate.get("results") or []
    if len(cases) != len(results) or len(cases) < 9:
        raise VerificationError("verification artifact lacks the required cases")
    if len({int(case["seed"]) for case in cases}) < 3:
        raise VerificationError("verification artifact lacks three independent seeds")
    if not {"random", "zeros", "boundary"}.issubset(
        str(case["pattern"]) for case in cases
    ):
        raise VerificationError("verification artifact lacks boundary patterns")
    if any(
        dict(case.get("parameters") or {}) != dict(workload_parameters)
        for case in cases
    ):
        raise VerificationError(
            "verification cases are not bound to workload parameters"
        )
    recorded_device = str(execution.get("device_type", ""))
    if recorded_device not in {"cpu", "cuda"}:
        raise VerificationError("verification artifact has no supported replay device")
    replay_device = device or recorded_device
    expected_backend = str(execution.get("backend", ""))
    actual_execution = _execution_identity(replay_device)
    if expected_backend not in {"cpu", "cuda", "rocm"}:
        raise VerificationError(
            "verification artifact has no execution backend identity"
        )
    if actual_execution.get("backend") != expected_backend:
        raise VerificationError(
            "verification replay backend differs from recorded backend"
        )
    if expected_backend == "rocm":
        for field in ("hip_version", "gfx_target"):
            if execution.get(field) != actual_execution.get(field):
                raise VerificationError(
                    f"verification replay {field} differs from recorded ROCm device"
                )
    module = _load_module(reference_path)
    graph = yaml.safe_load(graph_path.read_text()) or {}
    replay = _run_cases(
        getattr(module, str(reference_data["entry_point"])),
        getattr(module, str(reference_data["input_factory"])),
        graph,
        cases,
        atol=float(tolerance["atol"]),
        rtol=float(tolerance["rtol"]),
        required_matched_ratio=float(tolerance["required_matched_ratio"]),
        max_error_cap=(
            float(tolerance["max_error_cap"])
            if tolerance.get("max_error_cap") is not None
            else None
        ),
        allow_negative_inf=bool(tolerance["allow_negative_inf"]),
        device=replay_device,
        check_shapes=True,
    )
    for expected, actual in zip(results, replay):
        identity = ("seed", "pattern", "parameters_sha256")
        if any(expected.get(key) != actual.get(key) for key in identity):
            raise VerificationError("verification replay identity mismatch")


def _execution_identity(device: str) -> dict[str, Any]:
    """Identify the actual PyTorch backend selected by a device string."""
    import torch

    if not str(device).startswith("cuda"):
        return {"device_type": "cpu", "backend": "cpu", "device": str(device)}
    if not torch.cuda.is_available():
        raise VerificationError(f"requested CUDA/HIP device is unavailable: {device}")
    selected = torch.device(device)
    index = (
        selected.index if selected.index is not None else torch.cuda.current_device()
    )
    if index < 0 or index >= torch.cuda.device_count():
        raise VerificationError(f"requested CUDA/HIP device index is invalid: {device}")
    properties = torch.cuda.get_device_properties(index)
    hip_version = getattr(torch.version, "hip", None)
    if hip_version:
        gfx_target = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        if not gfx_target.startswith("gfx"):
            raise VerificationError(
                "HIP runtime selected a device without an AMD gfx target"
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


def _resolve_torch_operation(node_type: str) -> Callable[..., Any]:
    import torch
    import torch.nn.functional as functional

    normalized = node_type.lower().split(".")[-1]
    for owner in (torch, functional):
        function = getattr(owner, normalized, None)
        if callable(function):
            return function
    raise VerificationError(f"cannot resolve PyTorch reference operation {node_type!r}")


def verify_generated_handler(
    node_type: str,
    source_code: str,
    node_data: Mapping[str, Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Numerically validate generated handler code before it may be cached."""
    import torch

    safe_builtins = {
        "abs": abs,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "str": str,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "Dict": dict,
        "Any": Any,
    }
    exec(source_code, namespace, namespace)  # pylint: disable=exec-used
    name = f"create_{node_type}_subgraph"
    function = namespace.get(name)
    if not callable(function):
        raise VerificationError(f"generated handler does not define {name}")
    subgraph = function("verification_node", dict(node_data))
    if not isinstance(subgraph, Mapping) or not subgraph:
        raise VerificationError("generated handler returned an empty subgraph")
    input_shapes = [
        tuple(int(dim) for dim in shape) for shape in node_data.get("input_shapes", [])
    ]
    if not input_shapes:
        raise VerificationError("generated handler validation requires input shapes")
    dtypes = list(node_data.get("input_dtypes") or [])
    cases = []
    for seed, pattern in ((11, "random"), (29, "zeros"), (47, "boundary")):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        tensors = []
        for index, shape in enumerate(input_shapes):
            dtype_name = str(
                dtypes[index] if index < len(dtypes) else "torch.float32"
            ).split(".")[-1]
            dtype = getattr(torch, dtype_name, torch.float32)
            value = torch.randn(shape, generator=generator, dtype=dtype)
            tensors.append(value)
        inputs = _pattern_inputs(tuple(tensors), pattern)
        kwargs = {
            key: value
            for key, value in (node_data.get("module_args") or {}).items()
            if isinstance(value, (str, int, float, bool, tuple, list))
            and key != "raw_attributes"
        }
        expected = _resolve_torch_operation(node_type)(*_clone(inputs), **kwargs)
        graph = annotate_semantics({"layers": dict(subgraph)}, strict=True)
        actual = EinsumGraphExecutor(graph, check_shapes=True)(*_clone(inputs))
        _assert_close(actual, expected, atol, rtol)
        cases.append({"seed": seed, "pattern": pattern})
    return {
        "status": "passed",
        "verifier": "solar.generated_handler.v1",
        "cases": cases,
    }
