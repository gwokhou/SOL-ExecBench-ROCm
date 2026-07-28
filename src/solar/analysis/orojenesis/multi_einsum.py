# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from solar.analysis.orojenesis.errors import OrojenesisError
from solar.ir.contracts import layer_operation
from solar.schema_versions import OROJENESIS_MULTI_EINSUM_PROBLEM_SCHEMA_VERSION

_TOKEN = re.compile(r"[A-Za-z][0-9]*")


def _divisors(value: int) -> list[int]:
    if value <= 0:
        raise OrojenesisError("multi-einsum dimensions must be positive")
    small: list[int] = []
    large: list[int] = []
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            small.append(candidate)
            if candidate * candidate != value:
                large.append(value // candidate)
        candidate += 1
    return small + list(reversed(large))


def multi_einsum_mapper_role(layer_index: int, layer_count: int) -> str:
    """Map a chain position to the pinned ``_relax_io_kn`` FFMT variant."""
    if layer_count < 2 or layer_index not in range(layer_count):
        raise OrojenesisError("invalid multi-einsum chain position")
    if layer_index == 0:
        return "first"
    if layer_count == 2:
        return "second_last"
    if layer_index == 1:
        return "second"
    if layer_index == layer_count - 1:
        return "last"
    return "middle"


def _matmul_descriptor(
    layer_id: str,
    layer: Mapping[str, Any],
) -> dict[str, Any]:
    semantic = layer_operation(layer)
    if semantic.get("kind") != "einsum":
        raise OrojenesisError(
            "multi-einsum chains accept exact einsum layers only",
        )
    equation = str(semantic.get("equation", ""))
    if "->" not in equation:
        raise OrojenesisError(
            "multi-einsum equation must have an explicit output",
        )
    lhs, rhs = equation.split("->", 1)
    operands = lhs.split(",")
    operand_tokens = [_TOKEN.findall(operand) for operand in operands]
    output_tokens = _TOKEN.findall(rhs)
    if (
        len(operand_tokens) != 2
        or any(len(tokens) != 2 for tokens in operand_tokens)
        or len(output_tokens) != 2
    ):
        raise OrojenesisError(
            "multi-einsum currently requires binary rank-2 matmul",
        )
    m_token, k_token = operand_tokens[0]
    second_k, n_token = operand_tokens[1]
    if (
        k_token != second_k
        or output_tokens != [m_token, n_token]
        or len({m_token, k_token, n_token}) != 3
    ):
        raise OrojenesisError("multi-einsum layer is not a canonical matmul")
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    input_names = [str(name) for name in names.get("inputs") or []]
    output_names = [str(name) for name in names.get("outputs") or []]
    input_shapes = [list(shape) for shape in shapes.get("inputs") or []]
    output_shapes = [list(shape) for shape in shapes.get("outputs") or []]
    input_dtypes = [str(dtype) for dtype in dtypes.get("inputs") or []]
    output_dtypes = [str(dtype) for dtype in dtypes.get("outputs") or []]
    if not (
        len(input_names) == len(input_shapes) == len(input_dtypes) == 2
        and len(output_names) == len(output_shapes) == len(output_dtypes) == 1
    ):
        raise OrojenesisError("multi-einsum tensor metadata arity mismatch")
    m_size, k_size = (int(value) for value in input_shapes[0])
    second_k_size, n_size = (int(value) for value in input_shapes[1])
    if second_k_size != k_size or output_shapes[0] != [m_size, n_size]:
        raise OrojenesisError("multi-einsum matmul shapes are inconsistent")
    effects = semantic.get("effects") or {}
    if any(
        (
            effects.get("mutates"),
            effects.get("aliases"),
            effects.get("atomic"),
            effects.get("opaque_library_call"),
        ),
    ):
        raise OrojenesisError("multi-einsum chain contains observable effects")
    if len(set(input_dtypes + output_dtypes)) != 1:
        raise OrojenesisError(
            "multi-einsum chain requires one exact tensor dtype",
        )
    return {
        "id": str(layer_id),
        "equation": equation,
        "input": input_names[0],
        "weight": input_names[1],
        "output": output_names[0],
        "m": m_size,
        "k": k_size,
        "n": n_size,
        "dtype": input_dtypes[0],
    }


def multi_einsum_problem(
    chain: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Create a canonical, hashable linear-matmul-chain problem."""
    descriptors = [
        _matmul_descriptor(str(layer_id), layer) for layer_id, layer in chain
    ]
    if len(descriptors) < 2:
        raise OrojenesisError("multi-einsum proof requires at least two layers")
    first_m = descriptors[0]["m"]
    dtype = descriptors[0]["dtype"]
    for previous, current in itertools.pairwise(descriptors):
        if previous["output"] != current["input"]:
            raise OrojenesisError(
                "multi-einsum layers are not a producer-consumer chain",
            )
        if previous["m"] != current["m"] or previous["n"] != current["k"]:
            raise OrojenesisError("multi-einsum boundary shapes do not match")
        if current["m"] != first_m or current["dtype"] != dtype:
            raise OrojenesisError(
                "multi-einsum chain M dimension or dtype drifted",
            )
    return {
        "schema_version": (OROJENESIS_MULTI_EINSUM_PROBLEM_SCHEMA_VERSION),
        "chain": {"kind": "linear_matmul", "layers": descriptors},
    }


def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


@dataclass(frozen=True)
class _RegionTensorMetadata:
    input_names: list[str]
    output_name: str
    input_shapes: list[list[int]]
    output_shape: list[int]
    input_dtypes: list[str]
    output_dtype: str


def _region_einsum_tokens(
    layer: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, list[list[str]], list[str]]:
    semantic = layer_operation(layer)
    if semantic.get("kind") != "einsum":
        raise OrojenesisError(
            "multi-einsum regions accept exact einsum layers only",
        )
    equation = str(semantic.get("equation", ""))
    if "->" not in equation:
        raise OrojenesisError(
            "multi-einsum equation must have an explicit output",
        )
    lhs, rhs = equation.split("->", 1)
    operands = [_TOKEN.findall(operand) for operand in lhs.split(",")]
    output = _TOKEN.findall(rhs)
    if len(operands) != 2 or len(output) < 2:
        raise OrojenesisError(
            "multi-einsum region requires binary matrix contraction",
        )
    return semantic, equation, operands, output


def _region_tensor_metadata(
    layer: Mapping[str, Any],
    operand_tokens: Sequence[Sequence[str]],
    output_tokens: Sequence[str],
) -> _RegionTensorMetadata:
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    input_names = [str(name) for name in names.get("inputs") or []]
    output_names = [str(name) for name in names.get("outputs") or []]
    input_shapes = [list(shape) for shape in shapes.get("inputs") or []]
    output_shapes = [list(shape) for shape in shapes.get("outputs") or []]
    input_dtypes = [str(dtype) for dtype in dtypes.get("inputs") or []]
    output_dtypes = [str(dtype) for dtype in dtypes.get("outputs") or []]
    valid = (
        len(input_names) == len(input_shapes) == len(input_dtypes) == 2
        and len(output_names) == len(output_shapes) == len(output_dtypes) == 1
        and all(
            len(tokens) == len(shape)
            for tokens, shape in zip(
                operand_tokens,
                input_shapes,
                strict=True,
            )
        )
        and len(output_tokens) == len(output_shapes[0])
    )
    if not valid:
        raise OrojenesisError(
            "multi-einsum region tensor metadata arity mismatch",
        )
    return _RegionTensorMetadata(
        input_names=input_names,
        output_name=output_names[0],
        input_shapes=input_shapes,
        output_shape=output_shapes[0],
        input_dtypes=input_dtypes,
        output_dtype=output_dtypes[0],
    )


def _region_token_sizes(
    operand_tokens: Sequence[Sequence[str]],
    output_tokens: Sequence[str],
    metadata: _RegionTensorMetadata,
) -> dict[str, int]:
    token_sizes: dict[str, int] = {}
    token_shapes = [
        *zip(operand_tokens, metadata.input_shapes, strict=True),
        (output_tokens, metadata.output_shape),
    ]
    for tokens, shape in token_shapes:
        for token, raw_size in zip(tokens, shape, strict=True):
            size = int(raw_size)
            if size <= 0 or (
                token in token_sizes and token_sizes[token] != size
            ):
                raise OrojenesisError(
                    "multi-einsum region dimensions are inconsistent",
                )
            token_sizes[token] = size
    return token_sizes


def _region_matmul_axes(
    operand_tokens: Sequence[Sequence[str]],
    output_tokens: Sequence[str],
) -> tuple[int, int, str, str, list[str]]:
    reductions = (set(operand_tokens[0]) & set(operand_tokens[1])) - set(
        output_tokens,
    )
    if len(reductions) != 1:
        raise OrojenesisError(
            "multi-einsum region requires one reduction dimension",
        )
    reduction = next(iter(reductions))
    candidates: list[tuple[int, int, str, list[str]]] = []
    for activation_index, weight_index in ((0, 1), (1, 0)):
        activation_tokens = operand_tokens[activation_index]
        weight_tokens = operand_tokens[weight_index]
        weight_free = [token for token in weight_tokens if token != reduction]
        activation_free = [
            token for token in activation_tokens if token != reduction
        ]
        if (
            len(weight_tokens) == 2
            and len(weight_free) == 1
            and activation_free
            and set(output_tokens) == {*activation_free, weight_free[0]}
            and len(output_tokens) == len(set(output_tokens))
        ):
            candidates.append(
                (
                    activation_index,
                    weight_index,
                    weight_free[0],
                    activation_free,
                ),
            )
    if len(candidates) > 1:
        candidates = [item for item in candidates if item[0] == 0]
    if len(candidates) != 1:
        raise OrojenesisError(
            "multi-einsum region requires an unambiguous broadcast-weight matmul",
        )
    activation_index, weight_index, n_token, activation_free = candidates[0]
    row_tokens = [token for token in output_tokens if token != n_token]
    if set(row_tokens) != set(activation_free):
        raise OrojenesisError(
            "multi-einsum output does not preserve activation axes",
        )
    if (
        operand_tokens[activation_index][-1] != reduction
        or output_tokens[-1] != n_token
    ):
        raise OrojenesisError(
            "multi-einsum region requires row-major activation/output axes",
        )
    return activation_index, weight_index, reduction, n_token, row_tokens


def _region_dtype(
    semantic: Mapping[str, Any],
    metadata: _RegionTensorMetadata,
    activation_index: int,
    weight_index: int,
) -> str:
    effects = semantic.get("effects") or {}
    if any(
        (
            effects.get("mutates"),
            effects.get("aliases"),
            effects.get("atomic"),
            effects.get("opaque_library_call"),
        ),
    ):
        raise OrojenesisError("multi-einsum region contains observable effects")
    ordered = [
        metadata.input_dtypes[activation_index],
        metadata.input_dtypes[weight_index],
        metadata.output_dtype,
    ]
    if len(set(ordered)) != 1:
        raise OrojenesisError(
            "multi-einsum region requires one exact tensor dtype",
        )
    return ordered[0]


def _region_matmul_descriptor(
    layer_id: str,
    layer: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize rank-2 or broadcast-weight batched matrix multiplication."""
    semantic, equation, operands, output_tokens = _region_einsum_tokens(layer)
    metadata = _region_tensor_metadata(layer, operands, output_tokens)
    token_sizes = _region_token_sizes(operands, output_tokens, metadata)
    (
        activation_index,
        weight_index,
        reduction,
        n_token,
        row_tokens,
    ) = _region_matmul_axes(operands, output_tokens)
    dtype = _region_dtype(
        semantic,
        metadata,
        activation_index,
        weight_index,
    )
    row_shape = [token_sizes[token] for token in row_tokens]
    descriptor = {
        "id": str(layer_id),
        "equation": equation,
        "kind": "batched_matmul" if len(row_tokens) > 1 else "matmul",
        "input": metadata.input_names[activation_index],
        "weight": metadata.input_names[weight_index],
        "output": metadata.output_name,
        "activation_operand": activation_index,
        "weight_operand": weight_index,
        "activation_axes": operands[activation_index],
        "weight_axes": operands[weight_index],
        "output_axes": output_tokens,
        "row_axes": row_tokens,
        "row_shape": row_shape,
        "m": _shape_product(row_shape),
        "k": token_sizes[reduction],
        "n": token_sizes[n_token],
        "dtype": dtype,
    }
    if len(row_tokens) > 1:
        descriptor["batch_axes"] = row_tokens[:-1]
        descriptor["batch_shape"] = row_shape[:-1]
    else:
        descriptor["batch_axes"] = []
        descriptor["batch_shape"] = []
    return descriptor


def multi_einsum_layer_problem(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Build a Timeloop problem for one multi-einsum matmul descriptor."""
    return {
        "problem": {
            "instance": {
                "M": int(descriptor["m"]),
                "K": int(descriptor["k"]),
                "N": int(descriptor["n"]),
            },
            "shape": {
                "data-spaces": [
                    {"name": "Weights", "projection": [[["K"]], [["N"]]]},
                    {"name": "Inputs", "projection": [[["M"]], [["K"]]]},
                    {
                        "name": "Outputs",
                        "projection": [[["M"]], [["N"]]],
                        "read-write": True,
                    },
                ],
                "dimensions": ["M", "K", "N"],
            },
        },
    }


def find_multi_einsum_chains(
    layers: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    """Find complete endpoint-to-endpoint chains supported by tiled fusion."""
    producers = {
        str(name): str(layer_id)
        for layer_id, layer in layers.items()
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    for layer_id, layer in layers.items():
        for name in (layer.get("tensor_names") or {}).get("inputs") or []:
            consumers[str(name)].append(str(layer_id))
    einsums: dict[str, dict[str, Any]] = {}
    for layer_id, layer in layers.items():
        try:
            einsums[str(layer_id)] = _matmul_descriptor(str(layer_id), layer)
        except OrojenesisError:
            continue

    successor: dict[str, str] = {}
    predecessor: dict[str, str] = {}
    for layer_id, descriptor in einsums.items():
        output = str(descriptor["output"])
        output_consumers = consumers.get(output) or []
        if len(output_consumers) != 1 or output_consumers[0] not in einsums:
            continue
        consumer_id = output_consumers[0]
        if einsums[consumer_id]["input"] != output:
            continue
        successor[layer_id] = consumer_id
        predecessor[consumer_id] = layer_id

    chains: list[list[str]] = []
    for start in sorted(einsums):
        if start in predecessor or start not in successor:
            continue
        chain = [start]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        if len(chain) < 2:
            continue
        # Restrict dropped intermediate traffic to complete graph endpoints
        # and graph-input weights with a producer-consumer witness.
        first = einsums[chain[0]]
        last = einsums[chain[-1]]
        external_names = [
            first["input"],
            *(einsums[item]["weight"] for item in chain),
        ]
        if any(
            str(
                layers.get(producers.get(str(name), ""), {}).get("type", ""),
            ).lower()
            != "start"
            for name in external_names
        ):
            continue
        if consumers.get(str(last["output"])):
            continue
        try:
            multi_einsum_problem([(item, layers[item]) for item in chain])
        except OrojenesisError:
            continue
        chains.append(chain)
    return chains
