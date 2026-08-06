"""Timeloop problem and mapper-configuration builders."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod

from solar.analysis.orojenesis.configuration import (
    orojenesis_mapper_thread_count,
)
from solar.analysis.orojenesis.errors import OrojenesisError
from solar.ir.contracts import layer_operation
from solar.types import DynamicValue

_TOKEN = re.compile(r"[A-Za-z][0-9]*")
_AXIS = re.compile(r"\(([^()]*)\)|([A-Za-z][0-9]*)")
_DIRECT_CONVOLUTION_EQUATIONS = {
    "conv1d": frozenset({"BC(P+R),OCR->BOP", "BO(P+R),OCR->BOP"}),
    "conv2d": frozenset(
        {
            "BC(P+R)(Q+S),OCRS->BOPQ",
            "BO(P+R)(Q+S),OCRS->BOPQ",
        },
    ),
    "conv3d": frozenset({"BC(P+T)(Q+R)(U+S),OCTRS->BOPQU"}),
}


def compulsory_witness_streaming_dimension(
    layer: Mapping[str, DynamicValue],
    dimensions: list[str],
    *,
    capacity_bytes: int,
    word_bits: int,
) -> str | None:
    """Return an independently streamable batch dimension when proven."""
    semantic = layer.get("semantic_op") or {}
    proof_source = semantic.get("proof_source") or {}
    target = str(proof_source.get("target") or "")
    equation = str(semantic.get("equation") or "")
    effects = semantic.get("effects") or {}
    source_kind = str(proof_source.get("kind") or "")
    aten_pure = bool(
        source_kind == "aten"
        and effects.get("mutates") in (False, [])
        and not effects.get("aliases")
        and not effects.get("atomic")
        and not effects.get("opaque_library_call")
    )
    extended_native_pure = bool(
        source_kind == "extended_native"
        and target in _DIRECT_CONVOLUTION_EQUATIONS
        and effects
        in (
            {},
            {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": True,
            },
        )
    )
    pure = bool((aten_pure or extended_native_pure) and dimensions)
    if not pure:
        return None
    if equation in _DIRECT_CONVOLUTION_EQUATIONS.get(target, ()):
        return dimensions[0]
    if target != "bmm" or equation != "BMK,BKN->BMN":
        return None
    shapes = layer.get("tensor_shapes") or {}
    tensors = [*(shapes.get("inputs") or []), *(shapes.get("outputs") or [])]
    if (
        len(tensors) != 3
        or any(len(shape) != 3 for shape in tensors)
        or int(word_bits) <= 0
        or int(word_bits) % 8
    ):
        return None
    batch_sizes = {int(shape[0]) for shape in tensors}
    if len(batch_sizes) != 1 or next(iter(batch_sizes)) <= 0:
        return None
    slice_words = sum(
        prod(int(size) for size in shape[1:]) for shape in tensors
    )
    return (
        dimensions[0]
        if slice_words * (int(word_bits) // 8) <= int(capacity_bytes)
        else None
    )


@dataclass
class _DimensionRegistry:
    sizes: dict[str, int] = field(default_factory=dict)
    symbols: dict[str, str] = field(default_factory=dict)

    def add(self, token: str, size: DynamicValue) -> None:
        integer_size = int(size)
        if token in self.sizes and self.sizes[token] != integer_size:
            raise OrojenesisError(f"inconsistent dimension {token}")
        self.sizes[token] = integer_size

    def symbol(self, token: str) -> str:
        if token not in self.symbols:
            symbols = string.ascii_uppercase + string.ascii_lowercase
            if len(self.symbols) >= len(symbols):
                raise OrojenesisError(
                    "Orojenesis supports at most 52 distinct dimensions",
                )
            self.symbols[token] = symbols[len(self.symbols)]
        return self.symbols[token]


def problem_for_layer(
    layer: Mapping[str, DynamicValue],
) -> dict[str, DynamicValue]:
    """Translate one exact einsum layer into a Timeloop problem."""
    semantic = layer_operation(layer)
    if semantic.get("kind") != "einsum":
        raise OrojenesisError("Orojenesis accepts exact einsum layers only")
    left_hand_side, right_hand_side = str(
        semantic.get("equation", ""),
    ).split("->", 1)
    operands = left_hand_side.split(",")
    shapes = (layer.get("tensor_shapes") or {}).get("inputs") or []
    output_shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    if len(operands) != len(shapes) or len(output_shapes) != 1:
        raise OrojenesisError(
            "einsum operand arity does not match tensor metadata",
        )

    input_axes = [_operand_axes(operand) for operand in operands]
    output_axes = _operand_axes(right_hand_side)
    _validate_ranks(input_axes, shapes, output_axes, output_shapes[0])
    registry = _DimensionRegistry()
    for axes, shape in zip(input_axes, shapes, strict=True):
        _register_simple_dimensions(axes, shape, registry)
    _register_simple_dimensions(output_axes, output_shapes[0], registry)
    for token in registry.sizes:
        registry.symbol(token)
    data_spaces = [
        _input_data_space(index, axes, shape, registry)
        for index, (axes, shape) in enumerate(
            zip(input_axes, shapes, strict=True),
        )
    ]
    data_spaces.append(
        _output_data_space(output_axes, output_shapes[0], registry),
    )
    remapped_sizes = {
        registry.symbols[token]: size for token, size in registry.sizes.items()
    }
    return {
        "problem": {
            "instance": remapped_sizes,
            "shape": {
                "data-spaces": data_spaces,
                "dimensions": list(remapped_sizes),
            },
        },
    }


def _operand_axes(operand: str) -> list[tuple[str, ...]]:
    axes: list[tuple[str, ...]] = []
    position = 0
    for match in _AXIS.finditer(operand):
        if match.start() != position:
            raise OrojenesisError("unsupported einsum projection syntax")
        expression = match.group(1)
        tokens = (
            tuple(expression.split("+"))
            if expression is not None
            else (str(match.group(2)),)
        )
        if not tokens or any(
            _TOKEN.fullmatch(token) is None for token in tokens
        ):
            raise OrojenesisError("unsupported einsum projection syntax")
        axes.append(tokens)
        position = match.end()
    if position != len(operand):
        raise OrojenesisError("unsupported einsum projection syntax")
    return axes


def _validate_ranks(
    input_axes: Sequence[Sequence[tuple[str, ...]]],
    input_shapes: Sequence[Sequence[DynamicValue]],
    output_axes: Sequence[tuple[str, ...]],
    output_shape: Sequence[DynamicValue],
) -> None:
    if any(
        len(axes) != len(shape)
        for axes, shape in zip(input_axes, input_shapes, strict=True)
    ):
        raise OrojenesisError("einsum rank does not match input shape")
    if len(output_axes) != len(output_shape):
        raise OrojenesisError("einsum rank does not match output shape")


def _register_simple_dimensions(
    axes: Sequence[tuple[str, ...]],
    shape: Sequence[DynamicValue],
    registry: _DimensionRegistry,
) -> None:
    for tokens, size in zip(axes, shape, strict=True):
        if len(tokens) == 1:
            registry.add(tokens[0], size)


def _axis_projection(
    tokens: tuple[str, ...],
    size: DynamicValue,
    registry: _DimensionRegistry,
) -> list[list[str]]:
    if any(token not in registry.sizes for token in tokens):
        raise OrojenesisError("projected einsum dimension has no exact size")
    projected_size = 1 + sum(registry.sizes[token] - 1 for token in tokens)
    if projected_size != int(size):
        raise OrojenesisError("projected einsum dimension is inconsistent")
    return [[registry.symbol(token)] for token in tokens]


def _input_data_space(
    index: int,
    axes: Sequence[tuple[str, ...]],
    shape: Sequence[DynamicValue],
    registry: _DimensionRegistry,
) -> dict[str, DynamicValue]:
    return {
        "name": f"Input{index}",
        "projection": [
            _axis_projection(tokens, size, registry)
            for tokens, size in zip(axes, shape, strict=True)
        ],
    }


def _output_data_space(
    axes: Sequence[tuple[str, ...]],
    shape: Sequence[DynamicValue],
    registry: _DimensionRegistry,
) -> dict[str, DynamicValue]:
    return {
        "name": "Output",
        "projection": [
            _axis_projection(tokens, size, registry)
            for tokens, size in zip(axes, shape, strict=True)
        ],
        "read-write": True,
    }


def architecture(
    word_bits: int,
    *,
    buffer_capacity_bytes: int | None = None,
) -> dict[str, DynamicValue]:
    """Build the generic Orojenesis memory architecture."""
    return {
        "architecture": {
            "version": 0.2,
            "subtree": [
                {
                    "name": "System",
                    "local": [_main_memory(word_bits)],
                    "subtree": [
                        {
                            "name": "PE",
                            "local": [
                                _buffer(
                                    "Buffer",
                                    word_bits,
                                    capacity_bytes=buffer_capacity_bytes,
                                ),
                                _macc(word_bits),
                            ],
                        },
                    ],
                },
            ],
        },
    }


def multi_architecture(word_bits: int) -> dict[str, DynamicValue]:
    """Return the official two-buffer abstraction used for tiled fusion."""
    return {
        "architecture": {
            "version": 0.2,
            "subtree": [
                {
                    "name": "System",
                    "local": [_main_memory(word_bits)],
                    "subtree": [
                        {
                            "name": "PE",
                            "local": [
                                _buffer("InputOutputBuffer", word_bits),
                                _buffer("WeightBuffer", word_bits),
                                _macc(word_bits),
                            ],
                        },
                    ],
                },
            ],
        },
    }


def _main_memory(word_bits: int) -> dict[str, DynamicValue]:
    return {
        "name": "MainMemory",
        "class": "DRAM",
        "attributes": {"width": 64, "word-bits": int(word_bits)},
    }


def _buffer(
    name: str,
    word_bits: int,
    *,
    capacity_bytes: int | None = None,
) -> dict[str, DynamicValue]:
    if capacity_bytes is not None and (
        int(capacity_bytes) <= 0 or int(capacity_bytes) % 1024
    ):
        raise OrojenesisError(
            "witness buffer capacity must be a positive whole number of KiB",
        )
    return {
        "name": name,
        "class": "regfile",
        "attributes": {
            "sizeKB": (
                int(capacity_bytes) // 1024
                if capacity_bytes is not None
                else 2147483648
            ),
            "instances": 1,
            "word-bits": int(word_bits),
        },
    }


def _macc(word_bits: int) -> dict[str, DynamicValue]:
    return {
        "name": "MACC",
        "class": "intmac",
        "attributes": {"datawidth": int(word_bits)},
    }


def multi_mapper_config(
    row_tile: int,
    *,
    role: str,
) -> dict[str, DynamicValue]:
    """Build a fusion-friendly mapping sweep for a linear matmul chain."""
    if int(row_tile) <= 0:
        raise OrojenesisError("multi-einsum row tile must be positive")
    main_temporal = _main_temporal_constraint(role)
    return {
        "mapper": _mapper_policy(),
        "mapspace_constraints": [
            {
                "target": "MainMemory",
                "type": "datatype",
                "keep": ["Weights", "Inputs", "Outputs"],
                "bypass": [],
            },
            {
                "target": "InputOutputBuffer",
                "type": "datatype",
                "keep": ["Inputs", "Outputs"],
                "bypass": ["Weights"],
            },
            {
                "target": "WeightBuffer",
                "type": "datatype",
                "keep": ["Weights"],
                "bypass": ["Inputs", "Outputs"],
            },
            main_temporal,
            {
                "target": "InputOutputBuffer",
                "type": "temporal",
                "factors": "M=1",
                "permutation": "MNK",
            },
            {
                "target": "WeightBuffer",
                "type": "temporal",
                "factors": f"M={int(row_tile)}",
                "permutation": "MKN",
            },
        ],
    }


def _main_temporal_constraint(role: str) -> dict[str, DynamicValue]:
    constraints = {
        "first": (None, "KNM"),
        "second": ("N=1", "KNM"),
        "middle": ("K=1 N=1", "KNM"),
        "last": ("K=1", "KNM"),
        "second_last": (None, "NKM"),
    }
    if role not in constraints:
        raise OrojenesisError(f"invalid multi-einsum mapper role: {role}")
    factors, permutation = constraints[role]
    result: dict[str, DynamicValue] = {
        "target": "MainMemory",
        "type": "temporal",
        "permutation": permutation,
    }
    if factors is not None:
        result["factors"] = factors
    return result


def mapper_config(
    dimensions: list[str],
    spaces: list[str],
) -> dict[str, DynamicValue]:
    """Build a mapper configuration for the supplied problem dimensions."""
    return {
        "mapper": _mapper_policy(),
        "mapspace_constraints": [
            {
                "target": "Buffer",
                "type": "temporal",
                "permutation": "".join(dimensions),
            },
            {"target": "MainMemory", "type": "temporal"},
            {
                "target": "MainMemory",
                "type": "datatype",
                "keep": spaces,
                "bypass": [],
            },
        ],
    }


def compulsory_witness_mapper_config(
    instance: Mapping[str, DynamicValue],
    dimensions: list[str],
    spaces: list[str],
    *,
    streaming_dimension: str,
) -> dict[str, DynamicValue]:
    """Constrain one mapping that streams an independent outer dimension."""
    if streaming_dimension not in dimensions:
        raise OrojenesisError("streaming dimension is not in the problem")
    inner_factors = " ".join(
        f"{dimension}={1 if dimension == streaming_dimension else int(instance[dimension])}"
        for dimension in dimensions
    )
    outer_factors = " ".join(
        f"{dimension}={int(instance[dimension]) if dimension == streaming_dimension else 1}"
        for dimension in dimensions
    )
    permutation = "".join(dimensions)
    return {
        "mapper": _mapper_policy(),
        "mapspace_constraints": [
            {
                "target": "Buffer",
                "type": "datatype",
                "keep": spaces,
                "bypass": [],
            },
            {
                "target": "Buffer",
                "type": "temporal",
                "factors": inner_factors,
                "permutation": permutation,
            },
            {
                "target": "MainMemory",
                "type": "temporal",
                "factors": outer_factors,
                "permutation": permutation,
            },
            {
                "target": "MainMemory",
                "type": "datatype",
                "keep": spaces,
                "bypass": [],
            },
        ],
    }


def _mapper_policy() -> dict[str, DynamicValue]:
    return {
        "optimization-metrics": ["last-level-accesses"],
        "algorithm": "linear-pruned",
        "victory-condition": 0,
        "timeout": 0,
        "log-oaves": True,
        "num-threads": orojenesis_mapper_thread_count(),
        "log-oaves-mappings": False,
    }
