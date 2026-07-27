"""Timeloop problem and mapper-configuration builders."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from solar.analysis.orojenesis_common import OrojenesisError
from solar.common.types import DynamicValue
from solar.ir.contracts import layer_operation

_TOKEN = re.compile(r"[A-Za-z][0-9]*")


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

    registry = _DimensionRegistry()
    data_spaces = [
        _input_data_space(index, operand, shape, registry)
        for index, (operand, shape) in enumerate(
            zip(operands, shapes, strict=True),
        )
    ]
    data_spaces.append(
        _output_data_space(right_hand_side, output_shapes[0], registry),
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


def _input_data_space(
    index: int,
    operand: str,
    shape: Sequence[DynamicValue],
    registry: _DimensionRegistry,
) -> dict[str, DynamicValue]:
    tokens = _TOKEN.findall(operand)
    if len(tokens) != len(shape):
        raise OrojenesisError("einsum rank does not match input shape")
    for token, size in zip(tokens, shape, strict=True):
        registry.add(token, size)
    return {
        "name": f"Input{index}",
        "projection": [[[registry.symbol(token)]] for token in tokens],
    }


def _output_data_space(
    operand: str,
    shape: Sequence[DynamicValue],
    registry: _DimensionRegistry,
) -> dict[str, DynamicValue]:
    tokens = _TOKEN.findall(operand)
    if len(tokens) != len(shape):
        raise OrojenesisError("einsum rank does not match output shape")
    for token, size in zip(tokens, shape, strict=True):
        registry.add(token, size)
    return {
        "name": "Output",
        "projection": [[[registry.symbol(token)]] for token in tokens],
        "read-write": True,
    }


def architecture(word_bits: int) -> dict[str, DynamicValue]:
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
                                _buffer("Buffer", word_bits),
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


def _buffer(name: str, word_bits: int) -> dict[str, DynamicValue]:
    return {
        "name": name,
        "class": "regfile",
        "attributes": {
            "sizeKB": 2147483648,
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


def _mapper_policy() -> dict[str, DynamicValue]:
    return {
        "optimization-metrics": ["last-level-accesses"],
        "algorithm": "linear-pruned",
        "victory-condition": 0,
        "timeout": 0,
        "log-oaves": True,
        "num-threads": 8,
        "log-oaves-mappings": False,
    }
