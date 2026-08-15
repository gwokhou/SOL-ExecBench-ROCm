"""Typed parsing rules for Torchview function attributes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from solar.types import DynamicValue

type ParsedAttributes = dict[str, DynamicValue]
type ArgumentRule = Callable[
    [list[DynamicValue], Mapping[str, DynamicValue]],
    ParsedAttributes,
]

_TORCH_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "float8_e4m3fn",
        "float8_e5m2",
    }
)
_REDUCTION_OPERATIONS = frozenset(
    {
        "mean",
        "sum",
        "logsumexp",
        "prod",
        "amax",
        "amin",
        "any",
        "all",
        "norm",
        "std",
        "var",
    }
)


@dataclass(slots=True, kw_only=True)
class SemanticArgumentEncoder:
    """Encode ordered Torchview arguments into executable semantic values."""

    tensor_index: int = 0

    def encode(self, value: DynamicValue) -> DynamicValue:
        """Recursively encode one raw argument."""
        match value:
            case dict() as mapping if mapping.get("tensor_placeholder"):
                reference = {"tensor": self.tensor_index}
                self.tensor_index += 1
                return reference
            case dict() as mapping:
                return {
                    str(key): self.encode(item) for key, item in mapping.items()
                }
            case (list() | tuple()) as items:
                return [self.encode(item) for item in items]
            case str() as text if text.startswith("__torch_"):
                name = text.removeprefix("__torch_").removesuffix("__")
                if name in _TORCH_DTYPES:
                    return {"dtype": name}
                return {"value": name}
            case _:
                return {"value": value}


def parse_operation_attributes(
    node_name: str,
    arguments: Sequence[DynamicValue],
    kwargs: Mapping[str, DynamicValue],
) -> ParsedAttributes:
    """Return exact semantic arguments plus operation-specific diagnostics."""
    encoder = SemanticArgumentEncoder()
    result: ParsedAttributes = {
        "call_arguments": [encoder.encode(item) for item in arguments],
        "call_kwargs": {
            str(key): encoder.encode(value) for key, value in kwargs.items()
        },
    }
    scalar_arguments = [
        argument
        for argument in arguments
        if not (isinstance(argument, dict) and "tensor_placeholder" in argument)
    ]
    rule = _argument_rule(node_name)
    if rule is not None:
        result.update(rule(scalar_arguments, kwargs))
    return result


def _argument_rule(node_name: str) -> ArgumentRule | None:
    if node_name in _REDUCTION_OPERATIONS:
        return _reduction_arguments
    return {
        "transpose": _transpose_arguments,
        "permute": _permute_arguments,
        "t": _matrix_transpose_arguments,
        "view": _shape_arguments,
        "reshape": _shape_arguments,
    }.get(node_name)


def _transpose_arguments(
    arguments: list[DynamicValue],
    kwargs: Mapping[str, DynamicValue],
) -> ParsedAttributes:
    result: ParsedAttributes = {}
    dimensions = [item for item in arguments if isinstance(item, int)]
    if len(dimensions) >= 2:
        result.update(dim0=dimensions[0], dim1=dimensions[1])
    for name in ("dim0", "dim1"):
        if name in kwargs:
            result[name] = kwargs[name]
    if "dim0" in result and "dim1" in result:
        result["transpose_dims"] = [result["dim0"], result["dim1"]]
    return result


def _permute_arguments(
    arguments: list[DynamicValue],
    kwargs: Mapping[str, DynamicValue],
) -> ParsedAttributes:
    dimensions = [item for item in arguments if isinstance(item, int)]
    sequence = _first_integer_sequence(arguments)
    if sequence is not None:
        dimensions = sequence
    if "dims" in kwargs:
        dimensions = list(kwargs["dims"])
    return {"permute_dims": dimensions} if dimensions else {}


def _matrix_transpose_arguments(
    arguments: list[DynamicValue],
    kwargs: Mapping[str, DynamicValue],
) -> ParsedAttributes:
    del arguments, kwargs
    return {"dim0": 0, "dim1": 1, "transpose_dims": [1, 0]}


def _shape_arguments(
    arguments: list[DynamicValue],
    kwargs: Mapping[str, DynamicValue],
) -> ParsedAttributes:
    del kwargs
    shape = [item for item in arguments if isinstance(item, int)]
    sequence = _first_integer_sequence(arguments)
    if sequence is not None:
        shape = sequence
    return {"target_shape": shape} if shape else {}


def _reduction_arguments(
    arguments: list[DynamicValue],
    kwargs: Mapping[str, DynamicValue],
) -> ParsedAttributes:
    result: ParsedAttributes = {}
    if "dim" in kwargs:
        dimension = kwargs["dim"]
        result["dim"] = (
            [dimension] if isinstance(dimension, int) else list(dimension)
        )
    if "keepdim" in kwargs:
        result["keepdim"] = kwargs["keepdim"]
    if "dim" not in result:
        dimension = _first_dimension(arguments)
        if dimension is not None:
            result["dim"] = dimension
    return result


def _first_dimension(
    arguments: list[DynamicValue],
) -> list[int] | None:
    for argument in arguments:
        if isinstance(argument, int):
            return [argument]
        if isinstance(argument, (list, tuple)) and all(
            isinstance(item, int) for item in argument
        ):
            return list(argument)
    return None


def _first_integer_sequence(
    arguments: list[DynamicValue],
) -> list[int] | None:
    for argument in arguments:
        if isinstance(argument, (list, tuple)) and all(
            isinstance(item, int) for item in argument
        ):
            return list(argument)
    return None


__all__ = ["SemanticArgumentEncoder", "parse_operation_attributes"]
