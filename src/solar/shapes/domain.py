# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOLAR-owned shape-domain constraint AST and stage-1 validator.

This module is the L3-b stage-1 contract: an integer-affine constraint AST
(symbols, ranges, divisibility, derived dimensions, assumptions, interval
dimensions, and provenance source bindings) plus membership validation for
concrete instances. It deliberately owns its vocabulary: upstream
``ShapesSpec``/``SymBool`` representations are inputs to study, never an
executable protocol here. Domain validation prices nothing as tensor work;
tensor accounting stays entirely with the existing static-graph analyzer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from solar.artifacts import stable_json_checksum
from solar.types import GraphValue


class ShapeReasonCode(StrEnum):
    """Stable verdict vocabulary for shape-domain validation."""

    VALIDATED = "validated"
    OUT_OF_RANGE = "out_of_range"
    DIVISIBILITY_VIOLATED = "divisibility_violated"
    ASSUMPTION_VIOLATED = "assumption_violated"
    SHAPE_MISMATCH = "shape_mismatch"
    INTERVAL_EXCEEDED = "interval_exceeded"
    UNKNOWN_SYMBOL = "unknown_symbol"
    SHAPE_BRANCH_REJECTED = "shape_branch_rejected"


class AssumptionOp(StrEnum):
    """Relational predicates between affine dimension expressions."""

    LE = "<="
    EQ = "=="


class TensorRole(StrEnum):
    """Which side of the reference boundary a bound tensor sits on."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeSymbol:
    """One integer dimension symbol with range, divisibility, provenance."""

    name: str
    lower: int
    upper: int
    divisibility: int = 1
    provenance: str = ""

    def check(self, value: int) -> ShapeReasonCode | None:
        """Return the violated constraint, or None when in range."""
        if not self.lower <= value <= self.upper:
            return ShapeReasonCode.OUT_OF_RANGE
        if self.divisibility > 1 and value % self.divisibility:
            return ShapeReasonCode.DIVISIBILITY_VIOLATED
        return None

    def to_dict(self) -> dict[str, GraphValue]:
        """Return the canonical serialization of this symbol."""
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "divisibility": self.divisibility,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DimExpr:
    """An integer affine expression over named symbols."""

    terms: tuple[tuple[str, int], ...] = ()
    constant: int = 0

    @classmethod
    def symbol(cls, name: str) -> DimExpr:
        """Bind one symbol with unit coefficient."""
        return cls(terms=((name, 1),))

    @classmethod
    def fixed(cls, value: int) -> DimExpr:
        """Return one fixed integer dimension."""
        return cls(constant=value)

    def scaled(self, coefficient: int) -> DimExpr:
        """Multiply the whole expression by one integer coefficient."""
        return DimExpr(
            terms=tuple(
                (name, value * coefficient) for name, value in self.terms
            ),
            constant=self.constant * coefficient,
        )

    @property
    def symbols(self) -> frozenset[str]:
        """Return every symbol this expression reads."""
        return frozenset(name for name, _ in self.terms)

    def evaluate(self, assignment: Mapping[str, int]) -> int | None:
        """Evaluate under an assignment; None when a symbol is unbound."""
        total = self.constant
        for name, coefficient in self.terms:
            if (value := assignment.get(name)) is None:
                return None
            total += coefficient * value
        return total

    def to_dict(self) -> dict[str, GraphValue]:
        """Return the canonical serialization of this expression."""
        return {
            "terms": [
                {"symbol": name, "coefficient": c} for name, c in self.terms
            ],
            "constant": self.constant,
        }


def _expr_from_dict(payload: Mapping[str, GraphValue]) -> DimExpr:
    return DimExpr(
        terms=tuple(
            (item["symbol"], int(item["coefficient"]))
            for item in payload.get("terms", ())
        ),
        constant=int(payload.get("constant", 0)),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Assumption:
    """A relational constraint between affine expressions."""

    left: DimExpr
    op: AssumptionOp
    right: DimExpr

    def holds(self, assignment: Mapping[str, int]) -> bool:
        """Evaluate both sides under a complete assignment."""
        left = self.left.evaluate(assignment)
        right = self.right.evaluate(assignment)
        if left is None or right is None:
            return False
        return left <= right if self.op is AssumptionOp.LE else left == right

    def to_dict(self) -> dict[str, GraphValue]:
        """Return the canonical serialization of this assumption."""
        return {
            "left": self.left.to_dict(),
            "op": self.op.value,
            "right": self.right.to_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class IntervalDim:
    """A dimension known only between affine bounds (nonzero-style)."""

    symbol: str
    lower: DimExpr
    upper: DimExpr
    provenance: str = ""

    def check(
        self, value: int, assignment: Mapping[str, int]
    ) -> ShapeReasonCode | None:
        """Return INTERVAL_EXCEEDED when outside the evaluated bounds."""
        lower = self.lower.evaluate(assignment)
        upper = self.upper.evaluate(assignment)
        if lower is None or upper is None or not lower <= value <= upper:
            return ShapeReasonCode.INTERVAL_EXCEEDED
        return None

    def to_dict(self) -> dict[str, GraphValue]:
        """Return the canonical serialization of this interval dimension."""
        return {
            "symbol": self.symbol,
            "lower": self.lower.to_dict(),
            "upper": self.upper.to_dict(),
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorBinding:
    """Per-axis dimension expressions for one named tensor."""

    tensor: str
    role: TensorRole
    dims: tuple[DimExpr | int, ...]

    def to_dict(self) -> dict[str, GraphValue]:
        """Return the canonical serialization of this tensor binding."""
        return {
            "tensor": self.tensor,
            "role": self.role.value,
            "dims": [
                dim if isinstance(dim, int) else dim.to_dict()
                for dim in self.dims
            ],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeVerdict:
    """One validation outcome with a stable reason code."""

    status: ShapeReasonCode
    detail: str = ""
    assignment: dict[str, int] | None = None

    @property
    def validated(self) -> bool:
        """Return whether this verdict accepted the instance."""
        return self.status is ShapeReasonCode.VALIDATED


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeDomain:
    """A complete stage-1 shape-domain declaration."""

    domain_id: str
    symbols: Mapping[str, ShapeSymbol]
    tensors: Mapping[str, TensorBinding]
    assumptions: tuple[Assumption, ...] = ()
    intervals: tuple[IntervalDim, ...] = ()

    def stable_identity(self) -> str:
        """Return the canonical checksum of this domain declaration."""
        return stable_json_checksum(self.to_dict())

    def to_dict(self) -> dict[str, GraphValue]:
        """Return the canonical serialization of the whole domain."""
        return {
            "domain_id": self.domain_id,
            "symbols": {
                name: symbol.to_dict()
                for name, symbol in sorted(self.symbols.items())
            },
            "tensors": {
                name: binding.to_dict()
                for name, binding in sorted(self.tensors.items())
            },
            "assumptions": [item.to_dict() for item in self.assumptions],
            "intervals": [item.to_dict() for item in self.intervals],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, GraphValue]) -> ShapeDomain:
        """Rebuild one domain from its canonical serialization."""
        symbols = {
            name: ShapeSymbol(
                name=item["name"],
                lower=int(item["lower"]),
                upper=int(item["upper"]),
                divisibility=int(item.get("divisibility", 1)),
                provenance=str(item.get("provenance", "")),
            )
            for name, item in payload["symbols"].items()
        }
        tensors = {
            name: TensorBinding(
                tensor=item["tensor"],
                role=TensorRole(item["role"]),
                dims=tuple(
                    int(dim) if isinstance(dim, int) else _expr_from_dict(dim)
                    for dim in item["dims"]
                ),
            )
            for name, item in payload["tensors"].items()
        }
        assumptions = tuple(
            Assumption(
                left=_expr_from_dict(item["left"]),
                op=AssumptionOp(item["op"]),
                right=_expr_from_dict(item["right"]),
            )
            for item in payload.get("assumptions", ())
        )
        intervals = tuple(
            IntervalDim(
                symbol=item["symbol"],
                lower=_expr_from_dict(item["lower"]),
                upper=_expr_from_dict(item["upper"]),
                provenance=str(item.get("provenance", "")),
            )
            for item in payload.get("intervals", ())
        )
        return cls(
            domain_id=str(payload["domain_id"]),
            symbols=symbols,
            tensors=tensors,
            assumptions=assumptions,
            intervals=intervals,
        )

    def validate_assignment(
        self, assignment: Mapping[str, int]
    ) -> ShapeVerdict:
        """Check one concrete symbol assignment against every constraint."""
        expected = set(self.symbols) | {item.symbol for item in self.intervals}
        if set(assignment) != expected:
            return ShapeVerdict(
                status=ShapeReasonCode.UNKNOWN_SYMBOL,
                detail=f"assignment keys {sorted(set(assignment))} != {sorted(expected)}",
            )
        for name, value in sorted(assignment.items()):
            symbol = self.symbols.get(name)
            if symbol is None:
                continue
            violation = symbol.check(value)
            if violation is not None:
                return ShapeVerdict(status=violation, detail=f"{name}={value}")
        for assumption in self.assumptions:
            if not assumption.holds(assignment):
                return ShapeVerdict(
                    status=ShapeReasonCode.ASSUMPTION_VIOLATED,
                    detail=f"{assumption.left.to_dict()} {assumption.op.value} "
                    f"{assumption.right.to_dict()} under {dict(assignment)}",
                )
        for interval in self.intervals:
            violation = interval.check(assignment[interval.symbol], assignment)
            if violation is not None:
                return ShapeVerdict(
                    status=violation,
                    detail=f"{interval.symbol}={assignment[interval.symbol]}",
                )
        return ShapeVerdict(
            status=ShapeReasonCode.VALIDATED, assignment=dict(assignment)
        )


def bind_tensors(
    domain: ShapeDomain, shapes: Mapping[str, Sequence[int]]
) -> ShapeVerdict:
    """Solve the symbol assignment from concrete tensor shapes.

    Stage-1 solver: fixed dims must match exactly; an affine dim may
    introduce at most one new symbol and must verify once all symbols are
    bound. Anything else is a shape mismatch, not a guess.
    """
    solved: dict[str, int] = {}
    for name, binding in sorted(domain.tensors.items()):
        concrete = shapes.get(name)
        if concrete is None or len(concrete) != len(binding.dims):
            return ShapeVerdict(
                status=ShapeReasonCode.SHAPE_MISMATCH,
                detail=f"tensor {name}: arity"
                f" {None if concrete is None else len(concrete)}"
                f" != {len(binding.dims)}",
            )
        for axis, (expr, value) in enumerate(
            zip(binding.dims, concrete, strict=True)
        ):
            if isinstance(expr, int):
                if expr != value:
                    return ShapeVerdict(
                        status=ShapeReasonCode.SHAPE_MISMATCH,
                        detail=f"tensor {name} axis {axis}: fixed {expr} != {value}",
                    )
                continue
            unsolved = [n for n, _ in expr.terms if n not in solved]
            if not unsolved:
                if expr.evaluate(solved) != value:
                    return ShapeVerdict(
                        status=ShapeReasonCode.SHAPE_MISMATCH,
                        detail=f"tensor {name} axis {axis}: {expr.to_dict()}"
                        f" != {value}",
                    )
                continue
            if len(unsolved) > 1:
                return ShapeVerdict(
                    status=ShapeReasonCode.SHAPE_MISMATCH,
                    detail=f"tensor {name} axis {axis}: stage-1 solver needs all"
                    f" but one symbol bound, unsolved {unsolved}",
                )
            symbol = unsolved[0]
            coefficient = dict(expr.terms)[symbol]
            others = expr.constant + sum(
                c * solved[n] for n, c in expr.terms if n != symbol
            )
            remainder = value - others
            if coefficient == 0 or remainder % coefficient:
                return ShapeVerdict(
                    status=ShapeReasonCode.SHAPE_MISMATCH,
                    detail=f"tensor {name} axis {axis}: {expr.to_dict()} does not"
                    f" divide onto {value}",
                )
            solved[symbol] = remainder // coefficient
    return domain.validate_assignment(solved)


def _build_wiring_tables(
    layers: Mapping[str, GraphValue],
) -> tuple[dict[str, int], dict[str, str]]:
    layer_index = {str(name): index for index, name in enumerate(layers)}
    tensor_ids: dict[str, str] = {}
    for index, name in enumerate(layers):
        layer = layers[name]
        if not isinstance(layer, Mapping):
            continue
        outputs = (layer.get("tensor_names") or {}).get("outputs") or []
        for slot, tensor_name in enumerate(outputs):
            tensor_ids.setdefault(str(tensor_name), f"L{index}:O{slot}")
    ext_counter = 0
    for name in layers:
        layer = layers[name]
        if not isinstance(layer, Mapping):
            continue
        inputs = (layer.get("tensor_names") or {}).get("inputs") or []
        for tensor_name in inputs:
            key = str(tensor_name)
            if key not in tensor_ids:
                tensor_ids[key] = f"E{ext_counter}"
                ext_counter += 1
    return layer_index, tensor_ids


def _normalized_graph_outputs(
    graph: Mapping[str, GraphValue],
    tensor_ids: Mapping[str, str],
) -> list[str] | None:
    raw = graph.get("outputs")
    if raw is None and isinstance(graph.get("graph_signature"), Mapping):
        raw = graph["graph_signature"].get("joint_outputs")
    if not isinstance(raw, list):
        return None
    return [tensor_ids.get(str(item), f"U:{item}") for item in raw]


def structural_projection(
    graph: Mapping[str, GraphValue],
) -> list[dict[str, GraphValue]]:
    """Project one operator graph onto its shape-independent structure."""
    layers = graph.get("layers") or {}
    if not isinstance(layers, Mapping):
        return []
    layer_index, tensor_ids = _build_wiring_tables(layers)
    graph_outputs = _normalized_graph_outputs(graph, tensor_ids)
    projection: list[dict[str, GraphValue]] = []
    for idx, name in enumerate(layers):
        layer = layers[name]
        semantic = layer.get("semantic_op") or {}
        raw_args = semantic.get("arguments", semantic.get("operands", []))
        raw_kwargs = semantic.get("kwargs", semantic.get("attributes", {}))
        connections = layer.get("connections") or {}
        tensor_names = layer.get("tensor_names") or {}
        entry: dict[str, GraphValue] = {
            "type": layer.get("type"),
            "target": semantic.get("target"),
            "exact_target": semantic.get("exact_target"),
            "overload": semantic.get("overload"),
            "effects": semantic.get("effects"),
            "source_input_index": layer.get("source_input_index"),
            "connections": {
                "inputs": [
                    layer_index.get(str(c), f"U:{c}")
                    for c in (connections.get("inputs") or [])
                ],
                "outputs": [
                    layer_index.get(str(c), f"U:{c}")
                    for c in (connections.get("outputs") or [])
                ],
            },
            "tensor_inputs": [
                tensor_ids[str(t)] for t in (tensor_names.get("inputs") or [])
            ],
            "tensor_outputs": [
                tensor_ids[str(t)] for t in (tensor_names.get("outputs") or [])
            ],
            "tensor_arguments": _tensor_argument_slots(raw_args),
            "tensor_kwargs": _tensor_argument_slots(raw_kwargs),
            "constant_arguments": _constant_arguments(raw_args),
            "constant_kwargs": _constant_arguments(raw_kwargs),
        }
        if idx == 0 and graph_outputs is not None:
            entry["graph_outputs"] = graph_outputs
        projection.append(entry)
    return projection


def _tensor_argument_slots(value: GraphValue) -> GraphValue:
    """Keep exact tensor reference slot positions while masking constants."""
    if isinstance(value, Mapping):
        if set(value) == {"tensor"}:
            return {"tensor": value["tensor"]}
        return {
            key: _tensor_argument_slots(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_tensor_argument_slots(entry) for entry in value]
    return "<const>"


def _constant_arguments(value: GraphValue) -> GraphValue:
    """Drop tensor references, keeping every serialized constant argument."""
    if isinstance(value, Mapping):
        if set(value) == {"tensor"}:
            return None
        return {key: _constant_arguments(item) for key, item in value.items()}
    if isinstance(value, list):
        return [
            item
            for item in (_constant_arguments(entry) for entry in value)
            if item is not None
        ]
    return value


_STRUCTURAL_FIELDS = (
    "type",
    "target",
    "exact_target",
    "overload",
    "effects",
    "source_input_index",
    "connections",
    "tensor_inputs",
    "tensor_outputs",
    "tensor_arguments",
    "tensor_kwargs",
    "graph_outputs",
)


def certify_structure(
    projections: Mapping[str, list[dict[str, GraphValue]]],
    assignments: Mapping[str, Mapping[str, int]],
) -> ShapeVerdict:
    """Certify domain-wide structural stability across concrete instances."""
    keys = sorted(projections)
    if not keys:
        return ShapeVerdict(
            status=ShapeReasonCode.VALIDATED, detail="structure_stable"
        )
    reference_key = keys[0]
    reference = projections[reference_key]
    divergences: list[dict[str, GraphValue]] = []
    aligned_keys = [reference_key]
    for key in keys[1:]:
        other = projections[key]
        if len(other) != len(reference):
            divergences.append(
                {
                    "instances": [reference_key, key],
                    "kind": "layer_count",
                    "values": [len(reference), len(other)],
                }
            )
            continue
        aligned_keys.append(key)
        for position, (left, right) in enumerate(
            zip(reference, other, strict=True)
        ):
            for field in _STRUCTURAL_FIELDS:
                if left.get(field) != right.get(field):
                    divergences.append(
                        {
                            "instances": [reference_key, key],
                            "kind": f"layer[{position}].{field}",
                            "values": [left.get(field), right.get(field)],
                        }
                    )
    if len(aligned_keys) > 1:
        _check_global_constant_consistency(
            projections,
            assignments,
            aligned_keys,
            len(reference),
            divergences,
        )
    if divergences:
        return ShapeVerdict(
            status=ShapeReasonCode.SHAPE_BRANCH_REJECTED,
            detail=f"unexplained divergences: {divergences[:16]}",
        )
    return ShapeVerdict(
        status=ShapeReasonCode.VALIDATED, detail="structure_stable"
    )


def _check_global_constant_consistency(
    projections: Mapping[str, list[dict[str, GraphValue]]],
    assignments: Mapping[str, Mapping[str, int]],
    keys: Sequence[str],
    layer_count: int,
    divergences: list[dict[str, GraphValue]],
) -> None:
    shared_symbols = sorted(
        set.intersection(*(set(assignments.get(k, {})) for k in keys))
    )
    for position in range(layer_count):
        for field in ("constant_arguments", "constant_kwargs"):
            values = [projections[k][position].get(field) for k in keys]
            if all(val == values[0] for val in values[1:]):
                continue
            if (
                _globally_explained_by_symbol(
                    values, keys, assignments, shared_symbols
                )
                is None
            ):
                divergences.append(
                    {
                        "instances": list(keys),
                        "kind": f"layer[{position}].{field}",
                        "values": values,
                    }
                )


def _globally_explained_by_symbol(
    values: Sequence[GraphValue],
    keys: Sequence[str],
    assignments: Mapping[str, Mapping[str, int]],
    shared_symbols: Sequence[str],
) -> str | None:
    """Require one shared symbol to explain every instance simultaneously."""
    for symbol in shared_symbols:
        marked0 = _mark(values[0], assignments[keys[0]][symbol])
        if all(
            _mark(val, assignments[key][symbol]) == marked0
            for key, val in zip(keys[1:], values[1:], strict=True)
        ):
            return symbol
    return None


def _mark(value: GraphValue, expected: int) -> GraphValue:
    """Replace integer leaves equal to one symbol value with a marker."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return "<symbol>" if value == expected else value
    if isinstance(value, list):
        return [_mark(item, expected) for item in value]
    if isinstance(value, Mapping):
        return {key: _mark(item, expected) for key, item in value.items()}
    return value
