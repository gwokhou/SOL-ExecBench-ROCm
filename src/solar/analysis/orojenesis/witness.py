# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Selected-capacity compulsory witness validation for Orojenesis runs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from math import isfinite, prod
from pathlib import Path
from typing import Protocol

from solar.analysis.orojenesis.errors import OrojenesisError
from solar.artifacts import sha256_file
from solar.types import DynamicValue

WITNESS_OUTPUT_FILES = (
    "timeloop-mapper.map+stats.xml",
    "timeloop-mapper.map.txt",
    "timeloop-mapper.stats.txt",
    "timeloop-mapper.map.yaml",
)


class WitnessPlanView(Protocol):
    """Read-only view of a single-layer compulsory-witness run plan."""

    @property
    def output(self) -> Path:
        """Return the witness output directory."""
        ...

    @property
    def spaces(self) -> list[str]:
        """Return ordered Timeloop data-space names."""
        ...

    @property
    def streaming_dimension(self) -> str | None:
        """Return the selected streaming dimension token, if any."""
        ...

    @property
    def environment(self) -> dict[str, str] | None:
        """Return environment overrides passed to the mapper."""
        ...

    @property
    def selected_capacity_bytes(self) -> int | None:
        """Return the cache capacity bound in bytes."""
        ...


def exact_nonnegative_integer(value: DynamicValue, *, field: str) -> int:
    """Parse ``value`` as a finite nonnegative integer."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OrojenesisError(f"{field} is not numeric") from exc
    if not isfinite(number) or number < 0 or not number.is_integer():
        raise OrojenesisError(f"{field} must be a finite nonnegative integer")
    return int(number)


def _xml_vector(
    stats: ET.Element,
    name: str,
    spaces: Sequence[str],
) -> dict[str, int]:
    items = stats.findall(f"./{name}/PerDataSpace/item")
    if len(items) != len(spaces):
        raise OrojenesisError(f"mapper XML has invalid {name} arity")
    return {
        space: exact_nonnegative_integer(item.text, field=f"{name}.{space}")
        for space, item in zip(spaces, items, strict=True)
    }


def witness_level_stats(
    path: Path,
    spaces: Sequence[str],
    *,
    word_bits: int,
) -> dict[str, dict[str, dict[str, int]]]:
    """Extract Buffer and MainMemory statistics from mapper XML output."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise OrojenesisError("invalid mapper map+stats XML") from exc
    result: dict[str, dict[str, dict[str, int]]] = {}
    for level in root.findall(".//levels_/item/px"):
        name = level.findtext("./specs_/LevelSpecs/level_name")
        if name not in {"Buffer", "MainMemory"}:
            continue
        stats = level.find("./stats_")
        if stats is None:
            raise OrojenesisError(f"mapper XML omits {name} stats")
        xml_word_bits = exact_nonnegative_integer(
            level.findtext("./specs_/word_bits/t_"),
            field=f"{name}.word_bits",
        )
        if xml_word_bits != int(word_bits):
            raise OrojenesisError(f"mapper XML {name} word width mismatch")
        result[name] = {
            metric: _xml_vector(stats, metric, spaces)
            for metric in (
                "keep",
                "utilized_capacity",
                "utilized_instances",
                "reads",
                "updates",
                "fills",
            )
        }
    if set(result) != {"Buffer", "MainMemory"}:
        raise OrojenesisError("mapper XML omits witness memory levels")
    return result


def _tensor_element_counts(layer: Mapping[str, DynamicValue]) -> list[int]:
    shapes = layer.get("tensor_shapes") or {}
    ordered = [*(shapes.get("inputs") or []), *(shapes.get("outputs") or [])]
    counts = [prod(int(dimension) for dimension in shape) for shape in ordered]
    if not counts or any(count <= 0 for count in counts):
        raise OrojenesisError("proof tensor shapes must be positive")
    return counts


def audit_compulsory_witness(
    point: Mapping[str, DynamicValue],
    level_stats: Mapping[str, Mapping[str, Mapping[str, int]]],
    spaces: Sequence[str],
    element_counts: Sequence[int],
    *,
    capacity_bytes: int,
    word_bits: int,
) -> dict[str, DynamicValue]:
    """Verify that the witness reached exact compulsory DRAM traffic."""
    if len(spaces) != 3 or len(element_counts) != 3:
        raise OrojenesisError(
            "contraction witness requires two inputs and one output",
        )
    accesses = point.get("data_space_accesses_words") or {}
    expected = dict(zip(spaces, element_counts, strict=True))
    main_memory = level_stats["MainMemory"]
    buffer = level_stats["Buffer"]
    input_spaces = list(spaces[:-1])
    output_space = spaces[-1]
    main_accesses = {
        "reads": {
            space: expected[space] if space in input_spaces else 0
            for space in spaces
        },
        "updates": {
            space: expected[space] if space == output_space else 0
            for space in spaces
        },
        "fills": dict.fromkeys(spaces, 0),
    }
    keep = dict.fromkeys(spaces, 1)
    instances = dict.fromkeys(spaces, 1)
    buffer_words = sum(buffer["utilized_capacity"].values())
    buffer_bytes = (buffer_words * int(word_bits)) / 8.0
    if (
        buffer_bytes != float(point["buffer_bytes"])
        or buffer_bytes > float(capacity_bytes)
        or exact_nonnegative_integer(
            point["dram_accesses_words"],
            field="dram_accesses_words",
        )
        != sum(element_counts)
        or accesses != expected
        or buffer["keep"] != keep
        or main_memory["keep"] != keep
        or buffer["utilized_instances"] != instances
        or main_memory["utilized_instances"] != instances
        or any(
            main_memory[metric] != values
            for metric, values in main_accesses.items()
        )
    ):
        raise OrojenesisError(
            "contraction compulsory witness did not reach the "
            "selected-capacity optimum",
        )
    return {
        "expected": expected,
        "main_accesses": main_accesses,
        "buffer_capacity": buffer["utilized_capacity"],
        "instances": instances,
    }


def compulsory_witness_certificate(
    layer: Mapping[str, DynamicValue],
    point: Mapping[str, DynamicValue],
    level_stats: Mapping[str, Mapping[str, Mapping[str, int]]],
    spaces: Sequence[str],
    element_counts: Sequence[int],
    *,
    capacity_bytes: int,
    word_bits: int,
) -> dict[str, DynamicValue]:
    """Construct the auditable selected-capacity optimality certificate."""
    audit = audit_compulsory_witness(
        point,
        level_stats,
        spaces,
        element_counts,
        capacity_bytes=capacity_bytes,
        word_bits=word_bits,
    )
    return {
        "kind": "selected_capacity_compulsory_witness_v1",
        "scope": "selected_capacity_only",
        "pareto_curve_complete": False,
        "capacity_bytes": int(capacity_bytes),
        "buffer_bytes": int(point["buffer_bytes"]),
        "compulsory_accesses_words": sum(element_counts),
        "data_space_accesses_words": audit["expected"],
        "main_memory_accesses_words": audit["main_accesses"],
        "buffer_utilized_capacity_words": audit["buffer_capacity"],
        "buffer_utilized_instances": audit["instances"],
        "proof": "feasible traffic equals the universal compulsory lower bound",
        "portability": {
            "compulsory_lower_bound": "architecture_independent",
            "achievability": "selected_architecture_cache_only",
        },
        "theorem_inputs": {
            "proof_source": dict(
                (layer.get("semantic_op") or {}).get("proof_source") or {},
            ),
            "equation": str(
                (layer.get("semantic_op") or {}).get("equation") or "",
            ),
            "tensor_shapes": layer.get("tensor_shapes"),
            "data_spaces": list(spaces),
            "word_bits": int(word_bits),
            "dense": True,
            "first_read_elision": True,
            "instances": 1,
        },
    }


def attach_compulsory_witness(
    result: dict[str, DynamicValue],
    layer: Mapping[str, DynamicValue],
    plan: WitnessPlanView,
    curve: Sequence[Mapping[str, DynamicValue]],
    *,
    word_bits: int,
) -> None:
    """Validate and attach witness evidence files and certificate to ``result``."""
    witness_paths = {name: plan.output / name for name in WITNESS_OUTPUT_FILES}
    missing = [
        name for name, path in witness_paths.items() if not path.is_file()
    ]
    if missing:
        raise OrojenesisError(
            "compulsory witness omits mapper evidence: " + ", ".join(missing),
        )
    result["evidence_files"].update(
        {
            name: {"path": name, "sha256": sha256_file(path)}
            for name, path in witness_paths.items()
        },
    )
    result["environment"] = plan.environment
    result["optimality_certificate"] = compulsory_witness_certificate(
        layer,
        curve[0],
        witness_level_stats(
            witness_paths["timeloop-mapper.map+stats.xml"],
            plan.spaces,
            word_bits=word_bits,
        ),
        plan.spaces,
        _tensor_element_counts(layer),
        capacity_bytes=int(plan.selected_capacity_bytes or 0),
        word_bits=word_bits,
    )
    result["optimality_certificate"].update(
        {
            "streaming_axis": "B",
            "streaming_dimension": plan.streaming_dimension,
        },
    )
