"""Curve parsing and composition for tiled-fusion Orojenesis evidence."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from solar.analysis.orojenesis_common import OrojenesisError
from solar.common.types import DynamicValue


def parse_multi_mapping_records(
    path: str | Path,
    *,
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    """Parse mapping-level OAVES fields used by the fusion workflow."""
    source = Path(path)
    if not source.is_file():
        raise OrojenesisError(f"missing multi-einsum OAVES output: {source}")
    records: list[dict[str, DynamicValue]] = []
    with source.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 24:
                continue
            try:
                record: dict[str, DynamicValue] = {
                    "buffer_bytes": int(float(row[0])),
                    "dram_accesses_words": float(row[2]),
                    "mapping": str(row[3]),
                    "compute_ops": float(row[5]),
                    "weight_util_bytes": int(float(row[6])),
                    "input_util_bytes": int(float(row[10])),
                    "output_util_bytes": int(float(row[11])),
                    "weight_accesses_words": float(row[21]),
                    "input_accesses_words": float(row[22]),
                    "output_accesses_words": float(row[23]),
                }
            except ValueError:
                continue
            component_sum = sum(
                float(record[name])
                for name in (
                    "weight_accesses_words",
                    "input_accesses_words",
                    "output_accesses_words",
                )
            )
            difference = abs(
                component_sum - float(record["dram_accesses_words"]),
            )
            if difference > max(1e-6, component_sum * 1e-9):
                raise OrojenesisError(
                    "multi-einsum OAVES access fields disagree",
                )
            invalid = (
                int(record["buffer_bytes"]) <= 0
                or float(record["compute_ops"]) <= 0
                or int(record["input_util_bytes"]) <= 0
                or int(record["output_util_bytes"]) <= 0
                or any(
                    float(value) < 0
                    for name, value in record.items()
                    if name.endswith("words")
                )
            )
            if invalid:
                raise OrojenesisError("multi-einsum OAVES record is invalid")
            record["dram_bytes"] = component_sum * int(word_bytes)
            records.append(record)
    if not records:
        raise OrojenesisError(
            "multi-einsum OAVES output has no mapping records",
        )
    return records


def _mapping_tile(
    record: Mapping[str, DynamicValue],
    *,
    row_tile: int,
    word_bytes: int,
    side: str,
) -> tuple[int, int]:
    utilization = int(record[f"{side}_util_bytes"])
    denominator = int(row_tile) * int(word_bytes)
    if denominator <= 0 or utilization % denominator:
        raise OrojenesisError("multi-einsum mapping tile is not rectangular")
    feature_tile = utilization // denominator
    if feature_tile <= 0:
        raise OrojenesisError("multi-einsum mapping tile is empty")
    return int(row_tile), int(feature_tile)


def _region_mapping_candidates(
    schedule: Sequence[str],
    raw_paths: Mapping[str, Sequence[str | Path]],
    row_tiles_by_node: Mapping[str, Sequence[int]],
    word_bytes: int,
) -> dict[str, list[dict[str, DynamicValue]]]:
    candidates: dict[str, list[dict[str, DynamicValue]]] = {}
    for node_id in schedule:
        paths = list(raw_paths.get(node_id) or [])
        row_tiles = [int(item) for item in row_tiles_by_node.get(node_id) or []]
        if not paths or len(paths) != len(row_tiles):
            raise OrojenesisError(
                "multi-einsum region sweep matrix is incomplete",
            )
        node_candidates: list[dict[str, DynamicValue]] = []
        for path, row_tile in zip(paths, row_tiles, strict=True):
            records = parse_multi_mapping_records(
                path,
                word_bytes=int(word_bytes),
            )
            for raw_record in records:
                record = dict(raw_record)
                record["row_tile"] = row_tile
                for side in ("input", "output"):
                    record[f"{side}_tile"] = _mapping_tile(
                        record,
                        row_tile=row_tile,
                        word_bytes=word_bytes,
                        side=side,
                    )
                node_candidates.append(record)
        if not node_candidates:
            raise OrojenesisError(
                "multi-einsum region node has no mapping candidates",
            )
        candidates[node_id] = node_candidates
    return candidates


def _region_edge_maps(
    edges: Sequence[Mapping[str, DynamicValue]],
) -> dict[str, list[str]]:
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    return dict(successors)


def _region_record_is_compatible(
    edge: Mapping[str, DynamicValue] | None,
    state: Mapping[str, DynamicValue],
    record: Mapping[str, DynamicValue],
) -> bool:
    if edge is None:
        return True
    producer_record = state["assignments"].get(str(edge["producer"]))
    if producer_record is None:
        return False
    producer_tile = tuple(producer_record["output_tile"])
    consumer_tile = tuple(record["input_tile"])
    axis_map = [int(item) for item in edge["axis_map"]]
    transformed = tuple(producer_tile[index] for index in axis_map)
    return transformed == consumer_tile


def _extend_region_state(
    state: Mapping[str, DynamicValue],
    record: Mapping[str, DynamicValue],
    node_id: str,
    edge: Mapping[str, DynamicValue] | None,
    leaves: set[str],
) -> dict[str, DynamicValue] | None:
    if not _region_record_is_compatible(edge, state, record):
        return None
    accesses = float(record["weight_accesses_words"])
    if edge is None:
        accesses += float(record["input_accesses_words"])
    if node_id in leaves:
        accesses += float(record["output_accesses_words"])
    assignments = dict(state["assignments"])
    assignments[node_id] = record
    return {
        "assignments": assignments,
        "buffer_bytes": int(state["buffer_bytes"])
        + int(record["buffer_bytes"]),
        "dram_accesses_words": float(state["dram_accesses_words"]) + accesses,
        "compute_ops": float(state["compute_ops"])
        + float(record["compute_ops"]),
        "mappings": [*list(state["mappings"]), str(record["mapping"])],
    }


def _prune_region_states(
    states: Sequence[dict[str, DynamicValue]],
    schedule: Sequence[str],
    processed: set[str],
    successors: Mapping[str, Sequence[str]],
) -> list[dict[str, DynamicValue]]:
    active = [
        item
        for item in schedule
        if item in processed
        and any(child not in processed for child in successors.get(item) or [])
    ]
    best: dict[tuple[DynamicValue, ...], dict[str, DynamicValue]] = {}
    for state in states:
        key: tuple[DynamicValue, ...] = (
            int(state["buffer_bytes"]),
            *(
                (item, tuple(state["assignments"][item]["output_tile"]))
                for item in active
            ),
        )
        previous = best.get(key)
        if previous is None or float(state["dram_accesses_words"]) < float(
            previous["dram_accesses_words"],
        ):
            best[key] = state
    return list(best.values())


def _region_mapping_states(
    schedule: Sequence[str],
    edges: Sequence[Mapping[str, DynamicValue]],
    leaves: set[str],
    candidates: Mapping[
        str,
        Sequence[Mapping[str, DynamicValue]],
    ],
) -> list[dict[str, DynamicValue]]:
    edge_by_consumer = {str(edge["consumer"]): edge for edge in edges}
    successors = _region_edge_maps(edges)
    states: list[dict[str, DynamicValue]] = [
        {
            "assignments": {},
            "buffer_bytes": 0,
            "dram_accesses_words": 0.0,
            "compute_ops": 0.0,
            "mappings": [],
        },
    ]
    processed: set[str] = set()
    for node_id in schedule:
        edge = edge_by_consumer.get(node_id)
        next_states = [
            extension
            for state in states
            for record in candidates[node_id]
            if (
                extension := _extend_region_state(
                    state,
                    record,
                    node_id,
                    edge,
                    leaves,
                )
            )
            is not None
        ]
        if not next_states:
            raise OrojenesisError(
                f"multi-einsum region has no compatible mapping for {node_id}",
            )
        processed.add(node_id)
        states = _prune_region_states(
            next_states,
            schedule,
            processed,
            successors,
        )
    return states


def _mapping_curve_points(
    states: Sequence[Mapping[str, DynamicValue]],
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    points: list[dict[str, DynamicValue]] = []
    for state in states:
        accesses = float(state["dram_accesses_words"])
        intensity = (
            0.0
            if accesses == 0
            else float(state["compute_ops"]) / (accesses * word_bytes)
        )
        points.append(
            {
                "buffer_bytes": int(state["buffer_bytes"]),
                "operational_intensity": intensity,
                "dram_accesses_words": accesses,
                "dram_bytes": accesses * word_bytes,
                "mappings": list(state["mappings"]),
            },
        )
    return points


def _pareto_capacity_curve(
    points: Sequence[dict[str, DynamicValue]],
) -> list[dict[str, DynamicValue]]:
    best_by_capacity: dict[int, dict[str, DynamicValue]] = {}
    for point in points:
        capacity = int(point["buffer_bytes"])
        previous = best_by_capacity.get(capacity)
        if previous is None or float(point["dram_bytes"]) < float(
            previous["dram_bytes"],
        ):
            best_by_capacity[capacity] = point
    pareto: list[dict[str, DynamicValue]] = []
    best_traffic = float("inf")
    for point in sorted(
        best_by_capacity.values(),
        key=lambda item: int(item["buffer_bytes"]),
    ):
        if float(point["dram_bytes"]) < best_traffic:
            pareto.append(point)
            best_traffic = float(point["dram_bytes"])
    return pareto


def compose_multi_einsum_region_curve(
    descriptor: Mapping[str, DynamicValue],
    raw_paths: Mapping[str, Sequence[str | Path]],
    *,
    row_tiles_by_node: Mapping[str, Sequence[int]],
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    """Compose replayable mappings for a validated multi-einsum region."""
    if int(word_bytes) <= 0:
        raise OrojenesisError("multi-einsum region word width must be positive")
    schedule = [str(item) for item in descriptor["schedule"]]
    edges = descriptor["edges"]
    leaves = {str(item) for item in descriptor["leaves"]}
    candidates = _region_mapping_candidates(
        schedule,
        raw_paths,
        row_tiles_by_node,
        word_bytes,
    )
    states = _region_mapping_states(schedule, edges, leaves, candidates)
    pareto = _pareto_capacity_curve(
        _mapping_curve_points(states, int(word_bytes)),
    )
    if not pareto:
        raise OrojenesisError("multi-einsum region has no Pareto mapping")
    return pareto


def _initial_chain_states(
    records: Sequence[Mapping[str, DynamicValue]],
    row_tile: int,
) -> list[dict[str, DynamicValue]]:
    return [
        {
            "buffer_bytes": int(record["buffer_bytes"]),
            "dram_accesses_words": float(record["weight_accesses_words"])
            + float(record["input_accesses_words"]),
            "compute_ops": float(record["compute_ops"]),
            "output_util_bytes": int(record["output_util_bytes"]),
            "mappings": [str(record["mapping"])],
            "row_tile": int(row_tile),
        }
        for record in records
    ]


def _extend_chain_state(
    state: Mapping[str, DynamicValue],
    record: Mapping[str, DynamicValue],
    *,
    final_layer: bool,
    row_tile: int,
) -> dict[str, DynamicValue]:
    accesses = float(state["dram_accesses_words"]) + float(
        record["weight_accesses_words"],
    )
    if final_layer:
        accesses += float(record["output_accesses_words"])
    return {
        "buffer_bytes": int(state["buffer_bytes"])
        + int(record["buffer_bytes"]),
        "dram_accesses_words": accesses,
        "compute_ops": float(state["compute_ops"])
        + float(record["compute_ops"]),
        "output_util_bytes": int(record["output_util_bytes"]),
        "mappings": [*list(state["mappings"]), str(record["mapping"])],
        "row_tile": int(row_tile),
    }


def _advance_chain_states(
    states: Sequence[Mapping[str, DynamicValue]],
    records: Sequence[Mapping[str, DynamicValue]],
    *,
    final_layer: bool,
    row_tile: int,
) -> list[dict[str, DynamicValue]]:
    by_input: dict[int, list[Mapping[str, DynamicValue]]] = defaultdict(list)
    for record in records:
        by_input[int(record["input_util_bytes"])].append(record)
    next_states: dict[tuple[int, int], dict[str, DynamicValue]] = {}
    for state in states:
        compatible = by_input.get(int(state["output_util_bytes"]), [])
        for record in compatible:
            candidate = _extend_chain_state(
                state,
                record,
                final_layer=final_layer,
                row_tile=row_tile,
            )
            key = (
                int(candidate["buffer_bytes"]),
                int(candidate["output_util_bytes"]),
            )
            previous = next_states.get(key)
            if previous is None or float(
                candidate["dram_accesses_words"],
            ) < float(previous["dram_accesses_words"]):
                next_states[key] = candidate
    return list(next_states.values())


def _compose_chain_tile(
    per_layer: Sequence[Sequence[Mapping[str, DynamicValue]]],
    row_tile: int,
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    states = _initial_chain_states(per_layer[0], row_tile)
    for layer_index, records in enumerate(per_layer[1:], start=1):
        states = _advance_chain_states(
            states,
            records,
            final_layer=layer_index == len(per_layer) - 1,
            row_tile=row_tile,
        )
        if not states:
            break
    points = _mapping_curve_points(states, int(word_bytes))
    for point in points:
        point["row_tile"] = int(row_tile)
    return points


def compose_multi_einsum_curve(
    raw_paths: Sequence[Sequence[str | Path]],
    *,
    row_tiles: Sequence[int],
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    """Compose compatible per-layer mappings into a replayable joint curve."""
    if len(raw_paths) < 2 or any(
        len(paths) != len(row_tiles) for paths in raw_paths
    ):
        raise OrojenesisError("multi-einsum sweep matrix is incomplete")
    points: list[dict[str, DynamicValue]] = []
    for tile_index, row_tile in enumerate(row_tiles):
        per_layer = [
            parse_multi_mapping_records(
                paths[tile_index],
                word_bytes=word_bytes,
            )
            for paths in raw_paths
        ]
        points.extend(_compose_chain_tile(per_layer, row_tile, word_bytes))
    if not points:
        raise OrojenesisError(
            "multi-einsum sweeps contain no compatible tile path",
        )
    return _pareto_capacity_curve(points)


def _parse_serialized_curve(
    path: str | Path,
    *,
    word_bytes: int,
    include_row_tile: bool,
) -> list[dict[str, DynamicValue]]:
    source = Path(path)
    qualifier = "" if include_row_tile else " region"
    label = f"multi-einsum{qualifier} curve"
    if not source.is_file():
        raise OrojenesisError(f"missing {label}: {source}")
    expected_columns = 5 if include_row_tile else 4
    points: list[dict[str, DynamicValue]] = []
    with source.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != expected_columns:
                continue
            point = _serialized_curve_point(
                row,
                word_bytes=word_bytes,
                include_row_tile=include_row_tile,
            )
            if point is not None:
                points.append(point)
    if not points:
        validity = "valid " if include_row_tile else ""
        raise OrojenesisError(
            f"serialized {label} has no {validity}points",
        )
    return points


def _serialized_curve_point(
    row: Sequence[str],
    *,
    word_bytes: int,
    include_row_tile: bool,
) -> dict[str, DynamicValue] | None:
    try:
        accesses = float(row[2])
        mappings = json.loads(row[3])
        point: dict[str, DynamicValue] = {
            "buffer_bytes": int(float(row[0])),
            "operational_intensity": float(row[1]),
            "dram_accesses_words": accesses,
            "dram_bytes": accesses * int(word_bytes),
            "mappings": mappings,
        }
        if include_row_tile:
            point["row_tile"] = int(row[4])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    invalid = (
        point["buffer_bytes"] <= 0
        or point["dram_accesses_words"] < 0
        or (include_row_tile and int(point.get("row_tile", 0)) <= 0)
        or not isinstance(mappings, list)
        or any(not isinstance(item, str) for item in mappings)
    )
    if invalid:
        qualifier = "" if include_row_tile else " region"
        raise OrojenesisError(
            f"serialized multi-einsum{qualifier} curve is invalid",
        )
    return point


def parse_multi_einsum_curve(
    path: str | Path,
    *,
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    """Parse a serialized joint curve without trusting analysis YAML."""
    return _parse_serialized_curve(
        path,
        word_bytes=word_bytes,
        include_row_tile=True,
    )


def parse_multi_einsum_region_curve(
    path: str | Path,
    *,
    word_bytes: int,
) -> list[dict[str, DynamicValue]]:
    """Parse a region joint curve without trusting analysis YAML."""
    return _parse_serialized_curve(
        path,
        word_bytes=word_bytes,
        include_row_tile=False,
    )


__all__ = [
    "compose_multi_einsum_curve",
    "compose_multi_einsum_region_curve",
    "parse_multi_einsum_curve",
    "parse_multi_einsum_region_curve",
    "parse_multi_mapping_records",
]
