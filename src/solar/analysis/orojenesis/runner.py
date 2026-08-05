# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import csv
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite, prod
from pathlib import Path
from typing import Any

import yaml

from solar.analysis.orojenesis.configuration import (
    IDENTITY_POLICY,
    MULTI_EINSUM_COMPOSITION,
    MULTI_EINSUM_SOLVER,
    OROJENESIS_COMMIT,
    OROJENESIS_TRUSTED_MAPPER_SHA256,
)
from solar.analysis.orojenesis.curves import (
    compose_multi_einsum_curve,
    parse_multi_mapping_records,
)
from solar.analysis.orojenesis.errors import OrojenesisError
from solar.analysis.orojenesis.identity import validate_toolchain_identity
from solar.analysis.orojenesis.multi_einsum import (
    _divisors,
    find_multi_einsum_chains,
    multi_einsum_layer_problem,
    multi_einsum_mapper_role,
    multi_einsum_problem,
)
from solar.analysis.orojenesis.problem import (
    architecture as _build_architecture,
    compulsory_witness_mapper_config as _build_compulsory_witness_mapper_config,
    compulsory_witness_streaming_dimension as _compulsory_witness_streaming_dimension,
    mapper_config as _build_mapper_config,
    multi_architecture as _build_multi_architecture,
    multi_mapper_config as _build_multi_mapper_config,
    problem_for_layer as _build_problem_for_layer,
)
from solar.analysis.orojenesis.process import (
    invoke_mapper_process,
    run_mapper_process as _default_mapper_runner,
)
from solar.analysis.orojenesis.regions import (
    compose_multi_einsum_region_curve,
    find_multi_einsum_regions,
    multi_einsum_region_mapper_role,
    multi_einsum_region_problem,
)
from solar.artifacts import sha256_file

__all__ = [
    "find_multi_einsum_chains",
    "find_multi_einsum_regions",
    "multi_einsum_layer_problem",
    "multi_einsum_mapper_role",
    "multi_einsum_problem",
    "multi_einsum_region_mapper_role",
    "multi_einsum_region_problem",
]

_WITNESS_OUTPUT_FILES = (
    "timeloop-mapper.map+stats.xml",
    "timeloop-mapper.map.txt",
    "timeloop-mapper.stats.txt",
    "timeloop-mapper.map.yaml",
)


def _multi_word_bytes(word_bits: int) -> int:
    if int(word_bits) <= 0 or int(word_bits) % 8:
        raise OrojenesisError("multi-einsum word width must be byte aligned")
    return int(word_bits) // 8


def _write_yaml_documents(
    output: Path,
    documents: Mapping[str, Any],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = output / name
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        paths[name] = path
    return paths


def _write_multi_curve(
    path: Path,
    curve: Sequence[Mapping[str, Any]],
    *,
    include_row_tile: bool,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for point in curve:
            row = [
                point["buffer_bytes"],
                point["operational_intensity"],
                point["dram_accesses_words"],
                json.dumps(
                    point.get("mappings") or [],
                    separators=(",", ":"),
                ),
            ]
            if include_row_tile:
                row.append(point["row_tile"])
            writer.writerow(row)


def _multi_evidence(
    paths: Mapping[str, Path],
    output: Path,
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _exact_nonnegative_integer(value: Any, *, field: str) -> int:
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
        space: _exact_nonnegative_integer(item.text, field=f"{name}.{space}")
        for space, item in zip(spaces, items, strict=True)
    }


def _witness_level_stats(
    path: Path,
    spaces: Sequence[str],
    *,
    word_bits: int,
) -> dict[str, dict[str, dict[str, int]]]:
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
        xml_word_bits = _exact_nonnegative_integer(
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


def _tensor_element_counts(layer: Mapping[str, Any]) -> list[int]:
    shapes = layer.get("tensor_shapes") or {}
    ordered = [*(shapes.get("inputs") or []), *(shapes.get("outputs") or [])]
    counts = [prod(int(dimension) for dimension in shape) for shape in ordered]
    if not counts or any(count <= 0 for count in counts):
        raise OrojenesisError("proof tensor shapes must be positive")
    return counts


def _audit_compulsory_witness(
    point: Mapping[str, Any],
    level_stats: Mapping[str, Mapping[str, Mapping[str, int]]],
    spaces: Sequence[str],
    element_counts: Sequence[int],
    *,
    capacity_bytes: int,
    word_bits: int,
) -> dict[str, Any]:
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
    buffer_bytes = buffer_words * (int(word_bits) // 8)
    if (
        buffer_bytes != int(point["buffer_bytes"])
        or buffer_bytes > int(capacity_bytes)
        or _exact_nonnegative_integer(
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


def _compulsory_witness_certificate(
    layer: Mapping[str, Any],
    point: Mapping[str, Any],
    level_stats: Mapping[str, Mapping[str, Mapping[str, int]]],
    spaces: Sequence[str],
    element_counts: Sequence[int],
    *,
    capacity_bytes: int,
    word_bits: int,
) -> dict[str, Any]:
    audit = _audit_compulsory_witness(
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


def _attach_compulsory_witness(
    result: dict[str, Any],
    layer: Mapping[str, Any],
    plan: _LayerRunPlan,
    curve: Sequence[Mapping[str, Any]],
    *,
    word_bits: int,
) -> None:
    witness_paths = {name: plan.output / name for name in _WITNESS_OUTPUT_FILES}
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
    result["optimality_certificate"] = _compulsory_witness_certificate(
        layer,
        curve[0],
        _witness_level_stats(
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


@dataclass(frozen=True)
class _LayerRunPlan:
    output: Path
    paths: dict[str, Path]
    spaces: list[str]
    streaming_dimension: str | None
    environment: dict[str, str] | None
    selected_capacity_bytes: int | None

    @property
    def is_compulsory_witness(self) -> bool:
        return self.selected_capacity_bytes is not None


class OrojenesisRunner:
    """Run and parse Timeloop's OAVES/Orojenesis mode at a pinned commit."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        timeout_seconds: int = 7200,
    ) -> None:
        """Validate and configure a pinned Orojenesis installation."""
        configured = home or os.environ.get("SOLAR_OROJENESIS_HOME")
        if not configured:
            raise OrojenesisError(
                "Orojenesis is required; set --orojenesis-home or SOLAR_OROJENESIS_HOME",
            )
        self.home = Path(configured).resolve()
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise OrojenesisError("Orojenesis timeout must be positive")
        self.mapper = self.home / "bin" / "timeloop-mapper"
        self.toolchain_identity = self._validate_toolchain()

    def _run_mapper(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        runner = getattr(self, "_process_runner", _default_mapper_runner)
        return invoke_mapper_process(
            runner,
            command,
            cwd=cwd,
            timeout=self.timeout_seconds,
            env=env,
        )

    def _validate_toolchain(self) -> dict[str, Any]:
        policy = replace(
            IDENTITY_POLICY,
            trusted_mapper_sha256=OROJENESIS_TRUSTED_MAPPER_SHA256,
        )
        return validate_toolchain_identity(
            self.home,
            self.mapper,
            policy,
        )

    @staticmethod
    def problem_for_layer(layer: Mapping[str, Any]) -> dict[str, Any]:
        """Translate one exact einsum layer into a Timeloop problem."""
        return _build_problem_for_layer(layer)

    @staticmethod
    def architecture(
        word_bits: int,
        *,
        buffer_capacity_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Build the generic Orojenesis memory architecture."""
        return _build_architecture(
            word_bits,
            buffer_capacity_bytes=buffer_capacity_bytes,
        )

    @staticmethod
    def multi_architecture(word_bits: int) -> dict[str, Any]:
        """Return the official two-buffer abstraction used for tiled fusion."""
        return _build_multi_architecture(word_bits)

    @staticmethod
    def multi_mapper_config(row_tile: int, *, role: str) -> dict[str, Any]:
        """Build a fusion-friendly mapping sweep for a linear matmul chain."""
        return _build_multi_mapper_config(row_tile, role=role)

    @staticmethod
    def mapper_config(
        dimensions: list[str],
        spaces: list[str],
    ) -> dict[str, Any]:
        """Build a mapper configuration for the supplied problem dimensions."""
        return _build_mapper_config(dimensions, spaces)

    @staticmethod
    def parse_curve(
        path: str | Path,
        *,
        word_bytes: int,
        spaces: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Parse and Pareto-filter an OAVES traffic curve."""
        source = Path(path)
        if not source.is_file():
            raise OrojenesisError(f"missing OAVES output: {source}")
        best: dict[int, dict[str, Any]] = {}
        with source.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    buffer_bytes = _exact_nonnegative_integer(
                        row[0],
                        field="buffer_bytes",
                    )
                    intensity = float(row[1])
                    accesses = _exact_nonnegative_integer(
                        row[2],
                        field="dram_accesses_words",
                    )
                    if not isfinite(intensity) or intensity < 0:
                        raise OrojenesisError(
                            "operational intensity must be finite and nonnegative",
                        )
                except (ValueError, OrojenesisError):
                    continue
                point = {
                    "buffer_bytes": buffer_bytes,
                    "operational_intensity": intensity,
                    "dram_accesses_words": accesses,
                    "dram_bytes": accesses * word_bytes,
                }
                if spaces and len(row) >= 3 + len(spaces):
                    try:
                        point["data_space_accesses_words"] = dict(
                            zip(
                                spaces,
                                (
                                    _exact_nonnegative_integer(
                                        value,
                                        field=f"{space} accesses",
                                    )
                                    for space, value in zip(
                                        spaces,
                                        row[-len(spaces) :],
                                        strict=True,
                                    )
                                ),
                                strict=True,
                            ),
                        )
                    except (ValueError, OrojenesisError):
                        continue
                previous = best.get(buffer_bytes)
                if (
                    previous is None
                    or point["dram_bytes"] < previous["dram_bytes"]
                ):
                    best[buffer_bytes] = point
        if not best:
            raise OrojenesisError("OAVES output contains no valid curve points")
        pareto: list[dict[str, Any]] = []
        best_traffic = float("inf")
        for point in sorted(
            best.values(),
            key=lambda item: item["buffer_bytes"],
        ):
            if point["dram_bytes"] < best_traffic:
                pareto.append(point)
                best_traffic = float(point["dram_bytes"])
        return pareto

    def _prepare_layer_run(
        self,
        layer: Mapping[str, Any],
        output: Path,
        *,
        word_bits: int,
        selected_capacity_bytes: int | None = None,
    ) -> _LayerRunPlan:
        problem = self.problem_for_layer(layer)
        dimensions = list(problem["problem"]["shape"]["dimensions"])
        spaces = [
            item["name"] for item in problem["problem"]["shape"]["data-spaces"]
        ]
        streaming_dimension = (
            _compulsory_witness_streaming_dimension(
                layer,
                dimensions,
                capacity_bytes=int(selected_capacity_bytes),
                word_bits=word_bits,
            )
            if selected_capacity_bytes is not None
            else None
        )
        witness_capacity = (
            int(selected_capacity_bytes)
            if streaming_dimension is not None
            and selected_capacity_bytes is not None
            else None
        )
        witness = witness_capacity is not None
        mapper = (
            _build_compulsory_witness_mapper_config(
                problem["problem"]["instance"],
                dimensions,
                spaces,
                streaming_dimension=str(streaming_dimension),
            )
            if witness
            else self.mapper_config(dimensions, spaces)
        )
        inputs = {
            "problem.yaml": problem,
            "architecture.yaml": self.architecture(
                word_bits,
                buffer_capacity_bytes=witness_capacity,
            ),
            "mapper.yaml": mapper,
        }
        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        if witness:
            inputs["environment.yaml"] = environment
        return _LayerRunPlan(
            output=output,
            paths=_write_yaml_documents(output, inputs),
            spaces=spaces,
            streaming_dimension=streaming_dimension,
            environment=environment if witness else None,
            selected_capacity_bytes=witness_capacity,
        )

    def _execute_layer_run(self, plan: _LayerRunPlan) -> Path:
        try:
            completed = self._run_mapper(
                [
                    str(self.mapper),
                    str(plan.paths["architecture.yaml"]),
                    str(plan.paths["problem.yaml"]),
                    str(plan.paths["mapper.yaml"]),
                    "-o",
                    str(plan.output),
                ],
                cwd=plan.output,
                env=plan.environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError("Orojenesis execution failed") from exc
        if completed.returncode != 0:
            raise OrojenesisError(
                f"Orojenesis exited with status {completed.returncode}",
            )
        return plan.output / "timeloop-mapper.oaves.csv"

    def run_layer(
        self,
        layer: Mapping[str, Any],
        output_dir: str | Path,
        *,
        word_bits: int,
        selected_capacity_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Run the pinned mapper for one layer and return auditable evidence."""
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        plan = self._prepare_layer_run(
            layer,
            output,
            word_bits=word_bits,
            selected_capacity_bytes=selected_capacity_bytes,
        )
        raw = self._execute_layer_run(plan)
        curve = self.parse_curve(
            raw,
            word_bytes=max(1, word_bits // 8),
            spaces=plan.spaces,
        )
        result: dict[str, Any] = {
            "solver": "NVlabs/timeloop oaves_keep_max",
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "word_bits": int(word_bits),
            "curve": curve,
            "evidence_files": {
                **{
                    name: {"path": name, "sha256": sha256_file(path)}
                    for name, path in plan.paths.items()
                },
                "curve": {
                    "path": raw.name,
                    "sha256": sha256_file(raw),
                },
            },
        }
        if plan.is_compulsory_witness:
            _attach_compulsory_witness(
                result,
                layer,
                plan,
                curve,
                word_bits=word_bits,
            )
        return result

    def run_multi_chain(
        self,
        chain: Sequence[tuple[str, Mapping[str, Any]]],
        output_dir: str | Path,
        *,
        word_bits: int,
    ) -> dict[str, Any]:
        """Run and compose official fusion-friendly mappings for a matmul chain."""
        descriptor = multi_einsum_problem(chain)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        word_bytes = _multi_word_bytes(word_bits)
        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        paths = _write_yaml_documents(
            output,
            {
                "chain.yaml": descriptor,
                "architecture.yaml": self.multi_architecture(word_bits),
                "environment.yaml": environment,
            },
        )
        row_tiles = _divisors(int(descriptor["chain"]["layers"][0]["m"]))
        raw_paths, sweeps = self._collect_chain_sweeps(
            descriptor,
            output,
            paths,
            row_tiles=row_tiles,
            word_bits=word_bits,
            word_bytes=word_bytes,
            environment=environment,
        )
        curve = compose_multi_einsum_curve(
            raw_paths,
            row_tiles=row_tiles,
            word_bytes=word_bytes,
        )
        curve_path = output / "multi-einsum-curve.csv"
        _write_multi_curve(curve_path, curve, include_row_tile=True)
        paths["curve"] = curve_path
        return {
            "solver": MULTI_EINSUM_SOLVER,
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "composition": MULTI_EINSUM_COMPOSITION,
            "word_bits": int(word_bits),
            "environment": environment,
            "problem": descriptor,
            "sweeps": sweeps,
            "curve": curve,
            "evidence_files": _multi_evidence(paths, output),
        }

    def _collect_chain_sweeps(
        self,
        descriptor: Mapping[str, Any],
        output: Path,
        paths: dict[str, Path],
        *,
        row_tiles: Sequence[int],
        word_bits: int,
        word_bytes: int,
        environment: Mapping[str, str],
    ) -> tuple[list[list[Path]], list[dict[str, Any]]]:
        raw_paths: list[list[Path]] = []
        sweeps: list[dict[str, Any]] = []
        layers = descriptor["chain"]["layers"]
        for layer_index, layer_descriptor in enumerate(layers):
            role = multi_einsum_mapper_role(layer_index, len(layers))
            problem = multi_einsum_layer_problem(layer_descriptor)
            problem_name = f"problem-layer-{layer_index}.yaml"
            paths.update(
                _write_yaml_documents(output, {problem_name: problem}),
            )
            layer_paths = [
                self._run_multi_sweep(
                    output,
                    f"layer-{layer_index}-m-{row_tile}",
                    problem,
                    row_tile=row_tile,
                    role=role,
                    word_bits=word_bits,
                    word_bytes=word_bytes,
                    environment=environment,
                    context=f"layer {layer_index}",
                )
                for row_tile in row_tiles
            ]
            raw_paths.append(layer_paths)
            self._record_sweep_evidence(
                paths,
                output,
                f"layer-{layer_index}",
                row_tiles,
            )
            sweeps.append(
                {
                    "layer_id": str(layer_descriptor["id"]),
                    "row_tiles": list(row_tiles),
                    "role": role,
                },
            )
        return raw_paths, sweeps

    def run_multi_region(
        self,
        region: Mapping[str, Any],
        output_dir: str | Path,
        *,
        word_bits: int,
    ) -> dict[str, Any]:
        """Run independent mapper sweeps and compose an extended MatMul region."""
        descriptor = multi_einsum_region_problem(region)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        word_bytes = _multi_word_bytes(word_bits)
        environment = {"TIMELOOP_ENABLE_FIRST_READ_ELISION": "1"}
        paths = _write_yaml_documents(
            output,
            {
                "region.yaml": descriptor,
                "architecture.yaml": self.multi_architecture(word_bits),
                "environment.yaml": environment,
            },
        )
        raw_paths, row_tiles_by_node, sweeps = self._collect_region_sweeps(
            descriptor,
            output,
            paths,
            word_bits=word_bits,
            word_bytes=word_bytes,
            environment=environment,
        )
        curve = compose_multi_einsum_region_curve(
            descriptor,
            raw_paths,
            row_tiles_by_node=row_tiles_by_node,
            word_bytes=word_bytes,
        )
        curve_path = output / "multi-einsum-region-curve.csv"
        _write_multi_curve(curve_path, curve, include_row_tile=False)
        paths["curve"] = curve_path
        return {
            "solver": MULTI_EINSUM_SOLVER,
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "composition": descriptor["composition"],
            "word_bits": int(word_bits),
            "environment": environment,
            "problem": descriptor,
            "sweeps": sweeps,
            "curve": curve,
            "evidence_files": _multi_evidence(paths, output),
        }

    def _collect_region_sweeps(
        self,
        descriptor: Mapping[str, Any],
        output: Path,
        paths: dict[str, Path],
        *,
        word_bits: int,
        word_bytes: int,
        environment: Mapping[str, str],
    ) -> tuple[
        dict[str, list[Path]],
        dict[str, list[int]],
        list[dict[str, Any]],
    ]:
        raw_paths: dict[str, list[Path]] = {}
        row_tiles_by_node: dict[str, list[int]] = {}
        sweeps: list[dict[str, Any]] = []
        for node_index, node in enumerate(descriptor["nodes"]):
            node_id = str(node["id"])
            row_tiles = _divisors(int(node["m"]))
            row_tiles_by_node[node_id] = row_tiles
            role = multi_einsum_region_mapper_role(descriptor, node_id)
            problem = multi_einsum_layer_problem(node)
            problem_name = f"problem-node-{node_index}.yaml"
            paths.update(
                _write_yaml_documents(output, {problem_name: problem}),
            )
            raw_paths[node_id] = [
                self._run_multi_sweep(
                    output,
                    f"node-{node_index}-m-{row_tile}",
                    problem,
                    row_tile=row_tile,
                    role=role,
                    word_bits=word_bits,
                    word_bytes=word_bytes,
                    environment=environment,
                    context=f"node {node_id}",
                )
                for row_tile in row_tiles
            ]
            self._record_sweep_evidence(
                paths,
                output,
                f"node-{node_index}",
                row_tiles,
            )
            sweeps.append(
                {
                    "node_id": node_id,
                    "row_tiles": row_tiles,
                    "role": role,
                },
            )
        return raw_paths, row_tiles_by_node, sweeps

    def _run_multi_sweep(
        self,
        output: Path,
        prefix: str,
        problem: Mapping[str, Any],
        *,
        row_tile: int,
        role: str,
        word_bits: int,
        word_bytes: int,
        environment: Mapping[str, str],
        context: str,
    ) -> Path:
        sweep_dir = output / prefix
        sweep_dir.mkdir(parents=True, exist_ok=True)
        documents = _write_yaml_documents(
            sweep_dir,
            {
                "architecture.yaml": self.multi_architecture(word_bits),
                "mapper.yaml": self.multi_mapper_config(row_tile, role=role),
                "problem.yaml": problem,
            },
        )
        try:
            completed = self._run_mapper(
                [
                    str(self.mapper),
                    str(documents["architecture.yaml"]),
                    str(documents["problem.yaml"]),
                    str(documents["mapper.yaml"]),
                    "-o",
                    str(sweep_dir),
                ],
                cwd=sweep_dir,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError(
                "multi-einsum Orojenesis execution failed",
            ) from exc
        if completed.returncode != 0:
            raise OrojenesisError(
                "multi-einsum Orojenesis exited with status "
                f"{completed.returncode} for {context}, M={row_tile}",
            )
        raw_path = sweep_dir / "timeloop-mapper.oaves.csv"
        parse_multi_mapping_records(raw_path, word_bytes=word_bytes)
        return raw_path

    @staticmethod
    def _record_sweep_evidence(
        paths: dict[str, Path],
        output: Path,
        prefix: str,
        row_tiles: Sequence[int],
    ) -> None:
        for row_tile in row_tiles:
            sweep_dir = output / f"{prefix}-m-{row_tile}"
            for leaf in ("architecture.yaml", "mapper.yaml", "problem.yaml"):
                paths[f"{prefix}-m-{row_tile}-{leaf}"] = sweep_dir / leaf
            paths[f"{prefix}-m-{row_tile}-raw"] = (
                sweep_dir / "timeloop-mapper.oaves.csv"
            )


def select_capacity_point(
    curve: Sequence[Mapping[str, Any]],
    capacity_bytes: int,
) -> dict[str, Any] | None:
    """Return the minimum-traffic point within a buffer capacity."""
    candidates = [
        point for point in curve if int(point["buffer_bytes"]) <= capacity_bytes
    ]
    if not candidates:
        return None
    return dict(min(candidates, key=lambda point: float(point["dram_bytes"])))
