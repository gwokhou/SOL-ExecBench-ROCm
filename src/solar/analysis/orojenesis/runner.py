# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

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
from solar.analysis.orojenesis.errors import (
    OrojenesisError,
    OrojenesisTimeoutError,
)
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
from solar.analysis.orojenesis.witness import (
    attach_compulsory_witness as _attach_compulsory_witness,
    exact_nonnegative_integer as _exact_nonnegative_integer,
)
from solar.artifacts import sha256_file
from solar.artifacts.yaml import write_yaml_atomic
from solar.observation import ObservationStage, observed_stage

__all__ = [
    "find_multi_einsum_chains",
    "find_multi_einsum_regions",
    "multi_einsum_layer_problem",
    "multi_einsum_mapper_role",
    "multi_einsum_problem",
    "multi_einsum_region_mapper_role",
    "multi_einsum_region_problem",
]

MAX_MULTI_EINSUM_DIVISORS = 128


@dataclass(frozen=True, slots=True, kw_only=True)
class _MultiSweepSpec:
    row_tile: int
    role: str
    word_bits: int
    word_bytes: int
    environment: Mapping[str, str]
    context: str
    deadline: float | None = None


def _bounded_row_tiles(m_dim: int, *, context: str) -> list[int]:
    tiles = _divisors(int(m_dim))
    if len(tiles) > MAX_MULTI_EINSUM_DIVISORS:
        raise OrojenesisError(
            f"multi-einsum row tile divisor count {len(tiles)} exceeds "
            f"cap {MAX_MULTI_EINSUM_DIVISORS} for {context}"
        )
    return tiles


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
        write_yaml_atomic(path, document, no_aliases=False)
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


@dataclass(frozen=True, slots=True, kw_only=True)
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

    @observed_stage(ObservationStage.MAPPER)
    def _run_mapper(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        runner = getattr(self, "_process_runner", _default_mapper_runner)
        effective_timeout = (
            float(self.timeout_seconds) if timeout is None else float(timeout)
        )
        return invoke_mapper_process(
            runner,
            command,
            cwd=cwd,
            timeout=effective_timeout,
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
        word_bytes: float,
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
                except ValueError, OrojenesisError:
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
                    except ValueError, OrojenesisError:
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
            and selected_capacity_bytes > 0
            and selected_capacity_bytes % 1024 == 0
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
        except subprocess.TimeoutExpired as exc:
            raise OrojenesisTimeoutError(
                f"Orojenesis mapper exceeded {self.timeout_seconds}s timeout",
            ) from exc
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
        if int(word_bits) <= 0:
            raise OrojenesisError("word_bits must be positive")
        word_bytes: int | float = (
            int(word_bits) // 8
            if int(word_bits) % 8 == 0
            else int(word_bits) / 8.0
        )
        raw = self._execute_layer_run(plan)
        curve = self.parse_curve(
            raw,
            word_bytes=word_bytes,
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
        row_tiles = _bounded_row_tiles(
            int(descriptor["chain"]["layers"][0]["m"]),
            context="chain layer 0",
        )
        deadline = time.monotonic() + float(self.timeout_seconds)
        raw_paths, sweeps = self._collect_chain_sweeps(
            descriptor,
            output,
            paths,
            row_tiles=row_tiles,
            word_bits=word_bits,
            word_bytes=word_bytes,
            environment=environment,
            deadline=deadline,
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
        deadline: float,
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
                    _MultiSweepSpec(
                        row_tile=row_tile,
                        role=role,
                        word_bits=word_bits,
                        word_bytes=word_bytes,
                        environment=environment,
                        context=f"layer {layer_index}",
                        deadline=deadline,
                    ),
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
        deadline = time.monotonic() + float(self.timeout_seconds)
        raw_paths, row_tiles_by_node, sweeps = self._collect_region_sweeps(
            descriptor,
            output,
            paths,
            word_bits=word_bits,
            word_bytes=word_bytes,
            environment=environment,
            deadline=deadline,
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
        deadline: float,
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
            row_tiles = _bounded_row_tiles(
                int(node["m"]),
                context=f"node {node_id}",
            )
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
                    _MultiSweepSpec(
                        row_tile=row_tile,
                        role=role,
                        word_bits=word_bits,
                        word_bytes=word_bytes,
                        environment=environment,
                        context=f"node {node_id}",
                        deadline=deadline,
                    ),
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
        spec: _MultiSweepSpec,
    ) -> Path:
        remaining = (
            float(self.timeout_seconds)
            if spec.deadline is None
            else spec.deadline - time.monotonic()
        )
        if remaining <= 0:
            raise OrojenesisTimeoutError(
                "multi-einsum Orojenesis mapper exceeded "
                f"{self.timeout_seconds}s timeout for "
                f"{spec.context}, M={spec.row_tile}",
            )
        sweep_dir = output / prefix
        sweep_dir.mkdir(parents=True, exist_ok=True)
        documents = _write_yaml_documents(
            sweep_dir,
            {
                "architecture.yaml": self.multi_architecture(spec.word_bits),
                "mapper.yaml": self.multi_mapper_config(
                    spec.row_tile, role=spec.role
                ),
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
                env=spec.environment,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            raise OrojenesisTimeoutError(
                "multi-einsum Orojenesis mapper exceeded "
                f"{self.timeout_seconds}s timeout for "
                f"{spec.context}, M={spec.row_tile}",
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError(
                "multi-einsum Orojenesis execution failed",
            ) from exc
        if completed.returncode != 0:
            raise OrojenesisError(
                "multi-einsum Orojenesis exited with status "
                f"{completed.returncode} for {spec.context}, M={spec.row_tile}",
            )
        raw_path = sweep_dir / "timeloop-mapper.oaves.csv"
        parse_multi_mapping_records(raw_path, word_bytes=spec.word_bytes)
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
