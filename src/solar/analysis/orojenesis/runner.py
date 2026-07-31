# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
    def architecture(word_bits: int) -> dict[str, Any]:
        """Build the generic Orojenesis memory architecture."""
        return _build_architecture(word_bits)

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
                    buffer_bytes = int(float(row[0]))
                    intensity = float(row[1])
                    accesses = float(row[2])
                except ValueError:
                    continue
                point = {
                    "buffer_bytes": buffer_bytes,
                    "operational_intensity": intensity,
                    "dram_accesses_words": accesses,
                    "dram_bytes": accesses * word_bytes,
                }
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

    def run_layer(
        self,
        layer: Mapping[str, Any],
        output_dir: str | Path,
        *,
        word_bits: int,
    ) -> dict[str, Any]:
        """Run the pinned mapper for one layer and return auditable evidence."""
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        problem = self.problem_for_layer(layer)
        dimensions = list(problem["problem"]["shape"]["dimensions"])
        spaces = [
            item["name"] for item in problem["problem"]["shape"]["data-spaces"]
        ]
        inputs = {
            "problem.yaml": problem,
            "architecture.yaml": self.architecture(word_bits),
            "mapper.yaml": self.mapper_config(dimensions, spaces),
        }
        paths: dict[str, Path] = {}
        for name, data in inputs.items():
            path = output / name
            path.write_text(yaml.safe_dump(data, sort_keys=False))
            paths[name] = path
        try:
            completed = self._run_mapper(
                [
                    str(self.mapper),
                    str(paths["architecture.yaml"]),
                    str(paths["problem.yaml"]),
                    str(paths["mapper.yaml"]),
                    "-o",
                    str(output),
                ],
                cwd=output,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OrojenesisError("Orojenesis execution failed") from exc
        if completed.returncode != 0:
            raise OrojenesisError(
                f"Orojenesis exited with status {completed.returncode}",
            )
        raw = output / "timeloop-mapper.oaves.csv"
        curve = self.parse_curve(raw, word_bytes=max(1, word_bits // 8))
        return {
            "solver": "NVlabs/timeloop oaves_keep_max",
            "commit": OROJENESIS_COMMIT,
            "toolchain": self.toolchain_identity,
            "word_bits": int(word_bits),
            "curve": curve,
            "evidence_files": {
                **{
                    name: {"path": name, "sha256": sha256_file(path)}
                    for name, path in paths.items()
                },
                "curve": {
                    "path": raw.name,
                    "sha256": sha256_file(raw),
                },
            },
        }

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
