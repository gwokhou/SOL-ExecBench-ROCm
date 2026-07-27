# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from solar import schema_versions as _schemas
from solar.analysis.orojenesis_common import OrojenesisError
from solar.analysis.orojenesis_curves import (
    compose_multi_einsum_curve,
    parse_multi_einsum_curve,
    parse_multi_einsum_region_curve,
    parse_multi_mapping_records,
)
from solar.analysis.orojenesis_curves import (
    compose_multi_einsum_region_curve as _compose_multi_einsum_region_curve,
)
from solar.analysis.orojenesis_identity import (
    OrojenesisIdentityPolicy,
    validate_toolchain_identity,
)
from solar.analysis.orojenesis_problem import (
    architecture as _build_architecture,
)
from solar.analysis.orojenesis_problem import (
    mapper_config as _build_mapper_config,
)
from solar.analysis.orojenesis_problem import (
    multi_architecture as _build_multi_architecture,
)
from solar.analysis.orojenesis_problem import (
    multi_mapper_config as _build_multi_mapper_config,
)
from solar.analysis.orojenesis_problem import (
    problem_for_layer as _build_problem_for_layer,
)
from solar.analysis.orojenesis_process import (
    run_mapper_process as _default_mapper_runner,
)
from solar.ir.contracts import CONTRACTION_KIND, INPUT_KIND, layer_operation

OROJENESIS_COMMIT = "97d52178bf9a9c209bf79be96b87c164bcd35625"
OROJENESIS_REPOSITORY = "https://github.com/NVlabs/timeloop.git"
OROJENESIS_TREE_OID = "05b05ec5a2a2979b1fe92046b937556d9ad99847"
OROJENESIS_SOURCE_ARCHIVE_SHA256 = (
    "3a254ab201d92b7eba993d3c7dcf0bb148a31dc9e57ece020fbaa38ad67c7873"
)
OROJENESIS_BUILDER_IMAGE = (
    "ubuntu:24.04@sha256:"
    "4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"
)
OROJENESIS_UBUNTU_SNAPSHOT = "20260718T000000Z"
OROJENESIS_OPENSSL_BOOTSTRAP_SHA256 = (
    "9c79333ab21bce0fb8dd92304cd76b3b1c427b0f2fedc897257fb5cced37c39e"
)
OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256 = (
    "641de77d8f142cfd62a1a6f964ba67b20754d3337c480efb529d086075a06c9a"
)
OROJENESIS_SOURCE_DATE_EPOCH = 1753058729
OROJENESIS_COMPILER_WRAPPER_SHA256 = (
    "04363ce239f76a4763490c049de1d69e2265d59578d51bed753f688c6f75278d"
)
# No reviewed mapper artifact has been published for this release. Keeping the
# repository-owned allowlist empty makes formal bounds fail closed until a
# reproducible artifact digest is reviewed and added here.
OROJENESIS_TRUSTED_MAPPER_SHA256: frozenset[str] = frozenset()
OROJENESIS_PROVENANCE_FILENAME = "orojenesis-provenance.json"
MULTI_EINSUM_SOLVER = "NVlabs/Orojenesis tiled-fusion"
MULTI_EINSUM_COMPOSITION = "linear_matmul_compatible_tiles_sum_capacity_v1"
MULTI_EINSUM_LAYOUT_COMPOSITION = "linear_matmul_axis_map_tile_shape_v2"
MULTI_EINSUM_BATCH_COMPOSITION = "broadcast_batch_linear_tile_shape_v1"
MULTI_EINSUM_FANOUT_COMPOSITION = "matmul_fanout_tree_tile_shape_v1"
_TOKEN = re.compile(r"[A-Za-z][0-9]*")
_IDENTITY_POLICY = OrojenesisIdentityPolicy(
    schema_version=_schemas.OROJENESIS_IDENTITY_SCHEMA_VERSION,
    repository=OROJENESIS_REPOSITORY,
    commit=OROJENESIS_COMMIT,
    tree_oid=OROJENESIS_TREE_OID,
    source_archive_sha256=OROJENESIS_SOURCE_ARCHIVE_SHA256,
    compiler_wrapper_sha256=OROJENESIS_COMPILER_WRAPPER_SHA256,
    builder_image=OROJENESIS_BUILDER_IMAGE,
    ubuntu_snapshot=OROJENESIS_UBUNTU_SNAPSHOT,
    openssl_sha256=OROJENESIS_OPENSSL_BOOTSTRAP_SHA256,
    ca_certificates_sha256=OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256,
    source_date_epoch=OROJENESIS_SOURCE_DATE_EPOCH,
    provenance_filename=OROJENESIS_PROVENANCE_FILENAME,
    trusted_mapper_sha256=OROJENESIS_TRUSTED_MAPPER_SHA256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "sha256": _sha256(path),
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
        completed = runner(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            env=env,
        )
        if completed.stdout is not None:
            (cwd / "stdout.log").write_text(
                str(completed.stdout),
                encoding="utf-8",
            )
        if completed.stderr is not None:
            (cwd / "stderr.log").write_text(
                str(completed.stderr),
                encoding="utf-8",
            )
        return completed

    def _validate_toolchain(self) -> dict[str, Any]:
        policy = replace(
            _IDENTITY_POLICY,
            trusted_mapper_sha256=OROJENESIS_TRUSTED_MAPPER_SHA256,
        )
        identity = validate_toolchain_identity(
            self.home,
            self.mapper,
            policy,
        )
        if int(identity.get("schema_version", 0)) != policy.schema_version:
            raise OrojenesisError(
                "unsupported Orojenesis provenance schema",
            )
        return identity

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
                    name: {"path": name, "sha256": _sha256(path)}
                    for name, path in paths.items()
                },
                "curve": {
                    "path": raw.name,
                    "sha256": _sha256(raw),
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
                env={**os.environ, **environment},
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
    for previous, current in zip(descriptors, descriptors[1:], strict=False):
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
        "schema_version": (
            _schemas.OROJENESIS_MULTI_EINSUM_PROBLEM_SCHEMA_VERSION
        ),
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


@dataclass(frozen=True)
class _ViewMetadata:
    target: str
    semantic: Mapping[str, Any]
    input_name: str
    output_name: str
    input_shape: list[int]
    output_shape: list[int]
    dtype: str


def _view_metadata(layer: Mapping[str, Any]) -> _ViewMetadata | None:
    semantic = layer_operation(layer)
    target = str(semantic.get("target", ""))
    if semantic.get("kind") in {INPUT_KIND, CONTRACTION_KIND} or target not in {
        "view",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
    }:
        return None
    effects = semantic.get("effects") or {}
    if (
        effects.get("mutates")
        or effects.get("atomic")
        or effects.get("opaque_library_call")
    ):
        return None
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
        len(input_names) == len(output_names) == 1
        and len(input_shapes) == len(output_shapes) == 1
        and len(input_dtypes) == len(output_dtypes) == 1
        and input_dtypes[0] == output_dtypes[0]
        and _shape_product(input_shapes[0]) == _shape_product(output_shapes[0])
    ):
        return None
    aliases = effects.get("aliases")
    if (
        not isinstance(aliases, list)
        or len(aliases) != 1
        or not isinstance(aliases[0], Mapping)
    ):
        return None
    alias = aliases[0]
    if (
        alias.get("input") != 0
        or alias.get("output") != 0
        or alias.get("conditional") is not False
    ):
        return None
    return _ViewMetadata(
        target=target,
        semantic=semantic,
        input_name=input_names[0],
        output_name=output_names[0],
        input_shape=input_shapes[0],
        output_shape=output_shapes[0],
        dtype=input_dtypes[0],
    )


def _view_axis_map(metadata: _ViewMetadata) -> list[int] | None:
    target = metadata.target
    input_shape = metadata.input_shape
    output_shape = metadata.output_shape
    if target in {"view", "squeeze", "unsqueeze"}:
        input_flat = [_shape_product(input_shape[:-1]), input_shape[-1]]
        output_flat = [
            _shape_product(output_shape[:-1]),
            output_shape[-1],
        ]
        if input_flat != output_flat:
            return None
        return [0, 1]
    if len(input_shape) != 2 or len(output_shape) != 2:
        return None
    literal_arguments = [
        item.get("value")
        for item in metadata.semantic.get("arguments") or []
        if isinstance(item, Mapping) and "value" in item
    ]
    if target == "transpose":
        axis_map = _transpose_axis_map(literal_arguments)
    else:
        axis_map = _permutation_axis_map(literal_arguments)
    if axis_map is None:
        return None
    expected_output_shape = [input_shape[index] for index in axis_map]
    return axis_map if output_shape == expected_output_shape else None


def _transpose_axis_map(arguments: Sequence[Any]) -> list[int] | None:
    if len(arguments) != 2 or any(
        not isinstance(value, int) for value in arguments
    ):
        return None
    dimensions = [int(value) % 2 for value in arguments]
    if dimensions == [0, 1] or dimensions == [1, 0]:
        return [1, 0]
    if dimensions[0] == dimensions[1]:
        return [0, 1]
    return None


def _permutation_axis_map(arguments: Sequence[Any]) -> list[int] | None:
    permutation = arguments[-1] if arguments else None
    if (
        not isinstance(permutation, (list, tuple))
        or len(permutation) != 2
        or any(not isinstance(value, int) for value in permutation)
    ):
        return None
    axis_map = [int(value) % 2 for value in permutation]
    return axis_map if sorted(axis_map) == [0, 1] else None


def _internal_zero_copy_view(layer: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = _view_metadata(layer)
    if metadata is None:
        return None
    axis_map = _view_axis_map(metadata)
    if axis_map is None:
        return None
    return {
        "target": metadata.target,
        "input": metadata.input_name,
        "output": metadata.output_name,
        "input_shape": metadata.input_shape,
        "output_shape": metadata.output_shape,
        "dtype": metadata.dtype,
        "axis_map": axis_map,
    }


def _region_axis_map(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    bridges: Sequence[str],
    views: Mapping[str, Mapping[str, Any]],
) -> list[int] | None:
    producer_shape = [int(producer["m"]), int(producer["n"])]
    consumer_shape = [int(consumer["m"]), int(consumer["k"])]
    axis_map = [0, 1]
    for bridge in bridges:
        bridge_map = list(views[bridge]["axis_map"])
        axis_map = [axis_map[index] for index in bridge_map]
    mapped_shape = [producer_shape[index] for index in axis_map]
    return axis_map if mapped_shape == consumer_shape else None


@dataclass(frozen=True)
class _RegionDiscovery:
    layers: dict[str, Mapping[str, Any]]
    producers: dict[str, str]
    consumers: dict[str, list[str]]
    descriptors: dict[str, dict[str, Any]]
    views: dict[str, dict[str, Any]]


def _region_discovery(
    layers: Mapping[str, Mapping[str, Any]],
) -> _RegionDiscovery:
    layer_map = {str(key): value for key, value in layers.items()}
    producers = {
        str(name): str(layer_id)
        for layer_id, layer in layer_map.items()
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    descriptors: dict[str, dict[str, Any]] = {}
    views: dict[str, dict[str, Any]] = {}
    for layer_id, layer in layer_map.items():
        for name in (layer.get("tensor_names") or {}).get("inputs") or []:
            consumers[str(name)].append(layer_id)
        with contextlib.suppress(OrojenesisError):
            descriptors[layer_id] = _region_matmul_descriptor(layer_id, layer)
        view = _internal_zero_copy_view(layer)
        if view is not None:
            views[layer_id] = view
    return _RegionDiscovery(
        layers=layer_map,
        producers=producers,
        consumers=dict(consumers),
        descriptors=descriptors,
        views=views,
    )


def _trace_region_tensor(
    discovery: _RegionDiscovery,
    tensor: str,
) -> tuple[str | None, list[str]]:
    path: list[str] = []
    current = str(tensor)
    seen: set[str] = set()
    while True:
        producer = discovery.producers.get(current)
        if producer is None or producer in seen:
            return None, []
        seen.add(producer)
        if producer in discovery.descriptors:
            return producer, list(reversed(path))
        if producer not in discovery.views:
            return producer, list(reversed(path))
        view = discovery.views[producer]
        if len(discovery.consumers.get(str(view["output"])) or []) != 1:
            return None, []
        path.append(producer)
        current = str(view["input"])


def _candidate_region_edges(
    discovery: _RegionDiscovery,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], set[str]]:
    edges: list[dict[str, Any]] = []
    entry_bridges: dict[str, list[str]] = {}
    valid_nodes = set(discovery.descriptors)
    for consumer_id, descriptor in discovery.descriptors.items():
        producer_id, bridges = _trace_region_tensor(
            discovery,
            str(descriptor["input"]),
        )
        if producer_id is None or producer_id not in discovery.descriptors:
            source = discovery.layers.get(str(producer_id), {})
            if str(source.get("type", "")).lower() != "start":
                valid_nodes.discard(consumer_id)
            else:
                entry_bridges[consumer_id] = bridges
            continue
        axis_map = _region_axis_map(
            discovery.descriptors[producer_id],
            descriptor,
            bridges,
            discovery.views,
        )
        if axis_map is None:
            valid_nodes.discard(consumer_id)
            continue
        edges.append(
            {
                "producer": producer_id,
                "consumer": consumer_id,
                "tensor": str(discovery.descriptors[producer_id]["output"]),
                "bridges": bridges,
                "axis_map": axis_map,
                "layer_path": [producer_id, *bridges, consumer_id],
            },
        )
    return edges, entry_bridges, valid_nodes


def _filtered_region_edges(
    edges: Sequence[Mapping[str, Any]],
    valid_nodes: set[str],
) -> list[dict[str, Any]]:
    return [
        dict(edge)
        for edge in edges
        if edge["producer"] in valid_nodes and edge["consumer"] in valid_nodes
    ]


def _region_edge_maps(
    edges: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    predecessors = {
        str(edge["consumer"]): str(edge["producer"]) for edge in edges
    }
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    return predecessors, dict(successors)


def _validate_region_endpoints(
    discovery: _RegionDiscovery,
    edges: Sequence[Mapping[str, Any]],
    entry_bridges: dict[str, list[str]],
    valid_nodes: set[str],
) -> None:
    predecessors, _ = _region_edge_maps(edges)
    for node_id in list(valid_nodes):
        descriptor = discovery.descriptors[node_id]
        weight_producer, weight_bridges = _trace_region_tensor(
            discovery,
            str(descriptor["weight"]),
        )
        weight_source = discovery.layers.get(str(weight_producer), {})
        if str(weight_source.get("type", "")).lower() != "start":
            valid_nodes.discard(node_id)
            continue
        descriptor["weight_bridges"] = weight_bridges
        if node_id in predecessors:
            continue
        activation_producer, bridges = _trace_region_tensor(
            discovery,
            str(descriptor["input"]),
        )
        activation_source = discovery.layers.get(
            str(activation_producer),
            {},
        )
        if str(activation_source.get("type", "")).lower() != "start":
            valid_nodes.discard(node_id)
        else:
            entry_bridges[node_id] = bridges


def _region_components(
    edges: Sequence[Mapping[str, Any]],
) -> list[set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        left, right = str(edge["producer"]), str(edge["consumer"])
        neighbors[left].add(right)
        neighbors[right].add(left)
    components: list[set[str]] = []
    visited: set[str] = set()
    for seed in sorted(neighbors):
        if seed in visited:
            continue
        stack = [seed]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(neighbors[node], reverse=True))
        visited.update(component)
        components.append(component)
    return components


def _component_schedule(
    component: set[str],
    predecessors: Mapping[str, str],
    successors: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str]] | None:
    roots = sorted(node for node in component if node not in predecessors)
    if len(roots) != 1:
        return None
    schedule: list[str] = []
    ready = list(roots)
    while ready:
        node = ready.pop(0)
        schedule.append(node)
        ready.extend(sorted(successors.get(node) or []))
    return (roots, schedule) if len(schedule) == len(component) else None


def _region_kind(
    component: set[str],
    descriptors: Mapping[str, Mapping[str, Any]],
    successors: Mapping[str, Sequence[str]],
) -> tuple[str, str]:
    if any(len(successors.get(node) or []) > 1 for node in component):
        return "matmul_fanout_tree", MULTI_EINSUM_FANOUT_COMPOSITION
    if any(descriptors[node]["kind"] == "batched_matmul" for node in component):
        return (
            "broadcast_batch_linear_matmul",
            MULTI_EINSUM_BATCH_COMPOSITION,
        )
    return "linear_matmul_with_axis_maps", MULTI_EINSUM_LAYOUT_COMPOSITION


def _region_physical_paths(
    schedule: Sequence[str],
    edges: Sequence[Mapping[str, Any]],
    descriptors: Mapping[str, Mapping[str, Any]],
    entry_bridges: Mapping[str, Sequence[str]],
) -> list[list[str]]:
    paths = [list(edge["layer_path"]) for edge in edges]
    for node in schedule:
        entry = list(entry_bridges.get(node) or [])
        if entry:
            paths.append([*entry, node])
        weight = list(descriptors[node].get("weight_bridges") or [])
        if weight:
            paths.append([*weight, node])
    return paths


def _component_region(
    component: set[str],
    edges: Sequence[Mapping[str, Any]],
    discovery: _RegionDiscovery,
    entry_bridges: Mapping[str, Sequence[str]],
    legacy_sets: set[tuple[str, ...]],
) -> dict[str, Any] | None:
    predecessors, successors = _region_edge_maps(edges)
    component_edges = [
        edge
        for edge in edges
        if edge["producer"] in component and edge["consumer"] in component
    ]
    if len(component) < 2 or len(component_edges) != len(component) - 1:
        return None
    schedule_result = _component_schedule(
        component,
        predecessors,
        successors,
    )
    if schedule_result is None:
        return None
    roots, schedule = schedule_result
    leaves = sorted(node for node in component if not successors.get(node))
    if any(
        discovery.consumers.get(str(discovery.descriptors[node]["output"]))
        for node in leaves
    ):
        return None
    if (
        tuple(schedule) in legacy_sets
        and all(not edge["bridges"] for edge in component_edges)
        and all(
            discovery.descriptors[node]["kind"] == "matmul"
            for node in component
        )
    ):
        return None
    kind, composition = _region_kind(
        component,
        discovery.descriptors,
        successors,
    )
    ordered_edges = [
        edge
        for producer in schedule
        for consumer in schedule
        for edge in component_edges
        if str(edge["producer"]) == producer
        and str(edge["consumer"]) == consumer
    ]
    return {
        "schema_version": (
            _schemas.OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION
        ),
        "kind": kind,
        "composition": composition,
        "nodes": [discovery.descriptors[node] for node in schedule],
        "edges": ordered_edges,
        "roots": roots,
        "leaves": leaves,
        "schedule": schedule,
        "physical_paths": _region_physical_paths(
            schedule,
            component_edges,
            discovery.descriptors,
            entry_bridges,
        ),
    }


def find_multi_einsum_regions(
    layers: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find endpoint-proven MatMul regions beyond the legacy direct chain."""
    discovery = _region_discovery(layers)
    edges, entry_bridges, valid_nodes = _candidate_region_edges(discovery)
    edges = _filtered_region_edges(edges, valid_nodes)
    _validate_region_endpoints(
        discovery,
        edges,
        entry_bridges,
        valid_nodes,
    )
    edges = _filtered_region_edges(edges, valid_nodes)
    legacy_sets = {
        tuple(chain) for chain in find_multi_einsum_chains(discovery.layers)
    }
    regions: list[dict[str, Any]] = []
    for component in _region_components(edges):
        region = _component_region(
            component,
            edges,
            discovery,
            entry_bridges,
            legacy_sets,
        )
        if region is not None:
            regions.append(region)
    return regions


def multi_einsum_region_problem(region: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a supported extended MatMul region."""
    schema_version = _schemas.OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION
    try:
        descriptor = json.loads(json.dumps(region, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise OrojenesisError(
            "multi-einsum region is not serializable",
        ) from exc
    if int(descriptor.get("schema_version", 0)) != schema_version:
        raise OrojenesisError("unsupported multi-einsum region schema")
    compositions = {
        "linear_matmul_with_axis_maps": MULTI_EINSUM_LAYOUT_COMPOSITION,
        "broadcast_batch_linear_matmul": MULTI_EINSUM_BATCH_COMPOSITION,
        "matmul_fanout_tree": MULTI_EINSUM_FANOUT_COMPOSITION,
    }
    if compositions.get(str(descriptor.get("kind"))) != descriptor.get(
        "composition",
    ):
        raise OrojenesisError("multi-einsum region composition mismatch")
    nodes = descriptor.get("nodes") or []
    schedule = [str(item) for item in descriptor.get("schedule") or []]
    node_ids = [str(node.get("id")) for node in nodes]
    if (
        len(nodes) < 2
        or schedule != node_ids
        or len(node_ids) != len(set(node_ids))
    ):
        raise OrojenesisError("multi-einsum region schedule is invalid")
    for node in nodes:
        if (
            str(node.get("kind")) not in {"matmul", "batched_matmul"}
            or any(int(node.get(name, 0)) <= 0 for name in ("m", "k", "n"))
            or not str(node.get("dtype", ""))
        ):
            raise OrojenesisError("multi-einsum region node is invalid")
    positions = {node_id: index for index, node_id in enumerate(schedule)}
    predecessors: dict[str, str] = {}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in descriptor.get("edges") or []:
        producer = str(edge.get("producer"))
        consumer = str(edge.get("consumer"))
        axis_map = edge.get("axis_map")
        if (
            producer not in positions
            or consumer not in positions
            or positions[producer] >= positions[consumer]
            or consumer in predecessors
            or axis_map not in ([0, 1], [1, 0])
            or list(edge.get("layer_path") or [])[0:1] != [producer]
            or list(edge.get("layer_path") or [])[-1:] != [consumer]
        ):
            raise OrojenesisError("multi-einsum region edge is invalid")
        predecessors[consumer] = producer
        successors[producer].append(consumer)
    roots = sorted(
        node_id for node_id in schedule if node_id not in predecessors
    )
    leaves = sorted(
        node_id for node_id in schedule if not successors.get(node_id)
    )
    if (
        len(roots) != 1
        or len(predecessors) != len(nodes) - 1
        or descriptor.get("roots") != roots
        or descriptor.get("leaves") != leaves
    ):
        raise OrojenesisError("multi-einsum region is not an arborescence")
    if descriptor["composition"] != MULTI_EINSUM_FANOUT_COMPOSITION and any(
        len(items) > 1 for items in successors.values()
    ):
        raise OrojenesisError("linear multi-einsum region contains fan-out")
    if descriptor["composition"] == MULTI_EINSUM_BATCH_COMPOSITION and not any(
        node.get("kind") == "batched_matmul" for node in nodes
    ):
        raise OrojenesisError(
            "batched multi-einsum region has no batch dimension",
        )
    return descriptor


def multi_einsum_region_mapper_role(
    region: Mapping[str, Any],
    node_id: str,
) -> str:
    """Choose the pinned FFMT constraint variant for a region node."""
    descriptor = multi_einsum_region_problem(region)
    schedule = [str(item) for item in descriptor["schedule"]]
    edges = descriptor["edges"]
    predecessors = {
        str(edge["consumer"]): str(edge["producer"]) for edge in edges
    }
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    if node_id not in schedule:
        raise OrojenesisError("multi-einsum region mapper node is unknown")
    if node_id not in predecessors:
        return "first"
    if not successors.get(node_id):
        return "second_last" if len(schedule) == 2 else "last"
    parent = predecessors[node_id]
    if parent not in predecessors:
        return "second"
    return "middle"


def compose_multi_einsum_region_curve(
    region: Mapping[str, Any],
    raw_paths: Mapping[str, Sequence[str | Path]],
    *,
    row_tiles_by_node: Mapping[str, Sequence[int]],
    word_bytes: int,
) -> list[dict[str, Any]]:
    """Compose replayable mapping assignments for a linear or fan-out region."""
    descriptor = multi_einsum_region_problem(region)
    return _compose_multi_einsum_region_curve(
        descriptor,
        raw_paths,
        row_tiles_by_node=row_tiles_by_node,
        word_bytes=word_bytes,
    )


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
        # The official composition drops intermediate traffic.  Restrict it
        # to complete graph endpoints and graph-input weights so that every
        # dropped access has an explicit producer-consumer witness.
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


__all__ = [
    "MULTI_EINSUM_COMPOSITION",
    "MULTI_EINSUM_BATCH_COMPOSITION",
    "MULTI_EINSUM_FANOUT_COMPOSITION",
    "MULTI_EINSUM_LAYOUT_COMPOSITION",
    "MULTI_EINSUM_SOLVER",
    "OROJENESIS_COMMIT",
    "OROJENESIS_BUILDER_IMAGE",
    "OROJENESIS_CA_CERTIFICATES_BOOTSTRAP_SHA256",
    "OROJENESIS_COMPILER_WRAPPER_SHA256",
    "OROJENESIS_OPENSSL_BOOTSTRAP_SHA256",
    "OROJENESIS_PROVENANCE_FILENAME",
    "OROJENESIS_REPOSITORY",
    "OROJENESIS_SOURCE_ARCHIVE_SHA256",
    "OROJENESIS_TREE_OID",
    "OROJENESIS_TRUSTED_MAPPER_SHA256",
    "OROJENESIS_SOURCE_DATE_EPOCH",
    "OROJENESIS_UBUNTU_SNAPSHOT",
    "OrojenesisError",
    "OrojenesisRunner",
    "compose_multi_einsum_curve",
    "compose_multi_einsum_region_curve",
    "find_multi_einsum_chains",
    "find_multi_einsum_regions",
    "multi_einsum_layer_problem",
    "multi_einsum_mapper_role",
    "multi_einsum_problem",
    "multi_einsum_region_mapper_role",
    "multi_einsum_region_problem",
    "parse_multi_einsum_curve",
    "parse_multi_einsum_region_curve",
    "parse_multi_mapping_records",
    "select_capacity_point",
]
