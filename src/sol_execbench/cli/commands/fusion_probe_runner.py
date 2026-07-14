# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Built-in HIP driver for the representative fusion-validation slice.

This module is intentionally executed in a child process by ``hardware fusion
collect``.  It compiles the checked-in HIP harness once, then starts a fresh
process for every timing round, so timing state cannot leak between rounds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_graph.builder import (
    build_authority_bound_graph,
)
from sol_execbench.core.scoring.amd_sol import (
    fusion_signature_for_group,
)
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    extract_amdgpu_kernel_metadata,
)
from sol_execbench.core.scoring.fusion_validation import (
    FusionSignature,
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    performance_from_rounds,
)
from sol_execbench.core.integrity.checksums import sha256_file


_RMS_DEFINITION = "025_rmsnorm_h4096"
_TANH_DEFINITION = "061_tanh_gated_residual_add_backward"
_SOURCE_RESOURCES = resources.files("sol_execbench.data").joinpath("fusion_probes")


def main() -> int:
    """Collect the built-in gfx1200 probe evidence described by environment."""
    try:
        suite_path = Path(_required_env("SOL_EXECBENCH_FUSION_SUITE_MANIFEST"))
        root = Path(_required_env("SOL_EXECBENCH_FUSION_BENCHMARK_ROOT"))
        output = Path(_required_env("SOL_EXECBENCH_FUSION_OUTPUT"))
        architecture = _required_env("SOL_EXECBENCH_FUSION_ARCHITECTURE")
        device = int(_required_env("SOL_EXECBENCH_FUSION_DEVICE"))
        manifest = _json_object(suite_path)
        shapes = _selected_shapes(manifest, root, architecture)
        with TemporaryDirectory(prefix="sol-execbench-fusion-") as temporary:
            with resources.as_file(_SOURCE_RESOURCES) as source_dir:
                executable, command = _compile_runner(
                    Path(temporary), architecture, source_dir
                )
                binary = executable.read_bytes()
                resource_for = _resource_factory(
                    binary, command, architecture, source_dir
                )
                cases = tuple(
                    _collect_case(
                        executable=executable,
                        device=device,
                        shape=shape,
                        resource_for=resource_for,
                    )
                    for shape in shapes
                )
        artifact = FusionValidationArtifact(
            architecture=architecture,
            gpu_uuid=_required_env("SOL_EXECBENCH_FUSION_GPU_UUID"),
            rocm_version=_required_env("SOL_EXECBENCH_FUSION_ROCM_VERSION"),
            hipcc_version=_hipcc_version(),
            clocks_locked=os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED") == "1",
            suite_manifest_sha256=sha256_file(suite_path),
            benchmark_root_sha256=_selected_input_sha256(root),
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            cases=cases,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 1 if any(case.performance.status != "passed" for case in cases) else 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"built-in fusion probe failed: {exc}", file=sys.stderr)
        return 2


def _selected_shapes(
    manifest: dict[str, Any], root: Path, architecture: str
) -> tuple[dict[str, Any], ...]:
    selected = {
        row["workload_uuid"]: row["definition"]
        for row in manifest.get("workloads", [])
        if isinstance(row, dict)
        and isinstance(row.get("workload_uuid"), str)
        and row.get("definition") in {_RMS_DEFINITION, _TANH_DEFINITION}
    }
    if len(selected) != 30:
        raise ValueError(
            "representative manifest must select 14 RMSNorm and 16 tanh workloads"
        )
    rows: list[dict[str, Any]] = []
    for definition in (_RMS_DEFINITION, _TANH_DEFINITION):
        path = _workload_file(root, definition)
        definition_model = Definition.model_validate_json(
            (path.parent / "definition.json").read_text(encoding="utf-8")
        )
        for raw in path.read_text(encoding="utf-8").splitlines():
            workload = Workload.model_validate_json(raw)
            if str(workload.uuid) not in selected:
                continue
            axes = workload.axes
            batch = axes.get("batch_size")
            sequence = axes.get("seq_len", 1)
            if not isinstance(batch, int) or not isinstance(sequence, int):
                raise ValueError(f"workload {workload.uuid} has non-static shape")
            kind = "rms" if definition == _RMS_DEFINITION else "tanh"
            rows.append(
                {
                    "uuid": str(workload.uuid),
                    "kind": kind,
                    "batch": batch,
                    "sequence": sequence,
                    "signature": _workload_fusion_signature(
                        definition_model,
                        workload,
                        architecture=architecture,
                        kind=kind,
                    ),
                }
            )
    if {row["uuid"] for row in rows} != set(selected):
        raise ValueError(
            "representative manifest workloads are not present in benchmark root"
        )
    return tuple(sorted(rows, key=lambda row: str(row["uuid"])))


def _workload_fusion_signature(
    definition: Definition,
    workload: Workload,
    *,
    architecture: str,
    kind: str,
) -> FusionSignature:
    """Bind probe evidence to the exact fusion group extracted for a workload."""
    # Physical fusion evidence must bind the same semantic provider as the
    # publishable floor.  A static AST signature may be useful diagnostically,
    # but it cannot validate a torch.export-authority group.
    graph = build_authority_bound_graph(definition, workload)
    if any(
        warning in {"semantic_export_failed", "semantic_graph_provider_required"}
        for warning in graph.warnings
    ):
        raise ValueError(
            f"workload {workload.uuid} has no torch.export authority graph"
        )
    expected_ops = (
        ("torch.mean", "torch.add") if kind == "rms" else ("torch.sum", "torch.mul")
    )
    matches = _diagnostic_subgraph_matches(graph, expected_ops)
    if len(matches) != 1:
        raise ValueError(
            f"workload {workload.uuid} has {len(matches)} matching {kind} fusion groups"
        )
    group = matches[0]
    tile_contract: dict[str, object] = (
        {"workgroup_size": 256, "reduction": "mean", "epilogue": "epsilon_add"}
        if kind == "rms"
        else {
            "workgroup_size": 256,
            "reduction": "sum",
            "epilogue": "scalar_mul",
        }
    )
    return fusion_signature_for_group(
        group,
        graph.to_dict(),
        tile_contract=tile_contract,
    )


def _diagnostic_subgraph_matches(
    graph: Any, expected_ops: tuple[str, str]
) -> list[FusionGroup]:
    """Return exact adjacent authority subgraphs for a diagnostic HIP probe.

    ``semantic_component.v1`` intentionally absorbs longer data-flow chains for
    the authority floor.  A two-op HIP probe can validate an implementation
    candidate for the matching subgraph, but must not be re-labelled as proof
    that the whole component is physically fusable.  Its distinct pattern ID
    prevents accidental matching by the authority reducer.
    """
    nodes = tuple(graph.nodes)
    matches: list[FusionGroup] = []
    for first, second in zip(nodes, nodes[1:], strict=False):
        if (first.op_name, second.op_name) != expected_ops:
            continue
        internal = set(first.output_tensor_ids) & set(second.input_tensor_ids)
        if not internal:
            continue
        node_ids = (first.node_id, second.node_id)
        inputs = set(first.input_tensor_ids) | set(second.input_tensor_ids)
        outputs = set(first.output_tensor_ids) | set(second.output_tensor_ids)
        matches.append(
            FusionGroup(
                group_id=f"diagnostic_{first.node_id}_{second.node_id}",
                pattern_id="diagnostic_reduction_epilogue.v1",
                pattern_version=1,
                node_ids=node_ids,
                external_input_tensor_ids=tuple(sorted(inputs - internal)),
                external_output_tensor_ids=tuple(sorted(outputs - internal)),
                internal_tensor_ids=tuple(sorted(internal)),
                flops=0.0,
                external_read_bytes=0.0,
                external_write_bytes=0.0,
                eliminated_intermediate_bytes=0.0,
                required_lds_bytes=None,
                confidence=EstimateConfidence.INEXACT,
                warnings=("diagnostic_subgraph_not_authority_fusion",),
            )
        )
    return matches


def _workload_file(root: Path, definition: str) -> Path:
    candidates = tuple(root.rglob(f"{definition}/workload.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely locate workload file for {definition}")
    return candidates[0]


def _compile_runner(
    workspace: Path, architecture: str, source_dir: Path
) -> tuple[Path, tuple[str, ...]]:
    source = source_dir / "fusion_probe_runner.hip"
    executable = workspace / "fusion_probe_runner"
    command = (
        "hipcc",
        "-O3",
        f"--offload-arch={architecture}",
        f"-I{source_dir}",
        str(source),
        "-o",
        str(executable),
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "HIP probe compilation failed")
    return executable, command


def _resource_factory(
    binary: bytes,
    command: tuple[str, ...],
    architecture: str,
    source_dir: Path,
):
    metadata = {
        item.name: item
        for item in extract_amdgpu_kernel_metadata(
            binary, target_architecture=architecture
        )
    }
    source = (source_dir / "gfx1200_reduction_epilogue.hip").read_bytes()
    binary_sha256 = hashlib.sha256(binary).hexdigest()
    source_sha256 = hashlib.sha256(source).hexdigest()

    def build(
        kernel_name: str,
        *,
        occupancy: int,
        lds_limit: int,
        correct: bool,
    ) -> KernelResourceEvidence:
        item = metadata.get(kernel_name)
        if (
            item is None
            or item.vgpr_count is None
            or item.sgpr_count is None
            or item.private_segment_bytes is None
            or item.group_segment_bytes is None
        ):
            raise ValueError(f"missing complete code-object metadata for {kernel_name}")
        return KernelResourceEvidence(
            kernel_name=kernel_name,
            binary_sha256=binary_sha256,
            source_sha256=source_sha256,
            compile_command=command,
            architecture=architecture,
            vgpr_count=item.vgpr_count,
            sgpr_count=item.sgpr_count,
            vgpr_spill_count=item.vgpr_spill_count,
            sgpr_spill_count=item.sgpr_spill_count,
            private_segment_bytes=item.private_segment_bytes,
            static_lds_bytes=item.group_segment_bytes,
            dynamic_lds_bytes=0,
            lds_limit_bytes=lds_limit,
            active_blocks_per_multiprocessor=occupancy,
            launch_passed=True,
            correctness_passed=correct,
        )

    return build


def _collect_case(
    *, executable: Path, device: int, shape: dict[str, Any], resource_for: Any
) -> FusionValidationCase:
    kind = str(shape["kind"])
    rows = int(shape["batch"]) * int(shape["sequence"])
    samples = tuple(_run_round(executable, device, kind, rows) for _ in range(3))
    if not all(sample["correct"] for sample in samples):
        raise RuntimeError(f"correctness failed for {shape['uuid']}")
    performance = performance_from_rounds(
        tuple(float(median(sample["fused"])) for sample in samples),
        tuple(float(median(sample["unfused"])) for sample in samples),
    )
    occupancy = samples[0]["occupancy"]
    lds_limit = int(samples[0]["lds_limit"])
    if kind == "rms":
        signature = shape["signature"]
        fused_name, unfused_names, variant = (
            "sol_fusion_rms_mean_epsilon",
            ("sol_unfused_rms_mean", "sol_unfused_rms_epsilon"),
            "rmsnorm_fp32_mean_epsilon",
        )
    else:
        signature = shape["signature"]
        fused_name, unfused_names, variant = (
            "sol_fusion_tanh_backward_sum_scalar",
            ("sol_unfused_tanh_backward_sum", "sol_unfused_tanh_backward_scalar"),
            "tanh_backward_fp32_sum_scalar",
        )
    return FusionValidationCase(
        evidence_id=f"{kind}:{shape['uuid']}",
        workload_uuid=str(shape["uuid"]),
        variant_id=variant,
        signature=signature,
        fused=resource_for(
            fused_name,
            occupancy=int(occupancy[fused_name]),
            lds_limit=lds_limit,
            correct=True,
        ),
        unfused=tuple(
            resource_for(
                name,
                occupancy=int(occupancy[name]),
                lds_limit=lds_limit,
                correct=True,
            )
            for name in unfused_names
        ),
        capacity_status="passed",
        performance=performance,
    )


def _run_round(executable: Path, device: int, kind: str, rows: int) -> dict[str, Any]:
    completed = subprocess.run(
        (str(executable), str(device), kind, str(rows)),
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "HIP probe execution failed")
    parsed: dict[str, Any] = {"fused": [], "unfused": [], "occupancy": {}}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts[:1] == ["CORRECT"] and len(parts) == 2:
            parsed["correct"] = parts[1] == "1"
        elif parts[:1] == ["LDS_LIMIT"] and len(parts) == 2:
            parsed["lds_limit"] = int(parts[1])
        elif parts[:1] == ["OCCUPANCY"] and len(parts) == 3:
            parsed["occupancy"][parts[1]] = int(parts[2])
        elif parts[:1] == ["PAIR"] and len(parts) == 3:
            parsed["fused"].append(float(parts[1]))
            parsed["unfused"].append(float(parts[2]))
    if (
        not isinstance(parsed.get("correct"), bool)
        or not isinstance(parsed.get("lds_limit"), int)
        or len(parsed["fused"]) != 30
        or len(parsed["unfused"]) != 30
    ):
        raise ValueError("HIP probe returned incomplete timing evidence")
    return parsed


def _selected_input_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for definition in (_RMS_DEFINITION, _TANH_DEFINITION):
        path = _workload_file(root, definition)
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hipcc_version() -> str:
    output = subprocess.run(
        ("hipcc", "--version"), text=True, capture_output=True, check=False
    ).stdout
    match = re.search(r"(?:HIP|ROCm) version:\s*([^\s]+)", output, re.I)
    return match.group(1) if match else output.strip() or "unknown"


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"missing {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
