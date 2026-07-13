#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Compile and measure one benchmark workload with the real TorchInductor backend.

This is a performance-provider adapter, not a lower-bound generator.  It emits
a JSONL row consumable by ``report_amd_sol_heldout.py`` and an independently
checksummed raw-evidence JSON file.  The adapter verifies compiled output
against the eager reference before reporting a timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.scoring.amd_bound_graph.fx_helpers import _torch_dtype
from sol_execbench.core.scoring.amd_sol import PerformanceProviderResult
from sol_execbench.core.timestamps import utc_timestamp


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _workload(path: Path, workload_uuid: str) -> Workload:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            workload = Workload.model_validate_json(line)
            if str(workload.uuid) == workload_uuid:
                return workload
    raise ValueError(f"workload UUID {workload_uuid!r} not found in {path}")


def _inputs(definition: Definition, workload: Workload, torch: Any) -> tuple[Any, ...]:
    resolved_shapes = definition.get_input_shapes(workload.axes)
    result: list[Any] = []
    for name, spec in definition.inputs.items():
        shape = resolved_shapes[name]
        if shape is None:
            raise ValueError(f"input {name!r} has no concrete static shape")
        dtype = _torch_dtype(torch, spec.dtype)
        if dtype.is_floating_point or dtype.is_complex:
            result.append(torch.randn(shape, dtype=dtype, device="cuda"))
        else:
            result.append(torch.zeros(shape, dtype=dtype, device="cuda"))
    return tuple(result)


def _architecture(torch: Any) -> str:
    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "")
    if not isinstance(architecture, str) or not architecture.startswith("gfx"):
        raise RuntimeError("Torch CUDA device does not expose an AMD gfx architecture")
    return architecture.split(":", maxsplit=1)[0].lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", required=True, type=Path)
    parser.add_argument("--workload-uuid", required=True)
    parser.add_argument("--jsonl-output", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument(
        "--append",
        action="store_true",
        help="append one provider row instead of replacing the JSONL file",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 3:
        raise ValueError("warmup must be >= 1 and repetitions must be >= 3")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TorchInductor provider requires a visible ROCm CUDA device")
    definition_path = args.problem_dir / "definition.json"
    workload_path = args.problem_dir / "workload.jsonl"
    definition = Definition.model_validate_json(
        definition_path.read_text(encoding="utf-8")
    )
    workload = _workload(workload_path, args.workload_uuid)
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(definition.reference, str(definition_path), "exec"), namespace)
    reference = namespace.get("run")
    if not callable(reference):
        raise ValueError("definition reference must define callable run")
    torch.manual_seed(0)
    inputs = _inputs(definition, workload, torch)
    eager = reference(*inputs)
    compiled = torch.compile(reference, backend="inductor", fullgraph=True)
    for _ in range(args.warmup):
        compiled(*inputs)
    torch.cuda.synchronize()
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, eager)
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(args.repetitions):
        start = perf_counter()
        compiled(*inputs)
        torch.cuda.synchronize()
        samples_ms.append((perf_counter() - start) * 1_000.0)
    architecture = _architecture(torch)
    identity = _sha256_payload(
        {
            "definition": definition.model_dump(mode="json"),
            "workload": workload.model_dump(mode="json"),
            "backend": "torch-inductor",
            "fullgraph": True,
            "seed": 0,
        }
    )
    evidence = {
        "schema_version": "sol_execbench.torch_inductor_provider_evidence.v1",
        "created_at": utc_timestamp(),
        "definition": definition.name,
        "workload_uuid": str(workload.uuid),
        "target_architecture": architecture,
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip or "unknown",
        "device_name": torch.cuda.get_device_name(0),
        "input_identity_sha256": identity,
        "compile": {"backend": "inductor", "fullgraph": True},
        "warmup": args.warmup,
        "samples_ms": samples_ms,
        "measurement_ms": min(samples_ms),
    }
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = PerformanceProviderResult(
        provider_name="torch-inductor",
        provider_revision=torch.__version__,
        provider_schema_version="sol_execbench.torch_inductor_provider_evidence.v1",
        target_architecture=architecture,
        rocm_version=torch.version.hip or "unknown",
        input_identity_sha256=identity,
        status="supported",
        result_kind="measurement",
        is_theoretical_lower_bound=False,
        predicted_latency_ms=None,
        measured_latency_ms=min(samples_ms),
        warnings=(),
        raw_evidence_ref=str(args.evidence_output),
        raw_evidence_sha256=sha256_file(args.evidence_output),
        output_payload={"backend": "inductor", "fullgraph": True},
    )
    row = {
        "definition": definition.name,
        "workload_uuid": str(workload.uuid),
        "provider_result": result.to_dict(),
    }
    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_output.open(
        "a" if args.append else "w", encoding="utf-8"
    ) as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "definition": definition.name,
                "workload_uuid": str(workload.uuid),
                "measurement_ms": min(samples_ms),
                "output": str(args.jsonl_output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
