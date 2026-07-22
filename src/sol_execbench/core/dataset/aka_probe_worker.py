# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated live workload probe for target-aware AKA materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from sol_execbench.core.bench.io import ShiftingMemoryPoolAllocator
from sol_execbench.core.bench.reference_protocol import (
    MAX_REFERENCE_TENSOR_STORAGE_BYTES,
    ReferenceCase,
    reference_case_storage_bytes,
    reference_values_storage_bytes,
)
from sol_execbench.core.bench.reference_service import ReferenceService
from sol_execbench.core.bench.eval_output_integrity import stable_reference_outputs
from sol_execbench.core.bench.utils import call_and_collect_outputs
from sol_execbench.core.dataset.aka_compatibility import PROBE_RESULT_PREFIX
from sol_execbench.core.platform.runtime import (
    cache_clear_policy_for_device,
    detect_rocm_device,
)


def _emit(status: str, reason_code: str, detail: str = "", **metrics: int) -> None:
    payload = {
        "status": status,
        "reason_code": reason_code,
        "detail": detail[:4096],
        "metrics": metrics,
    }
    print(PROBE_RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _run_probe(args: argparse.Namespace) -> None:
    try:
        device_info = detect_rocm_device(args.device)
    except Exception as exc:
        _emit("infrastructure_error", "gpu_unavailable", str(exc))
        return
    if device_info.gfx_target != args.expected_arch:
        _emit(
            "infrastructure_error",
            "target_arch_mismatch",
            f"detected {device_info.gfx_target}, expected {args.expected_arch}",
        )
        return
    try:
        service = ReferenceService(
            args.problem_dir,
            device=args.device,
            definition_path=args.problem_dir / "definition.json",
        )
        if not 0 <= args.row_index < len(service.workloads):
            raise ValueError("row index is outside the workload inventory")
        workload = service.workloads[args.row_index]
        if workload.uuid != args.workload_uuid:
            raise ValueError("workload UUID does not match the selected row")
        inputs = service.prepare_inputs(workload, args.row_index, 0)
        input_bytes = reference_values_storage_bytes(inputs)
        if input_bytes > MAX_REFERENCE_TENSOR_STORAGE_BYTES:
            _emit(
                "incompatible",
                "reference_ipc_payload_limit",
                "input payload exceeds the trusted reference IPC limit",
                input_storage_bytes=input_bytes,
                ipc_limit_bytes=MAX_REFERENCE_TENSOR_STORAGE_BYTES,
            )
            return
        resolved_axes = service.definition.get_resolved_axes_values(workload.axes)
        outputs = call_and_collect_outputs(
            service.reference,
            inputs,
            destination_passing_style=False,
            definition=service.definition,
            resolved_axes=resolved_axes,
            device=args.device,
            output_names=service.output_names,
            output_dtypes=service.output_dtypes,
        )
        outputs = stable_reference_outputs(outputs, inputs)
        case_bytes = reference_case_storage_bytes(ReferenceCase(inputs, outputs))
        if case_bytes > MAX_REFERENCE_TENSOR_STORAGE_BYTES:
            _emit(
                "incompatible",
                "reference_ipc_payload_limit",
                "reference case exceeds the trusted reference IPC limit",
                reference_case_bytes=case_bytes,
                ipc_limit_bytes=MAX_REFERENCE_TENSOR_STORAGE_BYTES,
            )
            return
        cache_policy = cache_clear_policy_for_device(args.device)
        allocator = ShiftingMemoryPoolAllocator(inputs, outputs, 60)
        allocator.get_unique_args()
        cache = torch.empty(
            cache_policy.clear_buffer_bytes,
            dtype=torch.int8,
            device=args.device,
        )
        cache.zero_()
        torch.cuda.synchronize(args.device)
        _emit(
            "compatible",
            "probe_passed",
            input_storage_bytes=input_bytes,
            reference_case_bytes=case_bytes,
            cache_clear_bytes=cache_policy.clear_buffer_bytes,
        )
    except torch.cuda.OutOfMemoryError as exc:
        _emit("incompatible", "probe_oom", str(exc))
    except (RuntimeError, ValueError, TypeError) as exc:
        _emit("incompatible", "reference_execution_failed", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--workload-uuid", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-arch", required=True)
    _run_probe(parser.parse_args())


if __name__ == "__main__":
    main()
