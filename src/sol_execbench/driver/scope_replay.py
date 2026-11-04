# SPDX-License-Identifier: Apache-2.0
"""One candidate invocation with fresh trusted inputs, outside canonical timing."""

import ctypes
import os
from pathlib import Path

import torch

from sol_execbench.core.bench.eval_workload_runner import evaluate_workloads
from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.output_allocation import (
    allocate_outputs,
    normalize_outputs,
)
from sol_execbench.core.bench.reward_hack import (
    ThreadInjectionMonitor,
    check_thread_injection_from_monitor,
)
from sol_execbench.core.bench.rocm_profiler.evaluate_scope import (
    verify_staged_scope,
)
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_COLLECTOR_ENV,
    SCOPE_ENV,
    SCOPE_INTENT_FILE,
    SCOPE_RECEIPT_FILE,
    EvaluateScopeIntent,
    EvaluateScopeReceipt,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.checksums import stable_json_checksum


def run_scope_replay(request: WorkloadEvaluationRequest, staging: Path) -> None:
    """Bind one eager invocation, including all implicit library dispatches."""
    directory = Path(os.environ[SCOPE_ENV])
    intent = EvaluateScopeIntent.model_validate_json(
        (staging / SCOPE_INTENT_FILE).read_bytes()
    )
    if (
        len(request.workloads) != 1
        or request.workloads[0].uuid != intent.workload_uuid
    ):
        raise ValueError("scope_workload_identity_mismatch")
    verify_staged_scope(staging, intent)
    tool = ctypes.CDLL(os.environ[SCOPE_COLLECTOR_ENV])
    tool.sol_scope_boundary.argtypes = [ctypes.c_int]
    tool.sol_scope_boundary.restype = None
    tool.sol_scope_seal.argtypes = []
    tool.sol_scope_seal.restype = None
    # Prime the candidate with a separate, validated invocation before capture.
    _invoke(request, tool, scoped=False)
    input_sha256 = _invoke(request, tool, scoped=True)
    device_uuid = str(torch.cuda.get_device_properties(request.device).uuid)
    request.dependencies.check_integrity(
        request.dependencies.integrity_snapshot,
        request.dependencies.driver_globals,
    )
    verify_staged_scope(staging, intent)
    torch.cuda.synchronize(request.device)
    tool.sol_scope_seal()
    receipt = EvaluateScopeReceipt(
        intent_sha256=stable_json_checksum(intent.model_dump(mode="json")),
        input_sha256=input_sha256,
        process_id=os.getpid(),
        device_uuid=device_uuid,
        correct=True,
        completed=True,
    )
    atomic_write_json_value(
        directory / SCOPE_RECEIPT_FILE, receipt.model_dump(mode="json")
    )


def _invoke(
    request: WorkloadEvaluationRequest, tool: ctypes.CDLL, *, scoped: bool
) -> str:
    workload = request.workloads[0]
    case = request.dependencies.reference_client.timing_iteration_case(
        workload_uuid=workload.uuid,
        row_index=0,
        trial_index=0,
        iteration_index=int(scoped),
    )
    if case.outputs:
        raise ValueError("scope_reference_outputs_disclosed")
    axes = request.definition.get_resolved_axes_values(workload.axes)
    outputs = (
        allocate_outputs(request.definition, axes, request.device)
        if request.destination_passing_style
        else []
    )
    monitor = ThreadInjectionMonitor()
    with monitor:
        torch.cuda.synchronize(request.device)
        if scoped:
            tool.sol_scope_boundary(1)
        result = request.dependencies.user_fn(*case.inputs, *outputs)
        torch.cuda.synchronize(request.device)
        if scoped:
            tool.sol_scope_boundary(0)
    check_thread_injection_from_monitor(monitor)
    request.dependencies.check_integrity(
        request.dependencies.integrity_snapshot,
        request.dependencies.driver_globals,
    )
    if not request.destination_passing_style:
        normalized = normalize_outputs(
            result,
            device=torch.device(request.device),
            output_names=request.output_names,
            output_dtypes=request.output_dtypes_torch,
        )
        outputs = [normalized[name] for name in request.output_names]
    if not outputs or any(
        output.device != torch.device(request.device) for output in outputs
    ):
        raise ValueError("scope_outputs_wrong_device")
    request.dependencies.reference_client.validate_timing_outputs(outputs)
    return case.input_sha256


def evaluate_driver_workloads(
    request: WorkloadEvaluationRequest, staging: Path
) -> None:
    """Generated-driver seam; ordinary calls retain the original timing path."""
    if SCOPE_ENV in os.environ:
        run_scope_replay(request, staging)
    else:
        evaluate_workloads(request)
