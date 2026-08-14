# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current CLI, worker, and GPU qualification control-plane schemas."""

from enum import StrEnum


class ExecutionControlSchema(StrEnum):
    """Canonical execution control-plane artifact identifiers."""

    BATCH_GPU_QUALIFICATION = "sol_execbench.batch_gpu_qualification.v1"
    CLI_PROTOCOL = "sol_execbench.cli_protocol.v1"
    SOLAR_WORKER_IPC = "sol_execbench.solar_worker_ipc.v2"


class BatchGPUQualificationArtifactKind(StrEnum):
    """Artifacts in one large-batch qualification contract family."""

    GATE = "gate"
    RECEIPT = "receipt"


class CLIArtifactKind(StrEnum):
    """Machine-readable CLI protocol artifacts."""

    CONTRACT = "contract"
    RESPONSE = "response"


__all__ = [
    "BatchGPUQualificationArtifactKind",
    "CLIArtifactKind",
    "ExecutionControlSchema",
]
