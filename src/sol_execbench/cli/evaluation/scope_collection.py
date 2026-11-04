# SPDX-License-Identifier: Apache-2.0
"""Production candidate-only collection with a sanitized, local environment."""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.cli.evaluation.command import _evaluation_env
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileRequest,
    collect_rocprofv3_profile,
)
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_COLLECTOR_ENV,
    SCOPE_ENV,
    SCOPE_LAUNCH_FILE,
    SCOPE_LEDGER_ENV,
    SCOPE_LEDGER_FILE,
    SCOPE_MAX_ROWS_ENV,
    SCOPE_RUN_ENV,
    SCOPE_SCHEMA_ENV,
    SCOPE_TOOL_ENV,
    SCOPE_TRACE_DIRECTORY,
    EvaluateScopeIntent,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    MAX_SDK_API_RECORDS,
)
from sol_execbench.core.bench.rocm_profiler.models import Rocprofv3ProfileResult
from sol_execbench.core.bench.safetensors_io import flashinfer_safetensors_env
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded


def collect_evaluate_scope(
    *,
    staging: Path,
    directory: Path,
    tool: Path,
    intent: EvaluateScopeIntent,
    timeout: int,
) -> Rocprofv3ProfileResult:
    """Inject only parent-owned scope settings after environment sanitization."""
    environment = _evaluation_env(
        staging, flashinfer_safetensors_env, graceful_exit=True
    )
    injection = {
        SCOPE_ENV: str(directory),
        SCOPE_COLLECTOR_ENV: str(tool),
        SCOPE_TOOL_ENV: str(tool),
        SCOPE_LEDGER_ENV: str(directory / SCOPE_LEDGER_FILE),
        SCOPE_RUN_ENV: intent.run_id,
        SCOPE_SCHEMA_ENV: str(SchemaVersion.HIP_SDK_LEDGER),
        SCOPE_MAX_ROWS_ENV: str(MAX_SDK_API_RECORDS),
    }

    def run(
        args: Sequence[str], cwd: Path | None, limit: int | None
    ) -> subprocess.CompletedProcess[str]:
        atomic_write_json_value(
            staging / SCOPE_LAUNCH_FILE,
            {"command": list(args), "environment": injection},
        )
        return run_in_process_group_bounded(
            [sys.executable, "evaluation_orchestrator.py"],
            cwd=cwd,
            timeout=limit,
            env={**environment, SCOPE_ENV: str(directory)},
        )

    return collect_rocprofv3_profile(
        Rocprofv3ProfileRequest(
            application_command=intent.application_command,
            output_directory=directory / SCOPE_TRACE_DIRECTORY,
            output_file="scope",
            working_directory=staging,
            timeout_seconds=timeout,
        ),
        runner=run,
    )
