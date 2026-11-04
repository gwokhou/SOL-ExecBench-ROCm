# SPDX-License-Identifier: Apache-2.0
"""Collect separate SDK coverage evidence for actual eager PyTorch workloads."""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from scripts.internal.upgrade.torch_scope_workload import PROBLEMS
from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileRequest,
    collect_rocprofv3_profile,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    MAX_SDK_API_RECORDS,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    HIPTorchExpectation,
)
from sol_execbench.core.bench.rocm_profiler.hip_torch_scope import (
    qualify_torch_scope,
)
from sol_execbench.core.bench.rocm_profiler.models import Rocprofv3ProfileResult
from sol_execbench.core.bench.rocm_profiler.pmc_clock import (
    TARGET_ROCR_UUID,
    pmc_clock_window,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.integrity.schema_versions import SchemaVersion

REPETITIONS = 3
TIMEOUT_SECONDS = 60


def collect_case(tool: Path, directory: Path, case: str) -> dict:
    """Retain every row; qualification happens separately after normal exit."""
    directory.mkdir()
    nonce = uuid4().hex
    executor = Path(__file__).with_name("torch_scope_workload.py")
    definition = (
        executor.parents[3]
        / "problems/AMD_AKA/torch2hip"
        / PROBLEMS[case]
        / "definition.json"
    )
    command = (
        sys.executable,
        str(executor),
        case,
        str(tool),
        str(directory / "workload.json"),
        nonce,
    )
    intent = {
        "run_id": nonce,
        "workload_id": case,
        "device_uuid": TARGET_ROCR_UUID,
        "definition_sha256": sha256_file(definition),
        "executor_sha256": sha256_file(executor),
        "collector_sha256": sha256_file(tool),
        "application_command": list(command),
    }
    atomic_write_json_value(directory / "controller-intent.json", intent)
    result = _collect(command, directory, tool, nonce)
    atomic_write_json_value(directory / "collection.json", result.to_dict())
    if not result.succeeded or result.returncode != 0:
        raise RuntimeError(f"PyTorch scope collection failed: {case}")
    hashes = {
        name: sha256_file(directory / name)
        for name in ("torch_results.json", "ledger.json", "workload.json")
    }
    atomic_write_json_value(directory / "collection-binding.json", hashes)
    qualified = qualify_torch_scope(
        directory,
        HIPTorchExpectation.model_validate({**intent, "returncode": 0}),
        hashes,
        executor_path=executor,
        collector_path=tool,
        definition_path=definition,
    )
    if (
        len(qualified.links) != int(case == "event_chain")
        or not qualified.implicit_dispatch_ids
        or len(qualified.dispatches) <= len(qualified.implicit_dispatch_ids)
    ):
        raise RuntimeError(
            "capture does not cover the planned library/event cases"
        )
    atomic_write_json_value(
        directory / "scope.json", qualified.model_dump(mode="json")
    )
    return {
        "case": case,
        "run_id": nonce,
        "qualified": True,
        "dispatches": len(qualified.dispatches),
        "implicit_dispatches": len(qualified.implicit_dispatch_ids),
        "links": len(qualified.links),
    }


def _collect(
    command: tuple[str, ...], directory: Path, tool: Path, nonce: str
) -> Rocprofv3ProfileResult:
    """Keep SDK tool injection local to the separate diagnostic subprocess."""
    environment = {
        "ROCP_TOOL_LIBRARIES": str(tool),
        "SOL_SCOPE_LEDGER": str(directory / "ledger.json"),
        "SOL_SCOPE_RUN": nonce,
        "SOL_SCOPE_SCHEMA": str(SchemaVersion.HIP_SDK_LEDGER),
        "SOL_SCOPE_MAX_ROWS": str(MAX_SDK_API_RECORDS),
    }
    previous = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        result = collect_rocprofv3_profile(
            Rocprofv3ProfileRequest(
                application_command=command,
                output_directory=directory,
                output_file="torch",
                timeout_seconds=TIMEOUT_SECONDS,
            )
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result


def acquire(tool: Path, output: Path, cases: list[str]) -> None:
    """Borrow the production host guard and require verified STABLE_PEAK."""
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with (
        native_gpu_execution(timeout_seconds=30),
        pmc_clock_window(output) as lease,
    ):
        if not lease.locked:
            raise RuntimeError("STABLE_PEAK is required")
        for repeat in range(REPETITIONS):
            for case in cases:
                result = collect_case(tool, output / f"{repeat}-{case}", case)
                results.append({**result, "repetition": repeat})
                atomic_write_json_value(output / "results.json", results)
                print(f"COLLECTED {repeat}-{case}", flush=True)


def main() -> None:
    """Run under the existing UUID-scoped host guard, without timing claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", choices=PROBLEMS, nargs="+", default=list(PROBLEMS)
    )
    args = parser.parse_args()
    acquire(args.tool.resolve(), args.output.resolve(), args.cases)


if __name__ == "__main__":
    main()
