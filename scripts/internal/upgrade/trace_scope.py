# SPDX-License-Identifier: Apache-2.0
"""Opt-in acceptance for an audited native executor's complete HIP scope."""

import argparse
from pathlib import Path
from uuid import uuid4

from scripts.internal.upgrade.event_causality import EXPECTED_LINKS
from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileRequest,
    collect_rocprofv3_profile,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope import (
    qualify_hip_trace_scope,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeExpectation,
    HIPTraceScopeQualification,
)
from sol_execbench.core.bench.rocm_profiler.pmc_clock import (
    TARGET_ROCR_UUID,
    pmc_clock_window,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.checksums import sha256_bytes, sha256_file
from sol_execbench.core.integrity.schema_versions import SchemaVersion

CASES = ("reuse", "recreate", "independent")
REPETITIONS = 3
TIMEOUT_SECONDS = 45
SOURCE_NAMES = ("hip_trace_scope_probe.cpp", "hip_trace_scope_journal.h")


def collect_case(probe: Path, directory: Path, case: str) -> dict:
    """Save controller intent before collection, then require exact scope."""
    directory.mkdir()
    journal = directory / "journal.json"
    nonce = uuid4().hex
    binary_hash = sha256_file(probe)
    command = (
        str(probe),
        case,
        str(journal),
        nonce,
        binary_hash,
        str(SchemaVersion.HIP_TRACE_JOURNAL),
    )
    sources = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in SOURCE_NAMES
    }
    intent = {
        "run_id": nonce,
        "workload_id": case,
        "candidate_sha256": binary_hash,
        "executor_source_sha256": sha256_bytes(
            "".join(sources.values()).encode()
        ),
        "device_uuid": TARGET_ROCR_UUID,
        "application_command": list(command),
    }
    atomic_write_json_value(
        directory / "controller-intent.json",
        {**intent, "source_files": sources},
    )
    result = collect_rocprofv3_profile(
        Rocprofv3ProfileRequest(
            application_command=command,
            output_directory=directory,
            output_file="probe",
            timeout_seconds=TIMEOUT_SECONDS,
        )
    )
    atomic_write_json_value(directory / "collection.json", result.to_dict())
    if not result.succeeded or result.returncode != 0:
        raise RuntimeError(
            "controlled scope collection did not finish successfully"
        )
    expected = HIPScopeExpectation.model_validate(
        {**intent, "returncode": result.returncode}
    )
    trace = directory / "probe_results.json"
    retained = {
        "trace_sha256": sha256_file(trace),
        "journal_sha256": sha256_file(journal),
    }
    atomic_write_json_value(directory / "collection-binding.json", retained)
    qualified = qualify_hip_trace_scope(
        trace, journal, probe, expected, **retained
    )
    expected_dispatches, expected_links = validate_kernel_oracle(
        qualified, case
    )
    atomic_write_json_value(
        directory / "scope.json", qualified.model_dump(mode="json")
    )
    return {
        "case": case,
        "run_id": nonce,
        "dispatches": expected_dispatches,
        "links": expected_links,
        "scope_sha256": sha256_file(directory / "scope.json"),
    }


def validate_kernel_oracle(
    qualified: HIPTraceScopeQualification, case: str
) -> tuple[int, int]:
    """Compare every bound kernel and causal edge with the audited fixture plan."""
    expected_dispatches, expected_links = (
        (1, 0) if case == "independent" else (6, 3)
    )
    if (
        len(qualified.dispatches) != expected_dispatches
        or len(qualified.links) != expected_links
    ):
        raise RuntimeError("scope differs from the audited fixture plan")
    names = {
        item.dispatch_id: item.kernel_name for item in qualified.dispatches
    }
    pairs = {
        (names[link.predecessor_dispatch_id], names[link.successor_dispatch_id])
        for link in qualified.links
    }
    if pairs != (frozenset() if case == "independent" else EXPECTED_LINKS):
        raise RuntimeError(
            "event links differ from the independent kernel-role oracle"
        )
    return expected_dispatches, expected_links


def acquire(probe: Path, output: Path) -> None:
    """Use the existing host lock borrowing and strict STABLE_PEAK window."""
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with (
        native_gpu_execution(timeout_seconds=30),
        pmc_clock_window(output) as lease,
    ):
        if not lease.locked:
            raise RuntimeError("STABLE_PEAK is required")
        for repeat in range(REPETITIONS):
            for case in CASES if repeat % 2 == 0 else reversed(CASES):
                result = collect_case(probe, output / f"{repeat}-{case}", case)
                result["repetition"] = repeat
                results.append(result)
                atomic_write_json_value(output / "results.json", results)
                print(
                    f"PASS {repeat}-{case}: {result['dispatches']} bound dispatches",
                    flush=True,
                )


def main() -> None:
    """Run under the existing UUID-scoped host guard after executor audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.probe.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
