# SPDX-License-Identifier: Apache-2.0
"""Bounded opt-in GPU acceptance for diagnostic HIP event dispatch links."""

import argparse
import json
from pathlib import Path

from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileRequest,
    collect_rocprofv3_profile,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_artifacts import (
    read_hip_event_causality,
)
from sol_execbench.core.bench.rocm_profiler.pmc_clock import pmc_clock_window
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.checksums import sha256_file

CASES = ("reuse", "recreate", "independent", "unrecorded")
REPETITIONS = 3
TIMEOUT_SECONDS = 45
EXPECTED_LINKS = frozenset(
    {
        ("produce_first", "consume_first"),
        ("produce_first", "consume_fanout"),
        ("produce_second", "consume_second"),
    }
)


def validate_capture(path: Path, case: str) -> dict:
    """Compare inferred IDs with independently named fixture kernel roles."""
    captures = json.loads(path.read_text())["rocprofiler-sdk-tool"]
    if len(captures) != 1:
        raise ValueError("expected a single process capture")
    capture = captures[0]
    symbols = {
        item["kernel_id"]: item["kernel_name"].removesuffix(".kd")
        for item in capture["kernel_symbols"]
    }
    names = {
        item["dispatch_info"]["dispatch_id"]: symbols[
            item["dispatch_info"]["kernel_id"]
        ]
        for item in capture["buffer_records"]["kernel_dispatch"]
    }
    results, warnings = read_hip_event_causality(path)
    if warnings or any(item.reason_codes for item in results):
        raise ValueError(f"causal capture rejected: {warnings}")
    normalized = [item.model_dump(mode="json") for item in results]
    atomic_write_json_value(path.parent / "causality.json", normalized)
    links = [link for item in results for link in item.links]
    pairs = {
        (names[link.predecessor_dispatch_id], names[link.successor_dispatch_id])
        for link in links
    }
    expected = EXPECTED_LINKS if case in {"reuse", "recreate"} else frozenset()
    if pairs != expected or len(links) != len(expected):
        raise ValueError(f"unexpected dispatch edges: {pairs}")
    unresolved = [wait for item in results for wait in item.unresolved_waits]
    if len(unresolved) != int(case == "unrecorded"):
        raise ValueError("unexpected unresolved wait count")
    return {
        "case": case,
        "links": sorted(pairs),
        "unresolved_waits": len(unresolved),
        "artifact_sha256": sha256_file(path),
    }


def acquire(probe: Path, output: Path) -> None:
    """Require an owned/borrowed strict clock window and retain every attempt."""
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with (
        native_gpu_execution(timeout_seconds=30),
        pmc_clock_window(output) as lease,
    ):
        if not lease.locked:
            raise RuntimeError("STABLE_PEAK is required for live acceptance")
        for repetition in range(REPETITIONS):
            order = CASES if repetition % 2 == 0 else tuple(reversed(CASES))
            for case in order:
                directory = output / f"{repetition}-{case}"
                result = collect_rocprofv3_profile(
                    Rocprofv3ProfileRequest(
                        application_command=(str(probe), case),
                        output_directory=directory,
                        output_file="probe",
                        timeout_seconds=TIMEOUT_SECONDS,
                    )
                )
                atomic_write_json_value(
                    directory / "collection.json", result.to_dict()
                )
                if not result.succeeded or not any(
                    line.startswith('{"case":')
                    and json.loads(line).get("correct") is True
                    for line in result.stdout.splitlines()
                ):
                    raise RuntimeError(f"GPU fixture failed: {case}")
                evidence = validate_capture(
                    directory / "probe_results.json", case
                )
                evidence["repetition"] = repetition
                results.append(evidence)
                atomic_write_json_value(output / "results.json", results)
                print(
                    f"PASS {repetition}-{case}: {len(evidence['links'])} links",
                    flush=True,
                )


def main() -> None:
    """Run only when explicitly invoked inside the qualified host guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.probe.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
