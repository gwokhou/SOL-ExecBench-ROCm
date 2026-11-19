"""Run one development session with the existing evaluator and bounded scratch."""

import json
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from pilot_capture import CAPTURE_BYTES

from scripts.internal.solar.e2_contracts import (
    CORRECTNESS_TIMEOUT_SECONDS,
    COST_TIMEOUT_SECONDS,
    PRIVATE_FILE,
    Activity,
    Session,
)
from scripts.internal.solar.e2_runner import (
    _record_preparation,
    _session_result,
)
from scripts.internal.solar.e2_staging import instrument_drivers, package
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.evaluation.runtime import run_evaluation_runtime
from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_DEVICE,
    ENV_SOL_EXECBENCH_INPUT_NONCE,
)
from sol_execbench.core.process.resource_guard import evidence_size

SCRATCH = Path("/e2-scratch")
SCRATCH_BYTES = 256 * 1024**2
EVIDENCE_SESSION_BYTES = CAPTURE_BYTES + 16 * 1024**2
PILOT_IMPORT = (
    "from pilot_capture import serve_pilot_worker as serve_reference_worker"
)
ORIGINAL_IMPORT = "from scripts.internal.solar.e2_service import serve_observed_worker as serve_reference_worker"


def scratch_usage() -> tuple[int, int]:
    """Return allocated tmpfs bytes and the enforced capacity."""
    usage = os.statvfs(SCRATCH)
    return (
        usage.f_blocks - usage.f_bfree
    ) * usage.f_frsize, usage.f_blocks * usage.f_frsize


def bound_mount() -> None:
    """Require an empty and exactly sized kernel tmpfs."""
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    if not any(
        row.split()[4] == str(SCRATCH)
        and row.split(" - ")[1].split()[0] == "tmpfs"
        for row in mounts
    ):
        raise RuntimeError("pilot scratch is not a dedicated tmpfs")
    if scratch_usage() != (0, SCRATCH_BYTES):
        raise RuntimeError("pilot scratch size or starting occupancy differs")


def measure_scratch(stop: threading.Event, samples: list[int]) -> None:
    """Sample allocated temporary bytes while the evaluator runs."""
    while not stop.wait(0.025):
        samples[0] = max(samples[0], scratch_usage()[0])


def instrument(output: Path, fault: str | None) -> None:
    """Select the pilot service in a staged copy of the driver."""
    instrument_drivers(output)
    worker = output / "reference_worker.py"
    source = worker.read_text()
    if source.count(ORIGINAL_IMPORT) != 1:
        raise RuntimeError("pilot reference import seam drifted")
    worker.write_text(source.replace(ORIGINAL_IMPORT, PILOT_IMPORT))
    private = output / PRIVATE_FILE
    control = json.loads(private.read_text())
    atomic_write_json_value(private, {**control, "pilot_fault": fault})


def validate_artifacts(output: Path, result: dict[str, Any]) -> None:
    """Verify the observation digest and byte count after publication."""
    observations = result.get("observations", {})
    files = observations.get("observations", [])
    directory = output / "judge-observations"
    for row in files:
        path = directory / row["artifact"]
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("pilot artifact hash mismatch")
    if (
        observations
        and sum((directory / row["artifact"]).stat().st_size for row in files)
        != observations["capture_bytes"]
    ):
        raise ValueError("pilot capture byte accounting mismatch")


def run_session(
    repository: Path,
    output: Path,
    session: Session,
    *,
    device: str,
    fault: str | None = None,
) -> dict[str, Any]:
    """Run and account for one cold-start development session."""
    seed = session.case.require_development()
    bound_mount()
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    atomic_write_json_value(
        output / "started.json",
        {**session.metadata(), "fault": fault, "final_verdict": None},
    )
    with TemporaryDirectory(dir=SCRATCH) as scratch:
        result = _run_in_scratch(
            repository, output, session, device, Path(scratch), seed, fault
        )
    result["scratch_after_cleanup_bytes"] = scratch_usage()[0]
    result["session_wall_seconds"] = time.monotonic() - start
    result["capture_verified"] = False
    result["retained_bytes_before_result"] = evidence_size(
        output, EVIDENCE_SESSION_BYTES
    )
    atomic_write_json_value(output / "session.json", result)
    validate_artifacts(output, result)
    result["capture_verified"] = "observations" in result
    atomic_write_json_value(output / "session.json", result)
    if result["scratch_after_cleanup_bytes"] != 0:
        raise RuntimeError("scratch cleanup did not return usage to zero")
    if result["retained_bytes_before_result"] >= EVIDENCE_SESSION_BYTES:
        raise RuntimeError("pilot session exceeded retained-evidence allowance")
    return result


def _run_in_scratch(
    repository: Path,
    output: Path,
    session: Session,
    device: str,
    scratch: Path,
    seed: int,
    fault: str | None,
) -> dict[str, Any]:
    cache_names = {".cache": scratch / "cache", ".tmp": scratch / "tmp"}
    for name, path in cache_names.items():
        path.mkdir()
        (output / name).symlink_to(path, target_is_directory=True)
    database = scratch / "miopen-db"
    database.mkdir()
    env = {
        ENV_SOL_EXECBENCH_DEVICE: device,
        ENV_SOL_EXECBENCH_INPUT_NONCE: uuid4().hex + uuid4().hex,
        "MIOPEN_USER_DB_PATH": str(database),
    }
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    stop = threading.Event()
    samples = [0]
    monitor = threading.Thread(target=measure_scratch, args=(stop, samples))
    monitor.start()
    try:
        result = _evaluate(repository, output, session, device, seed, fault)
    finally:
        stop.set()
        monitor.join()
        for name in cache_names:
            (output / name).unlink()
        (output / PRIVATE_FILE).unlink(missing_ok=True)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    cache_files = [
        path
        for path in scratch.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    result["scratch"] = {
        "start_bytes": 0,
        "peak_sampled_allocated_bytes": max(samples[0], scratch_usage()[0]),
        "end_logical_bytes": sum(path.stat().st_size for path in cache_files),
        "comgr_files": sum("comgr" in path.parts for path in cache_files),
        "limit_bytes": SCRATCH_BYTES,
    }
    return result


def _evaluate(
    repository: Path,
    output: Path,
    session: Session,
    device: str,
    seed: int,
    fault: str | None,
) -> dict[str, Any]:
    start = time.monotonic()
    native = (
        native_gpu_execution(device, timeout_seconds=60, manage_clocks=True)
        if device == "cuda:0"
        else nullcontext()
    )
    with (
        native,
        package(
            repository,
            output,
            session,
            workload_uuid=str(uuid4()),
            private_seed=seed,
        ) as packager,
    ):
        if device == "cpu":
            packager.config.lock_clocks = False
            (output / "config.json").write_text(
                packager.config.model_dump_json()
            )
        command = packager.execute()
        instrument(output, fault)
        prepared = time.monotonic() - start
        limit = (
            COST_TIMEOUT_SECONDS
            if session.activity is Activity.COST
            else CORRECTNESS_TIMEOUT_SECONDS
        )
        runtime = run_evaluation_runtime(
            packager,
            eval_cmd=command,
            staging_dir=output,
            output_file=output / "trace.jsonl",
            timeout=max(1, int(limit - prepared)),
            profile=ProfileMode.NONE,
            expected_trace_count=1,
        )
        result = _session_result(session, output, runtime)
        _record_preparation(result, packager, prepared)
    result["evaluation_wall_seconds"] = time.monotonic() - start
    return result
