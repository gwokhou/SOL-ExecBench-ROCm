"""Launch the frozen E2 activity once and record the native GPU handback."""

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from scripts.internal.solar.e2_contracts import (
    EVIDENCE_BUDGET_BYTES,
    GPU_BUDGET_SECONDS,
)
from scripts.internal.solar.e2_evidence_budget import retained_evidence
from scripts.internal.solar.e2_successor_audit import _AUDIT_NAME
from scripts.internal.solar.e2_successor_entry import (
    _ENTRY_MODULE,
    _NATIVE_CLOSEOUT,
    _OWNER_START,
)
from sol_execbench.core.bench.clock_lock import observe_clock_levels
from sol_execbench.core.bench.clock_ownership import (
    ClockOwnerRecord,
    ClockOwnerState,
    clock_owner_path,
)
from sol_execbench.core.bench.gpu_admission import (
    ADMISSION_ENV,
    CLOCK_STATE_ENV,
    AdmissionDevice,
    AdmissionState,
    configured_admission,
)
from sol_execbench.core.bench.gpu_execution import PCI_DEVICES_ROOT
from sol_execbench.core.bench.gpu_lock_paths import (
    GPU_LOCK_DIR_ENV,
    GPU_LOCK_UUID_ENV,
    configured_gpu_locks,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.amd_smi import parse_processes
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.process.termination import (
    cleanup_on_termination,
    defer_termination,
)
from sol_execbench.core.scoring.release_environment import (
    CONTAINER_IMAGE_ID_ENV,
)
from sol_execbench.tools.docker_process import _remove_owned_container

_CONTAINER_REPO = "/sol-execbench"
_CONTAINER_BUNDLE = "/outputs"
_CONTAINER_FREEZE = f"{_CONTAINER_BUNDLE}/freeze.json"
_CONTAINER_PYTHON = "/venv/bin/python"
_MODULE_PREFIX = "scripts.internal.solar"
_HOST = "host-run"
_CPU_TIMEOUT_SECONDS = 7 * 3600
_PROBE_SECONDS = 20
_CAPTURE_BYTES = 16384
_CPU_MEMORY = "4g"
_TERMINAL_RECEIPT_RESERVE = 1024**2
_PYTHONPATH = f"/research:{_CONTAINER_REPO}/src:{_CONTAINER_REPO}"
_PYTHON_FLAGS = ("-B", "-P", "-m")
_CACHE_MOUNT = "/e2-scratch:rw,exec,nosuid,nodev,size=256m"
_TEMP_MOUNT = "/tmp:rw,exec,nosuid,nodev,size=128m"
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_NATIVE_ENV_KEYS = (
    GPU_LOCK_DIR_ENV,
    GPU_LOCK_UUID_ENV,
    CLOCK_STATE_ENV,
    ADMISSION_ENV,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class HostExecution:
    """A complete invocation bound to one already verified freeze digest."""

    repository: Path
    bundle: Path
    freeze_sha256: str
    spec: dict[str, Any]
    image_tag: str

    @property
    def activity(self) -> Path:
        """Use the only result directory sealed into this freeze."""
        return self.bundle / self.spec["activity_directory"]


def _mount(
    source: Path, target: str, *, writable: bool = False
) -> tuple[str, str]:
    mode = "" if writable else ",readonly"
    return "--mount", f"type=bind,source={source},target={target}{mode}"


def _entry_arguments(stage: str, activity: str) -> list[str]:
    common = ["--freeze", _CONTAINER_FREEZE]
    if stage == "report":
        return [
            f"{_MODULE_PREFIX}.e2_successor_report",
            "--evidence-root",
            f"{_CONTAINER_BUNDLE}/{activity}",
            *common,
        ]
    module = (
        f"{_MODULE_PREFIX}.e2_successor_audit"
        if stage == "audit"
        else _ENTRY_MODULE
    )
    return [
        module,
        *([stage] if stage != "audit" else []),
        "--repository",
        _CONTAINER_REPO,
        "--evidence-root",
        _CONTAINER_BUNDLE,
        *common,
        *(["--open-holdout"] if stage == "online" else []),
    ]


def _entry_mounts(run: HostExecution) -> list[str]:
    return [
        *_mount(run.bundle / "capsule", "/research"),
        *_mount(run.bundle / "capsule", f"{_CONTAINER_BUNDLE}/capsule"),
        *_mount(run.bundle / "freeze.json", _CONTAINER_FREEZE),
        *_mount(run.bundle / "readiness", f"{_CONTAINER_BUNDLE}/readiness"),
    ]


def _environment_args(values: Mapping[str, str]) -> list[str]:
    return [
        part
        for key, value in values.items()
        for part in ("-e", f"{key}={value}")
    ]


def phase_commands(run: HostExecution) -> dict[str, list[str]]:
    """Derive the exact native online and device-free stage arguments."""
    affinity = "--cpuset-cpus=" + ",".join(map(str, run.spec["cpu_affinity"]))
    online = [
        "bash",
        str(run.repository / "scripts/run_docker.sh"),
        "--record-container-validation",
        "--compatibility-entry",
        str(run.bundle / _HOST / "compatibility.json"),
        # The reference worker publishes after the candidate exits and is adopted.
        "--init",
        "--read-only",
        f"--memory={run.spec['worker_memory_bytes']}",
        f"--memory-swap={run.spec['worker_memory_bytes']}",
        affinity,
        "--tmpfs",
        _TEMP_MOUNT,
        "--tmpfs",
        _CACHE_MOUNT,
        *_entry_mounts(run),
        *_environment_args(
            {
                **_THREAD_ENV,
                "PYTHONPATH": _PYTHONPATH,
                "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
            }
        ),
        "--",
        _CONTAINER_PYTHON,
        *_PYTHON_FLAGS,
        *_entry_arguments("online", run.activity.name),
    ]
    cpu = [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        f"--memory={_CPU_MEMORY}",
        f"--memory-swap={_CPU_MEMORY}",
        affinity,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        _CONTAINER_REPO,
        "--entrypoint",
        _CONTAINER_PYTHON,
        "--tmpfs",
        _TEMP_MOUNT,
        *_mount(run.repository, _CONTAINER_REPO),
        *_mount(run.bundle, _CONTAINER_BUNDLE, writable=True),
        *_entry_mounts(run),
        *_environment_args(
            {
                **_THREAD_ENV,
                CONTAINER_IMAGE_ID_ENV: run.spec["image_id"],
                "PYTHONPATH": _PYTHONPATH,
                "SOL_EXECBENCH_SANDBOXED": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
            }
        ),
        run.spec["image_id"],
        *_PYTHON_FLAGS,
    ]
    return {
        "online": online,
        **{
            phase: [*cpu, *_entry_arguments(phase, run.activity.name)]
            for phase in ("offline", "audit", "report")
        },
    }


def _native_environment(run: HostExecution) -> dict[str, str]:
    locks = configured_gpu_locks()
    device = run.spec["device"]
    if locks.uuid != device["uuid"]:
        raise ValueError("E2 launch environment targets another GPU")
    native_id = (
        (PCI_DEVICES_ROOT / device["bdf"] / "unique_id").read_text().strip()
    )
    if not native_id:
        raise ValueError("E2 target lacks its native ROCr identity")
    image = run_in_process_group_bounded(
        [
            "docker",
            "image",
            "inspect",
            "--format={{.Id}}",
            f"sol-execbench:{run.image_tag}",
        ],
        timeout=_PROBE_SECONDS,
        max_capture_bytes=_CAPTURE_BYTES,
    )
    if image.returncode or image.stdout.strip() != run.spec["image_id"]:
        raise ValueError("E2 online image tag differs from the frozen image")
    return {
        **os.environ,
        "IMAGE_NAME": "sol-execbench",
        "IMAGE_TAG": run.image_tag,
        "SOL_EXECBENCH_OUTPUT_DIR": str(run.bundle),
        "SOL_EXECBENCH_HOST_PYTHON": sys.executable,
        "ROCR_VISIBLE_DEVICES": f"GPU-{native_id}",
        "HIP_VISIBLE_DEVICES": "0",
    }


def _locks_clear(path: Path, proc_locks: list[str]) -> bool:
    stat = path.stat(follow_symlinks=False)
    identity = (
        f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    )
    return all(identity not in row.split() for row in proc_locks)


def _record_handback(run: HostExecution, code: int, elapsed: float) -> bool:
    target = run.activity / _NATIVE_CLOSEOUT
    started = ClockOwnerRecord.model_validate_json(
        (run.activity / _OWNER_START).read_text()
    )
    locks = configured_gpu_locks()
    restored = ClockOwnerRecord.model_validate_json(
        clock_owner_path(locks.directory, started.lease_id).read_text()
    )
    clock_levels = observe_clock_levels(locks.uuid)
    active = run_in_process_group_bounded(
        [
            resolve_rocm_tool_command("amd-smi"),
            "process",
            "-g",
            locks.uuid,
            "--json",
        ],
        timeout=_PROBE_SECONDS,
        max_capture_bytes=_CAPTURE_BYTES,
    )
    admission = configured_admission()
    if admission is None or active.returncode:
        raise ValueError(
            "E2 cannot observe native recovery and device processes"
        )
    current = admission.read(
        AdmissionDevice(pci_bdf=run.spec["device"]["bdf"], smi_uuid=locks.uuid)
    )
    processes = parse_processes(active.stdout)
    live_locks = Path("/proc/locks").read_text().splitlines()
    levels = {
        "L1-ownership": _locks_clear(locks.ownership, live_locks),
        "L2-execution": _locks_clear(locks.execution, live_locks),
    }
    clean = (
        elapsed <= GPU_BUDGET_SECONDS
        and restored.state is ClockOwnerState.RESTORED
        and restored.release_blocker is None
        and restored.started_at == started.started_at
        and restored.devices == started.devices
        and current.state is AdmissionState.READY
        and current.owner is None
        and clock_levels == ("AMDSMI_DEV_PERF_LEVEL_AUTO",)
        and not processes
        and all(levels.values())
    )
    atomic_write_json_value(
        target,
        {
            "clean": clean,
            "returncode": code,
            "native_wrapper_wall_seconds": elapsed,
            "receipts": {
                f"{started.lease_id}.json": restored.model_dump(mode="json")
            },
            "admission": current.model_dump(mode="json"),
            "clocks": {
                "gpu_data": [{"perf_level": x} for x in (clock_levels or ())]
            },
            "gpu_process_count": len(processes),
            "locks": [
                {"level": level, "held": not clear}
                for level, clear in levels.items()
            ],
        },
    )
    return clean


def _run_stage(
    command: Sequence[str],
    receipt: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    start = time.monotonic()
    cid = receipt.with_suffix(".cid")
    online = env is not None
    argv = (
        list(command)
        if online
        else [*command[:2], "--cidfile", str(cid), *command[2:]]
    )
    result = None
    timeout = GPU_BUDGET_SECONDS if online else _CPU_TIMEOUT_SECONDS
    timed_out = False
    try:
        result = run_in_process_group_bounded(
            argv,
            env=env,
            timeout=timeout,
            max_capture_bytes=_CAPTURE_BYTES,
        )
        return result.returncode
    except TimeoutExpired:
        timed_out = True
        return 124
    finally:
        atomic_write_json_value(
            receipt,
            {
                "returncode": result.returncode
                if result
                else (124 if timed_out else None),
                "timed_out": timed_out,
                "wall_seconds": time.monotonic() - start,
                "stdout": result.stdout if result else None,
                "stderr": result.stderr if result else None,
            },
        )
        if cid.exists():
            with defer_termination():
                _remove_owned_container(argv[0], cid)


def _verify_terminal_outputs(run: HostExecution) -> dict[str, Any]:
    audit_path = run.activity / _AUDIT_NAME
    audit = load_json_value(audit_path)
    report = run.activity / "report"
    index_path = report / "report-index.json"
    index = load_json_value(index_path)
    correctness, cost = (
        load_json_value(report / name)
        for name in ("correctness.json", "cost.json")
    )
    if (
        audit["state"] != "completed"
        or audit["freeze_sha256"] != run.freeze_sha256
        or len(audit["online"]) != 1440
        or len(audit["offline"]) != 72
        or audit["roster_sha256"] != index["roster_sha256"]
        or index["output_sha256"]
        != {
            name: sha256_file(report / name)
            for name in ("correctness.json", "cost.json")
        }
        or correctness["logical_units"] != 432
        or correctness["attempt_states"] != {"completed": 1296}
        or cost["planned_blocks"] != 24
        or cost["planned_pairs"] != 72
        or cost["offline_states"] != {"completed": 72}
    ):
        raise ValueError(
            "E2 terminal reports do not cover the frozen experiment"
        )
    sizes = [
        retained_evidence(path) for path in (run.activity, run.bundle / _HOST)
    ]
    allocated = sum(x.allocated_bytes for x in sizes)
    logical = sum(x.logical_bytes for x in sizes)
    if (
        max(allocated, logical) + _TERMINAL_RECEIPT_RESERVE
        > EVIDENCE_BUDGET_BYTES
    ):
        raise ValueError(
            "E2 host and activity evidence exceeded the frozen capacity"
        )
    return {
        "state": "completed",
        "audit_sha256": sha256_file(audit_path),
        "report_index_sha256": sha256_file(index_path),
        "retained_allocated_bytes": allocated,
        "retained_logical_bytes": logical,
        "performance_authority": False,
    }


def _host_plan(
    run: HostExecution, commands: dict[str, list[str]]
) -> dict[str, Any]:
    return {
        "freeze_sha256": run.freeze_sha256,
        "host_entry_sha256": sha256_file(Path(__file__)),
        "image_tag": run.image_tag,
        "native_profile": {
            name: os.environ.get(name) for name in _NATIVE_ENV_KEYS
        },
        "commands": commands,
        "performance_authority": False,
    }


def _run_activity(
    run: HostExecution, commands: dict[str, list[str]], plan: dict[str, Any]
) -> int:
    host = run.bundle / _HOST
    if run.activity.exists():
        raise FileExistsError(run.activity)
    env = _native_environment(run)
    host.mkdir(exist_ok=False)
    atomic_write_json_value(host / "launch.json", plan)
    online = host / "online.json"
    with cleanup_on_termination():
        code = _run_stage(commands["online"], online, env=env)
        if not (run.activity / _OWNER_START).is_file():
            return code or 1
        if not _record_handback(
            run, code, load_json_value(online)["wall_seconds"]
        ):
            return 1
        offline = (
            _run_stage(commands["offline"], host / "offline.json")
            if code == 0
            else 1
        )
        audit = _run_stage(commands["audit"], host / "audit.json")
        report = _run_stage(commands["report"], host / "report.json")
    if any((code, offline, audit, report)):
        return 1
    atomic_write_json_value(
        host / "completed.json", _verify_terminal_outputs(run)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Use distinct authorization for the sole operation that can expose holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--host-plan", type=Path)
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "run" and not args.open_holdout:
        parser.error("E2 run requires explicit --open-holdout authorization")
    if args.action == "run" and args.host_plan is None:
        parser.error("E2 run requires its predeclared --host-plan")
    bundle = args.bundle.resolve(strict=True)
    path = bundle / "freeze.json"
    if sha256_file(path) != args.freeze_sha256:
        raise ValueError("E2 launch does not match its verified freeze digest")
    run = HostExecution(
        repository=args.repository.resolve(strict=True),
        bundle=bundle,
        freeze_sha256=args.freeze_sha256,
        spec=load_json_value(path),
        image_tag=args.image_tag,
    )
    commands = phase_commands(run)
    plan = _host_plan(run, commands)
    if args.action == "plan":
        print(json.dumps(plan, sort_keys=True))
        return 0
    if load_json_value(args.host_plan) != plan:
        raise ValueError("E2 host launch differs from its predeclared plan")
    return _run_activity(run, commands, plan)


if __name__ == "__main__":
    sys.exit(main())
