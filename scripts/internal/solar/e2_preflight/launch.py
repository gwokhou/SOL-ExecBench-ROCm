"""Launch the bounded GPU pilot through the repository's native owner."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.amd_smi import parse_processes
from sol_execbench.core.process import run_in_process_group_to_files
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)


def _default_admission_record(state_dir: Path) -> Path:
    candidates = sorted((state_dir / "gpu-admission").glob("*.json"))
    if len(candidates) == 1:
        return candidates[0]
    return state_dir / "gpu-admission/default.json"


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
HISTORY = ROOT / "data/outputs/solar-e2-20261108-r2"
STATE = Path.home() / ".local/state"
CLOCK_STATE = STATE / "gpu-leases/sol"
ADMISSION = _default_admission_record(STATE)
GPU_SECONDS = 1800
IMAGE_TAG = "solar-b1-20261108"
SMI = "/usr/bin/amd-smi"


def read_json(path: Path) -> dict[str, Any]:
    """Read an existing native receipt."""
    return json.loads(path.read_text())


def capture(
    out: Path,
    name: str,
    command: list[str],
    env: dict[str, str],
    timeout: int = 30,
) -> str:
    """Retain bounded host metadata and require a clean exit."""
    stdout = out / f"{name}.stdout"
    result = run_in_process_group_to_files(
        command, stdout, out / f"{name}.stderr", timeout=timeout, env=env
    )
    if result.returncode or stdout.stat().st_size > 1024**2:
        raise RuntimeError(f"pilot {name} failed: {result.returncode}")
    return stdout.read_text()


def verify(out: Path, env: dict[str, str], spec: dict[str, Any]) -> None:
    """Check the prerequisite and frozen image and GPU identity."""
    if read_json(out / "cpu/fault-checks.json")["status"] != "passed":
        raise RuntimeError("pilot CPU fixtures did not pass")
    image = capture(
        out,
        "image",
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            f"sol-execbench:{IMAGE_TAG}",
        ],
        env,
    ).strip()
    if image != spec["image_id"]:
        raise RuntimeError("frozen image drift")
    uuid = spec["device"]["uuid"]
    devices = json.loads(capture(out, "devices", [SMI, "list", "--json"], env))
    selected = [row for row in devices if row["uuid"].lower() == uuid]
    if len(selected) != 1 or selected[0]["bdf"] != spec["device"]["bdf"]:
        raise RuntimeError("frozen GPU identity drift")
    if parse_processes(
        capture(
            out, "processes-before", [SMI, "process", "-g", uuid, "--json"], env
        )
    ):
        raise RuntimeError("target GPU is in use")
    admission = read_json(ADMISSION)
    if admission["state"] != "ready" or admission["owner"] is not None:
        raise RuntimeError("native GPU admission is not ready")


def command(repo: Path, out: Path) -> list[str]:
    """Construct the existing native owner command with bounded pilot mounts."""
    return [
        "bash",
        str(repo / "scripts/run_docker.sh"),
        "--record-container-validation",
        "--compatibility-entry",
        str(out / "gpu-compatibility.json"),
        "--read-only",
        "--mount",
        f"type=bind,source={CODE},target=/pilot,readonly",
        "--memory=16081639424",
        "--memory-swap=16081639424",
        "--cpuset-cpus=0-31",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=128m",
        "--tmpfs",
        "/e2-scratch:rw,exec,nosuid,nodev,size=256m",
        "-e",
        "PYTHONPATH=/sol-execbench/src:/sol-execbench:/pilot",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "GIT_OPTIONAL_LOCKS=0",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "PYTORCH_ALLOC_CONF=expandable_segments:False",
        "--",
        "/venv/bin/python",
        "/pilot/run.py",
        "gpu",
    ]


def closeout(
    out: Path,
    env: dict[str, str],
    uuid: str,
    previous: set[str],
    returncode: int | None,
) -> None:
    """Record actual GPU and lock state after the owner returns."""
    receipts = {
        path.name: read_json(path)
        for path in (CLOCK_STATE / "clock-leases").glob("*.json")
        if path.name not in previous
    }
    admission = read_json(ADMISSION)
    clocks = json.loads(
        capture(
            out,
            "clocks-after",
            [SMI, "metric", "-l", "-g", uuid, "--json"],
            env,
        )
    )
    processes = parse_processes(
        capture(
            out, "processes-after", [SMI, "process", "-g", uuid, "--json"], env
        )
    )
    locks = []
    active_locks = Path("/proc/locks").read_text().splitlines()
    for level in ("L1-ownership", "L2-execution"):
        stat = (STATE / f"gpu-{uuid}.{level}.lock").stat()
        key = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        locks.append(
            {
                "level": level,
                "held": any(key in line.split() for line in active_locks),
            }
        )
    restored = bool(receipts) and all(
        row["state"] == "restored" and not row["release_blocker"]
        for row in receipts.values()
    )
    ready = admission["state"] == "ready" and admission["owner"] is None
    auto = all(
        row["perf_level"] == "AMDSMI_DEV_PERF_LEVEL_AUTO"
        for row in clocks["gpu_data"]
    )
    clean = (
        restored
        and ready
        and auto
        and not processes
        and not any(lock["held"] for lock in locks)
    )
    atomic_write_json_value(
        out / "gpu-closeout.json",
        {
            "clean": clean,
            "returncode": returncode,
            "receipts": receipts,
            "admission": admission,
            "clocks": clocks,
            "gpu_process_count": len(processes),
            "locks": locks,
        },
    )
    if not clean:
        raise RuntimeError("pilot GPU handback is unconfirmed")


def main(out: Path) -> None:
    """Launch one bounded GPU pilot on the frozen source."""
    spec = read_json(HISTORY / "freeze-e2.json")
    repo = HISTORY / "execution-source"
    verify_release_source_state(
        repo, expected_revision=spec["execution_commit"]
    )
    uuid = spec["device"]["uuid"]
    env = {
        **os.environ,
        "GPU_LOCK_DIR": str(STATE),
        "GPU_LOCK_UUID": uuid,
        "ROCR_VISIBLE_DEVICES": os.environ.get(
            "ROCR_VISIBLE_DEVICES", "GPU-<redacted>"
        ),
        "HIP_VISIBLE_DEVICES": "0",
        "SOL_EXECBENCH_ADMISSION_RECORD": str(ADMISSION),
        "SOL_EXECBENCH_CLOCK_STATE_DIR": str(CLOCK_STATE),
        "SOL_EXECBENCH_HOST_PYTHON": sys.executable,
        "SOL_EXECBENCH_OUTPUT_DIR": str(out),
        "IMAGE_TAG": IMAGE_TAG,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    verify(out, env, spec)
    plan = command(repo, out)
    atomic_write_json_value(
        out / "gpu-launch.json",
        {
            "command": plan,
            "image_id": spec["image_id"],
            "commit": spec["execution_commit"],
            "device": spec["device"],
            "gpu_seconds": GPU_SECONDS,
            "pilot_source_sha256": {
                p.name: sha256_file(p) for p in CODE.glob("*.py")
            },
            "performance_authority": False,
        },
    )
    previous = {
        path.name for path in (CLOCK_STATE / "clock-leases").glob("*.json")
    }
    result = None
    try:
        result = run_in_process_group_to_files(
            plan,
            out / "gpu.stdout",
            out / "gpu.stderr",
            timeout=GPU_SECONDS,
            env=env,
        )
    finally:
        closeout(
            out,
            env,
            uuid,
            previous,
            result.returncode if result is not None else None,
        )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(strict=True))
