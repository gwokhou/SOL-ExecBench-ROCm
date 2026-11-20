"""Host preparer, preflight verifier, and native launcher for C2 and C3 smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import copyfile
from subprocess import TimeoutExpired
from typing import Any

from sol_execbench.core.bench.clock_lock import observe_clock_levels
from sol_execbench.core.bench.clock_ownership import ClockOwnerState
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
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.source_tree import source_tree_manifest
from sol_execbench.core.platform.amd_smi import parse_processes
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command
from sol_execbench.core.process import (
    run_in_process_group_to_files,
)
from sol_execbench.core.process.subprocesses import (
    run_in_process_group_bounded,
)
from sol_execbench.core.process.termination import (
    cleanup_on_termination,
    defer_termination,
)
from sol_execbench.core.scoring.release_environment import (
    CONTAINER_IMAGE_ID_ENV,
    verify_release_source_state,
)
from sol_execbench.core.solar_bridge.coverage_models import (
    COVERAGE_SOURCE_SHA256,
    COVERAGE_TARGET_SHA256,
)
from sol_execbench.tools.docker_process import _remove_owned_container
from solar.analysis.orojenesis.runner import OrojenesisRunner

REPO = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = REPO / "data/outputs/solar-c2-c3-20261119"
B2_FREEZE = REPO / "data/outputs/solar-b2-20261108/freeze-r6.json"
B2_CHECKOUT = REPO / "data/outputs/solar-b2-20261108/execution-source"
E2_CHECKOUT = REPO / "data/outputs/solar-e2-20261108-r2/execution-source"
KNOWN_B2_FREEZE_SHA256 = (
    "b43c845bf141da84a9384bcc7bef61a09166de345a64df633c149cc8460aa495"
)
IMAGE_NAME = "sol-execbench"
IMAGE_TAG = "solar-b1-20261108"
IMAGE_REF = f"{IMAGE_NAME}:{IMAGE_TAG}"
CONTAINER_REPO = "/sol-execbench"
CONTAINER_OUTPUTS = "/outputs"
CONTAINER_PYTHON = "/venv/bin/python"
CAMPAIGN_MODULE = "scripts.internal.solar.c2_c3_campaign"
CAMPAIGN_SOURCE = REPO / "scripts/internal/solar/c2_c3_campaign.py"
DEFAULT_LOCK_DIR = Path.home() / ".local/state"
DEFAULT_ADMISSION = DEFAULT_LOCK_DIR / "gpu-admission/rx9060xt.json"
DEFAULT_CLOCK_STATE = DEFAULT_LOCK_DIR / "gpu-leases/sol"
CACHE_ENV = {
    "TRITON_CACHE_DIR": "/tmp/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
    "TORCH_EXTENSIONS_DIR": "/tmp/torch-extensions",
    "XDG_CACHE_HOME": "/tmp/cache",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "MIOPEN_USER_DB_PATH": "/tmp/miopen-db",
    "MIOPEN_CUSTOM_CACHE_DIR": "/tmp/miopen-cache",
}
_HOST_WRAPPER_GRACE_SECONDS = 120.0
_PROBE_TIMEOUT_SECONDS = 30.0
_CPU_CONTAINER_TIMEOUT_SECONDS = 1800.0
_CAPTURE_BYTES = 32768
_MANIFEST_REL = "problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml"
_DESCRIPTOR_REL = (
    "problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml"
)


class HostAction(StrEnum):
    """Supported host launcher actions."""

    PREPARE = "prepare"
    READINESS = "readiness"
    REPRESENTATIVES = "b2-representatives"
    C2 = "c2"
    C3_SMOKE = "c3-smoke"
    C3_RESUME = "c3-resume"
    AUDIT = "audit"


@dataclass(frozen=True, slots=True, kw_only=True)
class HostConfig:
    """Verified bundle and freeze configuration on the host."""

    bundle: Path
    freeze_sha256: str
    spec: dict[str, Any]

    @property
    def execution_checkout(self) -> Path:
        """Pinned detached execution checkout directory."""
        return Path(self.spec["execution_checkout"])

    @property
    def mapper_home(self) -> Path:
        """Pinned external Orojenesis mapper directory."""
        return Path(self.spec["mapper_home"])


def _probe(
    command: Sequence[str], *, timeout: float = _PROBE_TIMEOUT_SECONDS
) -> str:
    result = run_in_process_group_bounded(
        list(command), timeout=timeout, max_capture_bytes=_CAPTURE_BYTES
    )
    if result.returncode:
        raise RuntimeError(
            f"probe {list(command[:4])} failed ({result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def _docker_init_identity(image: str) -> dict[str, str]:
    program = (
        "import hashlib,json,os,pathlib; x=pathlib.Path('/proc/1/exe'); "
        "print(json.dumps(dict(target=os.readlink(x),"
        "sha256=hashlib.sha256(x.read_bytes()).hexdigest())))"
    )
    raw = _probe(
        [
            "docker",
            "run",
            "--pull=never",
            "--rm",
            "--init",
            "--read-only",
            "--network=none",
            "--cap-drop=ALL",
            "--pids-limit=16",
            "--security-opt=no-new-privileges",
            "--entrypoint=/venv/bin/python",
            image,
            "-B",
            "-c",
            program,
        ],
        timeout=45.0,
    )
    return json.loads(raw)


def _verify_static_hardware(expected: Mapping[str, Any]) -> None:
    smi = resolve_rocm_tool_command("amd-smi")
    devices = json.loads(_probe([smi, "list", "--json"]))
    matches = [d for d in devices if d["uuid"].lower() == expected["uuid"]]
    if len(matches) != 1 or matches[0]["bdf"] != expected["bdf"]:
        raise RuntimeError("physical GPU list identity drifted")
    asic = json.loads(
        _probe(
            [
                smi,
                "static",
                "-g",
                expected["uuid"],
                "--asic",
                "--vram",
                "--json",
            ]
        )
    )["gpu_data"][0]
    if (
        asic["asic"]["target_graphics_version"] != "gfx1200"
        or asic["asic"]["num_compute_units"]
        != expected["physical_compute_units"]
        or asic["vram"]["size"] != {"value": 16304, "unit": "MB"}
    ):
        raise RuntimeError("GPU ASIC/VRAM configuration drifted from B2 freeze")


def _verify_prereqs_for_prepare() -> dict[str, Any]:
    if sha256_file(B2_FREEZE) != KNOWN_B2_FREEZE_SHA256:
        raise RuntimeError("B2 freeze SHA-256 differs from pinned expectation")
    b2: dict[str, Any] = load_json_value(B2_FREEZE)
    verify_release_source_state(
        B2_CHECKOUT, expected_revision=b2["execution_commit"]
    )
    if not E2_CHECKOUT.is_dir():
        raise RuntimeError(f"missing E2 checkout: {E2_CHECKOUT}")
    _probe(["git", "-C", str(REPO), "diff", "--quiet", "--", "src", "problems"])
    head_sha = _probe(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    image_id = _probe(
        ["docker", "image", "inspect", "--format={{.Id}}", IMAGE_REF]
    )
    if image_id != b2["image_id"]:
        raise RuntimeError("pinned Docker image tag moved")
    toolchain = OrojenesisRunner(Path(b2["mapper_home"])).toolchain_identity
    if (
        toolchain["artifact"]["sha256"] != b2["mapper_binary_sha256"]
        or toolchain["provenance_sha256"] != b2["mapper_provenance_sha256"]
    ):
        raise RuntimeError("Orojenesis mapper digest drifted")
    _verify_static_hardware(b2["device"])
    init = _docker_init_identity(b2["image_id"])
    return {"b2": b2, "init": init, "head_sha": head_sha}


def _create_clean_checkout(
    bundle: Path, head_sha: str, b2: Mapping[str, Any]
) -> tuple[Path, str, dict[str, Any]]:
    checkout = bundle / "execution-source"
    _probe(
        ["git", "clone", "--no-local", str(REPO), str(checkout)], timeout=120.0
    )
    _probe(["git", "-C", str(checkout), "checkout", "--detach", head_sha])
    verify_release_source_state(checkout, expected_revision=head_sha)
    if sha256_file(checkout / _MANIFEST_REL) != COVERAGE_SOURCE_SHA256:
        raise RuntimeError("frozen LLM_CORE_V2 manifest digest drifted")
    if sha256_file(checkout / _DESCRIPTOR_REL) != COVERAGE_TARGET_SHA256:
        raise RuntimeError(
            "frozen standard-16gb target descriptor digest drifted"
        )
    for rel, digest in b2["frozen_inputs_sha256"].items():
        if sha256_file(checkout / rel) != digest:
            raise RuntimeError(f"frozen B2 input drifted in HEAD: {rel}")
    new_src = source_tree_manifest(checkout / "src")
    b2_src = source_tree_manifest(B2_CHECKOUT / "src")
    e2_src = source_tree_manifest(E2_CHECKOUT / "src")
    changed_vs_e2 = sorted(
        k for k in set(new_src) | set(e2_src) if new_src.get(k) != e2_src.get(k)
    )
    changed_vs_b2 = sorted(
        k for k in set(new_src) | set(b2_src) if new_src.get(k) != b2_src.get(k)
    )
    if changed_vs_e2 != ["sol_execbench/core/platform/hardware.py"]:
        raise RuntimeError(f"unexpected src diff vs E2-r2: {changed_vs_e2}")
    diff_record = {
        "execution_commit": head_sha,
        "new_src_manifest_sha256": stable_json_checksum(new_src),
        "b2_src_manifest_sha256": stable_json_checksum(b2_src),
        "e2_src_manifest_sha256": stable_json_checksum(e2_src),
        "changed_vs_e2_r2": changed_vs_e2,
        "changed_vs_b2_r6": changed_vs_b2,
    }
    atomic_write_json_value(
        bundle / "preflight/source-diff-vs-prior.json", diff_record
    )
    return checkout, stable_json_checksum(new_src), diff_record


def _build_freeze_payload(
    prereqs: Mapping[str, Any],
    *,
    checkout: Path,
    source_manifest_sha256: str,
    capsule: Path,
) -> dict[str, Any]:
    b2 = prereqs["b2"]
    return {
        "protocol": "LLM_CORE_C2_C3_SMOKE_20261119",
        "performance_authority": False,
        "b2_freeze_sha256": KNOWN_B2_FREEZE_SHA256,
        "execution_commit": prereqs["head_sha"],
        "execution_checkout": str(checkout),
        "source_manifest_sha256": source_manifest_sha256,
        "image_id": b2["image_id"],
        "container_semantic_runtime": b2["container_semantic_runtime"],
        "mapper_home": b2["mapper_home"],
        "mapper_binary_sha256": b2["mapper_binary_sha256"],
        "mapper_provenance_sha256": b2["mapper_provenance_sha256"],
        "device": b2["device"],
        "representatives": b2["representatives"],
        "exact_rejections": b2["exact_rejections"],
        "budgets": {
            key: b2["budgets"][key]
            for key in (
                "jobs",
                "mapper_threads",
                "required_cpu_affinity",
                "worker_memory_bytes",
                "readiness_seconds",
                "formal_seconds",
                "mapper_seconds",
                "c2_capacity_seconds",
                "c3_smoke_seconds",
                "c3_activity_seconds",
                "c3_evidence_bytes",
            )
        },
        "effective_container_memory_bytes": b2[
            "effective_container_memory_bytes"
        ],
        "cache_environment": CACHE_ENV,
        "docker_init": prereqs["init"],
        "frozen_inputs_sha256": {
            **b2["frozen_inputs_sha256"],
            _MANIFEST_REL: COVERAGE_SOURCE_SHA256,
            _DESCRIPTOR_REL: COVERAGE_TARGET_SHA256,
        },
        "host_driver_sha256": sha256_file(Path(__file__)),
        "host_wrapper_sha256": sha256_file(checkout / "scripts/run_docker.sh"),
        "capsule_sha256": {
            str(f.relative_to(capsule)): sha256_file(f)
            for f in sorted(capsule.rglob("*.py"))
        },
    }


def _mount(
    source: Path, target: str, *, writable: bool = False
) -> tuple[str, str]:
    suffix = "" if writable else ",readonly"
    return "--mount", f"type=bind,source={source},target={target}{suffix}"


def _env_flags(mapping: Mapping[str, str]) -> list[str]:
    return [
        part
        for key, value in mapping.items()
        for part in ("-e", f"{key}={value}")
    ]


def _online_command(
    cfg: HostConfig, phase: str, *, resume: bool = False
) -> list[str]:
    affinity = "--cpuset-cpus=" + ",".join(
        map(str, cfg.spec["budgets"]["required_cpu_affinity"])
    )
    mem = cfg.spec["budgets"]["worker_memory_bytes"]
    entry = [
        CONTAINER_PYTHON,
        "-B",
        "-m",
        CAMPAIGN_MODULE,
        phase,
        "--expected-freeze-sha",
        cfg.freeze_sha256,
        *(["--resume"] if resume else []),
    ]
    return [
        "bash",
        str(cfg.execution_checkout / "scripts/run_docker.sh"),
        "--record-container-validation",
        "--compatibility-entry",
        str(cfg.bundle / f"preflight/compatibility-{phase}.json"),
        "--init",
        "--read-only",
        f"--memory={mem}",
        f"--memory-swap={mem}",
        affinity,
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=2g",
        *_mount(cfg.mapper_home, str(cfg.mapper_home)),
        *_mount(cfg.bundle / "capsule", "/research"),
        *_env_flags(
            {
                "PYTHONPATH": f"{CONTAINER_REPO}/src:/research",
                "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "SOLAR_OROJENESIS_MAPPER_THREADS": str(
                    cfg.spec["budgets"]["mapper_threads"]
                ),
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                **CACHE_ENV,
            }
        ),
        "--",
        *entry,
    ]


def _device_free_command(cfg: HostConfig, phase: str) -> list[str]:
    affinity = "--cpuset-cpus=" + ",".join(
        map(str, cfg.spec["budgets"]["required_cpu_affinity"])
    )
    mem = cfg.spec["budgets"]["worker_memory_bytes"]
    mounts = [
        *_mount(cfg.execution_checkout, CONTAINER_REPO),
        *_mount(cfg.mapper_home, str(cfg.mapper_home)),
        *_mount(cfg.bundle / "capsule", "/research"),
    ]
    if phase == HostAction.READINESS.value:
        mounts.extend(
            [
                *_mount(
                    cfg.bundle / "freeze.json",
                    f"{CONTAINER_OUTPUTS}/freeze.json",
                ),
                *_mount(
                    cfg.bundle / "preflight",
                    f"{CONTAINER_OUTPUTS}/preflight",
                    writable=True,
                ),
            ]
        )
    else:
        mounts.extend(
            [
                *_mount(cfg.bundle, CONTAINER_OUTPUTS),
                *_mount(cfg.bundle / "audit-out", "/audit-out", writable=True),
            ]
        )
    return [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--init",
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        f"--memory={mem}",
        f"--memory-swap={mem}",
        affinity,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        CONTAINER_REPO,
        "--entrypoint",
        CONTAINER_PYTHON,
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev,size=128m",
        *mounts,
        *_env_flags(
            {
                "PATH": "/venv/bin:/usr/local/bin:/usr/bin:/bin",
                CONTAINER_IMAGE_ID_ENV: cfg.spec["image_id"],
                "PYTHONPATH": f"{CONTAINER_REPO}/src:/research",
                "PYTHONDONTWRITEBYTECODE": "1",
                "SOL_EXECBENCH_SANDBOXED": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
            }
        ),
        cfg.spec["image_id"],
        "-B",
        "-m",
        CAMPAIGN_MODULE,
        phase,
        "--expected-freeze-sha",
        cfg.freeze_sha256,
    ]


def prepare(bundle: Path) -> str:
    """Verify host prerequisites and write the sealed C2/C3 freeze bundle."""
    bundle = bundle.resolve()
    freeze_path = bundle / "freeze.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    prereqs = _verify_prereqs_for_prepare()
    for sub in (
        "preflight",
        "host-run",
        "audit-out",
        "capsule/scripts/internal/solar",
    ):
        (bundle / sub).mkdir(parents=True, exist_ok=True)
    checkout, src_sha, _ = _create_clean_checkout(
        bundle, prereqs["head_sha"], prereqs["b2"]
    )
    capsule = bundle / "capsule"
    copyfile(
        CAMPAIGN_SOURCE, capsule / "scripts/internal/solar/c2_c3_campaign.py"
    )
    spec = _build_freeze_payload(
        prereqs,
        checkout=checkout,
        source_manifest_sha256=src_sha,
        capsule=capsule,
    )
    atomic_write_json_value(freeze_path, spec)
    digest = sha256_file(freeze_path)
    cfg = HostConfig(bundle=bundle, freeze_sha256=digest, spec=spec)
    commands = {
        HostAction.READINESS.value: _device_free_command(
            cfg, HostAction.READINESS.value
        ),
        HostAction.REPRESENTATIVES.value: _online_command(
            cfg, HostAction.REPRESENTATIVES.value
        ),
        HostAction.C2.value: _online_command(cfg, HostAction.C2.value),
        HostAction.C3_SMOKE.value: _online_command(
            cfg, HostAction.C3_SMOKE.value
        ),
        HostAction.C3_RESUME.value: _online_command(
            cfg, HostAction.C3_RESUME.value
        ),
        HostAction.AUDIT.value: _device_free_command(
            cfg, HostAction.AUDIT.value
        ),
    }
    atomic_write_json_value(
        bundle / "host-plan.json",
        {
            "freeze_sha256": digest,
            "host_driver_sha256": spec["host_driver_sha256"],
            "commands": commands,
        },
    )
    print(digest)
    return digest


def _load_validated_config(
    bundle: Path, expected_sha: str | None
) -> HostConfig:
    bundle = bundle.resolve(strict=True)
    freeze_path = bundle / "freeze.json"
    actual_sha = sha256_file(freeze_path)
    plan: dict[str, Any] = load_json_value(bundle / "host-plan.json")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("freeze SHA-256 differs from --expected-freeze-sha")
    if plan["freeze_sha256"] != actual_sha:
        raise ValueError("host-plan freeze SHA-256 mismatch")
    spec: dict[str, Any] = load_json_value(freeze_path)
    checkout = Path(spec["execution_checkout"])
    if (
        sha256_file(Path(__file__)) != spec["host_driver_sha256"]
        or sha256_file(checkout / "scripts/run_docker.sh")
        != spec["host_wrapper_sha256"]
    ):
        raise ValueError("host launcher or wrapper source changed")
    verify_release_source_state(
        checkout, expected_revision=spec["execution_commit"]
    )
    for rel, digest in spec["capsule_sha256"].items():
        if sha256_file(bundle / "capsule" / rel) != digest:
            raise ValueError(f"capsule file drifted: {rel}")
    return HostConfig(bundle=bundle, freeze_sha256=actual_sha, spec=spec)


def _native_env(cfg: HostConfig) -> dict[str, str]:
    lock_dir = Path(os.environ.get(GPU_LOCK_DIR_ENV, DEFAULT_LOCK_DIR))
    admission = Path(os.environ.get(ADMISSION_ENV, DEFAULT_ADMISSION))
    clock_state = Path(os.environ.get(CLOCK_STATE_ENV, DEFAULT_CLOCK_STATE))
    device = cfg.spec["device"]
    unique_id = (
        (PCI_DEVICES_ROOT / device["bdf"] / "unique_id").read_text().strip()
    )
    if not unique_id:
        raise RuntimeError("missing native PCI unique_id for target GPU")
    image_id = _probe(
        ["docker", "image", "inspect", "--format={{.Id}}", IMAGE_REF]
    )
    if image_id != cfg.spec["image_id"]:
        raise RuntimeError("image ID drifted from freeze")
    return {
        **os.environ,
        GPU_LOCK_DIR_ENV: str(lock_dir),
        GPU_LOCK_UUID_ENV: device["uuid"],
        ADMISSION_ENV: str(admission),
        CLOCK_STATE_ENV: str(clock_state),
        "ROCR_VISIBLE_DEVICES": f"GPU-{unique_id}",
        "HIP_VISIBLE_DEVICES": "0",
        "SOL_EXECBENCH_HOST_PYTHON": sys.executable,
        "SOL_EXECBENCH_OUTPUT_DIR": str(cfg.bundle),
        "IMAGE_NAME": IMAGE_NAME,
        "IMAGE_TAG": IMAGE_TAG,
        "SOLAR_OROJENESIS_HOME": str(cfg.mapper_home),
        "SOLAR_OROJENESIS_MAPPER_THREADS": str(
            cfg.spec["budgets"]["mapper_threads"]
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _check_gpu_idle_and_ready(cfg: HostConfig, env: Mapping[str, str]) -> None:
    device = cfg.spec["device"]
    smi = resolve_rocm_tool_command("amd-smi")
    procs = parse_processes(
        _probe([smi, "process", "-g", device["uuid"], "--json"])
    )
    if procs:
        raise RuntimeError(f"target GPU has active processes: {procs}")
    levels = observe_clock_levels(device["uuid"])
    if levels != ("AMDSMI_DEV_PERF_LEVEL_AUTO",):
        raise RuntimeError(f"target GPU clocks are not AUTO: {levels}")
    os.environ[ADMISSION_ENV] = env[ADMISSION_ENV]
    os.environ[CLOCK_STATE_ENV] = env[CLOCK_STATE_ENV]
    os.environ[GPU_LOCK_DIR_ENV] = env[GPU_LOCK_DIR_ENV]
    os.environ[GPU_LOCK_UUID_ENV] = env[GPU_LOCK_UUID_ENV]
    admission = configured_admission()
    if admission is None:
        raise RuntimeError("GPU admission record unavailable")
    current = admission.read(
        AdmissionDevice(pci_bdf=device["bdf"], smi_uuid=device["uuid"])
    )
    if current.state is not AdmissionState.READY or current.owner is not None:
        raise RuntimeError(f"GPU admission is not ready: {current}")


def _locks_clear(path: Path, proc_locks: list[str]) -> bool:
    stat = path.stat(follow_symlinks=False)
    ident = (
        f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    )
    return all(ident not in row.split() for row in proc_locks)


def _record_closeout(
    cfg: HostConfig,
    stage: str,
    prior_leases: set[str],
    returncode: int | None,
    wall_seconds: float,
) -> dict[str, Any]:
    locks = configured_gpu_locks()
    clock_state = Path(os.environ.get(CLOCK_STATE_ENV, DEFAULT_CLOCK_STATE))
    receipts = {
        p.name: load_json_value(p)
        for p in sorted((clock_state / "clock-leases").glob("*.json"))
        if p.name not in prior_leases
    }
    admission = configured_admission()
    device = cfg.spec["device"]
    current = (
        admission.read(
            AdmissionDevice(pci_bdf=device["bdf"], smi_uuid=device["uuid"])
        ).model_dump(mode="json")
        if admission
        else None
    )
    smi = resolve_rocm_tool_command("amd-smi")
    procs = parse_processes(
        _probe([smi, "process", "-g", device["uuid"], "--json"])
    )
    levels = observe_clock_levels(device["uuid"])
    live_locks = Path("/proc/locks").read_text(encoding="utf-8").splitlines()
    clean = (
        bool(receipts)
        and all(
            r.get("state") == ClockOwnerState.RESTORED.value
            and r.get("release_blocker") is None
            for r in receipts.values()
        )
        and current is not None
        and current.get("state") == AdmissionState.READY.value
        and current.get("owner") is None
        and levels == ("AMDSMI_DEV_PERF_LEVEL_AUTO",)
        and not procs
        and _locks_clear(locks.ownership, live_locks)
        and _locks_clear(locks.execution, live_locks)
    )
    payload = {
        "stage": stage,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "clean": clean,
        "receipts": receipts,
        "admission": current,
        "clock_levels": list(levels or ()),
        "gpu_process_count": len(procs),
    }
    atomic_write_json_value(
        cfg.bundle / f"host-run/{stage}-closeout.json", payload
    )
    return payload


def run_device_free_stage(cfg: HostConfig, action: HostAction) -> int:
    """Execute readiness or independent audit in a device-free read-only container."""
    command = _device_free_command(cfg, action.value)
    receipt = cfg.bundle / f"host-run/{action.value}.json"
    cid = receipt.with_suffix(".cid")
    argv = [*command[:2], "--cidfile", str(cid), *command[2:]]
    started = time.monotonic()
    result = None
    timed_out = False
    try:
        result = run_in_process_group_bounded(
            argv,
            timeout=_CPU_CONTAINER_TIMEOUT_SECONDS,
            max_capture_bytes=_CAPTURE_BYTES,
        )
        code = result.returncode
    except TimeoutExpired:
        timed_out = True
        code = 124
    finally:
        atomic_write_json_value(
            receipt,
            {
                "action": action.value,
                "returncode": code if result is not None or timed_out else None,
                "timed_out": timed_out,
                "wall_seconds": time.monotonic() - started,
                "stdout": result.stdout if result else None,
                "stderr": result.stderr if result else None,
            },
        )
        if cid.exists():
            with defer_termination():
                _remove_owned_container("docker", cid)
    if code == 0 and action is HostAction.AUDIT:
        for name in ("cpu-audit.json", "smoke-summary.json"):
            src = cfg.bundle / "audit-out" / name
            if not src.is_file():
                raise RuntimeError(f"audit did not emit {name}")
            copyfile(src, cfg.bundle / "llm-core" / name)
    return code


def run_online_stage(
    cfg: HostConfig,
    action: HostAction,
    *,
    budget_seconds: float,
    resume: bool = False,
) -> int:
    """Execute one GPU-owned phase via scripts/run_docker.sh."""
    stage = action.value
    env = _native_env(cfg)
    _check_gpu_idle_and_ready(cfg, env)
    _verify_static_hardware(cfg.spec["device"])
    if (
        os.statvfs(cfg.bundle).f_bavail * os.statvfs(cfg.bundle).f_frsize
        < cfg.spec["budgets"]["c3_evidence_bytes"]
    ):
        raise OSError(
            "insufficient free filesystem space for C3 evidence budget"
        )
    command = _online_command(cfg, stage, resume=resume)
    atomic_write_json_value(
        cfg.bundle / f"host-run/{stage}-launch.json",
        {
            "stage": stage,
            "freeze_sha256": cfg.freeze_sha256,
            "command": command,
        },
    )
    clock_state = Path(env[CLOCK_STATE_ENV])
    prior = {p.name for p in (clock_state / "clock-leases").glob("*.json")}
    started = time.monotonic()
    returncode: int | None = None
    with cleanup_on_termination():
        try:
            res = run_in_process_group_to_files(
                command,
                cfg.bundle / f"host-run/{stage}.stdout",
                cfg.bundle / f"host-run/{stage}.stderr",
                timeout=budget_seconds + _HOST_WRAPPER_GRACE_SECONDS,
                env=env,
            )
            returncode = res.returncode
        finally:
            closeout = _record_closeout(
                cfg, stage, prior, returncode, time.monotonic() - started
            )
    if not closeout["clean"]:
        raise RuntimeError(f"native GPU handback failed after {stage}")
    return returncode if returncode is not None else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch prepare, readiness, c2, c3-smoke, c3-resume, or audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=tuple(a.value for a in HostAction))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--expected-freeze-sha")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    action = HostAction(args.action)
    if action is HostAction.PREPARE:
        prepare(args.bundle)
        return 0
    cfg = _load_validated_config(args.bundle, args.expected_freeze_sha)
    if action in {HostAction.READINESS, HostAction.AUDIT}:
        return run_device_free_stage(cfg, action)
    if action is HostAction.REPRESENTATIVES:
        return run_online_stage(cfg, action, budget_seconds=1800.0)
    if action is HostAction.C2:
        return run_online_stage(
            cfg,
            action,
            budget_seconds=cfg.spec["budgets"]["c2_capacity_seconds"],
        )
    if action is HostAction.C3_SMOKE:
        return run_online_stage(
            cfg,
            action,
            budget_seconds=cfg.spec["budgets"]["c3_smoke_seconds"],
            resume=args.resume,
        )
    return run_online_stage(
        cfg,
        action,
        budget_seconds=cfg.spec["budgets"]["c3_activity_seconds"],
        resume=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
