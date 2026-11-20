"""Prepare, preflight-verify, and run the frozen LayerNorm h0 confirmation."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from shutil import copyfile, copytree
from subprocess import TimeoutExpired
from typing import Any

from scripts.internal.solar import e2_successor_host as native
from scripts.internal.solar.e2_layernorm_holdout_gpu import (
    ACTIVITY_DIRECTORY,
    EVIDENCE_BYTES,
    GPU_SECONDS,
    PROTOCOL,
    REFERENCE,
    roster,
    roster_digest,
    stage_identities,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.process.termination import cleanup_on_termination
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)

REPO = Path(__file__).resolve().parents[3]
DATA_OUTPUTS = REPO / "data/outputs"
HISTORY = DATA_OUTPUTS / "solar-e2-successor-reaping-frozen-20261118"
DEV_BUNDLE = DATA_OUTPUTS / "solar-e2-layernorm-gpu-dev-20261119"
SOURCE = DATA_OUTPUTS / "solar-e2-20261108-r2/execution-source"
CPU_RECEIPT = (
    DATA_OUTPUTS
    / "solar-e2-layernorm-reference-cpu-20261119/retained-call-comparison.json"
)
DEFAULT_BUNDLE = DATA_OUTPUTS / "solar-e2-layernorm-gpu-holdout-20261119"
PROTOCOL_DOC = (
    REPO / "docs/internal/solar-e2-layernorm-holdout-protocol-2026-11-19.md"
)
IMAGE_TAG = "solar-b1-20261108"
KNOWN_PRIOR_FREEZE = (
    "5077e043805976ed817361650de8c09f5214db8fd1d101ed95e32ad86d427c94"
)
KNOWN_DEV_FREEZE = (
    "142b2502b7c5693da11d16b3b6bef417e28706e8e74f11c25d6b5bad301a9efd"
)
KNOWN_CPU_RECEIPT = (
    "b90bbc41be4a2a8be39d719e9b11087766c0b981bf927922e94d4e56870690b9"
)
PATCH_FILES = (
    "e2_staging.py",
    "e2_layernorm_reference.py",
    "e2_layernorm_dev_inputs.py",
    "e2_layernorm_holdout_inputs.py",
    "e2_layernorm_holdout_gpu.py",
    "e2_layernorm_holdout_audit.py",
    "e2_layernorm_holdout_host.py",
)
_READONLY_DOC_NAMES = ("protocol.md", "cpu-acceptance.json", "dev-audit.json")


def _probe(command: list[str], timeout: int = 15) -> str:
    result = run_in_process_group_bounded(
        command, timeout=timeout, max_capture_bytes=16384
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[:4]} probe failed: {result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def _docker_init(image: str) -> dict[str, str]:
    program = (
        "import hashlib,json,os,pathlib; x=pathlib.Path('/proc/1/exe'); "
        "print(json.dumps(dict(target=os.readlink(x),"
        "sha256=hashlib.sha256(x.read_bytes()).hexdigest())))"
    )
    return json.loads(
        _probe(
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
            40,
        )
    )


def _resolve_host_amd_unique_id() -> str:
    pci_root = Path("/sys/bus/pci/devices")
    bdf = os.environ.get("SOL_EXECBENCH_GPU_BDF", "").strip()
    if bdf:
        return (pci_root / bdf / "unique_id").read_text().strip()
    matches = sorted(pci_root.glob("*/unique_id"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly 1 AMD PCI unique_id under {pci_root}, "
            f"found {len(matches)}; set SOL_EXECBENCH_GPU_BDF explicitly"
        )
    return matches[0].read_text().strip()


def _observe_host() -> dict[str, str]:
    return {
        "kernel_release": os.uname().release,
        "docker": _probe(
            ["docker", "info", "--format={{.ServerVersion}}/{{.InitCommit.ID}}"]
        ),
        "native_roc_device": _resolve_host_amd_unique_id(),
    }


def command_plan(
    bundle: Path, spec: dict[str, Any], digest: str
) -> dict[str, list[str]]:
    """Derive exact host-bound commands for readiness, online, and audit."""
    execution = native.HostExecution(
        repository=SOURCE,
        bundle=bundle,
        freeze_sha256=digest,
        spec=spec,
        image_tag=IMAGE_TAG,
    )
    legacy = native.phase_commands(execution)
    online = list(legacy["online"])
    starter = online.index(native._ENTRY_MODULE)
    online[starter:] = [
        "scripts.internal.solar.e2_layernorm_holdout_gpu",
        "--repository",
        "/sol-execbench",
        "--bundle",
        "/outputs",
        "--expected-sha",
        digest,
        "--open-holdout",
    ]
    divider = online.index("--")
    extra = [
        token
        for name in _READONLY_DOC_NAMES
        for token in native._mount(bundle / name, f"/outputs/{name}")
    ]
    online[divider:divider] = extra
    audit = list(legacy["audit"])
    audit_start = audit.index("scripts.internal.solar.e2_successor_audit")
    audit[audit_start:] = [
        "scripts.internal.solar.e2_layernorm_holdout_audit",
        "--repository",
        "/sol-execbench",
        "--bundle",
        "/outputs",
        "--expected-sha",
        digest,
    ]
    readiness = [
        token.replace("target=/outputs", "target=/outputs,readonly")
        if token == f"type=bind,source={bundle},target=/outputs"
        else token
        for token in legacy["audit"]
    ]
    ready_start = readiness.index("scripts.internal.solar.e2_successor_audit")
    readiness[ready_start:] = [
        "scripts.internal.solar.e2_layernorm_holdout_gpu",
        "--repository",
        "/sol-execbench",
        "--bundle",
        "/outputs",
        "--expected-sha",
        digest,
        "--verify-only",
    ]
    return {"readiness": readiness, "online": online, "audit": audit}


def _copy_research_capsule(bundle: Path) -> Path:
    capsule = bundle / "capsule"
    copytree(
        HISTORY / "capsule",
        capsule,
        ignore=lambda _, names: [n for n in names if n == "__pycache__"],
    )
    solar = Path("scripts/internal/solar")
    for name in PATCH_FILES:
        copyfile(REPO / solar / name, capsule / solar / name)
    return capsule


def _verify_prior_receipts() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    if sha256_file(HISTORY / "freeze.json") != KNOWN_PRIOR_FREEZE:
        raise ValueError("prior reaping freeze SHA-256 changed")
    if sha256_file(DEV_BUNDLE / "freeze.json") != KNOWN_DEV_FREEZE:
        raise ValueError("prior LayerNorm dev freeze SHA-256 changed")
    if sha256_file(CPU_RECEIPT) != KNOWN_CPU_RECEIPT:
        raise ValueError("LayerNorm CPU receipt SHA-256 changed")
    dev_audit = load_json_value(DEV_BUNDLE / "audit.json")
    if (
        dev_audit.get("freeze_sha256") != KNOWN_DEV_FREEZE
        or dev_audit.get("acceptance_passed") is not True
        or dev_audit.get("gpu_handback_clean") is not True
    ):
        raise ValueError("LayerNorm development GPU acceptance is incomplete")
    prev = load_json_value(HISTORY / "freeze.json")
    cpu = load_json_value(CPU_RECEIPT)
    verify_release_source_state(
        SOURCE, expected_revision=prev["execution_commit"]
    )
    initial = load_json_value(HISTORY / "host-run/launch.json")
    inspected = _probe(
        [
            "docker",
            "image",
            "inspect",
            "--format={{.Id}}",
            f"sol-execbench:{IMAGE_TAG}",
        ]
    )
    if inspected != prev["image_id"]:
        raise ValueError("the local image tag moved")
    return prev, cpu, initial


def _build_freeze_spec(
    bundle: Path,
    capsule: Path,
    *,
    prev: dict[str, Any],
    initial: dict[str, Any],
    init: dict[str, str],
) -> dict[str, Any]:
    rows = [slot.metadata() for slot in roster()]
    identities = stage_identities(SOURCE)
    if len(rows) != 186 or len(identities) != 30:
        raise ValueError("LayerNorm holdout denominator count mismatch")
    return {
        "protocol": PROTOCOL,
        "reference": REFERENCE.value,
        "activity_directory": ACTIVITY_DIRECTORY,
        "performance_authority": False,
        "prior_freeze_sha256": KNOWN_PRIOR_FREEZE,
        "dev_freeze_sha256": KNOWN_DEV_FREEZE,
        "dev_audit_sha256": sha256_file(bundle / "dev-audit.json"),
        "cpu_acceptance_sha256": KNOWN_CPU_RECEIPT,
        "execution_commit": prev["execution_commit"],
        "image_id": prev["image_id"],
        "device": prev["device"],
        "docker_init": init,
        "host_runtime": _observe_host(),
        "container_semantic_runtime": prev["container_semantic_runtime"],
        "cpu_affinity": prev["cpu_affinity"],
        "worker_memory_bytes": prev["worker_memory_bytes"],
        "budgets": {
            "gpu_seconds": GPU_SECONDS,
            "evidence_bytes": EVIDENCE_BYTES,
        },
        "native_profile": initial["native_profile"],
        "host_driver_sha256": sha256_file(Path(__file__)),
        "host_wrapper_sha256": sha256_file(SOURCE / "scripts/run_docker.sh"),
        "host_helpers_sha256": sha256_file(Path(native.__file__)),
        "capsule_sha256": {
            str(f.relative_to(capsule)): sha256_file(f)
            for f in sorted(capsule.rglob("*.py"))
        },
        "protocol_sha256": sha256_file(bundle / "protocol.md"),
        "roster_sha256": roster_digest(),
        "staged_identities": identities,
        "roster": rows,
    }


def prepare(bundle: Path) -> str:
    """Freeze holdout metadata, capsule, and plan without opening holdout."""
    bundle = bundle.resolve()
    freeze_path = bundle / "freeze.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    prev, cpu, initial = _verify_prior_receipts()
    init = _docker_init(prev["image_id"])
    os.makedirs(bundle / "readiness", exist_ok=False)
    capsule = _copy_research_capsule(bundle)
    solar = Path("scripts/internal/solar")
    if (
        sha256_file(capsule / solar / "e2_staging.py")
        != cpu["research_staging_source_sha256"]
        or sha256_file(capsule / solar / "e2_layernorm_reference.py")
        != cpu["research_reference_source_sha256"]
    ):
        raise ValueError(
            "LayerNorm staging or reference disagrees with CPU receipt"
        )
    copyfile(PROTOCOL_DOC, bundle / "protocol.md")
    copyfile(CPU_RECEIPT, bundle / "cpu-acceptance.json")
    copyfile(DEV_BUNDLE / "audit.json", bundle / "dev-audit.json")
    spec = _build_freeze_spec(
        bundle, capsule, prev=prev, initial=initial, init=init
    )
    atomic_write_json_value(freeze_path, spec)
    digest = sha256_file(freeze_path)
    atomic_write_json_value(
        bundle / "host-plan.json",
        {
            "freeze_sha256": digest,
            "commands": command_plan(bundle, spec, digest),
            "host_driver_sha256": spec["host_driver_sha256"],
        },
    )
    print(digest)
    return digest


def _validate_host_state(
    bundle: Path, expected: str
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    path = bundle / "freeze.json"
    if sha256_file(path) != expected:
        raise ValueError("the LayerNorm holdout freeze changed")
    spec = load_json_value(path)
    if (
        _observe_host() != spec["host_runtime"]
        or sha256_file(Path(__file__)) != spec["host_driver_sha256"]
        or sha256_file(SOURCE / "scripts/run_docker.sh")
        != spec["host_wrapper_sha256"]
        or sha256_file(Path(native.__file__)) != spec["host_helpers_sha256"]
    ):
        raise ValueError("the host execution identity changed")
    verify_release_source_state(
        SOURCE, expected_revision=spec["execution_commit"]
    )
    plan = load_json_value(bundle / "host-plan.json")
    commands = command_plan(bundle, spec, expected)
    if plan["commands"] != commands or plan["freeze_sha256"] != expected:
        raise ValueError("the holdout launch commands changed")
    for rel, digest in spec["capsule_sha256"].items():
        if sha256_file(bundle / "capsule" / rel) != digest:
            raise ValueError(f"the launch overlay changed: {rel}")
    return spec, commands


def _stage(
    command: list[str],
    output: Path,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> int:
    started = time.monotonic()
    result = None
    code = None
    try:
        result = run_in_process_group_bounded(
            command, timeout=timeout, env=env, max_capture_bytes=32768
        )
        code = result.returncode
    except TimeoutExpired:
        code = 124
    finally:
        atomic_write_json_value(
            output,
            {
                "returncode": code,
                "wall_seconds": time.monotonic() - started,
                "stdout": result.stdout if result else "",
                "stderr": result.stderr if result else "",
            },
        )
    return code if code is not None else 1


def verify(bundle: Path, expected: str) -> int:
    """Run device-free container verification without opening holdout."""
    bundle = bundle.resolve()
    spec, commands = _validate_host_state(bundle, expected)
    if (bundle / ACTIVITY_DIRECTORY).exists():
        raise FileExistsError(bundle / ACTIVITY_DIRECTORY)
    receipt = bundle / "readiness/stage-receipt.json"
    verification = bundle / "readiness/verification.json"
    if receipt.exists() or verification.exists():
        raise FileExistsError(verification)
    code = native._run_stage(commands["readiness"], receipt)
    receipt.with_suffix(".cid").unlink(missing_ok=True)
    if (bundle / ACTIVITY_DIRECTORY).exists():
        raise FileExistsError("readiness check unexpectedly opened holdout")
    if code:
        raise RuntimeError(f"device-free readiness check failed with {code}")
    payload = json.loads(load_json_value(receipt)["stdout"])
    if payload != {
        "verified": True,
        "freeze_sha256": expected,
        "roster_slots": len(spec["roster"]),
        "holdout_opened": False,
    }:
        raise ValueError("unexpected readiness payload")
    atomic_write_json_value(
        verification,
        {
            **payload,
            "returncode": code,
            "receipt_sha256": sha256_file(receipt),
        },
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def run(bundle: Path, expected: str, *, open_holdout: bool) -> int:
    """Run online holdout execution and device-free audit only if authorized."""
    if not open_holdout:
        raise PermissionError(
            "LayerNorm holdout run requires explicit --open-holdout authorization"
        )
    bundle = bundle.resolve()
    spec, commands = _validate_host_state(bundle, expected)
    execution = native.HostExecution(
        repository=SOURCE,
        bundle=bundle,
        freeze_sha256=expected,
        spec=spec,
        image_tag=IMAGE_TAG,
    )
    if execution.activity.exists():
        raise FileExistsError(execution.activity)
    out = bundle / "host-run"
    out.mkdir(exist_ok=False)
    for name, value in spec["native_profile"].items():
        os.environ[name] = value
    env = native._native_environment(execution)
    if (
        os.statvfs(bundle).f_bavail * os.statvfs(bundle).f_frsize
        < EVIDENCE_BYTES
    ):
        raise OSError("insufficient evidence capacity for frozen activity")
    with cleanup_on_termination():
        try:
            code = _stage(
                commands["online"],
                out / "online.json",
                timeout=GPU_SECONDS,
                env=env,
            )
        finally:
            if (execution.activity / native._OWNER_START).exists():
                online = load_json_value(out / "online.json")
                native._record_handback(
                    execution, online["returncode"], online["wall_seconds"]
                )
        audit_code = native._run_stage(commands["audit"], out / "audit.json")
        (out / "audit.cid").unlink(missing_ok=True)
    print("GPU exit", code, "device-free audit exit", audit_code)
    if not audit_code:
        result = load_json_value(bundle / "audit.json")
        summary = {
            key: result[key]
            for key in (
                "acceptance_passed",
                "slot_states",
                "fixed_group_states",
                "permuted_group_states",
                "independent_draw_group_states",
                "negative_states",
                "negative_group_states",
                "gpu_handback_clean",
            )
        }
        print(json.dumps(summary, ensure_ascii=False))
    return 0 if code == audit_code == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """Dispatch prepare, verify, or explicitly authorized holdout execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "run"))
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--expected-sha")
    parser.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.bundle)
        return 0
    if not args.expected_sha:
        parser.error(f"{args.action} requires --expected-sha")
    if args.action == "verify":
        return verify(args.bundle, args.expected_sha)
    if not args.open_holdout:
        parser.error("run requires explicit --open-holdout authorization")
    return run(args.bundle, args.expected_sha, open_holdout=True)


if __name__ == "__main__":
    sys.exit(main())
