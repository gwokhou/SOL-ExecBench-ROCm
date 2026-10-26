"""Identity and evidence helpers for the non-public BASE-01 CPU receipt."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import re
import sys
import tomllib
from pathlib import Path
from typing import TypedDict

from sol_execbench.core.data.json_utils import atomic_write_json_value
from solar.artifacts import sha256_file
from solar.ir.contracts import IRPath

ROOT = Path(__file__).resolve().parents[3]
IMAGE_ID = (
    "sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd"
)
LOCK_SHA256 = "50668e6d0243eb80691d935424dedd2e2b4cf9c4b9d62937cd74dd71f2b83d37"
PROFILE = ROOT / "src/solar/rocm/profiles/RX_9060_XT.yaml"
ACCEPTANCE = ROOT / "docs/internal/solar-evolution-acceptance.md"
GPU_NODES = (Path("/dev/kfd"), Path("/dev/dri"))
LOCKED_PACKAGES = (
    "torch",
    "torchvision",
    "triton",
    "pydantic",
    "numpy",
    "PyYAML",
    "networkx",
    "graphviz",
    "pytest",
    "pytest-xdist",
    "ruff",
    "ty",
)


class SourceIdentity(TypedDict):
    """Source manifest separate from run results and operational logs."""

    files: dict[str, str]
    sha256: str


class UnrunPath(TypedDict):
    """An explicitly unexecuted real-workload slot."""

    ir_path: str
    status: str
    reason: str


class RealInputRecord(TypedDict):
    """Verified input identity without executing a reference program."""

    id: str
    directory: str
    uuid: str
    definition_sha256: str
    workload_sha256: str
    paths: list[UnrunPath]


def require_cpu_isolation() -> None:
    """Refuse collection if device nodes could permit real GPU execution."""
    nodes = [path for path in GPU_NODES if path.exists()]
    nodes.extend(Path("/dev").glob("nvidia*"))
    if nodes:
        raise RuntimeError(
            f"BASE-01 requires a container without GPU nodes: {nodes}"
        )


def environment_record() -> dict[str, object]:
    """Verify pinned lock inputs and installed key dependencies without sync."""
    require_cpu_isolation()
    if not sys.dont_write_bytecode or not sys.pycache_prefix:
        raise RuntimeError(
            "BASE-01 requires disabled writes and an isolated Python cache prefix"
        )
    lock = ROOT / "uv.lock"
    image_lock = Path("/sol-execbench/uv.lock")
    if (
        sha256_file(lock) != LOCK_SHA256
        or sha256_file(image_lock) != LOCK_SHA256
    ):
        raise RuntimeError("repository/image lock differs from BASE-01 pin")
    locked = {
        package["name"].lower().replace("_", "-"): package.get("version")
        for package in tomllib.loads(lock.read_text())["package"]
    }
    versions = {
        name: importlib.metadata.version(name) for name in LOCKED_PACKAGES
    }
    for name, version in versions.items():
        if version != locked[name.lower().replace("_", "-")]:
            raise RuntimeError(f"installed {name} does not match uv.lock")
    installed = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
        }
    )
    return {
        "required_image_id": IMAGE_ID,
        "python": sys.version,
        "python_cache_prefix": sys.pycache_prefix,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "platform": platform.platform(),
        "lock_sha256": LOCK_SHA256,
        "key_packages": versions,
        "installed_distributions": installed,
        "gpu_nodes_present": False,
        "device": "cpu",
        "torch_threads": 1,
        "publication_eligible": False,
    }


def file_inventory(root: Path) -> dict[str, str]:
    """Bind every retained artifact except Python caches."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def source_identity() -> SourceIdentity:
    """Bind current source, harness, tests and configuration before collection."""
    files: dict[str, str] = {}
    for directory in (
        "src",
        "tests/solar",
        "tests/sol_execbench/core/solar_bridge",
        "scripts/internal/solar",
    ):
        for name, digest in file_inventory(ROOT / directory).items():
            files[f"{directory}/{name}"] = digest
    for name in (
        "pyproject.toml",
        "uv.lock",
        "docker/Dockerfile",
        str(ACCEPTANCE.relative_to(ROOT)),
    ):
        files[name] = sha256_file(ROOT / name)
    payload = "".join(
        f"{name}\0{digest}\n" for name, digest in sorted(files.items())
    )
    return {
        "files": files,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def real_input_records() -> list[RealInputRecord]:
    """Verify frozen UUIDs and file hashes without executing any reference."""
    import json

    text = ACCEPTANCE.read_text()
    samples = re.findall(
        r"\| (R\d+) \| `([^`]+/[^`]+)` \| `(aka-[^`]+)` \|", text
    )
    hashes = {
        name: (definition, workload)
        for name, definition, workload in re.findall(
            r"\| (R\d+) \| `([0-9a-f]{64})` \| `([0-9a-f]{64})` \|", text
        )
    }
    if len(samples) != 8 or len(hashes) != 8:
        raise ValueError("frozen R inventory must contain exactly eight inputs")
    records: list[RealInputRecord] = []
    for name, directory, uuid in samples:
        root = ROOT / "problems/AMD_AKA" / directory
        actual = (
            sha256_file(root / "definition.json"),
            sha256_file(root / "workload.jsonl"),
        )
        rows = [
            json.loads(line)
            for line in (root / "workload.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if (
            actual != hashes[name]
            or sum(row.get("uuid") == uuid for row in rows) != 1
        ):
            raise ValueError(f"frozen real input drift: {name}")
        records.append(
            {
                "id": name,
                "directory": directory,
                "uuid": uuid,
                "definition_sha256": actual[0],
                "workload_sha256": actual[1],
                "paths": [
                    {
                        "ir_path": path.value,
                        "status": "not_run",
                        "reason": "real_workload_execution_outside_cpu_baseline",
                    }
                    for path in IRPath
                ],
            }
        )
    return records


def write_record(path: Path, value: object) -> None:
    """Atomically write an operational receipt without a public schema claim."""
    atomic_write_json_value(path, value)
