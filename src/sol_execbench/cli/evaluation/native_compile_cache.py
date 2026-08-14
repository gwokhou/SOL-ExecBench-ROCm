# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed native build reuse for isolated evaluations."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import torch

from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_CONTAINER_IMAGE_ID,
    ENV_SOL_EXECBENCH_NATIVE_COMPILE_CACHE,
    ENV_SOL_EXECBENCH_SOURCE_REVISION,
)

_CACHE_ARTIFACT = "benchmark_kernel.so"
_CACHE_ENTRY = "entry.json"
NATIVE_COMPILE_CACHE_FORMAT_VERSION: Final = 1
_STATIC_ARTIFACT_SUFFIXES = frozenset({".co", ".hsaco", ".o"})
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")
_COMPILE_ENVIRONMENT_NAMES = (
    "DEVICE_LIB_PATH",
    "HIP_CLANG_PATH",
    "HIP_PATH",
    "HIP_PLATFORM",
    "HSA_OVERRIDE_GFX_VERSION",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
    "PYTORCH_ROCM_ARCH",
    "ROCM_PATH",
    "VIRTUAL_ENV",
)


@dataclass(frozen=True, slots=True)
class NativeCompileCache:
    """One exact cache lookup bound to immutable build inputs."""

    root: Path
    key: str
    identity: dict[str, object]

    @classmethod
    def from_environment(
        cls,
        *,
        staging_dir: Path,
        command: Sequence[str],
        compile_environment: Mapping[str, str],
        compiler_path: str,
        compiler_sha256: str,
        compiler_version: str,
        environ: Mapping[str, str] | None = None,
    ) -> NativeCompileCache | None:
        """Build a strict lookup, or return ``None`` when caching is disabled."""
        environment = os.environ if environ is None else environ
        raw_root = environment.get(ENV_SOL_EXECBENCH_NATIVE_COMPILE_CACHE)
        if raw_root is None:
            return None
        root = Path(raw_root)
        if not root.is_absolute():
            raise ValueError("native compile cache root must be absolute")
        identity = _cache_identity(
            staging_dir=staging_dir,
            command=command,
            compile_environment=compile_environment,
            compiler_path=compiler_path,
            compiler_sha256=compiler_sha256,
            compiler_version=compiler_version,
            environ=environment,
        )
        return cls(
            root=root, key=stable_json_checksum(identity), identity=identity
        )

    def restore(self, destination: Path) -> bool:
        """Restore the verified build inventory while holding the cache lock."""
        with _locked_cache_root(self.root):
            entry_dir = self.root / self.key
            if not entry_dir.exists():
                return False
            if not _entry_uses_current_format(entry_dir):
                shutil.rmtree(entry_dir)
                return False
            inventory = _verified_entry_inventory(entry_dir, self)
            for relative_path, artifact in inventory:
                target = destination.parent / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink() or target.exists():
                    raise ValueError(
                        "native compile cache restore target already exists"
                    )
                shutil.copy2(artifact, target)
                if sha256_file(target) != sha256_file(artifact):
                    target.unlink(missing_ok=True)
                    raise ValueError(
                        "native compile cache restore hash mismatch"
                    )
            return True

    def store(self, artifact: Path) -> None:
        """Atomically publish a successful compile without overwriting entries."""
        inventory = _build_artifact_inventory(artifact)
        with _locked_cache_root(self.root):
            entry_dir = self.root / self.key
            if entry_dir.exists():
                if not _entry_uses_current_format(entry_dir):
                    shutil.rmtree(entry_dir)
                else:
                    _verified_entry_inventory(entry_dir, self)
                    return
            with TemporaryDirectory(
                prefix=f".{self.key}.", dir=self.root
            ) as temporary:
                staged = Path(temporary)
                artifacts = []
                for relative_path, source in inventory:
                    staged_artifact = staged / relative_path
                    staged_artifact.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, staged_artifact)
                    artifacts.append(
                        {
                            "relative_path": relative_path.as_posix(),
                            "sha256": sha256_file(staged_artifact),
                            "size_bytes": staged_artifact.stat().st_size,
                        }
                    )
                payload = {
                    "format_version": NATIVE_COMPILE_CACHE_FORMAT_VERSION,
                    "cache_key_sha256": self.key,
                    "identity": self.identity,
                    "artifacts": artifacts,
                }
                atomic_write_json_value(staged / _CACHE_ENTRY, payload)
                staged.rename(entry_dir)


def _cache_identity(
    *,
    staging_dir: Path,
    command: Sequence[str],
    compile_environment: Mapping[str, str],
    compiler_path: str,
    compiler_sha256: str,
    compiler_version: str,
    environ: Mapping[str, str],
) -> dict[str, object]:
    image_id = environ.get(ENV_SOL_EXECBENCH_CONTAINER_IMAGE_ID, "")
    source_revision = environ.get(ENV_SOL_EXECBENCH_SOURCE_REVISION, "")
    if _IMAGE_ID.fullmatch(image_id) is None:
        raise ValueError(
            "native compile cache requires immutable image identity"
        )
    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        raise ValueError("native compile cache requires exact source revision")
    return {
        "format_version": NATIVE_COMPILE_CACHE_FORMAT_VERSION,
        "solution_sha256": sha256_file(staging_dir / "solution.json"),
        "build_script_sha256": sha256_file(staging_dir / "build_ext.py"),
        "command": list(command),
        "compiler_path": compiler_path,
        "compiler_sha256": compiler_sha256,
        "compiler_version": compiler_version,
        "python_sha256": sha256_file(Path(sys.executable).resolve()),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "container_image_id": image_id,
        "source_revision": source_revision,
        "compile_environment": {
            name: compile_environment[name]
            for name in _COMPILE_ENVIRONMENT_NAMES
            if name in compile_environment
        },
    }


@contextmanager
def _locked_cache_root(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native compile cache root is not a regular directory")
    lock_path = root / ".lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_artifact_inventory(
    primary_artifact: Path,
) -> tuple[tuple[Path, Path], ...]:
    build_root = primary_artifact.parent.resolve()
    if primary_artifact.is_symlink() or not primary_artifact.is_file():
        raise ValueError("compiled artifact is not a regular file")
    paths = [primary_artifact.resolve()]
    paths.extend(
        path.resolve()
        for path in sorted(build_root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in _STATIC_ARTIFACT_SUFFIXES
    )
    return tuple(
        (path.relative_to(build_root), path) for path in dict.fromkeys(paths)
    )


def _verified_entry_inventory(
    entry_dir: Path,
    cache: NativeCompileCache,
) -> tuple[tuple[Path, Path], ...]:
    if entry_dir.is_symlink() or not entry_dir.is_dir():
        raise ValueError(
            "native compile cache entry is not a regular directory"
        )
    entry_path = entry_dir / _CACHE_ENTRY
    if entry_path.is_symlink() or not entry_path.is_file():
        raise ValueError("native compile cache metadata is not a regular file")
    payload = load_json_value(entry_path)
    expected = {
        "format_version": NATIVE_COMPILE_CACHE_FORMAT_VERSION,
        "cache_key_sha256": cache.key,
        "identity": cache.identity,
    }
    if not isinstance(payload, dict) or any(
        payload.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("native compile cache entry identity mismatch")
    raw_inventory = payload.get("artifacts")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise ValueError("native compile cache inventory is missing")
    inventory = tuple(
        _verified_inventory_item(entry_dir, item) for item in raw_inventory
    )
    relative_paths = tuple(item[0] for item in inventory)
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("native compile cache inventory paths are duplicated")
    if Path(_CACHE_ARTIFACT) not in set(relative_paths):
        raise ValueError("native compile cache primary artifact is missing")
    actual_paths = {
        path.relative_to(entry_dir)
        for path in entry_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != {*relative_paths, Path(_CACHE_ENTRY)}:
        raise ValueError("native compile cache inventory is not exact")
    return inventory


def _entry_uses_current_format(entry_dir: Path) -> bool:
    """Return whether a structurally readable cache entry is reusable."""
    if entry_dir.is_symlink() or not entry_dir.is_dir():
        raise ValueError(
            "native compile cache entry is not a regular directory"
        )
    entry_path = entry_dir / _CACHE_ENTRY
    if entry_path.is_symlink() or not entry_path.is_file():
        raise ValueError("native compile cache metadata is not a regular file")
    payload = load_json_value(entry_path)
    if not isinstance(payload, dict):
        raise ValueError("native compile cache metadata is invalid")
    return payload.get("format_version") == NATIVE_COMPILE_CACHE_FORMAT_VERSION


def _verified_inventory_item(
    entry_dir: Path,
    raw_item: object,
) -> tuple[Path, Path]:
    if not isinstance(raw_item, dict):
        raise ValueError("native compile cache inventory item is invalid")
    raw_path = raw_item.get("relative_path")
    if not isinstance(raw_path, str):
        raise ValueError("native compile cache inventory path is invalid")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("native compile cache inventory path escapes entry")
    artifact = entry_dir / relative_path
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("native compile cache artifact is not a regular file")
    if (
        raw_item.get("sha256") != sha256_file(artifact)
        or raw_item.get("size_bytes") != artifact.stat().st_size
    ):
        raise ValueError("native compile cache artifact integrity mismatch")
    return relative_path, artifact


__all__ = ["NativeCompileCache"]
