# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned, integrity-checked retrieval of AMD ISA XML bundles."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.tools.amd_isa.errors import (
    IsaDownloadError,
    IsaIntegrityError,
    IsaSpecUnavailableError,
)


_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
_LOCK_NAME = ".lock"


@dataclass(frozen=True)
class IsaSpecDescriptor:
    """Integrity-bound identity for one resolved ISA specification."""

    architecture: str
    family: str
    release: str
    path: Path
    sha256: str


def _cache_root() -> Path:
    configured = os.environ.get("SOL_EXECBENCH_AMD_ISA_CACHE")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (
        Path(xdg).expanduser() / "sol-execbench" / "amd-isa"
        if xdg
        else Path.home() / ".cache" / "sol-execbench" / "amd-isa"
    )


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class IsaSpecRepository:
    """Resolve project-supported gfx targets to downloaded, locked XML specs."""

    _GFX_PREFIXES = (
        ("gfx101", "rdna1"),
        ("gfx103", "rdna2"),
        ("gfx110", "rdna3"),
        ("gfx115", "rdna3_5"),
        ("gfx12", "rdna4"),
        ("gfx908", "cdna1"),
        ("gfx90a", "cdna2"),
        ("gfx94", "cdna3"),
        ("gfx95", "cdna4"),
    )

    def __init__(self, cache_root: Path | None = None) -> None:
        payload = json.loads(
            resources.files("sol_execbench.tools.amd_isa")
            .joinpath("releases.json")
            .read_text(encoding="utf-8")
        )
        if payload.get("schema_version") != "sol_execbench.amd_isa_release_lock.v1":
            raise IsaIntegrityError("unsupported AMD ISA release-lock schema")
        self._releases: dict[str, dict[str, Any]] = payload["releases"]
        self._default_release = str(payload["default_release"])
        self.cache_root = cache_root or _cache_root()

    @classmethod
    def architecture_family(cls, architecture: str) -> str:
        token = architecture.lower().split(":", maxsplit=1)[0].strip()
        if token in {family for _, family in cls._GFX_PREFIXES}:
            return token
        for prefix, family in cls._GFX_PREFIXES:
            if token.startswith(prefix):
                return family
        raise IsaSpecUnavailableError(
            f"no machine-readable AMD ISA family is declared for '{architecture}'"
        )

    def status(self, release: str | None = None) -> dict[str, object]:
        release = release or self._default_release
        entry = self._release(release)
        root = self._release_root(release)
        return {
            "release": release,
            "available": self._is_complete(root, entry),
            "cache_root": str(root),
            "families": sorted(entry["files"]),
        }

    def spec_path(
        self,
        architecture: str,
        *,
        release: str | None = None,
        allow_download: bool = True,
    ) -> Path:
        return self.resolve(
            architecture,
            release=release,
            allow_download=allow_download,
        ).path

    def resolve(
        self,
        architecture: str,
        *,
        release: str | None = None,
        allow_download: bool = True,
    ) -> IsaSpecDescriptor:
        """Resolve an architecture to an integrity-bound local spec."""

        selected_release = release or self._default_release
        family = self.architecture_family(architecture)
        root = self.ensure(
            release=selected_release,
            allow_download=allow_download,
        )
        spec = self._release(selected_release)["files"][family]
        return IsaSpecDescriptor(
            architecture=architecture.lower().split(":", maxsplit=1)[0].strip(),
            family=family,
            release=selected_release,
            path=root / spec["name"],
            sha256=str(spec["sha256"]),
        )

    def ensure(
        self, *, release: str | None = None, allow_download: bool = True
    ) -> Path:
        release = release or self._default_release
        entry = self._release(release)
        root = self._release_root(release)
        if self._is_complete(root, entry):
            return root
        if not allow_download or os.environ.get("SOL_EXECBENCH_AMD_ISA_OFFLINE") == "1":
            raise IsaSpecUnavailableError(
                f"AMD ISA release {release} is absent from {root}; downloads are disabled"
            )
        with _file_lock(root.parent / _LOCK_NAME):
            if self._is_complete(root, entry):
                return root
            self._download_and_extract(root, entry)
        return root

    def purge(self, release: str | None = None) -> None:
        """Delete one locally cached release; never touches a source checkout."""
        release = release or self._default_release
        root = self._release_root(release)
        with _file_lock(root.parent / _LOCK_NAME):
            shutil.rmtree(root, ignore_errors=True)

    def _release(self, release: str) -> dict[str, Any]:
        try:
            return self._releases[release]
        except KeyError as exc:
            raise IsaSpecUnavailableError(
                f"unknown AMD ISA release '{release}'"
            ) from exc

    def _release_root(self, release: str) -> Path:
        return self.cache_root / "specs" / release

    @staticmethod
    def _is_complete(root: Path, entry: dict[str, Any]) -> bool:
        for spec in entry["files"].values():
            path = root / spec["name"]
            if not path.is_file() or sha256_file(path) != spec["sha256"]:
                return False
        return True

    def _download_and_extract(self, root: Path, entry: dict[str, Any]) -> None:
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="download-", dir=root.parent))
        archive = temporary / "bundle.zip"
        try:
            request = Request(
                entry["url"], headers={"User-Agent": "sol-execbench-amd-isa"}
            )
            try:
                with (
                    urlopen(request, timeout=60) as response,
                    archive.open("wb") as output,
                ):
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > _MAX_ARCHIVE_BYTES:
                            raise IsaIntegrityError(
                                "AMD ISA archive exceeds size limit"
                            )
                        output.write(chunk)
            except URLError as exc:
                raise IsaDownloadError(
                    f"unable to download AMD ISA archive: {exc}"
                ) from exc
            if sha256_file(archive) != entry["archive_sha256"]:
                raise IsaIntegrityError(
                    "AMD ISA archive checksum does not match release lock"
                )
            expected = {
                spec["name"]: spec["sha256"] for spec in entry["files"].values()
            }
            destination = temporary / "specs"
            destination.mkdir()
            with zipfile.ZipFile(archive) as bundle:
                names = {item.filename for item in bundle.infolist()}
                if names != set(expected) or any(
                    item.is_dir() for item in bundle.infolist()
                ):
                    raise IsaIntegrityError("AMD ISA archive has unexpected members")
                if (
                    sum(item.file_size for item in bundle.infolist())
                    > _MAX_EXTRACTED_BYTES
                ):
                    raise IsaIntegrityError(
                        "AMD ISA archive exceeds extracted size limit"
                    )
                for name, checksum in expected.items():
                    target = destination / name
                    with bundle.open(name) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if sha256_file(target) != checksum:
                        raise IsaIntegrityError(
                            f"AMD ISA XML checksum mismatch: {name}"
                        )
            if root.exists():
                shutil.rmtree(root)
            destination.replace(root)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
