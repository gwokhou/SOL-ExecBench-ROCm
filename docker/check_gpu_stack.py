"""Verify that the official image supplies the GPU packages in the lock."""

from __future__ import annotations

import importlib.metadata
import sys
import tomllib
from pathlib import Path

GPU_PACKAGES = frozenset({"torch", "torchvision", "triton"})
GPU_PACKAGE_PREFIXES = ("rocm", "amd-torch-device-", "amd-torchvision-device-")


def main(lock_path: Path) -> None:
    """Reject a base GPU stack that differs from any required locked wheel."""
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    expected = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in GPU_PACKAGES
        or package["name"].startswith(GPU_PACKAGE_PREFIXES)
    }
    if not expected:
        raise SystemExit("The lock has no GPU packages to verify")
    mismatches = []
    for name, version in expected.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed = "missing"
        if observed != version:
            mismatches.append(
                f"{name}: expected {version}, observed {observed}"
            )
    if mismatches:
        raise SystemExit(
            "Official GPU stack mismatch: " + "; ".join(mismatches)
        )
    print(
        f"Verified {len(expected)} inherited GPU packages against {lock_path}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
