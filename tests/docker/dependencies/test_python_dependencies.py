# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"

FORBIDDEN = {
    "cuda-tile",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "cupti-python",
    "pytorch-cu130",
    "download.pytorch.org/whl/cu",
}


def test_rocm_dependency_sources():
    data = tomllib.loads(PYPROJECT.read_text())

    dependencies = data["project"]["dependencies"]
    assert "torch[device-gfx1200]==2.14.0+rocm10.1.0" in dependencies
    assert "torchvision[device-gfx1200]==0.29.0a0+rocm10.1.0" in dependencies
    assert "triton==3.8.0+git669b31ac.rocm10.1.0" in dependencies
    assert "rocm[devel,device-gfx1200]==10.1.0" in dependencies
    assert all("sys_platform != 'linux'" not in dep for dep in dependencies)
    for forbidden in FORBIDDEN:
        assert all(forbidden not in dep for dep in dependencies), forbidden

    indexes = data["tool"]["uv"]["index"]
    rocm_index = next(idx for idx in indexes if idx["name"] == "amd-rocm1010")
    assert rocm_index["url"] == "https://stable.repo.amd.com/rocm/whl-next/"

    sources = data["tool"]["uv"]["sources"]
    for package in ("torch", "torchvision", "triton", "rocm"):
        assert package in sources
        assert sources[package]["index"] == "amd-rocm1010"
