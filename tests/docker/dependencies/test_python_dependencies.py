# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tomllib

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
    assert (
        "torch==2.10.0; sys_platform != 'linux' or platform_machine != 'x86_64'"
        in dependencies
    )
    assert (
        "torch==2.11.0+rocm7.2; sys_platform == 'linux' and platform_machine == 'x86_64'"
        in dependencies
    )
    assert (
        "torchvision==0.25.0; sys_platform != 'linux' or platform_machine != 'x86_64'"
        in dependencies
    )
    assert (
        "torchvision==0.26.0+rocm7.2; sys_platform == 'linux' and platform_machine == 'x86_64'"
        in dependencies
    )
    for forbidden in FORBIDDEN:
        assert all(forbidden not in dep for dep in dependencies), forbidden

    indexes = data["tool"]["uv"]["index"]
    assert any(idx["name"] == "pytorch-rocm72" for idx in indexes)
    rocm_index = next(idx for idx in indexes if idx["name"] == "pytorch-rocm72")
    assert rocm_index["url"] == "https://download.pytorch.org/whl/rocm7.2"
    assert rocm_index["explicit"] is True

    sources = data["tool"]["uv"]["sources"]
    for package in ("torch", "torchvision"):
        assert package in sources
        assert any(src["index"] == "pytorch-rocm72" for src in sources[package])
