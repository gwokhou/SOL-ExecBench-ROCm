from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.process.environment import (
    ENV_PYTORCH_ALLOC_CONF,
    sanitized_subprocess_env,
)


def test_preserves_explicit_rocm_and_thread_runtime_configuration(
    tmp_path: Path,
):
    base = {
        "HIP_CLANG_PATH": "/opt/rocm/llvm/bin",
        "OMP_NUM_THREADS": "3",
        "MIOPEN_USER_DB_PATH": "/var/cache/miopen",
        "DEVICE_LIB_PATH": "/opt/rocm/amdgcn/bitcode",
        "SOL_EXECBENCH_DEVICE": "cuda:1",
        "OPENAI_API_KEY": "secret",
    }

    result = sanitized_subprocess_env(base, staging_dir=tmp_path)

    assert result["HIP_CLANG_PATH"] == "/opt/rocm/llvm/bin"
    assert result["OMP_NUM_THREADS"] == "3"
    assert result["MIOPEN_USER_DB_PATH"] == "/var/cache/miopen"
    assert result["DEVICE_LIB_PATH"] == "/opt/rocm/amdgcn/bitcode"
    assert result["SOL_EXECBENCH_DEVICE"] == "cuda:1"
    assert result["ROCPROF_TMPDIR"] == str(tmp_path / ".tmp")
    assert "OPENAI_API_KEY" not in result


@pytest.mark.parametrize(
    "allocator_environment",
    [
        {},
        {ENV_PYTORCH_ALLOC_CONF: "expandable_segments:True"},
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        },
    ],
    ids=["default", "inherited-current-option", "inherited-legacy-options"],
)
def test_evaluation_disables_unqualified_expandable_segments(
    tmp_path: Path, allocator_environment: dict[str, str]
) -> None:
    """Keep VMM out of both workers even when the parent enables it."""
    result = sanitized_subprocess_env(
        allocator_environment, staging_dir=tmp_path
    )

    assert result[ENV_PYTORCH_ALLOC_CONF] == "expandable_segments:False"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in result
    assert "PYTORCH_HIP_ALLOC_CONF" not in result
