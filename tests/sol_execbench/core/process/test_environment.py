from __future__ import annotations

from pathlib import Path

from sol_execbench.core.process.environment import sanitized_subprocess_env


def test_preserves_explicit_rocm_and_thread_runtime_configuration(tmp_path: Path):
    base = {
        "HIP_CLANG_PATH": "/opt/rocm/llvm/bin",
        "OMP_NUM_THREADS": "3",
        "MIOPEN_USER_DB_PATH": "/var/cache/miopen",
        "DEVICE_LIB_PATH": "/opt/rocm/amdgcn/bitcode",
        "OPENAI_API_KEY": "secret",
    }

    result = sanitized_subprocess_env(base, staging_dir=tmp_path)

    assert result["HIP_CLANG_PATH"] == "/opt/rocm/llvm/bin"
    assert result["OMP_NUM_THREADS"] == "3"
    assert result["MIOPEN_USER_DB_PATH"] == "/var/cache/miopen"
    assert result["DEVICE_LIB_PATH"] == "/opt/rocm/amdgcn/bitcode"
    assert "OPENAI_API_KEY" not in result
