"""Concurrent atomic cache publication and identity invalidation; CPU files only."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import patch

import pytest

from sol_execbench.cli.evaluation import compilation, native_compile_cache
from sol_execbench.cli.evaluation.native_compile_cache import NativeCompileCache


def _cache(tmp_path, **overrides):
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    for name in ("solution.json", "build_ext.py"):
        (staging / name).write_text("input")
    kwargs: dict[str, Any] = {
        "staging_dir": staging,
        "command": ["python", "build_ext.py"],
        "compile_environment": {},
        "compiler_path": "/hipcc",
        "compiler_sha256": "a" * 64,
        "compiler_version": "test",
        "environ": {
            "SOL_EXECBENCH_NATIVE_COMPILE_CACHE": str(tmp_path / "cache"),
            "SOL_EXECBENCH_CONTAINER_IMAGE_ID": "sha256:" + "b" * 64,
            "SOL_EXECBENCH_SOURCE_REVISION": "c" * 40,
        },
    }
    kwargs.update(overrides)
    return NativeCompileCache.from_environment(**kwargs)


@pytest.mark.parametrize(
    "changed",
    ["compiler_version", "compiler_sha256", "command", "compile_environment"],
    ids=["tool-version", "tool-bytes", "flags", "target"],
)
def test_cache_key_invalidates_build_inputs(tmp_path, changed):
    first = _cache(tmp_path)
    values = {
        "compiler_version": "changed",
        "compiler_sha256": "d" * 64,
        "command": ["python", "build_ext.py", "--changed"],
        "compile_environment": {"PYTORCH_ROCM_ARCH": "gfx1201"},
    }
    assert first.key != _cache(tmp_path, **{changed: values[changed]}).key


def test_cache_atomic_concurrent_publication_and_restore(tmp_path):
    cache = _cache(tmp_path)
    artifacts = []
    for n in range(2):
        folder = tmp_path / str(n)
        folder.mkdir()
        artifact = folder / "benchmark_kernel.so"
        artifact.write_bytes(b"same binary")
        (folder / "probe.o").write_bytes(b"object")
        artifacts.append(artifact)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(cache.store, artifacts))
    target = tmp_path / "target" / "benchmark_kernel.so"
    assert cache.restore(target)
    assert target.read_bytes() == b"same binary"
    assert (target.parent / "probe.o").read_bytes() == b"object"
    assert len(list(cache.root.glob("*/entry.json"))) == 1
    target.write_bytes(b"not a cache edit")
    third = tmp_path / "third" / target.name
    assert cache.restore(third)
    assert third.read_bytes() == b"same binary"


def test_cache_failed_store_removes_partial_entry(tmp_path):
    cache = _cache(tmp_path)
    artifact = tmp_path / "benchmark_kernel.so"
    artifact.write_bytes(b"binary")
    with (
        patch.object(
            native_compile_cache.shutil,
            "copy2",
            side_effect=OSError("disk error"),
        ),
        pytest.raises(OSError, match="disk error"),
    ):
        cache.store(artifact)
    assert not (cache.root / cache.key).exists()
    assert not list(cache.root.glob(f".{cache.key}.*"))
    cache.store(artifact)
    assert cache.restore(tmp_path / "target" / artifact.name)


def test_cache_restore_conflict_does_not_copy_prefix(tmp_path):
    cache = _cache(tmp_path)
    source = tmp_path / "build"
    source.mkdir()
    primary = source / "benchmark_kernel.so"
    primary.write_bytes(b"binary")
    (source / "later.o").write_bytes(b"object")
    cache.store(primary)
    target = tmp_path / "target"
    target.mkdir()
    (target / "later.o").write_bytes(b"user object")
    with pytest.raises(ValueError, match="already exists"):
        cache.restore(target / primary.name)
    assert not (target / primary.name).exists()
    assert (target / "later.o").read_bytes() == b"user object"


def test_compile_max_jobs_is_capped_and_not_runtime_passthrough(
    tmp_path, monkeypatch
):
    import subprocess
    from types import SimpleNamespace

    from sol_execbench.core.process.environment import sanitized_subprocess_env

    observed = []
    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(compilation, "compile_job_limit", lambda n: min(n, 2))
    packager = SimpleNamespace(
        _is_cpp=True, compile=lambda: (["build"], str(tmp_path / "kernel.so"))
    )

    def runner(command, **kwargs):
        observed.append(kwargs["env"])
        return subprocess.CompletedProcess(command, 0, "", "")

    compilation.run_compile_phase(
        cast(compilation.CompilePackager, packager),
        staging_dir=tmp_path,
        compile_timeout=5,
        runner=runner,
    )
    assert observed[0]["MAX_JOBS"] == "2"
    assert "MAX_JOBS" not in sanitized_subprocess_env(
        {"MAX_JOBS": "8"}, staging_dir=tmp_path
    )


def test_cache_rejects_corrupt_artifact_without_copy(tmp_path):
    cache = _cache(tmp_path)
    artifact = tmp_path / "benchmark_kernel.so"
    artifact.write_bytes(b"original")
    cache.store(artifact)
    (cache.root / cache.key / artifact.name).write_bytes(b"tampered")
    target = tmp_path / "restored" / artifact.name
    with pytest.raises(ValueError, match="integrity mismatch"):
        cache.restore(target)
    assert not target.exists()
