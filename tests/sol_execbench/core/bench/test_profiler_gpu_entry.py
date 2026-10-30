"""Live profiler APIs hold native exclusion before their first subprocess."""

from __future__ import annotations

from functools import partial

import pytest

from sol_execbench.core.bench.gpu_lock import (
    GPULockVerificationError,
    acquire_gpu_lock,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3CollectionRequest,
    Rocprofv3ProfileRequest,
    collect_rocprofv3_counters,
    collect_rocprofv3_profile,
    collect_rocprofv3_timing,
)
from sol_execbench.core.bench.timing_policy import (
    TimingSourceType,
    select_timing_policy,
)


@pytest.mark.parametrize("entry", ["profile", "timing", "counters"])
@pytest.mark.parametrize(
    "available", [True, False], ids=["live", "unavailable"]
)
def test_profiler_entry_excludes_competitors_before_runner(
    tmp_path, monkeypatch, entry, available
):
    locks = tmp_path / "locks"
    monkeypatch.setenv("GPU_LOCK_DIR", str(locks))
    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST", raising=False)
    monkeypatch.setattr(
        "sol_execbench.core.bench.rocm_profiler.counter_collection.resolve_rocm_tool",
        lambda _: tmp_path / "rocprofv3-avail",
    )

    def runner(*_args):
        with (
            pytest.raises(TimeoutError),
            acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pass
        raise ValueError("CPU runner stopped before device execution")

    if entry == "timing":
        request = Rocprofv3CollectionRequest(
            application_command=("cpu-fixture",),
            output_directory=tmp_path / "output",
            output_file="fixture",
            policy=select_timing_policy(TimingSourceType.TRITON),
            tool_version="CPU fixture",
            gpu_architecture="gfx1200",
        )
        collect = partial(
            collect_rocprofv3_timing,
            request,
            runner=runner,
            rocprofv3_available=available,
        )
    else:
        request = Rocprofv3ProfileRequest(
            application_command=("cpu-fixture",),
            output_directory=tmp_path / "output",
            output_file="fixture",
        )
        collect = partial(
            collect_rocprofv3_profile
            if entry == "profile"
            else collect_rocprofv3_counters,
            request,
            runner=runner,
            rocprofv3_available=available,
        )
    if available:
        with pytest.raises(ValueError, match="CPU runner stopped"):
            collect()
        with acquire_gpu_lock(timeout_seconds=0.01):
            pass
    else:
        collect()
        assert not locks.exists()


def test_internal_profiler_runner_requires_native_context(
    tmp_path, monkeypatch
):
    from sol_execbench.core.bench.rocm_profiler.commands import default_runner

    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    with pytest.raises(GPULockVerificationError, match="active native"):
        default_runner(["must-not-execute"])


pytestmark = pytest.mark.usefixtures("native_gpu_cpu_observation")
