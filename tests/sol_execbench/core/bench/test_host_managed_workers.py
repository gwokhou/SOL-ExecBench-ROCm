"""CPU subprocess coverage for host exclusion plus worker-stage serialization."""

from __future__ import annotations

import fcntl
import select
import subprocess
import sys
from contextlib import contextmanager

import pytest

from sol_execbench.core.bench.clock_ownership import ClockDevice
from sol_execbench.core.bench.gpu_execution import _natural_clock_owner
from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock

_WORKER = """
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from sol_execbench.core.bench.gpu_lock import acquire_evaluation_gpu_lock
from sol_execbench.core.solar_bridge.formal_device import device_execution_policy
from solar.pipeline import analysis
from sol_execbench.core.bench import gpu_execution
from sol_execbench.core.bench.clock_ownership import ClockDevice

gpu_execution.resolve_gpu_device = lambda: ClockDevice(bdf='0000:05:00.0', uuid='11111111-2222-3333-4444-555555555555')
gpu_execution.observe_clock_levels = lambda _: ('AMDSMI_DEV_PERF_LEVEL_AUTO',)
gpu_execution.synchronize_initialized_device = lambda: None

root, mode, explicit = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'yes'
policy = device_execution_policy(
    'cuda:0', root / 'gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock' if explicit else None, 0.15
)

def cleanup():
    with (root / 'cleaned').open('a') as handle:
        handle.write(mode + '\\n')

def extract(*args):
    with acquire_evaluation_gpu_lock(timeout_seconds=0.05):
        print('entered', flush=True)
        if mode == 'hold':
            sys.stdin.readline()
    raise ValueError('CPU stage finished')

analysis.extract_request_graph = extract
request = SimpleNamespace(execution_policy=replace(policy, device_stage_cleanup=cleanup))
try:
    analysis.run_pipeline(request, None, root)
except analysis.PipelineStageError as error:
    if isinstance(error.error, TimeoutError):
        print('blocked', flush=True)
        sys.exit(42)
    if isinstance(error.error, ValueError) and str(error.error) == 'CPU stage finished':
        sys.exit(0)
    raise
"""


@pytest.mark.parametrize("host_managed", [False, True], ids=["native", "host"])
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "shared"])
def test_solar_workers_serialize_under_host_lock(
    tmp_path, monkeypatch, host_managed, explicit
):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("GPU_LOCK_FD", raising=False)
    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST", raising=False)
    command = [sys.executable, "-B", "-c", _WORKER, str(tmp_path)]
    option = "yes" if explicit else "no"
    with _parent_owner(host_managed, monkeypatch):
        first = subprocess.Popen(
            [*command, "hold", option],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert first.stdout is not None
            assert select.select([first.stdout], [], [], 10)[0]
            assert first.stdout.readline().strip() == "entered"
            blocked = subprocess.run(
                [*command, "probe", option],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert blocked.returncode == 42, blocked.stdout + blocked.stderr
            assert blocked.stdout.strip() == "blocked"
            assert not (tmp_path / "cleaned").exists()
            _, error = first.communicate("release\n", timeout=10)
            assert first.returncode == 0, error
            after = subprocess.run(
                [*command, "probe", option],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert after.returncode == 0, after.stdout + after.stderr
            assert after.stdout.strip() == "entered"
            assert (tmp_path / "cleaned").read_text().splitlines() == [
                "hold",
                "probe",
            ]
            if host_managed:
                with (
                    (
                        tmp_path
                        / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
                    ).open("a+") as handle,
                    pytest.raises(BlockingIOError),
                ):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if first.poll() is None:
                first.kill()
            first.communicate(timeout=5)
    with acquire_gpu_lock(timeout_seconds=0.1):
        pass


@contextmanager
def _parent_owner(host_managed, monkeypatch):
    if not host_managed:
        yield
        return
    device = ClockDevice(
        bdf="0000:05:00.0", uuid="11111111-2222-3333-4444-555555555555"
    )
    with acquire_gpu_lock(), _natural_clock_owner(device):
        monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")
        try:
            yield
        finally:
            monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST")


pytestmark = pytest.mark.usefixtures("native_gpu_cpu_observation")
