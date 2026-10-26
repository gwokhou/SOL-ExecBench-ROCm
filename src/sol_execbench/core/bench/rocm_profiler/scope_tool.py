# SPDX-License-Identifier: Apache-2.0
"""Build the packaged SDK collector once per independent diagnostic archive."""

from importlib.resources import files
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.platform.runtime import discover_rocm_root
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

COLLECTOR_SOURCE = "scope_collector.cpp"
COLLECTOR_LIBRARY = "libscope-collector.so"
COLLECTOR_BUILD_TIMEOUT = 120


def build_scope_collector(directory: Path) -> Path:
    """Retain source and bounded build logs; never load caller-provided binaries."""
    source = directory / COLLECTOR_SOURCE
    source.write_bytes(
        files("sol_execbench.core.bench.rocm_profiler")
        .joinpath(COLLECTOR_SOURCE)
        .read_bytes()
    )
    rocm = discover_rocm_root()
    if rocm is None:
        raise RuntimeError("scope_rocm_root_unavailable")
    target = directory / COLLECTOR_LIBRARY
    command = [
        "g++",
        "-D__HIP_PLATFORM_AMD__",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        f"-I{rocm}/include",
        str(source),
        f"-L{rocm}/lib",
        "-lrocprofiler-sdk",
        f"-Wl,-rpath,{rocm}/lib",
        "-o",
        str(target),
    ]
    result = run_in_process_group_bounded(
        command, timeout=COLLECTOR_BUILD_TIMEOUT
    )
    atomic_write_json_value(
        directory / "collector-build.json",
        {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode != 0:
        raise RuntimeError("scope_collector_build_failed")
    return target
