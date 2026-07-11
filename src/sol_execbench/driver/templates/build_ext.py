# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import shutil

import torch.utils.cpp_extension as ext

from sol_execbench.core.data.solution import Solution

HERE = Path.cwd().resolve()
ENVIRON = __import__("os").environ

# Parse solution — validates sources (e.g. forbidden keywords) at compile time.
solution = Solution(**json.loads((HERE / "solution.json").read_text()))
compile_options = solution.spec.compile_options


def _active_rocm_root() -> Path | None:
    """Resolve the installed ROCm tree from HIPCC rather than assuming /opt/rocm."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        return None
    resolved = Path(hipcc).resolve()
    # HIPCC normally lives at <ROCM_ROOT>/bin/hipcc.  Keep this defensive for
    # distro wrappers that do not resolve into the ROCm installation.
    return resolved.parent.parent if resolved.parent.name == "bin" else None


ROCM_ROOT = _active_rocm_root()
if ROCM_ROOT is not None:
    ENVIRON.setdefault("CXX", str(ROCM_ROOT / "bin" / "hipcc"))


def _rebase_rocm_flag(flag: str) -> str:
    """Map portable solution flags to the actual installed ROCm root."""
    if ROCM_ROOT is None:
        return flag
    if flag == "-I/opt/rocm/include":
        return f"-I{ROCM_ROOT / 'include'}"
    if flag == "-L/opt/rocm/lib":
        return f"-L{ROCM_ROOT / 'lib'}"
    return flag


# set flags
hip_cflags = list(compile_options.hip_cflags) if compile_options else []
cflags = (
    [_rebase_rocm_flag(flag) for flag in compile_options.cflags]
    if compile_options
    else []
)
ld_flags = (
    [_rebase_rocm_flag(flag) for flag in compile_options.ld_flags]
    if compile_options
    else []
)

rocm_arches = [
    target.value
    for target in solution.spec.target_hardware
    if target.value.startswith("gfx")
]
if rocm_arches and "PYTORCH_ROCM_ARCH" not in ENVIRON:
    ENVIRON["PYTORCH_ROCM_ARCH"] = ";".join(dict.fromkeys(rocm_arches))

# Collect HIP/C++ source files from current directory
sources = [
    str(p)
    for p in HERE.iterdir()
    if p.suffix in (".hip", ".cpp", ".cc", ".cxx", ".c") and p.is_file()
]
if not sources:
    raise RuntimeError("No HIP/C++ source files found in working directory")

extra_include_paths = [str(HERE)]

ext.load(
    name="benchmark_kernel",
    sources=sources,
    # PyTorch extension API uses this keyword for device compiler flags on ROCm too.
    extra_cuda_cflags=hip_cflags,
    extra_cflags=cflags,
    extra_ldflags=ld_flags,
    extra_include_paths=extra_include_paths,
    build_directory=str(HERE),
    verbose=True,
)

# Rename platform-suffixed .so → benchmark_kernel.so
so_files = [
    f for f in HERE.glob("benchmark_kernel*.so") if f.name != "benchmark_kernel.so"
]
if so_files:
    so_files[0].rename("benchmark_kernel.so")
elif not (HERE / "benchmark_kernel.so").exists():
    raise FileNotFoundError("benchmark_kernel.so not produced by compilation")

print("benchmark_kernel.so ready", flush=True)
