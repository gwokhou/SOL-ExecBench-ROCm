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

"""Build a staged native solution as a Torch extension."""

from pathlib import Path

import torch.utils.cpp_extension as ext

from sol_execbench.core.data.solution import Solution
from sol_execbench.core.platform.runtime import discover_rocm_root

HERE = Path.cwd().resolve()
ENVIRON = __import__("os").environ

# Parse solution — validates sources (e.g. forbidden keywords) at compile time.
solution = Solution.model_validate_json(
    (HERE / "solution.json").read_text(),
)
compile_options = solution.spec.compile_options


ROCM_ROOT = discover_rocm_root()
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
native_languages = set(solution.spec.languages)
if native_languages & {"ck", "rocwmma"}:
    # PyTorch defines these for its own HIP extension headers, but CK and
    # rocWMMA require the standard HIP half conversions and operators.
    for flag in (
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
    ):
        if flag not in hip_cflags:
            hip_cflags.append(flag)
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
    target
    for target in solution.spec.target_hardware
    if target.startswith("gfx")
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
    # PyTorch uses this keyword for ROCm device compiler flags too.
    extra_cuda_cflags=hip_cflags,
    extra_cflags=cflags,
    extra_ldflags=ld_flags,
    extra_include_paths=extra_include_paths,
    build_directory=str(HERE),
    verbose=True,
)

# Rename platform-suffixed .so → benchmark_kernel.so
so_files = [
    f
    for f in HERE.glob("benchmark_kernel*.so")
    if f.name != "benchmark_kernel.so"
]
if so_files:
    so_files[0].rename("benchmark_kernel.so")
elif not (HERE / "benchmark_kernel.so").exists():
    raise FileNotFoundError("benchmark_kernel.so not produced by compilation")

print("benchmark_kernel.so ready", flush=True)
