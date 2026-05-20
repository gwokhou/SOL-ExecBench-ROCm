# Phase 01: ROCm Environment Baseline - Research

**Researched:** 2026-05-21  
**Domain:** ROCm Docker, Python GPU dependencies, container dependency tests  
**Confidence:** HIGH for Docker and environment-test shape; MEDIUM for exact PyTorch/Triton wheel pinning because ROCm/PyTorch wheel support changes quickly.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
No explicit locked implementation decisions were recorded. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]

### the agent's Discretion
All implementation choices are at the agent's discretion — pure infrastructure phase. Preserve existing package layout, Docker script conventions, and benchmark semantics while making the minimum environment changes needed to establish a ROCm-only baseline. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]

### Deferred Ideas (OUT OF SCOPE)
- HIP/C++ solution staging and compiler flag migration are deferred to Phase 2. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]
- ROCm evaluation runtime, timing, profiling, and hardware introspection are deferred to Phase 3. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]
- Public example and library replacement decisions are deferred to Phase 4. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]
- Full test suite and RDNA 4/CDNA 3 hardware validation are deferred to Phase 5. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | Developer can build a ROCm >= 7.0 Docker image for this repository. [VERIFIED: .planning/REQUIREMENTS.md] | Use a pinned AMD ROCm Ubuntu 24.04 image tag and preserve `./scripts/run_docker.sh --build`. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags] |
| ENV-02 | Docker image includes ROCm runtime, HIP compiler tooling, ROCm profiling tools, AMD system management tools, and required ROCm libraries. [VERIFIED: .planning/REQUIREMENTS.md] | Validate `rocminfo`, `hipcc`, `rocprofv3`, `amd-smi` or `rocm-smi`, and selected library linkage. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html] |
| ENV-03 | Docker image installs PyTorch for ROCm and Triton for ROCm without CUDA wheel dependencies. [VERIFIED: .planning/REQUIREMENTS.md] | Replace `pytorch-cu130` with a ROCm wheel index and remove NVIDIA-only packages. [CITED: https://pytorch.org/get-started/previous-versions/] |
| ENV-04 | Docker dependency tests verify ROCm, HIP, PyTorch ROCm, Triton ROCm, and selected ROCm libraries are importable or executable. [VERIFIED: .planning/REQUIREMENTS.md] | Replace CUDA/CUTLASS/cuDNN/cuTile smoke tests with ROCm command/import/compile smoke tests and clear skip/fail classification. [VERIFIED: tests/docker/dependencies/] |
| SCFG-03 | NVIDIA/CUDA-only dependency declarations are removed or replaced with ROCm equivalents. [VERIFIED: .planning/REQUIREMENTS.md] | Remove `cuda-tile`, `nvidia-cudnn-frontend`, `nvidia-cutlass-dsl[cu13]`, `cupti-python`, and the `cu130` index from Phase 1 dependency scope. [VERIFIED: pyproject.toml] |
</phase_requirements>

## Summary

The current baseline is explicitly CUDA/NVIDIA: `docker/Dockerfile` uses `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`, exports CUDA paths, installs NVIDIA CUTLASS, and grants sudo access to `nvidia-smi`. [VERIFIED: docker/Dockerfile] `pyproject.toml` points `torch` and `torchvision` at the `pytorch-cu130` index and declares NVIDIA-only packages. [VERIFIED: pyproject.toml] `tests/docker/dependencies/` currently validates CUDA, CUTLASS, cuDNN, CuTe DSL, cuTile, and basic Triton import. [VERIFIED: tests/docker/dependencies/]

The most plannable Phase 1 approach is to keep the existing container flow, replace the base image and Python wheel source, and rewrite only Docker dependency tests. [ASSUMED] Use an exact ROCm image tag rather than `latest`; exact tags make the environment reproducible and prevent silent toolchain drift. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html] Prefer ROCm 7.1.1 for implementation planning because it satisfies ROCm >= 7.0, matches the host toolchain observed in this session, and aligns with stable PyTorch 2.10.0 ROCm 7.1 installation commands. [VERIFIED: hipcc --version] [CITED: https://pytorch.org/get-started/previous-versions/]

**Primary recommendation:** build from `rocm/dev-ubuntu-24.04:7.1.1-complete`, configure `torch==2.10.0` and `torchvision==0.25.0` from `https://download.pytorch.org/whl/rocm7.1`, remove NVIDIA-only Python packages, and replace Docker tests with ROCm runtime/tool/compiler/library checks. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags] [CITED: https://pytorch.org/get-started/previous-versions/]

## Project Constraints (from AGENTS.md)

- Source lives under `src/sol_execbench/`; tests live under `tests/`, Docker support under `docker/`, and helper scripts under `scripts/`. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff style; keep changes consistent with nearby modules and avoid broad refactors. [VERIFIED: AGENTS.md]
- Use `uv sync --all-groups`, `uv run pytest tests/`, `uv run ruff check .`, and `./scripts/run_docker.sh --build` as standard commands. [VERIFIED: AGENTS.md]
- Pytest is the test framework; environment-sensitive tests should use existing marker patterns and live next to related coverage. [VERIFIED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, or downloaded datasets. [VERIFIED: AGENTS.md]
- ROCm >= 7.0 is the supported software baseline; RDNA 4 and CDNA 3 must eventually pass adapted tests. [VERIFIED: AGENTS.md]
- Preserve SOL ExecBench benchmark semantics and public schemas unless ROCm requires a documented change. [VERIFIED: AGENTS.md]
- NVIDIA/CUDA paths may be removed instead of maintained as a dual backend. [VERIFIED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Container build baseline | Docker / Container | Python packaging | Docker owns ROCm OS packages and compiler/runtime availability; `uv` owns Python wheels. [VERIFIED: docker/Dockerfile] |
| ROCm host device passthrough | Docker run script | OS / Driver | `scripts/run_docker.sh` owns runtime device flags; host ROCm kernel driver state is outside the repo. [VERIFIED: scripts/run_docker.sh] |
| Python ROCm package resolution | Python packaging | Docker | `pyproject.toml` and `uv.lock` own wheel indexes; Docker consumes the locked environment. [VERIFIED: pyproject.toml] |
| Dependency smoke tests | Pytest | Docker | `tests/docker/dependencies/` is the existing boundary for container capability checks. [VERIFIED: tests/docker/dependencies/] |
| Clock/profiling runtime behavior | Deferred runtime layer | Docker | Phase 1 should verify tools exist but defer runtime use and benchmark semantics to Phase 3. [VERIFIED: .planning/ROADMAP.md] |

## Standard Stack

### Core

| Library / Tool | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| `rocm/dev-ubuntu-24.04` | `7.1.1-complete` | ROCm development container base with ROCm toolchain and libraries. [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags] | Pinned AMD ROCm image avoids `nvidia/cuda` and keeps Dockerfile close to current Ubuntu 24.04 baseline. [VERIFIED: docker/Dockerfile] |
| `hipcc` | ROCm 7.1.1 toolchain | HIP compiler smoke-test target. [VERIFIED: hipcc --version] | `hipcc` is the standard ROCm compiler driver for HIP C++ compilation. [CITED: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html] |
| `torch` | `2.10.0+rocm7.1` | PyTorch ROCm runtime dependency. [CITED: https://pytorch.org/get-started/previous-versions/] | Official PyTorch install docs publish ROCm-specific wheel indexes. [CITED: https://pytorch.org/get-started/previous-versions/] |
| `torchvision` | `0.25.0+rocm7.1` | Companion PyTorch package currently declared by the project. [VERIFIED: pyproject.toml] | Version must match the selected PyTorch release family. [CITED: https://pytorch.org/get-started/previous-versions/] |
| `triton` | PyTorch-compatible ROCm build | Triton import and minimal ROCm kernel smoke target. [CITED: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html] | Triton is already part of the benchmark solution surface and should be validated in the container. [VERIFIED: tests/docker/dependencies/test_triton.py] |

### Supporting

| Library / Tool | Version | Purpose | When to Use |
|----------------|---------|---------|-------------|
| `rocminfo` | From ROCm base image | Validate ROCm runtime and visible HSA agents. [VERIFIED: rocminfo] | Use in Docker dependency tests before higher-level Python checks. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html] |
| `amd-smi` or `rocm-smi` | From ROCm base image | Validate AMD system-management tooling. [VERIFIED: rocm-smi --version] | Use presence/version smoke tests in Phase 1; clock behavior is Phase 3. [VERIFIED: .planning/ROADMAP.md] |
| `rocprofv3` | From ROCm base image | Validate profiler tooling availability. [CITED: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html] | Verify executable presence only in Phase 1; profiling integration is Phase 3. [VERIFIED: .planning/ROADMAP.md] |
| `rocBLAS`, `hipBLASLt`, `MIOpen` | From ROCm libraries | Selected library linkage/import smoke targets. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/package-manager-integration.html] | Use link or `ctypes.util.find_library` smoke checks; full library replacement decisions are Phase 4. [VERIFIED: .planning/ROADMAP.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rocm/dev-ubuntu-24.04:7.1.1-complete` | `rocm/dev-ubuntu-24.04:7.0.2-complete` | Closer to minimum ROCm 7.0 requirement, but stable PyTorch ROCm 7.0 wheel guidance is less clear than stable ROCm 7.1 commands. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/install/3rd-party/pytorch-install.html] |
| `rocm/dev-ubuntu-24.04:*` plus `uv sync` | `rocm/pytorch:*` framework image | Framework image may simplify PyTorch, but it risks fighting `uv sync` and project lockfile ownership. [ASSUMED] |
| Python standalone `triton` pin | Let PyTorch ROCm resolve compatible Triton | Standalone PyPI `triton` exists, but the planner should avoid forcing a mismatched backend unless the lock resolver proves it is ROCm-compatible. [VERIFIED: pip index versions triton] [ASSUMED] |

**Installation:**
```bash
uv lock
./scripts/run_docker.sh --build -- pytest tests/docker/dependencies
```

**Version verification performed:** `pip index versions torch` returned latest `2.12.0`; `pip index versions torchvision` returned latest `0.27.0`; `pip index versions triton` returned latest `3.7.0`. [VERIFIED: pip index versions] These registry latest versions do not prove ROCm wheel compatibility; use official ROCm wheel-index docs for ROCm pins. [CITED: https://pytorch.org/get-started/previous-versions/]

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `torch` | PyPI / PyTorch ROCm wheel index | Established project. [CITED: https://pypi.org/project/torch/] | Not captured by local tools. [ASSUMED] | `github.com/pytorch/pytorch` via official project metadata. [CITED: https://pypi.org/project/torch/] | OK. [VERIFIED: slopcheck install torch torchvision triton] | Approved when installed from the official ROCm index. [CITED: https://pytorch.org/get-started/previous-versions/] |
| `torchvision` | PyPI / PyTorch ROCm wheel index | Established project. [CITED: https://pypi.org/project/torchvision/] | Not captured by local tools. [ASSUMED] | `github.com/pytorch/vision` via official project metadata. [CITED: https://pypi.org/project/torchvision/] | OK. [VERIFIED: slopcheck install torch torchvision triton] | Approved when version-matched to `torch`. [CITED: https://pytorch.org/get-started/previous-versions/] |
| `triton` | PyPI / PyTorch dependency resolution | Established project. [CITED: https://pypi.org/project/triton/] | Not captured by local tools. [ASSUMED] | `github.com/triton-lang/triton` via official project metadata. [CITED: https://pypi.org/project/triton/] | OK. [VERIFIED: slopcheck install torch torchvision triton] | Approved only if resolver selects a ROCm-compatible build. [ASSUMED] |

**Packages removed due to slopcheck [SLOP] verdict:** none. [VERIFIED: slopcheck install torch torchvision triton]  
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: slopcheck install torch torchvision triton]

Note: `slopcheck install ... --json` was unavailable in installed slopcheck 0.6.1, so the audit used plain `slopcheck install`. [VERIFIED: slopcheck install --help] Slopcheck attempted to install packages after checking and failed under PEP 668, but the pre-install package classifications were emitted before that failure. [VERIFIED: slopcheck install torch torchvision triton]

## Architecture Patterns

### System Architecture Diagram

```text
Host AMD GPU + ROCm kernel driver
        |
        v
scripts/run_docker.sh
  - passes /dev/kfd and /dev/dri
  - sets IPC, ulimits, project mount
        |
        v
docker/Dockerfile from pinned ROCm image
  - ROCm runtime/tools/libs
  - uv-managed Python environment
        |
        v
tests/docker/dependencies/
  |--> command checks: rocminfo, hipcc, rocprofv3, amd-smi/rocm-smi
  |--> Python checks: torch.version.hip, torch.cuda.is_available(), triton import
  |--> compile/link checks: minimal HIP, selected ROCm libraries
        |
        v
Phase 2+ consumers: native build, eval runtime, profiling, examples
```

### Recommended Project Structure

```text
docker/
├── Dockerfile              # ROCm base image and uv environment
├── entrypoint.sh           # Startup checks; avoid CUDA clock-lock hard failure in Phase 1
scripts/
└── run_docker.sh           # AMD device passthrough and build/run entry point
tests/docker/dependencies/
├── test_rocm_runtime.py    # rocminfo, amd-smi/rocm-smi, rocprofv3
├── test_hip.py             # hipcc compile/link smoke test
├── test_pytorch_rocm.py    # torch ROCm import/runtime smoke test
├── test_triton_rocm.py     # triton import and optional minimal kernel
└── test_rocm_libraries.py  # selected rocBLAS/hipBLASLt/MIOpen discovery/link checks
```

### Pattern 1: ROCm Docker Base With `uv` Ownership

**What:** Keep `uv` as the project dependency installer while replacing only the image base, ROCm env vars, and CUDA-specific dependency declarations. [VERIFIED: docker/Dockerfile]  
**When to use:** Use for Phase 1 because dependency ownership already lives in `pyproject.toml` and `uv.lock`. [VERIFIED: pyproject.toml]

**Example:**
```dockerfile
# Source: AMD ROCm Docker tags + existing Dockerfile pattern
FROM rocm/dev-ubuntu-24.04:7.1.1-complete AS base

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}
```

### Pattern 2: AMD Docker Runtime Flags

**What:** Replace NVIDIA Container Toolkit `--gpus all` with explicit AMD device access. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html]  
**When to use:** Use in `scripts/run_docker.sh` so existing `./scripts/run_docker.sh --build` remains the user-facing entry point. [VERIFIED: scripts/run_docker.sh]

**Example:**
```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  sol-execbench:latest
```

### Pattern 3: Dependency Tests Classify Missing Tooling

**What:** Tests should fail with clear messages for missing ROCm tools and distinguish unavailable hardware from broken installs. [ASSUMED]  
**When to use:** Use in `tests/docker/dependencies/` because Phase 1 success requires missing ROCm tooling to be distinguishable from test failures. [VERIFIED: .planning/ROADMAP.md]

**Example:**
```python
# Source: pytest + subprocess pattern already used in tests/docker/dependencies/test_cuda.py
import shutil
import subprocess


def require_tool(name: str) -> str:
    path = shutil.which(name)
    assert path, f"Missing ROCm tool: {name}"
    return path


def test_rocminfo_reports_gpu_agent():
    rocminfo = require_tool("rocminfo")
    result = subprocess.run([rocminfo], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Agent" in result.stdout and "gfx" in result.stdout, result.stdout
```

### Anti-Patterns to Avoid

- **Leaving `torch>=2.10.0` on PyPI default indexes:** Default PyPI resolution can select CPU/CUDA-incompatible wheels instead of ROCm-specific wheels. [CITED: https://pytorch.org/get-started/previous-versions/]
- **Keeping CUDA package names as optional dependencies:** `cuda-tile`, `nvidia-cudnn-frontend`, `nvidia-cutlass-dsl[cu13]`, and `cupti-python` pull CUDA/NVIDIA assumptions into a ROCm-only environment. [VERIFIED: pyproject.toml]
- **Replacing all source-level `torch.cuda` references in Phase 1:** PyTorch ROCm intentionally still exposes GPU APIs through the `torch.cuda` namespace, so broad string replacement would be unsafe. [CITED: https://docs.pytorch.org/docs/2.12/notes/hip.html]
- **Testing only `import torch`:** Import success does not prove ROCm backend availability; assert `torch.version.hip` and a small GPU tensor operation. [CITED: https://docs.pytorch.org/docs/2.12/notes/hip.html]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ROCm runtime packaging | Custom ROCm apt install from scratch | AMD-published ROCm Docker image | Official images encode compatible OS/toolchain packaging. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html] |
| HIP compiler detection | Manual filesystem probing only | `hipcc --version` and compile smoke test | The compiler can exist but fail to compile/link; smoke tests catch both. [VERIFIED: hipcc --version] |
| PyTorch ROCm resolution | Ad hoc wheel URLs | Official PyTorch ROCm wheel index | Wheel tags and dependency compatibility change by release. [CITED: https://pytorch.org/get-started/previous-versions/] |
| ROCm GPU visibility | Custom `/dev` scanning | `rocminfo` plus PyTorch ROCm runtime checks | `rocminfo` validates HSA runtime visibility; PyTorch validates Python stack visibility. [VERIFIED: rocminfo] |
| Selected library availability | Hard-coded path checks only | `ldconfig`, `ctypes.util.find_library`, or compile/link smoke tests | ROCm library paths can differ by image and linker configuration. [ASSUMED] |

**Key insight:** Phase 1 should prove the ROCm environment exists and resolves correctly, not port benchmark runtime behavior. [VERIFIED: .planning/ROADMAP.md]

## Common Pitfalls

### Pitfall 1: ROCm Version And PyTorch Wheel Mismatch

**What goes wrong:** Container builds but PyTorch reports no HIP backend or fails at runtime. [ASSUMED]  
**Why it happens:** PyTorch ROCm wheels are tied to ROCm-specific wheel indexes and release families. [CITED: https://pytorch.org/get-started/previous-versions/]  
**How to avoid:** Pin ROCm image and PyTorch wheel index together, then assert `torch.version.hip` in Docker tests. [CITED: https://docs.pytorch.org/docs/2.12/notes/hip.html]  
**Warning signs:** `torch.__version__` lacks `+rocm`, `torch.version.hip` is `None`, or `uv.lock` contains `cu130`/`nvidia-*` packages. [VERIFIED: uv run python torch check] [VERIFIED: uv.lock]

### Pitfall 2: Treating `torch.cuda` Names As NVIDIA-Only

**What goes wrong:** Planner over-scopes Phase 1 into source API migration. [ASSUMED]  
**Why it happens:** PyTorch uses the `torch.cuda` namespace for HIP-enabled builds. [CITED: https://docs.pytorch.org/docs/2.12/notes/hip.html]  
**How to avoid:** Leave source runtime migration to later phases and only update environment tests in Phase 1. [VERIFIED: .planning/ROADMAP.md]  
**Warning signs:** Plans propose sweeping `torch.cuda` renames in Phase 1. [ASSUMED]

### Pitfall 3: Docker Device Flags Stay NVIDIA-Specific

**What goes wrong:** ROCm image builds but cannot see AMD GPUs at runtime. [ASSUMED]  
**Why it happens:** Current script uses `--gpus all`, which targets NVIDIA container runtime behavior. [VERIFIED: scripts/run_docker.sh]  
**How to avoid:** Pass `/dev/kfd` and `/dev/dri`, add the video group, and keep existing IPC/ulimit settings. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html]  
**Warning signs:** `rocminfo` sees only CPU agents or PyTorch reports no GPU. [VERIFIED: rocminfo]

### Pitfall 4: Replacing CUDA Library Tests With Too Much Library Migration

**What goes wrong:** Phase 1 turns into CUTLASS/cuDNN example migration. [ASSUMED]  
**Why it happens:** Existing Docker tests are library-specific CUDA examples. [VERIFIED: tests/docker/dependencies/]  
**How to avoid:** Replace them with availability/linkage tests for ROCm libraries and defer feature-level examples to Phase 4. [VERIFIED: .planning/ROADMAP.md]

## Code Examples

### PyTorch ROCm Smoke Test

```python
# Source: PyTorch HIP semantics docs
import torch


def test_torch_rocm_backend_available():
    assert torch.version.hip is not None, "PyTorch is not a ROCm build"
    assert torch.cuda.is_available(), "No ROCm GPU visible through PyTorch"
    x = torch.ones((4,), device="cuda")
    torch.testing.assert_close(x + 1, torch.full((4,), 2, device="cuda"))
```

### HIP Compile Smoke Test

```python
# Source: HIP compiler smoke pattern adapted from current CUDA compile test
import os
import subprocess
import tempfile

HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void add_one(float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) x[i] += 1.0f;
}

int main() {
  int runtime_version = 0;
  hipRuntimeGetVersion(&runtime_version);
  printf("HIP runtime version: %d\n", runtime_version);
  printf("PASS\n");
  return 0;
}
'''


def test_hipcc_compile_and_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.hip")
        exe = os.path.join(tmpdir, "test")
        with open(src, "w") as f:
            f.write(HIP_SRC)
        result = subprocess.run(["hipcc", src, "-o", exe], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        assert "PASS" in result.stdout
```

### `pyproject.toml` ROCm Index Shape

```toml
# Source: PyTorch previous versions ROCm wheel-index pattern
[[tool.uv.index]]
name = "pytorch-rocm71"
url = "https://download.pytorch.org/whl/rocm7.1"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-rocm71", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-rocm71", marker = "sys_platform == 'linux'" }]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| CUDA 13.1.1 base image | Pinned ROCm Ubuntu 24.04 dev image | Phase 1 migration | Removes dependency on `nvidia/cuda`. [VERIFIED: docker/Dockerfile] |
| `--gpus all` Docker runtime | `/dev/kfd` and `/dev/dri` device passthrough | ROCm Docker runtime | Enables AMD GPU visibility in containers. [CITED: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html] |
| PyTorch CUDA wheel index `cu130` | PyTorch ROCm wheel index `rocm7.1` | Phase 1 migration | Prevents CUDA wheel dependencies. [VERIFIED: pyproject.toml] [CITED: https://pytorch.org/get-started/previous-versions/] |
| CUDA dependency tests | ROCm runtime/compiler/Python/library tests | Phase 1 migration | Verifies environment baseline without porting examples. [VERIFIED: tests/docker/dependencies/] |

**Deprecated/outdated for this ROCm-only port:**
- `nvidia/cuda:*` Docker base: replaced by AMD ROCm base image. [VERIFIED: docker/Dockerfile] [CITED: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags]
- `pytorch-cu130` uv index: replaced by a ROCm wheel index. [VERIFIED: pyproject.toml] [CITED: https://pytorch.org/get-started/previous-versions/]
- `nvidia-cudnn-frontend`, `nvidia-cutlass-dsl[cu13]`, `cuda-tile`, `cupti-python`: remove from Phase 1 dependencies because they are NVIDIA/CUDA-specific. [VERIFIED: pyproject.toml]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Use `rocm/dev-ubuntu-24.04:7.1.1-complete` as the primary baseline instead of the minimum 7.0.x tag. | Summary / Standard Stack | Planner may choose a baseline newer than a strict minimum-version test matrix expects. |
| A2 | Let PyTorch ROCm resolution choose a compatible Triton build unless lockfile proves a standalone pin is needed. | Standard Stack | Triton smoke tests may fail if the resolver selects a CPU/CUDA-oriented wheel. |
| A3 | `ldconfig`, `ctypes.util.find_library`, or compile/link checks are sufficient for selected ROCm library availability in Phase 1. | Don't Hand-Roll | Library presence may pass while later functional examples still fail. |
| A4 | Framework image `rocm/pytorch:*` is less desirable because it may conflict with `uv` dependency ownership. | Alternatives Considered | Planner may do extra dependency work compared with starting from a framework image. |

## Open Questions

1. **Should the project freeze exactly at ROCm 7.0.x or allow a newer exact ROCm 7.x patch?**
   - What we know: The project requires ROCm >= 7.0. [VERIFIED: AGENTS.md]
   - What's unclear: Whether validation must prove the lowest supported version specifically. [ASSUMED]
   - Recommendation: Plan against exact ROCm 7.1.1 unless the user explicitly requires lowest-version validation. [ASSUMED]

2. **Which selected ROCm libraries should Phase 1 test?**
   - What we know: Requirements mention selected ROCm libraries; later phases mention rocBLAS, hipBLASLt, MIOpen, Composable Kernel, rocWMMA, hipCUB, rocPRIM, and rocThrust. [VERIFIED: .planning/REQUIREMENTS.md]
   - What's unclear: Whether Phase 1 must install every future candidate. [ASSUMED]
   - Recommendation: In Phase 1, test `rocBLAS`, `hipBLASLt`, and `MIOpen` discovery/linkage as representative core libraries; defer broader library replacement to Phase 4. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Docker | Build/run container | yes | 29.4.3 linux x86_64. [VERIFIED: docker --version] | none |
| Host ROCm HIP compiler | Local audit reference | yes | HIP 7.1.52802 / ROCm 7.1.1. [VERIFIED: hipcc --version] | Container toolchain |
| Host ROCm runtime | Local audit reference | yes | HSA runtime 1.18 with gfx1200 GPU visible. [VERIFIED: rocminfo] | Container runtime after Docker build |
| Host ROCm SMI | Local audit reference | yes | ROCM-SMI 4.0.0, ROCM-SMI-LIB 7.8.0. [VERIFIED: rocm-smi --version] | `amd-smi` if image provides it |
| Current local Python env | Baseline comparison | CUDA build | `torch 2.10.0+cu130`, `torch.version.hip is None`, `triton 3.6.0`. [VERIFIED: uv run python torch check] | Replace in Phase 1 |
| Context7 docs tool | Documentation lookup | no | `ctx7` not found. [VERIFIED: which ctx7] | Official docs via web |
| Knowledge graph | Codebase graph context | no | `.planning/graphs/graph.json` absent. [VERIFIED: find .planning/graphs] | Codebase docs and grep |

**Missing dependencies with no fallback:** none for research. [VERIFIED: docker/ROCm probes above]  
**Missing dependencies with fallback:** Context7 is missing; official docs were used directly. [VERIFIED: which ctx7]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest with pytest-xdist. [VERIFIED: pyproject.toml] |
| Config file | `pyproject.toml`. [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/docker/dependencies -n 0` [ASSUMED] |
| Full suite command | `uv run pytest tests/` [VERIFIED: AGENTS.md] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| ENV-01 | Docker builds from ROCm base and not `nvidia/cuda`. | smoke/static | `./scripts/run_docker.sh --build -- true` plus Dockerfile grep. [ASSUMED] | Existing Dockerfile yes; test gap yes. [VERIFIED: docker/Dockerfile] |
| ENV-02 | ROCm tools and selected libraries available. | integration | `uv run pytest tests/docker/dependencies/test_rocm_runtime.py -n 0` [ASSUMED] | no; Wave 0 gap. [VERIFIED: tests/docker/dependencies/] |
| ENV-03 | PyTorch ROCm and Triton ROCm install without CUDA wheels. | integration/static | `uv run pytest tests/docker/dependencies/test_pytorch_rocm.py tests/docker/dependencies/test_triton_rocm.py -n 0` [ASSUMED] | partial Triton import exists; PyTorch ROCm test missing. [VERIFIED: tests/docker/dependencies/test_triton.py] |
| ENV-04 | Docker dependency tests cover ROCm/HIP/PyTorch/Triton/libraries with clear failure messages. | integration | `uv run pytest tests/docker/dependencies -n 0` [ASSUMED] | existing tests are CUDA-oriented. [VERIFIED: tests/docker/dependencies/] |
| SCFG-03 | NVIDIA-only Python dependencies and CUDA wheel index removed/replaced. | static/unit | `uv run python -c 'import tomllib; ...'` or grep-based pytest. [ASSUMED] | no; Wave 0 gap. [VERIFIED: pyproject.toml] |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/docker/dependencies -n 0` for Docker test edits; `uv lock --check` after dependency edits. [ASSUMED]
- **Per wave merge:** `./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0`. [ASSUMED]
- **Phase gate:** Docker build succeeds and in-container dependency tests pass in ROCm environment. [VERIFIED: .planning/ROADMAP.md]

### Wave 0 Gaps

- [ ] `tests/docker/dependencies/test_rocm_runtime.py` — covers ENV-02 and ENV-04. [ASSUMED]
- [ ] `tests/docker/dependencies/test_hip.py` — covers ENV-02 and ENV-04. [ASSUMED]
- [ ] `tests/docker/dependencies/test_pytorch_rocm.py` — covers ENV-03 and ENV-04. [ASSUMED]
- [ ] `tests/docker/dependencies/test_rocm_libraries.py` — covers ENV-02 and ENV-04. [ASSUMED]
- [ ] Static dependency assertion for absence of CUDA/NVIDIA package declarations — covers SCFG-03. [ASSUMED]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in this phase. [VERIFIED: .planning/codebase/INTEGRATIONS.md] |
| V3 Session Management | no | No session surface in this phase. [VERIFIED: .planning/codebase/INTEGRATIONS.md] |
| V4 Access Control | yes | Docker device and privileged flags should remain limited to the GPU evaluation container. [VERIFIED: scripts/run_docker.sh] |
| V5 Input Validation | yes | Dependency tests should use fixed commands and avoid shell interpolation. [VERIFIED: tests/docker/dependencies/test_cuda.py] |
| V6 Cryptography | no | No cryptographic behavior in this phase. [VERIFIED: .planning/codebase/INTEGRATIONS.md] |

### Known Threat Patterns for Docker/GPU Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Over-broad privileged container use | Elevation of privilege | Preserve existing use only where required for GPU tooling, document it, and avoid mounting secrets or datasets beyond project paths. [VERIFIED: scripts/run_docker.sh] [ASSUMED] |
| Dependency confusion / wrong GPU wheel index | Tampering | Use explicit PyTorch ROCm index and remove CUDA/NVIDIA package declarations. [CITED: https://pytorch.org/get-started/previous-versions/] |
| Accidental credential or dataset commit | Information disclosure | Follow AGENTS.md prohibition on credentials, tokens, proprietary kernels, and downloaded datasets. [VERIFIED: AGENTS.md] |

## Sources

### Primary (HIGH confidence)

- `AGENTS.md` - project layout, commands, testing, ROCm constraints. [VERIFIED: AGENTS.md]
- `.planning/phases/01-rocm-environment-baseline/01-CONTEXT.md` - phase boundary and deferred scope. [VERIFIED: .planning/phases/01-rocm-environment-baseline/01-CONTEXT.md]
- `.planning/REQUIREMENTS.md` and `.planning/ROADMAP.md` - requirement mapping and success criteria. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: .planning/ROADMAP.md]
- `docker/Dockerfile`, `scripts/run_docker.sh`, `pyproject.toml`, `tests/docker/dependencies/` - current implementation. [VERIFIED: repository grep]
- AMD ROCm Docker docs: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
- AMD ROCm Docker image support matrix: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html
- PyTorch previous versions ROCm install commands: https://pytorch.org/get-started/previous-versions/
- PyTorch HIP semantics: https://docs.pytorch.org/docs/2.12/notes/hip.html

### Secondary (MEDIUM confidence)

- Docker Hub ROCm dev image tags: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags
- AMD ROCm 7.0.2 PyTorch install page: https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/install/3rd-party/pytorch-install.html
- ROCprofiler SDK `rocprofv3` docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- PyPI project metadata for `torch`, `torchvision`, and `triton`: https://pypi.org/project/torch/ , https://pypi.org/project/torchvision/ , https://pypi.org/project/triton/

### Tertiary (LOW confidence)

- Local assumptions about exact selected library test set and Triton resolver behavior; both are flagged in the Assumptions Log. [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - ROCm image and PyTorch ROCm index are officially documented, but exact ROCm/PyTorch/Triton compatibility changes quickly. [CITED: https://pytorch.org/get-started/previous-versions/]
- Architecture: HIGH - Current file boundaries are clear and Phase 1 scope is narrow. [VERIFIED: docker/Dockerfile] [VERIFIED: pyproject.toml] [VERIFIED: tests/docker/dependencies/]
- Pitfalls: HIGH - Existing repo is heavily CUDA-oriented and official docs confirm key ROCm differences. [VERIFIED: repository grep] [CITED: https://docs.pytorch.org/docs/2.12/notes/hip.html]

**Research date:** 2026-05-21  
**Valid until:** 2026-06-20 for Docker/test architecture; 2026-05-28 for PyTorch/Triton ROCm package pins.
