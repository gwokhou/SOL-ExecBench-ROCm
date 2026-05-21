---
phase: 01-rocm-environment-baseline
plan: 02
subsystem: infra
tags: [uv, pytorch, rocm, dependencies]
requires:
  - phase: 01-rocm-environment-baseline
    provides: ROCm Docker baseline consuming uv sync --frozen
provides:
  - ROCm PyTorch dependency declarations
  - Synchronized uv.lock without CUDA/NVIDIA dependency residue
  - Static dependency regression guard
affects: [docker, dependencies, phase-2, phase-3]
tech-stack:
  added: [torch-2.10.0-rocm7.1, torchvision-0.25.0-rocm7.1, triton-rocm-3.6.0]
  patterns: [explicit-uv-index-mapping, static-pyproject-guard]
key-files:
  created:
    - tests/docker/dependencies/test_python_dependencies.py
  modified:
    - pyproject.toml
    - uv.lock
key-decisions:
  - "Pin Linux/Windows GPU framework packages to the ROCm local versions that uv resolves."
  - "Keep non-Linux fallback torch/torchvision pins at upstream 2.10.0/0.25.0."
  - "Pin triton-rocm from the PyTorch root wheel index because PyTorch ROCm depends on triton-rocm==3.6.0."
patterns-established:
  - "Use tomllib-based static dependency tests for pyproject invariants."
requirements-completed: [ENV-03, SCFG-03]
duration: 45min
completed: 2026-05-21
---

# Phase 1 Plan 02: ROCm Dependency Baseline Summary

**PyTorch ROCm dependency declarations and lockfile resolved without CUDA/NVIDIA package residue**

## Performance

- **Duration:** 45 min
- **Started:** 2026-05-21T04:50:00Z
- **Completed:** 2026-05-21T05:36:48Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Removed CUDA/NVIDIA-only Python dependencies from `pyproject.toml`.
- Added explicit ROCm PyTorch and PyTorch root indexes for `torch`, `torchvision`, and `triton-rocm`.
- Regenerated `uv.lock`, removing CUDA package families and locking ROCm wheels.
- Added a static dependency test that guards the ROCm index/source mapping.

## Task Commits

1. **Task 1-3: ROCm dependency migration and lock refresh** - `aa4d075` (feat)

## Files Created/Modified

- `pyproject.toml` - ROCm framework pins, Python range `>=3.12,<3.14`, and uv source mappings.
- `uv.lock` - ROCm lock data for torch, torchvision, and triton-rocm.
- `tests/docker/dependencies/test_python_dependencies.py` - Static dependency invariant checks.

## Decisions Made

The PyTorch ROCm wheels use local versions such as `2.10.0+rocm7.1`, so the project pins platform-specific dependency entries instead of a single global `torch==2.10.0`. `requires-python` is constrained to `<3.14` because uv could not resolve the ROCm wheel set for Python 3.14.

## Deviations from Plan

The plan originally described `torch==2.10.0` and `torchvision==0.25.0` as the release family. The resolved wheel versions require `+rocm7.1` local versions on Linux/Windows, plus an explicit `triton-rocm==3.6.0` source mapping.

## Issues Encountered

`uv lock` initially failed until `triton-rocm` was mapped to `https://download.pytorch.org/whl/`. After that, `uv lock` resolved 70 packages and removed the old CUDA/NVIDIA lock entries.

## User Setup Required

None.

## Next Phase Readiness

Docker frozen installs now resolve ROCm framework packages. Later phases should avoid reintroducing CUDA wheel indexes or NVIDIA-only Python packages.

## Verification

- `uv lock --check` - PASS
- `python -m pytest -p no:xdist -o addopts='' tests/docker/dependencies/test_python_dependencies.py -q` - PASS
- `rg -n "pytorch-cu130|download\\.pytorch\\.org/whl/cu130|cuda-tile|cupti-python|nvidia-cudnn-frontend|nvidia-cutlass-dsl" uv.lock pyproject.toml` - PASS, no package-source residue found

## Self-Check: PASSED

---
*Phase: 01-rocm-environment-baseline*
*Completed: 2026-05-21*
