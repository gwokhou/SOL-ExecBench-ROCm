---
status: partial
phase: 01-rocm-environment-baseline
source:
  - .planning/phases/01-rocm-environment-baseline/01-01-SUMMARY.md
  - .planning/phases/01-rocm-environment-baseline/01-02-SUMMARY.md
  - .planning/phases/01-rocm-environment-baseline/01-03-SUMMARY.md
  - .planning/phases/01-rocm-environment-baseline/01-04-SUMMARY.md
  - .planning/phases/01-rocm-environment-baseline/01-VERIFICATION.md
started: 2026-05-21T07:13:44Z
updated: 2026-05-21T08:12:43Z
---

## Current Test

[testing paused - 4 items outstanding]

## Tests

### 1. Docker ROCm Image Build
expected: Running `./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0` starts from the ROCm base image `rocm/dev-ubuntu-24.04:7.1.1-complete` and completes the Docker image build without CUDA/NVIDIA base image dependencies.
result: blocked
blocked_by: docker-storage
reason: "`./scripts/run_docker.sh --build -- pytest tests/docker/dependencies -n 0` connected to Docker after approval. The first attempt failed with a BuildKit parent snapshot error; after `docker builder prune -f` reclaimed 13.28GB, the retry passed the ROCm base/tooling layer (`hipcc --version`, `rocminfo`, `rocprofv3`, and `amd-smi` were found). The build then failed at Dockerfile line 75 during `uv sync --frozen --no-install-project --all-groups`: uv could not extract `torch==2.10.0+rocm7.1` because creating a hipBLASLt Tensile library file failed with `No space left on device (os error 28)`. Container runtime and pytest validation cannot run until Docker storage is expanded or more space is reclaimed."
latest_retry: "After Docker Root Dir was moved to `/home/docker-data`, `docker builder prune -f` reclaimed 28.05GB and the full verification command was rerun. Docker confirmed `Docker Root Dir: /home/docker-data`; `/home` had 530GB available and only 5% inode usage. The build again passed the ROCm base/tooling layer, then failed at Dockerfile line 75 during `uv sync --frozen --no-install-project --all-groups`: uv could not extract `torch==2.10.0+rocm7.1` because creating `torch/lib/hipblaslt/library/TensileLibrary_F8NF8N_BF8N_HA_Bias_SABV_SAV_UA_Type_F8NB_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx942.co` failed with `No space left on device (os error 28)`. Since host `/home` space and inodes are sufficient, the remaining blocker is likely Docker Desktop/BuildKit internal storage accounting or the BuildKit cache mount used for `/home/guohao/.cache/uv`, not the project code."

### 2. AMD Device Visibility
expected: Inside the built container, ROCm sees AMD GPU device access through `/dev/kfd` and `/dev/dri`, and startup output reports ROCm/HIP availability instead of CUDA-only device messaging.
result: [pending]

### 3. Docker Dependency Test Suite
expected: All six tests under `tests/docker/dependencies/` pass in the ROCm container when run with `pytest tests/docker/dependencies -n 0`.
result: [pending]

### 4. Actionable ROCm Failure Output
expected: If a ROCm dependency check fails, pytest output identifies the missing ROCm tool, library, or device access problem by name, including relevant stdout or stderr.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 1

## Gaps

[none yet]
