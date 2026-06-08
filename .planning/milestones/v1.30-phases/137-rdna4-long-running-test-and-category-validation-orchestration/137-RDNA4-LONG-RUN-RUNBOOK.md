# Phase 137 RDNA4 Long-Running Validation Runbook

## Purpose

This runbook defines how to run RDNA4 adapted pytest and category validation
without losing evidence from jobs that may run for many hours. It supports
Phase 137 only. Full dataset denominator execution belongs to Phase 138, timing
authority belongs to Phase 139, and public claim closure belongs to Phase 141.

## Preflight

Run these commands before starting long-running validation:

```bash
rocminfo
rocm-smi
lspci
hipcc --version
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from pathlib import Path; import torch; print('/dev/kfd', Path('/dev/kfd').exists()); print('/dev/dri', Path('/dev/dri').exists()); print('hip', torch.version.hip); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

Expected RDNA4 target evidence:

- `rocminfo` reports a GPU agent named `gfx1200`.
- PyTorch ROCm reports HIP support and a visible GPU in the same environment
  that will run pytest.
- `/dev/kfd` and `/dev/dri` are visible inside the same execution environment.

If host tools see `gfx1200` but `uv run` cannot see `/dev/kfd` or `/dev/dri`,
classify the run as an execution-environment device-passthrough boundary. Do
not report it as RDNA4 test failure or RDNA4 pass evidence.

## Adapted Pytest Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m requires_rdna4 -q -rs \
  --junitxml .planning/phases/137-rdna4-long-running-test-and-category-validation-orchestration/137-rdna4-pytest.xml
```

Polling policy:

- Poll every 30 minutes for healthy long-running jobs.
- Preserve process id, command line, start time, latest artifact path, and the
  last observed output summary in a checkpoint note.
- Do not terminate a healthy process solely due to elapsed time.
- If the process exits, classify every failure/skip before rerunning.

## Focused Category Commands

CPU-safe metadata and documentation guardrails:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rocm_library_examples.py \
  tests/sol_execbench/test_rocm_library_readiness_docs.py \
  tests/sol_execbench/test_rocm_diagnostics_reporting.py -q \
  --junitxml .planning/phases/137-rdna4-long-running-test-and-category-validation-orchestration/137-category-guardrails.xml
```

Example/category execution surface:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_examples.py \
  -k "hipblas or miopen or ck or rocwmma or triton or hip_cpp" -q -rs \
  --junitxml .planning/phases/137-rdna4-long-running-test-and-category-validation-orchestration/137-category-examples.xml
```

## Classification

Use these labels in Phase 137 artifacts:

| Label | Meaning |
| --- | --- |
| `passed` | Command completed and all selected checks passed. |
| `expected_skip` | Check was intentionally skipped by marker or known boundary. |
| `dependency_boundary` | Required library/header/tool is unavailable. |
| `execution_environment_boundary` | Host hardware exists but the command environment cannot access it. |
| `compile_failure` | Native or Triton build failed. |
| `runtime_failure` | Built code ran but failed at runtime. |
| `correctness_failure` | Execution completed with incorrect outputs. |
| `timeout` | Command or shard exceeded the configured timeout. |
| `known_gap` | Accepted gap with an evidence reference and next action. |

## Resume Rules

- Reuse JUnit XML and logs only when command arguments, repository commit,
  hardware target, and environment preflight match.
- Rerun failed category checks after fixing dependencies or code.
- Do not convert skipped `requires_rdna4` tests into pass evidence.
- Keep Phase 137 category evidence separate from Phase 138 dataset closure.

## Claim Boundary

Phase 137 evidence can show RDNA4 adapted-test or category readiness. It cannot
claim full dataset validation, benchmark-grade timing, upstream SOLAR parity,
leaderboard authority, CDNA3/MI300X validation, or CDNA4 validation.
