---
phase: 79-docker-matrix-selection-and-preflight
reviewed: 2026-05-28T06:47:14Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - docker/rocm-targets.json
  - src/sol_execbench/core/docker_matrix.py
  - tests/sol_execbench/test_docker_matrix_targets.py
  - tests/sol_execbench/test_docker_matrix_preflight.py
  - docker/Dockerfile
  - scripts/run_docker.sh
  - tests/sol_execbench/test_run_docker_matrix_script.py
findings:
  critical: 1
  warning: 2
  info: 0
  total: 3
status: issues_found
---

# Phase 79: Code Review Report

**Reviewed:** 2026-05-28T06:47:14Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the Docker target manifest, matrix/preflight Python helper, Dockerfile, run script, and related tests. The main defect is a claim-boundary violation: the preflight JSON explicitly says a not-tested target is not benchmark eligible, but the shell launcher proceeds to run benchmark commands anyway. I also found preflight robustness gaps around device permission checks and invalid boolean inputs.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Launcher ignores `benchmark_allowed=false` and runs not-tested targets

**Classification:** BLOCKER
**File:** `scripts/run_docker.sh:191`
**Issue:** `classify_docker_preflight_json` returns Matrix decision fields including `benchmark_allowed`, and `not_tested` decisions are explicitly non-authoritative and not benchmark eligible. The launcher only checks `status == runtime_unavailable` before continuing, so a preflight result with `status: not_tested` and `benchmark_allowed: false` still reaches `docker run`. This allows commands such as `sol-execbench ... --solution ...` to run under a Matrix decision that says benchmark execution is not allowed, violating the claim boundary that the Python module and tests assert. I confirmed this path with dry-run preflight overrides: the script emitted a `docker run ... sol-execbench ...` command while the corresponding preflight status would be `not_tested` with `benchmark_allowed: false`.
**Fix:**
```bash
PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${PREFLIGHT_JSON}" "benchmark_allowed")"
if [ "${PREFLIGHT_STATUS}" = "runtime_unavailable" ] || [ "${PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
    echo "${PREFLIGHT_JSON}"
    exit 1
fi
```
Use the JSON boolean text format actually emitted by `matrix_json_value` (`True`/`False` from Python today), or update `matrix_json_value` to emit normalized lowercase JSON literals and compare against `true`. Add a script test that sets preflight overrides to the available/not-tested case without `--preflight-only` and asserts no `docker run` line is emitted when `benchmark_allowed` is false.

## Warnings

### WR-01: `/dev/kfd` and `/dev/dri` checks can report runtime availability without usable device permissions

**Classification:** WARNING
**File:** `scripts/run_docker.sh:113`
**Issue:** The preflight marks `/dev/kfd` accessible with `-r /dev/kfd` and `/dev/dri` accessible with `-r /dev/dri`. ROCm device use requires write access to `/dev/kfd`, and `/dev/dri` being a readable directory does not prove any render/card device node inside it is usable by the current Docker invocation. This can produce a false "no runtime blockers" preflight result and then fail later inside Docker.
**Fix:**
```bash
dev_kfd_accessible="$(preflight_bool SOL_EXECBENCH_DEV_KFD_ACCESSIBLE "$(bool_text "$([ -r /dev/kfd ] && [ -w /dev/kfd ] && echo 1 || echo 0)")")"
dev_dri_accessible="$(preflight_bool SOL_EXECBENCH_DEV_DRI_ACCESSIBLE "$(bool_text "$([ -x /dev/dri ] && find /dev/dri -maxdepth 1 -type c \( -name 'renderD*' -o -name 'card*' \) -readable -writable -print -quit | grep -q . && echo 1 || echo 0)")")"
```
Add tests for the shell preflight override path or factor the permission probing into a small function that can be unit-tested with temporary device-like fixtures/mocks.

### WR-02: Invalid preflight boolean input escapes as a Python traceback

**Classification:** WARNING
**File:** `src/sol_execbench/core/docker_matrix.py:520`
**Issue:** `_parse_bool` raises `argparse.ArgumentTypeError`, but it is called after `parse_args()` has already completed. Invalid values from environment-driven script arguments, for example `SOL_EXECBENCH_GPU_ACCESSIBLE=maybe`, produce a Python traceback instead of a controlled CLI error. That makes the shell preflight less diagnosable and leaks implementation details into user-facing output.
**Fix:**
```python
try:
    gpu_accessible = (
        None if args.gpu_accessible is None else _parse_bool(args.gpu_accessible)
    )
except argparse.ArgumentTypeError as exc:
    _build_parser().error(str(exc))
```
Better, register boolean arguments with `type=_parse_bool` in `_build_parser()` so argparse owns validation for all preflight booleans. Add a module-main test for an invalid boolean value that asserts a non-zero exit and no traceback.

---

_Reviewed: 2026-05-28T06:47:14Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
