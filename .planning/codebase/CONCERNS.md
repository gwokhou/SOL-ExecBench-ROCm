# Codebase Concerns

**Analysis Date:** 2026-05-28

## Summary

This repository has strong schema, documentation, and CPU-safe guardrail
coverage, but its highest-risk areas are the boundaries where benchmark
semantics meet mutable external systems: ROCm hardware, Docker images, PyTorch
ROCm wheels, native extension compilation, optional profilers, downloaded
datasets, and user-supplied solution code.

Most concerns are managed through explicit claim boundaries, sidecar-only
evidence, marker-gated tests, and dry-run script coverage. The remaining risk
is concentrated in large orchestration modules and live-environment validation
that cannot be fully exercised by default CI.

## Technical Debt

### Large Orchestration Modules

Several modules carry broad orchestration responsibility and should be treated
carefully during changes:

| Area | File | Approx. Size | Concern |
| --- | --- | ---: | --- |
| Dataset batch execution | `scripts/run_dataset.py` | 1708 lines | Combines CLI fan-out, trace interpretation, derived report generation, skip handling, and artifact closure. Small behavior changes can affect many evidence surfaces. |
| Main benchmark CLI | `src/sol_execbench/cli/main.py` | 908 lines | Owns CLI parsing, subprocess staging, trace output, profile/static/environment sidecars, and user-facing errors. |
| Generated eval driver | `src/sol_execbench/driver/templates/eval_driver.py` | 695 lines | Runs reference/user code, correctness, timing, reward-hack checks, and JSON trace emission in a subprocess template. |
| Compatibility contract | `src/sol_execbench/core/compatibility.py` | 594 lines | Central strict model for ROCm compatibility evidence and claim boundaries. |
| Docker wrapper | `scripts/run_docker.sh` | 491 lines | Shell orchestration for target selection, host preflight, dependency preflight, image build/run, and evidence sidecars. |
| Runtime evidence | `src/sol_execbench/core/runtime_evidence.py` | 502 lines | Collects diagnostic evidence across host/container/Python/toolchain/GPU scopes. |

Risk pattern: these files mix IO, policy, serialization, and environment
handling. Prefer small, targeted helpers and focused tests when changing them.

### Historical ROCm Port Residue

The repo intentionally keeps legacy CUDA/NVIDIA examples and schema fixtures in
places such as `examples/cutlass/`, `examples/cudnn/`, `examples/cute_dsl/`,
`examples/cutile/`, and sample solution files like
`tests/sol_execbench/samples/rmsnorm/solution_cuda.json`. Guardrail tests such
as `tests/sol_execbench/test_rocm_migration_residue_audit.py`,
`tests/sol_execbench/test_rocm_schema_build_audit.py`, and
`tests/sol_execbench/test_rocm_library_examples.py` distinguish allowed
historical references from active supported paths.

Risk pattern: search hits for CUDA/NVIDIA terms are not automatically bugs, but
new active-runtime references should be audited against the existing allowlist
tests and docs wording.

### Sidecar Proliferation

The project has multiple optional evidence sidecars: environment snapshots,
rocprofv3 profiles, static kernel evidence, AMD score reports, SOL/SOLAR
derivation artifacts, Docker matrix entries, dependency preflight output, and
runtime compatibility matrices. These are intentionally kept out of canonical
trace JSONL.

Risk pattern: new evidence features must not mutate `Trace` semantics in
`src/sol_execbench/core/data/trace.py`, correctness behavior in
`src/sol_execbench/core/bench/correctness.py`, timing behavior in
`src/sol_execbench/core/bench/timing.py`, or score authority rules in
`src/sol_execbench/core/scoring_guardrails.py`.

## Testing Gaps And Environment Risk

### Live ROCm Coverage Is Marker-Gated

Default CI and many local runs are CPU-safe. Hardware-sensitive checks are
behind pytest markers in `tests/conftest.py` and documented in
`docs/TESTING.md`:

- `requires_rocm`
- `requires_rocm_dev`
- `requires_ck`
- `requires_rocwmma`
- `requires_rdna4`
- `requires_cdna3`
- `timing_serial`

Risk pattern: logic can pass CPU-safe tests while real ROCm execution,
extension compilation, profiling, or device timing still fails on a live host.
Changes touching `src/sol_execbench/driver/templates/build_ext.py`,
`src/sol_execbench/driver/templates/eval_driver.py`,
`src/sol_execbench/core/bench/timing.py`, `tests/docker/dependencies/`, or
native library examples should include a ROCm-capable validation note.

### Docker Matrix Is Diagnostic, Not Native Host Validation

The v1.18 matrix work explicitly documents that Docker Matrix Entries validate
container ROCm user-space on recorded host driver/devices. They do not prove
native host ROCm validation. This boundary is enforced in `docs/CLAIMS.md`,
`docs/TESTING.md`, and tests such as
`tests/sol_execbench/test_rocm_matrix_docs.py` and
`tests/sol_execbench/test_matrix_claim_guardrails.py`.

Risk pattern: wording regressions can overstate evidence. Treat docs changes
around "validated", "host", "native", "paper parity", "score authority", or
"leaderboard" as behavior-affecting changes.

### Dependency Availability Changes Externally

The default dependency stack pins PyTorch ROCm 7.1 wheels in `pyproject.toml`
and `uv.lock`, while Docker target policies live in `docker/rocm-targets.json`
and preflight classification code in
`src/sol_execbench/core/dependency_matrix.py`.

Risk pattern: PyTorch ROCm wheel indexes, Triton ROCm packages, Docker images,
and ROCm libraries are external moving parts. Matrix policy and dependency
guardrails should be updated together when any target stack changes.

## Security And Isolation Concerns

### User-Supplied Code Execution

Benchmark solutions can execute Python or native HIP/C++ code. Isolation is
process and staging-directory based through
`src/sol_execbench/driver/problem_packager.py` and
`src/sol_execbench/driver/templates/eval_driver.py`; it is not a hard security
sandbox. `SECURITY.md` correctly frames reports around evaluation isolation,
schema validation, build staging, and unsafe usage.

Risk pattern: do not treat subprocess staging as untrusted-code containment.
Changes that expand file access, environment propagation, subprocess command
construction, or artifact copying should be reviewed as security-sensitive.

### Shell Boundary In `scripts/run_docker.sh`

The Docker wrapper coordinates user arguments, environment variables, Docker
commands, target selection, preflight checks, and evidence output. It has
CPU-safe tests in `tests/sol_execbench/test_run_docker_matrix_script.py`,
`tests/sol_execbench/test_run_docker_dependency_preflight.py`, and
`tests/sol_execbench/test_run_docker_runtime_evidence.py`.

Risk pattern: shell quoting, path handling, and environment override behavior
are fragile. Run `bash -n scripts/run_docker.sh` and the wrapper tests for any
change to this script.

### Dataset And Token Handling

Dataset acquisition and inspection code lives in
`scripts/download_solexecbench.py`, `scripts/download_data.sh`,
`scripts/inspect_dataset.py`, and `src/sol_execbench/core/dataset/`.
Repository guidance says not to commit Hugging Face tokens, proprietary
kernels, or downloaded benchmark assets.

Risk pattern: download, cache, manifest, and artifact-report changes should
avoid writing credentials or data payloads into `.planning/`, docs, traces, or
sidecars.

## Performance And Reliability Hotspots

### Timing And Clock Policy

Timing code uses HIP-backed PyTorch CUDA compatibility APIs in
`src/sol_execbench/core/bench/timing.py`, clock controls in
`src/sol_execbench/core/bench/clock_lock.py`, and CLI flags in
`src/sol_execbench/cli/main.py`.

Risk pattern: timing changes can silently affect benchmark comparability.
Prefer preserving existing trace fields and add focused tests under
`tests/sol_execbench/core/bench/test_timing.py` plus live `timing_serial`
validation when possible.

### Reward-Hack Detection

Reward-hack behavior is guarded by `src/sol_execbench/core/bench/reward_hack.py`
and E2E cases in `tests/sol_execbench/test_e2e.py` and
`tests/sol_execbench/core/bench/test_reward_hack.py`.

Risk pattern: changes to eval driver imports, tensor cloning, output checking,
or reference invocation can weaken cheat detection. Keep malicious sample tests
green when modifying evaluation flow.

### Native Extension Build Surface

HIP/C++ build behavior spans `src/sol_execbench/driver/problem_packager.py`,
`src/sol_execbench/driver/templates/build_ext.py`, solution schema validation
in `src/sol_execbench/core/data/solution.py`, and examples under
`examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`,
and `examples/rocwmma/`.

Risk pattern: compiler flags, architecture detection, staging paths, and
library availability are environment-sensitive. Pair schema/build changes with
driver tests and marker-gated ROCm checks.

## Documentation Risk

The documentation is part of the product contract. Files such as
`docs/CLAIMS.md`, `docs/TESTING.md`, `docs/RESEARCHER-GUIDE.md`,
`docs/rocm_toolchain_routing.md`, `docs/static_kernel_evidence.md`, and
`docs/analysis.md` define what evidence means and what users may claim.

Risk pattern: code can be correct while docs overclaim. Keep docs guardrail
tests close to any new public wording around evidence authority, ROCm matrix
validation, AMD-native scores, static evidence, or paper parity.

## Watch List

- Native host ROCm 7.0.x / 7.1.x / 7.2.x validation remains deferred unless
  direct host evidence is archived.
- CDNA 3 and CDNA 4 validation claims remain deferred without real hardware
  artifacts.
- Full 235-problem paper validation and upstream SOLAR equivalence remain out
  of scope for current evidence.
- The large CLI, dataset runner, and generated eval driver are the most likely
  places for accidental behavior coupling.
- Any new compatibility or profiling artifact should be sidecar-only unless a
  future milestone explicitly changes the public contract.
