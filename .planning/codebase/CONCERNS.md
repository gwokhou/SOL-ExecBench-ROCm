---
last_mapped_commit: f4de6692ee7468e150c112e0cbdcc8842dd0c709
mapped_at: 2026-06-06
---

# Concerns

## Executive Summary

This repository is heavily test-covered and has explicit ROCm migration
guardrails, but its highest-risk areas remain the places where benchmark
semantics meet untrusted code, real GPU hardware, and derived evidence:

- Evaluation executes user-supplied Python/HIP code in subprocesses. The code
  has staging, schema, compile-flag, and reward-hack defenses, but this remains
  an isolation boundary rather than a security sandbox.
- Full ROCm validation is hardware-dependent. Many tests and the most important
  behavior are skipped unless ROCm devices, headers, and target architectures
  are present.
- Dataset execution and scoring are operationally complex. The largest scripts
  and modules are `scripts/run_dataset.py` at 3075 lines,
  `src/sol_execbench/core/scoring/solar_derivation.py` at 2570 lines, and
  `src/sol_execbench/cli/main.py` at 1116 lines.
- CDNA3/gfx942 validation has active known issues: timeouts without traces,
  static source-review blocks, Quant low-precision reference failures, low
  derived score coverage, and provisional hardware modeling.
- Dataset licensing and redistribution risks are real because source benchmark
  assets come from NVIDIA SOL-ExecBench and FlashInfer Trace. The repository has
  policy checks, but generated/local artifact boundaries need continued care.

## Security And Isolation Concerns

### User Code Execution Is Inherently Unsafe

The benchmark must import and execute submitted solution code. The generated
driver in `src/sol_execbench/driver/templates/eval_driver.py` imports the
reference implementation, snapshots critical evaluation functions, imports the
user function, and executes workloads in-process. The import helpers in
`src/sol_execbench/core/bench/eval_runtime.py` use
`importlib.util.spec_from_file_location(...).loader.exec_module(...)` for both
reference code and user code.

Controls exist:

- Python user solutions are staged under a temporary directory by
  `src/sol_execbench/driver/problem_packager.py`.
- Native ROCm solutions are compiled first and loaded as `benchmark_kernel.so`.
- `src/sol_execbench/core/bench/eval_runtime.py` blocks
  `torch.utils.cpp_extension.load()` and `load_inline()` for Python solutions.
- `src/sol_execbench/core/bench/reward_hack.py` and the generated driver check
  for monkey-patching, thread injection, lazy outputs, semantic output caching,
  and other reward-hack patterns.
- `SECURITY.md` explicitly frames evaluation isolation and staging as security
  reporting concerns.

Residual risk:

- This is not a hard sandbox. User Python code can still perform arbitrary
  process-local actions available to the evaluation subprocess, including file
  I/O, network if the host permits it, expensive allocations, and interaction
  with imported modules before runtime checks catch some mutations.
- Native HIP/C++ code runs inside the Python extension process and can crash or
  corrupt the process. The timeout catches hangs at the subprocess boundary, but
  not all local resource impacts.
- Reward-hack detection is heuristic and policy-based. The known CDNA3 notes in
  `.planning/quick/260604-cdna3-gfx942-validation-attempt-record/KNOWN_ISSUES.md`
  already show static source-review blocks for migrated reference code, which
  means the policy can be both protective and a source of false positives.

### Compile And Staging Boundary Is Sensitive

`src/sol_execbench/core/data/solution.py` validates source paths and compile
flags. It rejects absolute paths, `..`, response files, path injection flags,
  runtime linker flags, and legacy CUDA compile options. This is important and
  should stay high-priority because `src/sol_execbench/driver/templates/build_ext.py`
  passes solution compile options into `torch.utils.cpp_extension.load()`.

Residual risk:

- The allowlist intentionally permits ROCm system include/library paths such as
  `/opt/rocm/include` and `/opt/rocm/lib`. That is pragmatic but means builds
  still depend on host toolchain content.
- The compile path auto-injects offload architecture flags in
  `src/sol_execbench/driver/problem_packager.py` based on target hardware and
  local detection through `rocm_agent_enumerator` or `rocminfo`. Incorrect host
  detection or mixed-GPU systems can produce misleading or nonportable builds.
- `ProblemPackager.close()` removes the staging directory with
  `shutil.rmtree(..., ignore_errors=True)`. Cleanup failures are intentionally
  silent, so leaked staging directories are possible if the OS refuses cleanup.

### Safetensors Staging Relies On Path Policy

`src/sol_execbench/driver/problem_packager.py` stages repo-local safetensors by
symlink or copy. It rejects absolute paths and `..`, then searches the repo root
and `FLASHINFER_TRACE_DIR`.

Residual risk:

- The search strategy can map suffixes from a workload path to files under a
  configured external trace root. That is necessary for downloaded assets, but
  it means `FLASHINFER_TRACE_DIR` is part of the trust boundary.
- Symlink fallback to `shutil.copy2()` can turn large trace assets into large
  temporary staging trees.

## Correctness And Benchmark Semantics Risks

### Evaluation Driver Is Complex And Central

`src/sol_execbench/driver/templates/eval_driver.py` is a generated 583-line
script that owns the benchmark-critical sequence: load problem, review sources,
load reference, snapshot integrity, load solution, generate inputs, run ten
correctness rounds, measure reference and candidate latency, check outputs, and
emit canonical trace JSONL.

Fragile areas:

- The driver redirects stdout to stderr before importing PyTorch/Triton, keeping
  canonical JSONL on the original stdout. This is necessary but delicate; any
  future code that writes directly to file descriptors can affect trace parsing.
- Correctness behavior depends on `gen_inputs()`, `call_and_collect_outputs()`,
  destination-passing style, output allocation, shape/dtype checks, and repeated
  rounds staying aligned.
- Reference code and user code are both imported into the same process. Integrity
  snapshots reduce mutation risk but do not provide process isolation between
  reference and solution modules.

Known recent fragility:

- The CDNA3 known-issues file records a fixed bug where custom input factories
  were ignored for random workload inputs in `src/sol_execbench/core/bench/io.py`,
  causing reference-vs-reference numerical failures for
  `L2/033_multi_scale_feature_pyramid`.
- The same notes record a fixed bug where valid stdout traces from nonzero CLI
  exits were lost before `src/sol_execbench/core/dataset/runner.py` preserved
  them.

### Timing Semantics Are Operationally Fragile

`src/sol_execbench/core/bench/timing.py` uses PyTorch's `torch.cuda.Event` API as
the ROCm-compatible event timing path. It also clears L2 cache by allocating a
buffer at least twice the detected L2 cache size and uses
`ShiftingMemoryPoolAllocator` to vary tensor data pointers across iterations.

Risks:

- The code still uses the historical `torch.cuda` namespace because PyTorch ROCm
  exposes HIP devices there. This is correct for PyTorch ROCm but is an ongoing
  source of naming confusion and migration residue.
- Event timing assumes GPU availability and valid device selection. CPU timing is
  only available through `SOL_EXECBENCH_ALLOW_CPU_TIMING=1` in
  `src/sol_execbench/core/bench/eval_runtime.py`, so tests can pass CPU fallback
  paths that are not representative of real GPU timing.
- Timing-sensitive tests are marked `timing_serial` and skipped by default in
  `tests/conftest.py`. This protects regular CI but leaves timing regressions
  dependent on explicit hardware runs.
- `scripts/run_dataset.py` distinguishes `traces`, `derived`, and `timing`
  phases. The known-issues notes warn that derived-phase wall-clock speed should
  not be interpreted as timing evidence.

### Trace Parsing Can Hide Non-JSON Output

`src/sol_execbench/driver/problem_packager.py` parses only stdout lines starting
with `{` as traces and skips other lines. This allows library noise to be
ignored, but it also means malformed partial JSON or unexpected stdout text can
produce a no-trace diagnostic rather than a precise parse failure. The CLI
mitigates this by writing bounded no-trace sidecars in
`src/sol_execbench/cli/main.py`.

## ROCm Hardware And Validation Gaps

### Hardware-Gated Coverage Is Easy To Miss

`tests/conftest.py` skips ROCm tests when `/dev/kfd` or `/dev/dri` are missing,
when PyTorch is not a ROCm build, when target architecture markers do not match,
or when ROCm development headers are unavailable. It also skips
`requires_cutile` permanently because this is a ROCm-only port.

Risk:

- The suite can look green on non-GPU or sandboxed hosts while many important
  execution, timing, native extension, CK, rocWMMA, RDNA4, and CDNA3 paths are
  skipped.
- Architecture markers (`requires_rdna4`, `requires_cdna3`) validate target
  behavior only when the matching hardware is present.

### CDNA3/gfx942 Validation Is Not Closed

`.planning/quick/260604-cdna3-gfx942-validation-attempt-record/KNOWN_ISSUES.md`
records active CDNA3/gfx942 issues from a full validation artifact:

- 209 problems total.
- 170 OK and 39 FAIL at validation summary level.
- 3106 workloads total.
- 2694 passed workloads and 424 failed workloads.
- 197 trace files and 39 CLI logs.

Open classes:

- Timeout problems without traces, including examples under `L1/076`,
  `L1/094`, `L2/004`, `L2/005`, `L2/023`, `L2/024`, `L2/025`, `L2/026`,
  `L2/047`, `L2/055`, `L2/077`, and `L2/078`.
- Static source review blocks classified as `REWARD_HACK`, especially
  `precision_downgrade` and `semantic_output_cache` hits.
- Quant NVFP4/FP8 reference paths failing as `INVALID_REFERENCE` because CUDA
  scaled GEMM behavior is unavailable on ROCm.
- Derived scoring coverage is low: 196 scored entries and 2910 unsupported /
  unscored entries represented inside `scores[]`.
- Hardware model evidence falls back to `gfx1200` provisional modeling instead
  of a true `gfx942` AMD SOL model.

Impact:

- CDNA3 behavior is partially validated but not release-grade for full paper
  parity or final AMD-native scoring.
- Any public claim should preserve the existing scope language in `README.md`,
  `docs/CLAIMS.md`, `docs/research_preview.md`, and the milestone notes.

### CDNA4 Low-Precision Validation Is Deferred

`README.md`, `docs/solution.md`, and
`src/sol_execbench/core/dataset/low_precision.py` treat NVFP4/MXFP4 and
Blackwell-specific low-precision behavior as compatibility/ineligible or
deferred until CDNA4-class hardware is available.

Risk:

- Compatibility abstractions can preserve schema and shape behavior without
  proving real hardware equivalence, performance, or paper parity.
- The readiness classifier in `src/sol_execbench/core/dataset/readiness.py` and
  documentation guardrails need to remain aligned so deferred low-precision
  support does not accidentally become a validation claim.

## Scoring And Evidence Risks

### Derived Scoring Has Low Effective Coverage

The scoring stack spans multiple modules under `src/sol_execbench/core/scoring/`,
including `amd_sol.py`, `amd_sol_v2.py`, `amd_bound_graph.py`,
`amd_bound_estimates.py`, `solar_derivation.py`, and `amd_score.py`.

Known CDNA3 evidence shows:

- SOLAR sidecars: 3106.
- AMD SOL bound sidecars: 3106.
- AMD-native `scores`: 3106.
- `scored_count`: 196.
- `unscored_count`: 2910.

Main causes recorded in the known-issues notes:

- incomplete semantic evidence,
- unsupported semantic evidence,
- unsupported or inexact operators such as `getitem`, `torch.cat`,
  data movement, dtype conversion, and reductions,
- dynamic trace failures,
- provisional baseline usage.

Impact:

- The pipeline can complete structurally while still producing mostly
  unsupported or provisional scoring results.
- Reports must distinguish completed sidecar generation from authoritative
  scoring coverage.

### Hardware Model Selection Is Fragile

`src/sol_execbench/core/scoring/amd_hardware_models.py` and the AMD SOL
derivation path need explicit model coverage for target architectures. The
known CDNA3 notes say trace environments report generic `"AMD Radeon Graphics"`
and scoring warnings use `model_validation:gfx1200:provisional` even when
`--gpu-architecture gfx942` is passed.

Risk:

- Architecture CLI inputs can document the intended target but not guarantee the
  selected scoring model is architecture-correct.
- Adding `gfx942` or future CDNA4 models should be treated as both a model
  implementation task and a claim-boundary task.

### Release And Claim Guardrails Are Broad But Coupled

Release checks in `scripts/check_prerelease_readiness.py`,
`scripts/release_candidate_validation.py`,
`scripts/build_prerelease_artifact_bundle.py`, and reporting modules under
`src/sol_execbench/core/` enforce claim boundaries, provenance, consistency, and
redistribution policy.

Risk:

- Guardrails are distributed across many scripts and tests. Updating claim
  wording, source classifications, or evidence schemas can require coordinated
  changes across docs, scripts, fixtures, and tests.
- `scripts/check_prerelease_readiness.py` dynamically imports
  `scripts/check_dataset_redistribution.py` by path. That is fine for local
  release tooling, but it should stay outside untrusted-input execution paths.

## Dataset, Licensing, And Redistribution Risks

### Downloaded Dataset Artifacts Must Stay Out Of The Repo

The active milestone in `.planning/milestones/v1.29-ROADMAP.md` focuses on
dataset migration and compliance for NVIDIA SOL-ExecBench and FlashInfer Trace.
`scripts/download_solexecbench.py`, `src/sol_execbench/core/dataset/migration.py`,
`src/sol_execbench/core/dataset/manifest.py`, and
`src/sol_execbench/core/dataset/readiness.py` all operate near the boundary
between local-only source data and generated/publishable metadata.

Risks:

- NVIDIA source dataset content and derivative migrated problem artifacts must
  not be committed or bundled for release unless explicitly allowed by policy.
- Local manifests, checksums, ready subsets, and blocker reports are useful and
  safer to share, but they need clear source identifiers and redistribution
  metadata.
- `data/` is excluded from Ruff and intended for downloaded assets; accidental
  commits remain a process risk rather than something code style tools will
  catch.

Mitigations already present:

- `provenance.toml` and `THIRD_PARTY_NOTICES.txt` document source and license
  boundaries.
- `scripts/check_dataset_redistribution.py` and
  `scripts/check_prerelease_readiness.py` enforce redistribution policy.
- Tests such as `tests/sol_execbench/test_dataset_redistribution_policy.py`,
  `tests/sol_execbench/test_provenance_policy.py`, and
  `tests/sol_execbench/test_prerelease_readiness.py` cover the policy layer.

## Maintainability Hotspots

### Large Orchestration Modules

The following files are large enough to be fragile during focused changes:

- `scripts/run_dataset.py` at 3075 lines: dataset discovery, phase selection,
  closure records, reuse policy, trace preservation, derived evidence, timing
  evidence, summaries, and CLI argument parsing are all in one script.
- `src/sol_execbench/core/scoring/solar_derivation.py` at 2570 lines: parsing,
  schema evolution, semantic evidence, formula inputs, status aggregation, and
  compatibility behavior are concentrated in one module.
- `src/sol_execbench/cli/main.py` at 1116 lines: Click CLI, staging, compile,
  evaluation, profiling, static evidence, diagnostics, environment snapshots,
  and output formatting are tightly coupled.
- `src/sol_execbench/core/environment.py` at 620 lines: environment snapshot and
  diagnostics span tool probing, PyTorch ROCm checks, timing probes, and report
  serialization.
- `src/sol_execbench/driver/templates/eval_driver.py` at 583 lines: generated
  script template combines security, correctness, timing, and trace emission.

Suggested direction:

- Continue extracting pure helpers out of `scripts/run_dataset.py` into
  importable package modules under `src/sol_execbench/core/dataset/`.
- Keep generated driver changes extremely small and covered by subprocess-level
  tests under `tests/sol_execbench/driver/` and
  `tests/sol_execbench/core/bench/`.
- Treat scoring schema changes as compatibility migrations with fixture coverage.

### Migration Residue Is Intentional But Confusing

The ROCm port still uses names such as `torch.cuda`, `cuda_events`, and
`is_cuda_available()` in `src/sol_execbench/core/utils.py`,
`src/sol_execbench/core/bench/timing.py`, and tests. This is often technically
correct for PyTorch ROCm, but it is easy to mistake for unported CUDA behavior.

The project also keeps compatibility examples and tests for legacy categories:

- `examples/cutlass/gemm/solution_cutlass.json`
- `examples/cudnn/softmax/solution_cudnn.json`
- `examples/cutile/jamba_attn_proj/solution_cutile.json`
- `tests/sol_execbench/test_rocm_library_examples.py`

Risk:

- New contributors may remove intentional compatibility references or add new
  CUDA-only paths without realizing the difference.
- Docs such as `docs/original_parity.md`, `docs/solution.md`, and `README.md`
  should remain the source of truth for which CUDA/NVIDIA terms are historical,
  rejected, or compatibility-only.

## Performance And Operational Risks

### Full Dataset Runs Are Expensive And Partially Recoverable

`scripts/run_dataset.py` supports phased execution and trace reuse, but real GPU
trace and timing phases are still expensive. Known CDNA3 results include timeout
problems with no traces, which are not recoverable downstream without rerunning
the trace phase.

Risks:

- A single long-running problem can block full validation unless users shard,
  limit workloads, or use targeted reruns.
- Timeout values that are safe for CI may be insufficient for heavy benchmark
  shapes on CDNA3.
- Derived phases can be parallelized with `--jobs`, but timing phases should
  remain serial for reliable measurements.

### Static Evidence And Profiling Are Diagnostic, Not Authority

`src/sol_execbench/cli/main.py` can collect optional `rocprofv3` profiling and
static kernel evidence. These paths write sidecars and intentionally do not
change canonical trace JSONL correctness or performance authority.

Risk:

- Users may overinterpret static evidence artifacts or profiler sidecars as
  correctness, score, or paper-parity evidence.
- The code mitigates this with explicit sidecar claim-boundary fields, but docs
  and report wording need to keep reinforcing that distinction.

## Testing Gaps And Fragile Coverage

The repository has broad test coverage: 120 Python test files were found under
`tests/`. The suite includes schema, driver, reward-hack, dataset migration,
readiness, release guardrail, scoring, Docker, and hardware-marker tests.

Remaining gaps:

- Many GPU behaviors are skipped without ROCm hardware and device nodes.
- Native extension paths require ROCm development headers, CK headers, or
  rocWMMA headers.
- Timing tests are skipped unless selected explicitly with `-m timing_serial`.
- Full CDNA3 and future CDNA4 evidence require external hardware and artifact
  archives, not just local unit tests.
- Dataset validation depends on downloaded local assets under `data/`, which are
  intentionally not committed.

## Priority Watchlist

1. Keep evaluation isolation and reward-hack defenses under focused regression
   coverage whenever `src/sol_execbench/driver/templates/eval_driver.py`,
   `src/sol_execbench/core/bench/eval_runtime.py`, or
   `src/sol_execbench/core/bench/reward_hack.py` changes.
2. Resolve or explicitly defer the CDNA3 known issues in
   `.planning/quick/260604-cdna3-gfx942-validation-attempt-record/KNOWN_ISSUES.md`.
3. Add or select a real `gfx942` AMD hardware model before treating CDNA3
   AMD-native scores as release-grade.
4. Improve semantic evidence extraction in
   `src/sol_execbench/core/scoring/solar_derivation.py` and related AMD bound
   modules so derived scoring coverage rises beyond the current low coverage.
5. Continue shrinking `scripts/run_dataset.py` by moving tested logic into
   package modules.
6. Keep dataset redistribution checks mandatory before release bundles or public
   artifact publication.
