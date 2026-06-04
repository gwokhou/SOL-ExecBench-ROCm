---
generated_at: 2026-06-04
last_mapped_commit: ac6505f6511818160d36bc6935328ff0bd9468a6
focus: concerns
scope: full repo
---

# Concerns

## Status Map

- Current state: v1.29 is shipped and archived as of 2026-06-04, and no active
  milestone is defined.
- Highest-risk active surfaces are the generated evaluator, subprocess/native
  build boundary, static reward-hack guardrails, dataset migration/closure
  evidence, and proliferating derived authority reports.
- Externally blocked or explicitly deferred concerns remain paper-scale
  validation, upstream SOLAR parity, full MI300X/CDNA3 validation, CDNA4
  validation, hosted leaderboard/service readiness, hard sandboxing, stable
  benchmark authority, and NVIDIA dataset redistribution.
- The working tree currently shows `.planning/codebase/*.md` as deleted before
  this mapping pass. This file recreates only the requested concerns map and
  does not resolve the other deleted codebase documents.

## Release Claim Boundaries

- Full 235-problem paper validation is not complete and must not be claimed.
- Upstream NVlabs/SOLAR parity is not established and must not be claimed.
- Hosted leaderboard readiness, score authority, and stable benchmark authority
  are not established.
- Docker Matrix rows are ROCm user-space/container evidence only; they are not
  native-host validation or paper-equivalent evidence.
- MI300X is the concrete CDNA3 `gfx942` target, but full-suite MI300X validation
  remains deferred without archived real-hardware evidence.
- CDNA3-family hardware-validation claims must remain deferred until real
  full-suite `gfx94*` evidence is archived.
- CDNA4 validation and performance authority remain unavailable because suitable
  hardware is not currently accessible.
- NVFP4/MXFP4/Blackwell-style low-precision paths are compatibility code and
  readiness evidence, not CDNA4 validation.

## Execution Isolation

- The evaluator runs user-supplied Python, Triton, and native HIP/C++ code in a
  subprocess, but this is not a hardened sandbox.
- Reward-hack checks in `src/sol_execbench/core/bench/reward_hack.py` and the
  generated driver are guardrails, not a multi-tenant security boundary.
- Running untrusted submissions still requires an isolated ROCm host, VM, or
  container with an explicit threat model.
- Native compilation remains sensitive because source files and flags are
  staged then passed to `torch.utils.cpp_extension`; compile flag validation in
  `src/sol_execbench/core/data/solution.py` should stay strict when adding new
  ROCm library categories.
- `ProblemPackager._write_sources()` writes solution-provided relative paths
  into the staging directory. Path traversal and absolute paths are blocked, but
  future generated staging filenames must avoid collisions with allowed source
  names.
- The Docker image grants passwordless `amd-smi`/`rocm-smi` access when present
  so clock tooling can work. That is operationally useful, but it should remain
  documented as a privileged local evaluation environment, not a sandbox.

## Generated Driver Complexity

- `src/sol_execbench/driver/templates/eval_driver.py` is still behavior-dense
  and is the critical correctness/timing/reward-hack execution path.
- stdout/stderr routing is fragile: strict Trace JSONL depends on redirecting
  stdout to stderr before importing GPU libraries and writing canonical traces
  through a saved file descriptor.
- The driver performs source review, reference import, user import, integrity
  snapshots, safetensors lookup, repeated correctness rounds, timing, thread
  checks, and trace emission in one generated script. Small edits need focused
  driver regression tests.
- CPU fallback behavior is useful for tests but increases the chance that a
  CPU-safe pass is misread as ROCm hardware validation.
- Exceptions inside reference/user execution are captured into trace logs, but
  non-trace subprocess failures still rely on bounded diagnostic sidecars and
  stdout parsing heuristics.

## Reward-Hack Guardrails

- Static source review intentionally blocks broad families: streams, caches,
  file/process/network access, dynamic loading, embedded payload decoding, and
  float32 precision downgrade.
- These rules are conservative and can reject legitimate benchmark/reference
  patterns. `run_dataset` currently renames exact `stream` identifiers before
  static review, which is a fragile workaround around broad textual rules.
- Static review is stronger for Python because it uses AST inspection; non-Python
  sources still rely on pattern scanning after simple comment stripping.
- The checks guard known exploit classes but cannot prove semantic honesty of an
  arbitrary GPU kernel.
- New legitimate ROCm APIs that require streams, external loaders, or caching
  need careful policy decisions instead of ad hoc allow-list expansion.

## Timing And Performance Fragility

- Timing depends on HIP-backed PyTorch CUDA APIs, ROCm clocks, allocator state,
  device synchronization semantics, and optional `rocprofv3` evidence.
- `timing_serial` tests are skipped by default and require explicit selection,
  so timing regressions can escape routine CPU-safe runs.
- Clock-lock checks depend on environment variables, `amd-smi`/`rocm-smi`,
  Docker entrypoint behavior, and host permissions.
- `rocprofv3` profile/timing sidecars are diagnostic evidence only. They must
  not be upgraded into correctness, score, or paper-parity authority.
- Reference latency is optional through benchmark config; reports that use
  speedup or AMD score fields must preserve whether reference timing was
  actually collected and meaningful.

## Hardware And Dependency Matrix

- The package lock targets PyTorch ROCm 7.1 on Linux x86_64, while Docker target
  metadata spans multiple ROCm user-space versions. Mixed-version diagnostics
  exist but cannot create clean validation authority.
- ROCm device discovery depends on `/dev/kfd`, `/dev/dri`, PyTorch ROCm builds,
  and `gcnArchName`/`gfx_arch_name` metadata.
- `requires_rocm`, `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`,
  `requires_rdna4`, and `requires_cdna3` tests are marker-gated. This keeps CI
  CPU-safe but leaves real GPU/native extension paths dependent on local or
  Docker hardware runs.
- CDNA3 tests verify marker behavior and readiness language without real full
  validation unless executed on actual `gfx94*` hardware.
- CDNA4/low-precision compatibility code has CPU-safe semantic coverage but no
  real hardware or performance validation.

## Dataset And Provenance Boundaries

- NVIDIA SOL-ExecBench source and migrated dataset payloads are local-only or
  excluded under the current provenance policy; they must not be committed,
  bundled, hosted, or redistributed by this project.
- FlashInfer Trace has separate Apache-2.0 provenance and notice obligations;
  it should not be conflated with NVIDIA SOL-ExecBench policy.
- Migration manifests, readiness reports, ready subsets, execution closure, and
  paper denominator reports are derived evidence and must preserve source IDs,
  checksums, license boundaries, denominator accounting, blocker reasons, and
  requested-evidence drift.
- Dataset runner skip/reuse paths are correctness-sensitive because they can
  silently change denominators if manifest, readiness, solution mode, timeout,
  or evidence requests drift.
- Full dataset execution remains bounded by local assets, hardware access,
  safetensors/blob availability, and ROCm readiness blockers.

## Scoring And Derived Evidence

- AMD-native scoring, AMD SOL bounds, SOLAR derivation evidence, bound graphs,
  sanity reports, stability reports, consistency reports, claim-upgrade reports,
  trust summaries, Matrix reports, and closure reports are derived authority
  layers around canonical Trace JSONL.
- These modules are large and domain-heavy; `solar_derivation.py`,
  `amd_bound_graph.py`, and `amd_bound_estimates.py` are notable maintenance
  hotspots.
- Missing metadata, unsupported operators, ambiguous graph extraction, or
  degraded model evidence must stay visible rather than becoming fabricated
  supported estimates.
- Any new sidecar/report needs explicit authority-class wording and tests so it
  is not mistaken for benchmark score, paper parity, leaderboard readiness, or
  hardware validation.

## Legacy NVIDIA Residue

- CUDA/NVIDIA runtime values are rejected in ROCm schemas, but legacy examples,
  samples, docs, tests, and attribution remain for migration, compatibility, or
  provenance reasons.
- Directories such as `examples/cutlass/`, `examples/cudnn/`,
  `examples/cutile/`, and `examples/cute_dsl/` are legacy or compatibility
  surfaces and should not be presented as supported ROCm runtime paths.
- NVIDIA SPDX notices are tracked by `provenance.toml`; future cleanup must keep
  source headers, allowed notice paths, cleanup candidates, and prerelease
  guardrails synchronized.
- Text searches for CUDA/NVIDIA terms remain noisy because many occurrences are
  intentional claim boundaries, migration docs, or guardrail tests.

## Test Coverage Gaps

- Routine test coverage is intentionally CPU-safe and marker-gates GPU,
  hardware-architecture, native extension, CK, rocWMMA, and timing surfaces.
- Full driver e2e, native HIP/C++ builds, ROCm library examples, `rocprofv3`
  behavior, clock locking, and real dataset batches need ROCm/Docker hardware
  validation.
- Existing adversarial tests cover known reward-hack classes, but the guardrail
  model should be expanded whenever new solution APIs or timing policies are
  added.
- Public-contract and prerelease guardrail tests are essential because most
  current risk is overclaiming derived evidence, not just functional failure.
- Several project-state quick task artifacts are marked missing in
  `.planning/STATE.md`; that is planning hygiene debt even though the latest
  milestone audit passed.

## Operational Fragility

- `scripts/run_dataset.py` is large and mixes CLI orchestration, solution
  wrapping, source sanitization, profiling evidence, closure construction,
  readiness integration, and scoring sidecars.
- `src/sol_execbench/cli/main.py` is also large and owns user-facing command
  dispatch, staging lifecycle, compilation, execution, profiling, static
  evidence, no-trace diagnostics, and output formatting.
- Many subprocess paths capture bounded logs, but failures can still require
  manual correlation across trace JSONL, no-trace diagnostics, profile sidecars,
  static evidence sidecars, closure JSON, and Docker/preflight logs.
- The package supports Python `>=3.12,<3.14` while depending on exact ROCm wheel
  versions and platform markers. Dependency relocking should be treated as a
  high-risk validation event.

## Suggested Milestone Candidates

- Hardening milestone: shrink `eval_driver.py`, move more reusable behavior into
  importable helpers, and expand generated-driver regression tests for stdout
  routing, source review, correctness rounds, and no-trace diagnostics.
- Security boundary milestone: document and test the exact trust model for
  untrusted submissions, source-path collisions, native build flags, Docker
  privileges, file/network/process denial, and residual non-sandbox risks.
- Hardware validation milestone: run and archive full MI300X/CDNA3 evidence on
  real `gfx942`, including environment, clock, trace, sidecar, failure, and
  denominator artifacts.
- CDNA4/low-precision milestone: validate FP4/NVFP4/MXFP4-compatible paths on
  actual CDNA4-class hardware before any performance or hardware-equivalence
  claim.
- Dataset authority milestone: execute a complete local migrated denominator
  with readiness/closure/paper-denominator artifacts and explicit
  redistribution-safe release notes.
- Provenance stewardship milestone: reconcile remaining NVIDIA notice cleanup
  candidates, legacy examples, and upstream comparison without rewriting history
  or weakening license boundaries.
