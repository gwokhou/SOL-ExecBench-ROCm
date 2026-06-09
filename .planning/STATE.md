---
gsd_state_version: 1.0
milestone: v1.34
milestone_name: RDNA4 Readiness Blocker Closure
status: ready_to_plan
stopped_at: Phase 170 complete (1/1) — ready to discuss Phase 171
last_updated: 2026-06-09T13:56:03.792Z
last_activity: 2026-06-09 — Milestone 1.34 completed and archived
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-06-09)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.
**Current focus:** Phase 171 — custom input coverage recompute

## Current Position

Phase: 171
Plan: Not started
Status: Ready to plan
Last activity: 2026-06-09

## Recent Trend

- Phase 173 completed on 2026-06-09 with FlashInfer semantic readiness split into
  PyTorch-compatible simple cases and runtime-dependent runtime buckets:
  `pytorch_compatible`, `paged_decode`, `paged_prefill`, `ragged_prefill`,
  `mla_paged`, `moe_fp8_block_scale`, and `unknown_flashinfer_runtime`.

- Phase 174 completed on 2026-06-09 by finalizing readiness-closure evidence
  tracking for all original readiness blockers, including guardrails for no
  execution claim upgrade and residual blocker provenance.

- v1.23 shipped Phases 106-109 on 2026-06-01.
- v1.24 shipped Phases 110-113 on 2026-06-01.
- v1.25 shipped Phases 114-118 on 2026-06-01.
- v1.26 shipped Phases 119-122 on 2026-06-02.
- Quick task 260602-jjy fixed GitHub Actions Ty/CPU-safe CI failures on
  2026-06-02.

- Quick task 260602-mbt added a pre-commit-managed git pre-push Ty check hook
  on 2026-06-02.

- Quick task 260602-miz made `pre-commit install` enable pre-commit,
  commit-msg, and pre-push hooks by default on 2026-06-02.

- Quick task 260602-mmo added `pre-commit` to the dev dependency group on
  2026-06-02.

- Quick task 260602-mqi fixed stale CI, Docker, pre-commit, dependency marker,
  and documentation configuration on 2026-06-02.

- Quick task 260602-mqr tightened hook locking, Ruff excludes, Docker runtime
  dependency groups, Linux x86_64 ROCm markers, and configuration docs on
  2026-06-02.

- Quick task 260602-msd removed unnecessary README prose line wrapping while
  preserving command formatting on 2026-06-02.

- Quick task 260603-ffd resolved remaining GSD health warnings by indexing
  archived v1.27 phases and moving validation handoff files into the milestone
  archive on 2026-06-03.

- Quick task 260604-cdna3-gfx942-validation-attempt-record recorded a real
  `gfx942` CDNA3 validation attempt on 2026-06-04. The attempt confirmed ROCm
  runtime readiness and CDNA3 marker behavior, but full validation remains
  blocked by Triton static-review handling, a CPU-device synchronization bug,
  and a HIP RMSNorm runtime failure on `gfx942`.

- Quick task 260604-fix-cdna3-validation-blockers fixed the local blockers
  identified by the `gfx942` attempt: CPU-device synchronization in
  `call_and_collect_outputs`, Triton `triton.language.load` static-review
  handling, and the RMSNorm HIP example's wave-size-specific reduction. CDNA3
  validation still requires cloud revalidation before any claim upgrade.

- CDNA3 `gfx942` cloud revalidation passed the full adapted pytest suite on
  2026-06-04 at repository HEAD `0d6c3e1` with `1401 passed, 62 skipped`.
  Broader MI300X benchmark-grade validation still requires dataset, clock-lock,
  timing, AMD score, FP8, and deferred low-precision evidence.

- v1.28 CDNA3 Test and Documentation Readiness started on 2026-06-04 with
  Phase 127-130 scoped for CDNA3 hardware-gated tests, MI300X evidence
  contracts, deferred-validation guardrails, and public documentation closure.

- Phase 127 completed on 2026-06-04 with concrete `requires_cdna3` hardware
  marker tests, CPU-safe skip behavior checks, AST marker-use audit coverage,
  and CDNA3 metadata tests.

- Phase 128 completed on 2026-06-04 with expanded MI300X evidence artifacts,
  result categories, diagnostics blockers, and handoff/readiness documentation.

- Phase 129 completed on 2026-06-04 with CPU-safe public contract guardrails
  preserving CDNA3 deferred-validation wording after readiness expansion.

- Phase 130 completed on 2026-06-04 with testing, ROCm support, and
  contributor documentation for CDNA3 marker readiness and deferred MI300X
  validation evidence.

- Milestone v1.28 passed audit and was archived on 2026-06-04.

- Milestone v1.29 Dataset Migration and Compliance started on 2026-06-04 with
  Phases 131-135 scoped for dataset license/provenance policy, local migration,
  ROCm readiness classification, NVIDIA/Blackwell low-precision compatibility,
  and dataset-runner/public guardrails.

- Phase 131 completed on 2026-06-04 with machine-readable dataset provenance
  policy, redistribution classes, staged/release guardrails, prerelease
  readiness integration, and docs preserving NVIDIA Evaluation Dataset License
  and Apache-2.0 FlashInfer Trace boundaries.

- Phase 132 completed on 2026-06-04 with deterministic local SOL-ExecBench and
  FlashInfer Trace migration commands, checksum manifests, generated artifact
  refs, license-boundary metadata, and explicit blockers for missing blobs,
  safetensors refs, traces, and solutions.

- Phase 133 completed on 2026-06-04 with static ROCm readiness classes,
  deterministic blocker reports, auditable ready-subset denominator/exclusion
  metadata, closure inputs, and claim-boundary guardrails.

- Phase 134 completed on 2026-06-04 with CPU-safe low-precision compatibility
  helpers for NVFP4/MXFP4-like E2M1 semantics, scale metadata, explicit
  unvalidated-CDNA4 evidence markers, readiness blocker integration, and
  semantic round-trip tests.

- Phase 135 completed on 2026-06-04 with dataset runner migration-manifest
  provenance, license-boundary metadata, readiness/ready-subset denominator
  summaries, blocker-visible execution closure rows, deterministic no-ready
  summaries, public-safe cookbook workflow docs, and CPU-safe guardrail tests.

- Phase 136 completed on 2026-06-07 with RDNA4 validation scope documentation,
  default-off long-tail exclusion config for `scripts/run_dataset.py`,
  `excluded_long_tail` closure accounting, provenance/source-ref wiring, and
  CPU-safe guardrail tests preserving claim boundaries.

- Phase 137 completed on 2026-06-07 with a long-running RDNA4 validation
  runbook, host `gfx1200` preflight evidence, category guardrails, JUnit
  artifacts, and an accepted execution-environment boundary showing that
  `uv`/pytest could not access `/dev/kfd` or `/dev/dri` despite host ROCm
  tools seeing the RDNA4 GPU.

- Phase 138 blocked on 2026-06-07 before full dataset execution because
  `data/SOL-ExecBench/benchmark` is absent and `uv run` cannot see ROCm device
  nodes. See
  `.planning/phases/138-rdna4-full-dataset-execution-and-denominator-closure/138-BLOCKED.md`.

- Phase 138 blocker rechecked on 2026-06-07T15:41:14Z with the same result:
  only `data/` exists, and `uv run` reports `/dev/kfd False`, `/dev/dri False`,
  HIP `7.1.25424`, `torch.cuda.is_available() False`, and device count `0`.

- Phase 138 completed on 2026-06-08 after the dataset was downloaded and an
  escalated RDNA4 `uv run` environment executed all 121 ready problems. The
  bounded ready-subset denominator contains 3957 workload records: 1907 ready
  workloads attempted on RDNA4 and 2050 readiness-blocked workloads not
  attempted. The execution result was 86 OK problems, 35 FAIL problems, 1761
  passed workloads, 146 failed workloads, and complete closure accounting
  including 12 explicit `missing_trace` workload records.

- Phase 139 completed on 2026-06-08 with RDNA4 `gfx1200` environment evidence,
  `rocprofv3` availability evidence, explicit clock-lock blocker
  classification, 121 timing sidecars, and an evaluation stability report. GPU
  SCLK/MCLK lock and reset commands still require a sudo password, likely due
  incomplete sudoers coverage, so timing remains non-authoritative. All 121
  timing sidecars selected PyTorch/device-event fallback rather than
  profiler-backed `rocprofv3` kernel activity timing.

- Phase 140 completed on 2026-06-08 with RDNA4 AMD-native score report
  generation, 1839 AMD SOL v2 sidecars, 1839 SOLAR derivation sidecars,
  paper-denominator/parity/AMD-bound/consistency/claim/trust reports, and
  `out/rdna4-derived-reports/bundle/evidence-bundle.json`. Heavy derived
  generation was moved into `scripts/run_derived_isolated.py` using
  `systemd-run --user` with `MemoryMax=20G` and `MemorySwapMax=0`, preventing
  OOM sidecar builders from taking down Codex. The Phase 140 bundle records 56
  temporarily excluded sidecar workloads and zero unexcluded missing sidecars.

- Quick task 260604-vjx reduced heavyweight script memory and I/O overhead on
  2026-06-04 by streaming long subprocess output to bounded logs, caching
  prerelease checksums, avoiding redundant SOLAR sidecar rereads, and streaming
  workload prefix reads while preserving benchmark execution semantics.

- Quick task 260607-ivd simplified README wording for developers and
  researchers on 2026-06-07 by shortening dense prose, grouping the
  documentation index, and preserving ROCm validation boundaries.

- Quick task 260607-j8c improved README-linked documentation readability on
  2026-06-07 by clarifying configuration/test/research entry points,
  centralizing authority-class wording, reducing historical-log overload, and
  preserving claim guardrails.

- Quick task 260608-jan clarified RDNA4 timing fallback semantics on
  2026-06-08 by documenting source-policy fallback versus profiler-unavailable
  fallback and adding HIP native source collection coverage for profiler-backed
  `rocprofv3` timing.

- Phase 141 completed on 2026-06-08 with public RDNA4 claim closure across
  README, `docs/CLAIMS.md`, `docs/research_preview.md`,
  `docs/release_candidate_validation.md`, and `docs/rocm.md`. The final public
  wording cites the bounded `gfx1200` denominator and derived evidence counts,
  keeps timing non-authoritative, and prevents paper-parity, upstream SOLAR,
  NVIDIA B200, leaderboard, CDNA3/MI300X, or CDNA4 claim upgrades.

- Phase 149 blocked on 2026-06-08 after its first real full-problem
  replacement attempt collected `rocprofv3` kernel rows but produced 19
  `PASSED` traces and 1 `INVALID_REFERENCE` trace due to HIP OOM on the
  16 GiB RDNA4 host.

- Phase 150 started on 2026-06-08 to classify replacement attempts as full,
  partial, or blocked while preserving the rule that only complete
  profiler-backed sidecars count toward full timing coverage.

- Phase 159 planned on 2026-06-08 to replace unsafe full-problem profiler
  timing attempts with workload-sharded execution, deterministic manifests, and
  complete-workload aggregation into problem-level profiler-backed timing.

- Phase 159 completed on 2026-06-08 with workload-sharded import-only
  aggregation for `L1/037_flux_feedforward_gelu_approximate`. The final v2
  evidence imported all 16 existing `rocprofv3` workload slices, produced a
  problem-level profiler-backed aggregate sidecar, and moved coverage from
  60/235 to 61/235 profiler-backed problems.

- Phase 160 planned on 2026-06-08 to turn the Phase 159 single-problem
  workload-sharded aggregation path into a generalized audit and closure
  workflow for remaining partial/blocked profiler targets.

- Phase 160 completed on 2026-06-08 with a conservative remaining-target
  audit covering 14 partial/blocked profiler targets. A bounded real sharded
  attempt on `L1/026_video_patch_embedding_projection` showed the target moves
  from `profiler_blocked` to `partial_profiler_backed`, but not full
  `profiler_backed`, because workloads produce `INVALID_REFERENCE`,
  `RUNTIME_ERROR`, and one profiler-failed slice.

- Phase 161 planned on 2026-06-08 to classify the remaining partial profiler
  targets by correctness/runtime/profiler failure mode before spending more
  profiler time on them.

- Phase 161 completed on 2026-06-08 with a partial profiler failure
  classification ledger. The 10 partial targets split into 6
  `blocked_on_correctness`, 2 `blocked_on_runtime`, and 2
  `blocked_on_mixed_failures`; no target remains in a missing-slice-only
  closure bucket.

- Phase 162 completed on 2026-06-08 by promoting diagnosed RDNA4
  reference/gen_inputs OOM failures into the profiler timing denominator as
  `reference_oom_blocked`. The latest 235-problem coverage report accounts for
  61 `profiler_backed`, 10 `reference_oom_blocked`, 4 `profiler_blocked`, 46
  `timing_fallback`, and 114 `readiness_blocked` problems.

- Milestone v1.33 RDNA4 Benchmark-Grade Evidence Closure started on
  2026-06-08 with six ordered phases: denominator policy hardening, memory
  readiness classifier hardening, coverage recompute, `rocprofv3` timing
  closure, clock-lock evidence, and release evidence bundle.

- Phase 163 completed on 2026-06-08 with
  `docs/internal/RDNA4-DENOMINATOR-POLICY.md`, a `docs/CLAIMS.md` policy link,
  and CPU-safe regression coverage proving `reference_oom_blocked` is accounted
  but not complete profiler-backed timing.

- Phase 164 completed on 2026-06-08 with detailed blocker classes in profiler
  timing coverage reports and partial failure ledgers, distinguishing reference
  OOM, input-generation OOM, user-solution OOM, OOM plus profiler gaps,
  timeout, profiler gaps, and non-OOM workload failures.

- Phase 165 completed on 2026-06-08 by regenerating
  `out/rdna4-coverage-recompute-20260608/` with 235 denominator rows, 61
  profiler-backed problems, 174 non-passing/blocker ledger rows, and blocker
  class counts for reference/input/user/profiler-gap OOM cases.

- Phase 166 blocked on 2026-06-08. Full-problem and workload-sharded
  `rocprofv3` closure attempts produced partial/profiler-blocked evidence for
  `L1/028_hybrid_attention_mask_preparation` and
  `L1/053_gaussian_topk_sparse_activation`; `L1/053` offset 9 still OOMs during
  correctness error-stat computation on the current 16GB RDNA4 device.

- Phase 166 completed on 2026-06-09 after accepting
  `profiler_closure_oom_blocked` as a current-device blocker class. Accepted
  coverage reports 61 `profiler_backed`, 13 `reference_oom_blocked`, 2
  `profiler_blocked`, 45 `timing_fallback`, and 114 `readiness_blocked`
  problems.

- Phase 167 completed on 2026-06-09 with RDNA4 `rocm-smi`
  clock-control/reset evidence. Manual/SCLK/MCLK commands and reset succeeded,
  but observed clocks did not prove stable maximum-frequency lock, so RDNA4
  timing remains below benchmark-grade authority.

## Accumulated Context

### Decisions

- v1.25 is an engineering prerelease / release-candidate milestone, not a
  paper-scale validation or hosted-service milestone.

- MI300X is the CDNA3 hardware target; MI300X and CDNA3 are not separate
  validation targets.

- v1.28 should add real CDNA3 test readiness and documentation without claiming
  actual CDNA3 hardware validation on the current machine.

- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

- Full 235-problem paper-scale validation, upstream SOLAR parity, hosted
  leaderboard, hard sandboxing, large dependency relocking, and Docker
  privilege redesign remain deferred unless explicitly reopened.

- v1.29 may implement complete ROCm-equivalent code paths for
  NVIDIA/Blackwell low-precision semantics, but real CDNA4 validation,
  performance authority, and hardware-equivalence claims remain deferred until
  a complete hardware evidence chain exists.

- v1.29 must not redistribute NVIDIA/SOL-ExecBench original or derivative
  dataset content; users must download and migrate that data locally under
  their own applicable license rights.

- FlashInfer Trace provenance is tracked separately as Apache-2.0 content from
  `flashinfer-ai/flashinfer-trace`; required notices must be preserved when
  redistributing any FlashInfer Trace material.

### Roadmap Evolution

- Phase 159 added: RDNA4 workload-sharded profiler-backed timing aggregation
- Phase 160 added: RDNA4 generalized workload-sharded profiler closure
- Phase 161 added: RDNA4 partial profiler failure classification
- Phase 162 added: RDNA4 memory-footprint denominator policy
- Phase 163 added: RDNA4 denominator policy hardening
- Phase 164 added: RDNA4 memory readiness classifier hardening
- Phase 165 added: RDNA4 coverage recompute
- Phase 166 added: RDNA4 rocprof timing closure
- Phase 167 added: RDNA4 clock-lock evidence
- Phase 168 added: RDNA4 release evidence bundle

### Pending Todos

- Audit and archive milestone v1.33 with `/gsd-audit-milestone` and
  `/gsd-complete-milestone`.

### Blockers/Concerns

- Phase 138 failures are real RDNA4 execution findings and must remain visible
  in Phase 140 reports and Phase 141 public wording.

- RDNA4 manual DPM clock commands are superseded by the confirmed
  `amd-smi set -l STABLE_PEAK` runtime path. Sudoers coverage should track the
  exact `amd-smi version`, `set -l STABLE_PEAK`, and `set -l AUTO` commands.

- RDNA4 timing remains bounded by the v1.33 authority-gap closure: STABLE_PEAK
  gives stable near-maximum timing evidence, but it is not a full
  benchmark-grade authority upgrade.

- The full Phase 149 replacement objective remains blocked. Phase 150 may
  reduce the opaque fallback bucket by classifying partial and blocked
  replacement attempts, but it must not count those attempts as complete
  `profiler_backed` timing coverage.

- RDNA4 profiler OOM/timeout closure should no longer rely on serial timeout
  escalation. Phase 159 should make workload-level execution and complete
  aggregation the primary mitigation, with memory caps only as circuit breakers.

- Phase 166 accepted current-device profiler-closure OOM blockers. Full
  profiler-backed timing coverage remains false. Phase 167 recorded
  clock-control and reset evidence, but observed clocks did not prove stable
  maximum-frequency lock, so timing authority remains below benchmark-grade.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260609-he6 | Align clock locking scripts, Docker, docs, and tests around amd-smi STABLE_PEAK | 2026-06-09 | pending | [260609-he6-align-clock-locking-scripts-docker-docs-](./quick/260609-he6-align-clock-locking-scripts-docker-docs-/) |
| 260609-rdna4-clock-lock-workload | RDNA4 clock-lock workload test — confirmed firmware overrides clock lock under load | 2026-06-09 | pending | [260609-rdna4-clock-lock-workload](./quick/260609-rdna4-clock-lock-workload/) |
| 260608-k79 | Improve RDNA4 validation completeness guardrails | 2026-06-08 | f354c32 | [260608-k79-rdna4-rdna4](./quick/260608-k79-rdna4-rdna4/) |
| 260608-kjo | Add RDNA4 profiler-backed timing smoke slice | 2026-06-08 | 6615606 | [260608-kjo-rdna4-profiler-backed-timing-smoke-slice](./quick/260608-kjo-rdna4-profiler-backed-timing-smoke-slice/) |
| 260608-kum | Fix RDNA4 profiler timing finalization | 2026-06-08 | a945fc5 | [260608-kum-rdna4-profiler-timing-smoke-eval-driver-](./quick/260608-kum-rdna4-profiler-timing-smoke-eval-driver-/) |
| 260608-l01 | Add RDNA4 reference OOM classifier | 2026-06-08 | pending | [260608-l01-rdna4-reference-oom-classifier](./quick/260608-l01-rdna4-reference-oom-classifier/) |

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Paper validation | Full 235-problem paper-scale validation and upstream SOLAR parity | Deferred | v1.26 scope |
| Hardware validation | Full MI300X validation on the CDNA3 `gfx942` target without a complete evidence chain | Deferred | v1.26 scope |
| Hardware validation | Actual CDNA3/MI300X full-suite execution during v1.28 on the current machine | Deferred | v1.28 scope |
| Hardware validation | CDNA4 validation because suitable hardware is unavailable | Deferred | v1.26 scope |
| Operations | Hosted leaderboard or remote submission service | Deferred | v1.26 scope |
| Security | Hard sandbox or multi-tenant adversarial execution | Deferred | v1.26 scope |
| Release authority | Stable benchmark authority release | Deferred | v1.26 scope |
| Dataset redistribution | Publishing or hosting NVIDIA/SOL-ExecBench original or derivative dataset content | Deferred | v1.29 scope |
| Hardware validation | Real CDNA4 validation or performance authority for NVFP4/Blackwell semantics | Deferred | v1.29 scope |
| quick_task | 260531-rdf-add-run-dataset-closure-e2e-gaps | completed | v1.29 close artifact audit |
| quick_task | 260531-uki-add-remaining-requires-rocm-e2e-coverage | completed | v1.29 close artifact audit |
| quick_task | 260602-mqi-fix-stale-project-configuration-audit-fi | resolved | v1.31 close artifact hygiene |
| quick_task | 260602-mqr-fix-second-pass-configuration-audit-findings | resolved | v1.31 close artifact hygiene |
| quick_task | 260602-msd-unwrap-readme-prose-lines | resolved | v1.31 close artifact hygiene |
| debug | 260607-remote-ci-failure | resolved | v1.31 close artifact hygiene |
| quick_task | 260604-vjx-scripts-benchmark | resolved | v1.31 close artifact hygiene |
| quick_task | 260605-port-nvfp4-reference-scaled-mm | resolved | v1.31 close artifact hygiene |
| quick_task | 260606-clarify-cdna3-mi300x-hierarchy | resolved | v1.31 close artifact hygiene |
| quick_task | 260606-clarify-mi308x-cdna3-validation-docs | resolved | v1.31 close artifact hygiene |
| quick_task | 260606-wis-fix-current-pytest-failures-after-codeba | resolved | v1.31 close artifact hygiene |

## Session Continuity

Last session: 2026-06-09T08:59:46.651Z
Stopped at: Phase 173 context gathered
Resume file: .planning/phases/173-flashinfer-readiness-split/173-CONTEXT.md

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
