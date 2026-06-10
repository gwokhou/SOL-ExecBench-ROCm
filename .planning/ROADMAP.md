# Roadmap: SOL ExecBench ROCm Port

## Milestones

- **v1.35 Script Parallelism and Safety Hardening** - Phases 175-179 (in progress)
- **v1.34 RDNA4 Readiness Blocker Closure** - Phases 170-174 (shipped 2026-06-09)
- Complete **v1.33 RDNA4 Benchmark-Grade Evidence Closure** - Phases 163-169
  See `.planning/milestones/v1.33-ROADMAP.md`.
- Complete **v1.32 RDNA4 Profiler Timing Coverage Closure** - Phases 148-162
  See `.planning/milestones/v1.32-ROADMAP.md`.
- Complete **v1.31 RDNA4 Follow-Up Evidence Hardening** - Phases 142-147
  See `.planning/milestones/v1.31-ROADMAP.md`.
- Complete **v1.30 RDNA4 Benchmark-Grade Validation Closure** - Phases 136-141
  See `.planning/milestones/v1.30-ROADMAP.md`.
- Complete **v1.29 Dataset Migration and Compliance** - Phases 131-135
  See `.planning/milestones/v1.29-ROADMAP.md`.
- Earlier milestones are archived under `.planning/milestones/`.

## Current Position

**Status:** v1.35 in progress.

<details>
<summary>v1.34 RDNA4 Readiness Blocker Closure (Phases 170-174) -- SHIPPED 2026-06-09</summary>

- [x] Phase 170: Custom Input Evaluator Readiness (1/1 plans)
- [x] Phase 171: Custom Input Coverage Recompute (1/1 plans)
- [x] Phase 172: Quant Readiness Triage (1/1 plans)
- [x] Phase 173: FlashInfer Readiness Split (1/1 plans)
- [x] Phase 174: RDNA4 Readiness Closure Report and Claim Guardrails (1/1 plans)

</details>

### v1.35 Script Parallelism and Safety Hardening (In Progress)

**Milestone Goal:** Refactor key scripts for internal CPU-parallel/GPU-serial
concurrency, prevent unsafe multi-instance execution, and harden execution
environment independence and reproducibility for statistics-sensitive scripts.

## Phases

**Phase Numbering:**
- Integer phases (175, 176, ...): Planned milestone work
- Decimal phases (175.1, ...): Urgent insertions (marked with INSERTED)

- [x] **Phase 175: PID Lock Module** - Standalone fcntl.flock-based multi-instance prevention module
- [x] **Phase 176: Timing Isolation Audit** - Pre-flight GPU contention detection and environment audit for reproducibility (completed 2026-06-10)
- [x] **Phase 177: Profiler Timing Batch Parallelism** - CPU-parallel staging with GPU-serial profiling for the profiler timing batch script (completed 2026-06-10)
- [ ] **Phase 178: Derived Script Parallelism** - ThreadPoolExecutor-based parallel dispatch for the derived isolation script
- [ ] **Phase 179: Evaluation Stability Extension and Integration Tests** - New reason codes and end-to-end integration validation

## Phase Details

### Phase 175: PID Lock Module ✅ COMPLETE

**Goal**: Scripts that must run exclusively can detect and reject concurrent instances with a kernel-managed lock that never leaves stale state
**Depends on**: Nothing (first phase, zero dependencies)
**Requirements**: INST-01, INST-02, INST-03
**Success Criteria** (what must be TRUE):
  1. A script calling `acquire_pid_lock(output_dir)` gets an exclusive lock and proceeds; a second concurrent invocation exits immediately with a clear diagnostic naming the holding process
  2. Killing the holding process (SIGKILL, OOM) releases the lock automatically so the next invocation succeeds without manual cleanup
  3. `run_rdna4_profiler_timing_batch.py` and `run_rdna4_profiler_overhead_calibration.py` acquire the lock unconditionally at startup; `run_derived_isolated.py` acquires it only when `--pid-lock` flag is passed
**Plans**: 1 plan
**Plan List**:
- [x] 175-01-PLAN.md — Create fcntl.flock-based PID lock module with unit/integration tests and script integration

### Phase 176: Timing Isolation Audit ✅ COMPLETE

**Goal**: Profiling scripts verify their execution environment is clean before collecting timing-sensitive measurements and record that state for reproducibility audits
**Depends on**: Existing `clock_lock` module (no phase dependency)
**Requirements**: ISOL-01, ISOL-02, ISOL-03, ISOL-04
**Success Criteria** (what must be TRUE):
  1. Before profiling starts, the script detects concurrent GPU processes via `rocm-smi`/`amd-smi` and warns or aborts depending on severity
  2. Clock lock state is verified at batch start and rechecked between problems during long batch runs, with a logged warning if state drifts
  3. `torch.cuda.empty_cache()` is called at subprocess boundaries, reducing inter-problem GPU memory state leakage
  4. Batch summary sidecar includes an environment snapshot (GPU processes, clock state, lock status) enabling post-hoc reproducibility audit
**Plans**: 1 plan
**Plan List**:
- [x] 176-01-PLAN.md — Create timing_isolation.py module with concurrent GPU process detection, clock verification, cache clearing, and environment snapshot, plus comprehensive tests and script integration

### Phase 177: Profiler Timing Batch Parallelism ✅ COMPLETE

**Goal**: The profiler timing batch script stages problems in parallel CPU threads while keeping GPU profiling strictly serial, eliminating the manual multi-instance workflow and its timing bias
**Depends on**: Phase 175 (PID lock), Phase 176 (timing isolation)
**Requirements**: PRFL-01, PRFL-02, PRFL-03, PRFL-04, PRFL-05, PRFL-06
**Success Criteria** (what must be TRUE):
  1. CPU-side staging (JSON parsing, ProblemPackager construction, temp directory setup) runs in parallel via ThreadPoolExecutor while GPU profiling subprocess calls remain strictly serial
  2. No configuration flag or code path can enable concurrent GPU subprocess execution -- exclusivity is architecturally enforced
  3. Target list is pre-partitioned across worker threads by index so each worker owns exclusive targets with no file-based coordination or TOCTOU race
  4. Existing `--resume` deduplication semantics are preserved -- completed targets are skipped atomically under parallel execution
  5. Keyboard interrupt produces structured partial-completion output where interrupted targets are clearly distinguishable from completed or blocked targets
  6. Final output order is deterministic (problem-sorted) regardless of parallel completion order
**Plans**: 1 plan
**Plan List**:
- [x] 177-01-PLAN.md — Refactor run_rdna4_profiler_timing_batch.py with ThreadPoolExecutor-based parallel staging, comprehensive tests covering all six requirements

### Phase 178: Derived Script Parallelism

**Goal**: The derived isolation script runs multiple problem subprocesses concurrently, improving throughput for CPU-bound derived sidecar generation without affecting GPU correctness
**Depends on**: Phase 175 (PID lock, optional flag)
**Requirements**: DERV-01, DERV-02, DERV-03, DERV-04
**Success Criteria** (what must be TRUE):
  1. `run_derived_isolated.py` dispatches per-problem subprocesses concurrently via ThreadPoolExecutor instead of a serial for-loop
  2. Status JSONL writes are thread-safe -- concurrent workers never produce interleaved or corrupted lines
  3. Existing `--resume` and `--continue-on-failure` semantics produce identical results under parallel execution as under serial execution
  4. `--jobs` flag controls concurrency level with a sensible default (e.g., `min(os.cpu_count(), 4)`)
**Plans**: 1 plan
**Plan List**:
- [ ] 178-01-PLAN.md — Refactor run_derived_isolated.py with ThreadPoolExecutor-based parallel dispatch, thread-safe JSONL writes, and comprehensive tests

### Phase 179: Evaluation Stability Extension and Integration Tests

**Goal**: Evaluation stability diagnostics recognize new concurrency-related failure modes and integration tests verify the complete parallelism and safety hardening system end-to-end
**Depends on**: Phase 175, Phase 176, Phase 177, Phase 178
**Requirements**: STAB-01, STAB-02
**Success Criteria** (what must be TRUE):
  1. Evaluation stability diagnostics include `gpu_contention` and `multi_instance_interference` reason codes for concurrent GPU access detection
  2. Integration tests verify PID lock contention (second instance rejected), parallel staging with serial profiling (no GPU overlap), and isolation audit output (environment snapshot present and well-formed)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 175 -> 176 -> 177 -> 178 -> 179

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 175. PID Lock Module | 1/1 | Complete    | 2026-06-10 |
| 176. Timing Isolation Audit | 1/1 | Complete    | 2026-06-10 |
| 177. Profiler Timing Batch Parallelism | 1/1 | Complete    | 2026-06-10 |
| 178. Derived Script Parallelism | 0/1 | Planning complete | - |
| 179. Evaluation Stability Extension and Integration Tests | 0/? | Not started | - |
