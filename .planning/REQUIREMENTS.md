# Requirements: SOL ExecBench ROCm Port — v1.35

**Defined:** 2026-06-10
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1 Requirements

### Instance Safety

- [x] **INST-01**: Script acquires an exclusive `fcntl.flock` on a per-output-directory lock file at startup and exits with a clear diagnostic message if another instance holds the lock
- [x] **INST-02**: Lock is auto-released on process exit including SIGKILL and OOM kill with no stale-lock cleanup required
- [x] **INST-03**: PID lock is mandatory for `run_rdna4_profiler_timing_batch.py` and `run_rdna4_profiler_overhead_calibration.py`; optional via flag for `run_derived_isolated.py`

### Timing Environment Isolation

- [x] **ISOL-01**: Pre-flight audit detects concurrent GPU processes via `rocm-smi` or `amd-smi` before profiling starts and warns or aborts
- [x] **ISOL-02**: Clock lock state is verified at batch start and rechecked between problems during long batch runs
- [x] **ISOL-03**: `torch.cuda.empty_cache()` is called at subprocess boundaries to reduce inter-problem GPU state leakage
- [x] **ISOL-04**: Batch summary sidecar records environment snapshot (GPU processes, clock state, lock status) for reproducibility audit

### Profiler Timing Batch Parallelism

- [x] **PRFL-01**: `run_rdna4_profiler_timing_batch.py` uses ThreadPoolExecutor for CPU-side staging (JSON parsing, ProblemPackager construction, temp directory setup) while GPU profiling subprocess runs remain serial
- [x] **PRFL-02**: GPU profiling exclusivity is architecturally enforced — no configuration or CLI flag can enable concurrent GPU subprocess execution
- [x] **PRFL-03**: Target list is pre-partitioned across worker threads by index so each worker owns exclusive targets with no file-based coordination
- [x] **PRFL-04**: Existing `--resume` deduplication semantics are preserved — completed targets are skipped atomically
- [x] **PRFL-05**: Keyboard interrupt produces structured partial-completion output with interrupted targets clearly distinguishable from completed or blocked targets
- [x] **PRFL-06**: Output order is deterministic regardless of parallel completion order — results are collected and written in problem-sorted order

### Derived Script Parallelism

- [ ] **DERV-01**: `run_derived_isolated.py` uses ThreadPoolExecutor for concurrent per-problem subprocess dispatch
- [ ] **DERV-02**: Status JSONL writes are thread-safe via lock or batch-flush semantics
- [ ] **DERV-03**: Existing `--resume` and `--continue-on-failure` semantics are preserved under parallel execution
- [ ] **DERV-04**: `--jobs` flag controls concurrency level with a sensible default

### Evaluation Stability Extension

- [ ] **STAB-01**: New reason codes (`gpu_contention`, `multi_instance_interference`) are added to evaluation stability diagnostics for concurrent GPU access detection
- [ ] **STAB-02**: Integration tests verify PID lock contention, parallel staging + serial profiling, and isolation audit output

## v2 Requirements

### Deferred

- **WARM-01**: Configurable GPU warmup/idle barrier with L2 cache flush between profiling invocations
- **WARM-02**: Empirical L2 cache pollution audit comparing kernel activity durations with/without parallel CPU staging
- **CONF-01**: User-configurable `--workers` CLI flag for both scripts
- **MULTI-01**: Multi-GPU parallel profiling support

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-GPU parallel profiling | Requires fundamental architecture changes; separate milestone |
| Rich progress bar during parallel execution | Post-processing concern, not core correctness |
| Changing public CLI interface or sidecar format | Preserving backward compatibility is a hard constraint |
| Hardening report-only scripts (report_*.py) | No GPU interaction, no concurrency risk |
| ProcessPoolExecutor-based parallelism | Unsafe after torch import at module level (fork deadlock) |
| PID-in-file locking | Unreliable due to PID recycling; flock is strictly superior |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-01 | Phase 175 | Complete |
| INST-02 | Phase 175 | Complete |
| INST-03 | Phase 175 | Complete |
| ISOL-01 | Phase 176 | Complete |
| ISOL-02 | Phase 176 | Complete |
| ISOL-03 | Phase 176 | Complete |
| ISOL-04 | Phase 176 | Complete |
| PRFL-01 | Phase 177 | Complete |
| PRFL-02 | Phase 177 | Complete |
| PRFL-03 | Phase 177 | Complete |
| PRFL-04 | Phase 177 | Complete |
| PRFL-05 | Phase 177 | Complete |
| PRFL-06 | Phase 177 | Complete |
| DERV-01 | Phase 178 | Pending |
| DERV-02 | Phase 178 | Pending |
| DERV-03 | Phase 178 | Pending |
| DERV-04 | Phase 178 | Pending |
| STAB-01 | Phase 179 | Pending |
| STAB-02 | Phase 179 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0

---
*Requirements defined: 2026-06-10*
*Last updated: 2026-06-10 after roadmap creation*
