---
gsd_state_version: 1.0
milestone: v1.18
milestone_name: ROCm Version Matrix via Docker
status: Awaiting next milestone
stopped_at: Completed 80-02-PLAN.md
last_updated: "2026-05-28T11:46:07.267Z"
last_activity: 2026-05-28 — Milestone v1.18 completed and archived
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 9
  completed_plans: 9
  percent: 100
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-05-28)

**Core value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.
**Current focus:** v1.18 milestone audit

## Current Position

Phase: Milestone v1.18 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-05-28 — Milestone v1.18 completed and archived

## Performance Metrics

**Velocity:**

- Current milestone plans completed: 3
- Current milestone phases completed: 1
- Average duration: 5.7min
- Total execution time: 17min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 78. Matrix Contract And Claim Guardrails | 2/2 | 11min | 5.5min |
| 79. Docker Matrix Selection And Preflight | 1/2 | 6min | 6min |
| 80. uv And PyTorch ROCm Wheel Coordination | 0/TBD | Not started | n/a |
| 81. Runtime Evidence And Compatibility Reports | 0/TBD | Not started | n/a |
| 82. Validation Workflow, Docs, And CI Guardrails | 0/TBD | Not started | n/a |
| 78 | 2 | - | - |
| 79 | 2 | - | - |
| 80 | 2 | - | - |

**Recent Trend:**

- Last milestone: v1.17 shipped on 2026-05-26
- Trend: v1.18 starts with five planned phases for Docker-based ROCm version matrix evidence.

| Phase 78 P1 | 5min | 2 tasks | 2 files |
| Phase 78 P2 | 6min | 3 tasks | 6 files |
| Phase 79 P1 | 6min | 3 tasks | 4 files |
| Phase 79 P2 | 8min | 3 tasks | 3 files |
| Phase 80 P1 | 8min | 3 tasks | 7 files |
| Phase 80 P2 | 5min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

- v1.18 starts at Phase 78 because v1.17 ended at Phase 77.
- v1.18 uses Target and Matrix Entry language for compatibility evidence.
- Illegal mixed-version Targets are blocked during preflight by default.
- Mixed-version debug override may continue probes or smoke only, without clean validation, score, paper-parity, or leaderboard claims.
- Docker container validation must be described as container ROCm user-space validation on recorded host driver/devices, never native host validation.
- [Phase 78]: Target/requested values and observed host, container, Python dependency, toolchain, and GPU evidence are modeled as separate required objects.
- [Phase 78]: Score, paper-parity, and leaderboard authority are literal false fields on every Matrix Entry claim boundary.
- [Phase 78]: Compatibility Matrix Entries are strict sidecars separate from canonical trace, timing, scoring, and benchmark result schemas.
- [Phase 78]: Mixed-version Matrix Entries are blocked before benchmark execution by default.
- [Phase 78]: Docker/container scoped Matrix Entries cannot claim host_validated or native_host_validated=true.
- [Phase 78]: Explicit mixed-version debug override allows probes and smoke execution only, while clean validation and authority flags remain false.
- [Phase 78]: Compatibility matrix evidence remains sidecar-only and absent from canonical Definition, Workload, and Trace payloads.
- [Phase 79]: Declared Docker Target selection is repo-owned JSON policy, not runtime Docker tag discovery.
- [Phase 79]: Unknown Docker Targets require explicit unsafe/untested override and remain not_tested with authority fields false.
- [Phase 79]: Docker preflight failures classify as runtime_unavailable before benchmark execution.
- [Phase 79]: The Dockerfile keeps rocm/dev-ubuntu-24.04:7.1.1-complete as the default through pre-FROM build args.
- [Phase 79]: The Docker wrapper resolves Target selection before host preflight so unknown Targets fail before Docker build/run.
- [Phase 79]: Script tests use SOL_EXECBENCH_RUN_DOCKER_DRY_RUN=1 and env-injected observations to avoid live Docker or ROCm hardware.
- [Phase 80]: Dependency policy is stored next to each declared Docker Target and serialized inside Matrix observed evidence. — Plan 80-01 needs auditable Target-adjacent PyTorch ROCm wheel/index policy and strict Matrix Entry evidence.
- [Phase 80]: ROCm 7.1 remains the project-default uv path; ROCm 7.0 and 7.2 are explicit Target workflows only. — The plan preserves pyproject.toml and uv.lock while recording per-Target dependency workflows.
- [Phase 80]: The Docker wrapper delegates dependency policy to dependency_matrix.py preflight JSON before Docker build/run.
- [Phase 80]: Mixed-version dependency debug uses --allow-mixed-version-dependencies or SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES=1, separate from --allow-unknown-target, and grants no benchmark or authority claims.
- [Phase 81]: Runtime compatibility evidence is emitted only as explicit sidecars; default benchmark traces, score/timing/correctness schemas, defaults, and exit semantics remain unchanged.
- [Phase 81]: Per-target sidecars and aggregate matrix reports reuse `sol_execbench.rocm_compatibility_matrix.v1` and keep host, container, Python dependency, dependency policy, toolchain, and GPU evidence in separate observed scopes.
- [Phase 81]: Setup/runtime, dependency, benchmark correctness, and benchmark performance evidence are distinct diagnostic categories and are not converted into benchmark failures.
- [Phase 82]: ROCm Matrix documentation now guards against overstating Docker container user-space evidence as native host validation.
- [Phase 82]: CPU-safe Matrix guardrails are documented as the default validation path; live ROCm validation remains marker-gated.
- [Phase 82]: Current host ROCm 7.1.x evidence may be recorded as observed evidence, while native-host ROCm 7.0.x/7.2.x validation requires matching hosts and is not a default reinstall requirement.

### Pending Todos

None found.

### Blockers/Concerns

- PyTorch ROCm wheel availability for ROCm 7.0.x and 7.2.x must be verified during Phase 80 planning/execution.
- Host-driver/container compatibility, Triton ROCm readiness, and image digest capture remain implementation research items for Phases 79-81.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Dataset extraction | Original paper 124-model / 235-problem extraction | Deferred | v1.10 scope |
| Hardware validation | MI300X, CDNA 3, and CDNA 4 real-hardware validation | Deferred | v1.10 scope |
| Native host matrix | Host reinstall or separate-host ROCm 7.0.x/7.1.x/7.2.x validation | Deferred | v1.18 scope |
| Public service | Hosted leaderboard or submission service | Deferred | v1.10 scope |

## Session Continuity

Last session: 2026-05-28T09:38:04.361Z
Stopped at: Completed 80-02-PLAN.md
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
