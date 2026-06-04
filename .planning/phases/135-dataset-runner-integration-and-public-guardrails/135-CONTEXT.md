# Phase 135: Dataset Runner Integration and Public Guardrails - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous single-pass discuss/context

<domain>
## Phase Boundary

Integrate migrated local SOL-ExecBench and FlashInfer Trace dataset artifacts
with `scripts/run_dataset.py`, execution closure, documentation, and public
release guardrails. The phase must preserve local-only NVIDIA dataset
boundaries, ready-subset denominator accounting, and explicit deferred hardware
validation boundaries for CDNA3/CDNA4 and NVIDIA/Blackwell low-precision
semantics.

</domain>

<decisions>
## Implementation Decisions

### Runner Scope

Use the existing dataset runner and execution-closure seams rather than adding
a second runner. Preserve single-problem and ordinary dataset-root behavior
unless migration manifest, readiness, ready-subset, or closure reporting
arguments are provided.

### Public Safety

Commands and docs should describe local user download and local migration only.
Do not add examples that imply NVIDIA/SOL-ExecBench source or migrated
derivative dataset payloads are shipped by this repository or safe to
redistribute.

### Readiness and Evidence

Runner closure should retain skipped, filtered, blocked, missing, and
unvalidated low-precision workloads in the denominator. Unvalidated CDNA3/CDNA4
and NVIDIA-only performance semantics remain blockers or claim-boundary notes,
not validation evidence.

### Idempotency

Preserve existing resume and stale-provenance behavior. Existing trace reuse is
allowed only when closure provenance still matches the selected migration
manifest, readiness classification, ready subset, solution mode, runtime config,
and requested evidence.

</decisions>

<code_context>
## Existing Code Insights

- `scripts/run_dataset.py` already accepts `--ready-subset`, `--readiness`,
  `--dataset-manifest`, and `--execution-closure`, filters selected workloads,
  records closure sidecars, and compares provenance for reuse.
- `src/sol_execbench/core/dataset/run_closure.py` owns reuse decisions and
  closure record construction.
- `src/sol_execbench/core/dataset/execution_closure.py` defines the closure
  contract and provenance comparison keys.
- Phase 132 manifests expose `migration_kind`, source metadata,
  `license_boundary`, blockers, denominators, artifacts, and
  `manifest_checksum`.
- Phase 133 readiness and ready subsets expose workload readiness class,
  status, blocker reports, denominator metadata, and claim boundaries.
- Phase 134 low-precision support marks NVFP4/Blackwell compatibility as
  semantic and unvalidated on CDNA4.

</code_context>

<specifics>
## Specific Ideas

- Enrich execution closure provenance with source dataset, migration kind,
  source revision/root, license boundary, redistribution flags, readiness and
  ready-subset denominator metadata, and claim boundaries.
- Enrich per-workload closure records with readiness class and blocker details
  so skipped/blocker/not-ready workloads are visible without consulting a
  second file.
- Add deterministic summary behavior for closure-only ready-subset runs that
  have no runnable workloads.
- Add CPU-safe tests around migrated manifest metadata, ready-subset filtering,
  readiness-blocked closure rows, stale provenance for evidence/migration
  drift, and docs/guardrail strings.
- Update docs with local migration to ready subset to bounded runner workflow.

</specifics>

<deferred>
## Deferred Ideas

- Real CDNA3/MI300X full-suite execution.
- Real CDNA4/NVFP4 hardware validation or performance authority.
- High-performance FlashInfer CUDA-kernel ROCm tuning.
- Hosted leaderboard, remote submissions, or public dataset redistribution.

</deferred>
