# Phase 114: Release-Candidate Validation - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase creates a bounded release-candidate validation path for the
engineering prerelease. It should let maintainers run CPU-safe checks,
focused ROCm/Docker smoke checks, and a small dataset slice, then classify the
results as blocking, deferred, or diagnostic-only. It should not expand paper
parity, leaderboard, MI300X-on-CDNA3 full-suite, CDNA4, or hard-sandbox claims.

</domain>

<decisions>
## Implementation Decisions

### Validation Scope
- Keep the validation path bounded and repeatable rather than paper-scale.
- Prefer existing local commands, scripts, and sidecar report generators over
  adding new benchmark semantics.
- Treat live ROCm, Docker, profiler, and dataset availability as evidence that
  can be present, skipped, or unavailable; do not convert unavailable optional
  evidence into false success.
- Preserve canonical Trace, Definition, Workload, Solution, correctness,
  timing, score, and evaluator contract schemas.

### Evidence And Failure Classification
- Classify release-candidate outcomes into blocking, deferred, and
  diagnostic-only buckets with explicit next actions.
- Keep trace JSONL as the canonical run artifact and keep profile, static,
  environment, trust, and closure reports as sidecar evidence.
- Keep log tails bounded and avoid dumping full environments or token-like
  values into shareable artifacts.
- Do not treat `rocprofv3`, static evidence, Matrix entries, or Docker rows as
  correctness, timing, score, paper-parity, leaderboard, or native-host
  authority.

### Hardware And Dataset Boundaries
- RDNA 4 evidence can be prerelease evidence when direct artifacts exist.
- Docker/container user-space validation remains distinct from native-host
  validation.
- MI300X is the CDNA3 hardware target; MI300X-on-CDNA3 full-suite validation stays
  deferred unless a complete real-hardware evidence chain exists.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

### the agent's Discretion
Use the existing codebase patterns to decide whether this phase is best
implemented as a script, documentation checklist, pytest coverage, or a
combination. Keep the implementation minimal and release-focused.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/internal/analysis.md` documents trace collection, dataset runs, execution
  closure, parity gaps, timing evidence, clock locking, reward-hack review,
  profiling, and result interpretation.
- `docs/user/CLAIMS.md` defines current allowed claims and forbidden stronger
  claims for ROCm-port, runtime, profiling, static, Matrix, AMD-native score,
  research-preview, and leaderboard authority.
- `docs/internal/v1_19_evidence_guide.md` and `docs/internal/v1_20_evidence_quality_guide.md`
  describe evidence surfaces and trust-quality reports used for release review.
- `scripts/run_dataset.py`, `scripts/inspect_dataset.py`,
  `scripts/report_parity_gaps.py`, `scripts/report_consistency.py`,
  `scripts/report_evaluation_stability.py`, `scripts/report_claim_upgrade.py`,
  and `scripts/report_trust_summary.py` already provide release-adjacent
  evidence generation paths.
- `tests/sol_execbench/` includes CPU-safe guardrail coverage for contracts,
  dataset closure, Docker matrix behavior, scoring/reporting, and docs
  wording.

### Established Patterns
- Release credibility is built through explicit sidecars and claim-boundary
  wording rather than implicit authority upgrades.
- Hardware-sensitive behavior is guarded by pytest markers such as
  `requires_rocm`, `requires_rocm_dev`, `requires_rdna4`, `requires_cdna3`,
  `requires_ck`, `requires_rocwmma`, and `timing_serial`.
- Dataset-scale behavior separates discovered, parsed, ready, attempted,
  passed, failed, scored, degraded, unscored, unavailable, and evidence-missing
  states.
- Optional profiling and static evidence are diagnostic-only and must remain
  separate from canonical trace JSONL.

### Integration Points
- Add or update release validation documentation under `docs/` and protect it
  with focused docs/guardrail tests under `tests/sol_execbench/`.
- If a command wrapper is useful, keep it under `scripts/` and make it compose
  existing commands instead of embedding new benchmark policy.
- If report fixtures are needed, keep them small and deterministic so CPU-safe
  tests can validate shape and wording.

</code_context>

<specifics>
## Specific Ideas

- Define a single prerelease validation checklist or command path that records
  CPU-safe test results, ROCm/Docker smoke status, dataset-slice evidence, and
  failure classification.
- Keep the bounded dataset slice explicitly separate from full 235-problem
  paper validation.
- Prefer wording such as "engineering prerelease evidence" and "bounded local
  validation" over "release authority" or "paper parity".

</specifics>

<deferred>
## Deferred Ideas

- Full 235-problem paper-scale validation and upstream SOLAR parity.
- MI300X-on-CDNA3 full-suite hardware validation without complete evidence.
- CDNA4 validation while suitable hardware is unavailable.
- Hosted leaderboard, remote submissions, and hard multi-tenant sandboxing.
- Large PyTorch/ROCm dependency relocking or Docker privilege redesign unless
  validation exposes a blocking issue.

</deferred>
