# Phase 87: AMD SOL/SOLAR Bound Sanity Evidence - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds diagnostic AMD SOL/SOLAR bound sanity evidence over already
available sidecars. It must help researchers inspect evidence completeness,
aggregate status, warnings, and provisional model risk without changing score
eligibility, canonical benchmark contracts, Docker behavior, dependency locks,
or hardware-validation claims.

The report target is `amd_bound_sanity.v1`. It is a sidecar/reporting artifact,
not a new benchmark result, leaderboard result, upstream SOLAR equivalence
claim, model-validation claim, or paper-parity claim.

</domain>

<decisions>
## Implementation Decisions

### Input Scope
- Read existing artifacts only: traces, execution closure, AMD SOL v2 sidecars,
  SOLAR derivation sidecars, AMD-native score reports, compatibility Matrix
  reports, and source refs/checksums.
- Missing artifacts must be represented as `missing_evidence` or related
  diagnostic gaps.
- Do not trigger GPU execution, Docker probes, Docker privilege changes, new
  hardware probes, dependency relocking, or regenerated benchmark evidence.

### Diagnostic Status Vocabulary
- Use a sanity-specific diagnostic status layer:
  `scored`, `degraded`, `unscored`, `unsupported`, `provisional`, and
  `missing_evidence`.
- The report explains evidence completeness and risk only.
- It must not reinterpret correctness, timing, canonical traces, AMD-native
  score semantics, or score eligibility rules.
- Existing AMD score fields such as `supported`, AMD SOL aggregate status,
  SOLAR aggregate status, and warnings should be consumed as source evidence,
  not rewritten.

### Claim Boundaries
- Include explicit machine-readable and human-readable boundaries.
- Surface provisional RDNA 4 model risk with a field such as
  `provisional_rdna4_model_risk=true` when evidence depends on the existing
  RDNA 4 model assumptions.
- Fixed false/negative boundaries must include: upstream SOLAR equivalence,
  AMD SOL/SOLAR model validation, paper parity, leaderboard authority, score
  authority upgrade, CDNA 3 validation, MI300X validation, CDNA 4 validation,
  native-host validation, and new-hardware validation.
- Wording should use "sanity", "diagnostic", "existing evidence", and "risk";
  avoid "validated", "equivalent", "paper result", or "leaderboard ready"
  except in explicit negative boundaries.

### the agent's Discretion
- The agent may choose exact helper/model names and whether to place the core
  contract under `src/sol_execbench/core/scoring/` or a nearby reporting module.
- Prefer strict Pydantic models and deterministic JSON/Markdown helpers,
  following Phase 84/85 sidecar patterns.
- Add a thin script wrapper rather than a primary `sol-execbench` CLI option
  unless planning discovers an established local pattern requiring otherwise.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/scoring/amd_sol_v2.py` defines
  `sol_execbench.amd_sol_bound.v2`, aggregate statuses
  `scored/degraded/unscored`, coverage summaries, warnings, and hardware model
  metadata.
- `src/sol_execbench/core/scoring/solar_derivation.py` defines
  `sol_execbench.solar_derivation.v1`, aggregate status, formula/coverage
  evidence, missing evidence, and score-eligibility signals.
- `src/sol_execbench/core/scoring/amd_score.py` defines
  `sol_execbench.amd_native_score.v1`, derived AMD-native score reports,
  `supported`, warning aggregation, evidence summaries, and provisional
  hardware/model warnings.
- `src/sol_execbench/core/compatibility.py` and Phase 85 Matrix tooling already
  model diagnostic-only claim boundaries for Docker/native-host evidence.
- `src/sol_execbench/core/dataset/execution_closure.py` and
  `src/sol_execbench/core/dataset/paper_denominator.py` provide strict sidecar
  and bounded-source-ref patterns.

### Established Patterns
- v1.19 reporting artifacts are sidecar-only and use deterministic JSON with
  checksums, bounded refs, and explicit authority-false boundaries.
- Thin scripts under `scripts/` expose research reports without adding primary
  `sol-execbench` CLI options.
- CPU-safe tests should construct fixture payloads and should not require ROCm,
  Docker, GPU access, dependency downloads, or hardware probes.

### Integration Points
- Likely core module: `src/sol_execbench/core/scoring/amd_bound_sanity.py`.
- Likely script: `scripts/report_amd_bound_sanity.py`.
- Likely tests: `tests/sol_execbench/test_amd_bound_sanity.py`,
  `tests/sol_execbench/test_amd_bound_sanity_script.py`, and public contract
  guardrails.
- Phase 88 will document the final artifact, so Phase 87 should keep names and
  reason/status codes stable.

</code_context>

<specifics>
## Specific Ideas

Plan should start with CPU-safe tests for the JSON contract and diagnostic
status rollups, then add script/Markdown output tests and guardrails proving no
primary CLI exposure and no canonical schema changes. The implementation should
read optional artifacts by path, record bounded refs/checksums, summarize
availability and warnings, and never execute benchmarks or probes.

</specifics>

<deferred>
## Deferred Ideas

- Full 235-problem paper validation, upstream SOLAR equivalence comparison,
  AMD SOL/SOLAR model validation, leaderboard authority, CDNA3-family including MI300X, CDNA 4,
  native-host validation, new hardware validation, Docker privilege changes, and
  dependency relocking remain out of scope.
- Any changes to canonical Trace, Definition, Workload, Solution, correctness,
  timing, score, evaluator, or AMD-native score eligibility semantics remain
  out of scope.

</deferred>
