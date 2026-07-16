# Decision Sidecar Contract

This document records the contract for the **Decision sidecar**, which turns
diagnostic-only data-layer facts into structured optimization guidance. The
`sol_execbench.decision.v1` schema and run path are implemented in
`src/sol_execbench/core/bench/decision/` (emits `<trace>.decision.json` via
`--decision auto`); this file records the contract boundary and design intent.

For the modeling survey and source-credibility assessment that informs this
contract, see `docs/internal/decision-modeling-research.md`.

## Why a separate sidecar

The data layer (environment capability budgets and static resource footprints)
carries raw, neutrally-named facts: `vgpr_used`, `sgpr_used`, `lds_bytes`,
`vgpr_limit`, `wavefront_size`, and so on. Decision semantics — bottleneck
classification (`high_register_pressure`, `low_occupancy_estimate`, ...),
recommendations, and confidence-weighted hints — are a separate modeling concern.
Keeping them out of the data schemas prevents factual fields from drifting into
judgmental ones and lets the decision model evolve without bumping the data
schemas that feed it.

## Inputs

The Decision sidecar consumes facts from the diagnostic data sidecars:

- `EnvironmentSnapshot.capability_budgets[]` — arch ISA resource budgets
  (`sol_execbench.environment_snapshot.v2` with packaged
  `sol_execbench.arch_capability_budget.v1` budgets).
- `StaticKernelEvidenceSidecar.footprints[]` and `kernels[].footprint` —
  per-kernel resource usage (`sol_execbench.static_kernel_evidence.v3`).
- Optional `profile_summary.v2` runtime bottleneck hints and `agent_feedback.v2`
  aggregate items, after each passes its own freshness and authority checks.

**Applicability.** The Decision sidecar requires
`StaticKernelEvidenceSidecar.footprints[]`, which the static-evidence path
collects only for HIP/C++ solutions (its `is_cpp` gate). PyTorch and Triton
solutions produce no footprints, so no decision sidecar is written for them.

## Outputs

- A closed bottleneck taxonomy spanning compile-time, runtime, and resource
  dimensions, merged across data sources with documented precedence (runtime
  measured > static inferred; static informs the pre-run and no-profile
  fallback cases).
- Prompt-safe `recommendation` strings, `confidence`, `limitations[]`, and
  `evidence_refs[]` per hint.

## Boundary

- The data layer carries facts only and never embeds decision fields.
- The decision layer consumes facts and must not re-assert or override benchmark
  authority (correctness, timing, score, paper-parity, leaderboard).
- Canonical Trace JSONL remains the only authority surface; the Decision sidecar
  is `diagnostic` only, like the other sidecars.
- `vgpr_limit` is the architected *addressing* limit, not the physical register
  file; derivation uses it as a static pressure proxy. Physical occupancy uses
  `register_file_per_cu_bytes`. See decision-modeling-research.md §11.
