# Phase 191: Structured Profile Summary Evidence - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 191 expands `profile_summary.sidecar.v1` from bounded status and artifact
metadata into structured diagnostic profiling evidence. The phase should add
workload/kernel metric records and stable AMD-oriented bottleneck hints while
preserving the explicit diagnostic-only authority boundary.

This phase does not create official benchmark score evidence or measured
baseline provenance. Those belong to phases 192 and 193.

</domain>

<decisions>
## Implementation Decisions

### Metric Source Scope

- **D-01:** Parse text profiler artifacts only: CSV and JSON files registered
  by Phase 190.
- **D-02:** `.rocpd` and database artifacts should be cited but not parsed in
  Phase 191. Parsing those artifacts is deferred because ROCm schema/version
  compatibility would make the phase larger and riskier.
- **D-03:** Metrics should also use canonical Trace JSONL/profile metadata where
  bounded and already available, especially workload identity, artifact
  coverage, dispatch counts, and duration fields.

### Bottleneck Hint Taxonomy

- **D-04:** Use a conservative closed hint taxonomy:
  `compute_bound`, `memory_l2_bound`, `lds_bound`, `launch_overhead`,
  `insufficient_counters`, and `unknown`.
- **D-05:** Do not emit fine-grained occupancy, VGPR/SGPR pressure, cache, or
  bandwidth conclusions unless enough counters are present to justify them in a
  later phase.
- **D-06:** Missing or insufficient counters must degrade to
  `insufficient_counters` or `unknown`, not speculative optimization claims.

### Schema And Fixtures

- **D-07:** Phase 191 may add backward-compatible fields to
  `profile_summary.sidecar.v1` and should update fixtures, docs, and tests.
- **D-08:** New fields must remain diagnostic-only. They cannot become
  correctness, timing, performance, score, evidence-tier, release-gate,
  cutover, paper-parity, leaderboard, or claim-upgrade authority.
- **D-09:** HIP-facing docs should describe the new fields as bounded adapter
  input and require consumers to downgrade unknown hint categories.

</decisions>

<canonical_refs>
## Canonical References

### Milestone Scope

- `.planning/REQUIREMENTS.md` — PSUM-01, PSUM-02, PSUM-03.
- `.planning/ROADMAP.md` — Phase 191 goal and success criteria.
- `.planning/research/SUMMARY.md` — v1.38 research summary.
- `.planning/research/FEATURES.md` — expected structured profile summary
  features.
- `.planning/research/ARCHITECTURE.md` — data flow and integration points.
- `.planning/research/PITFALLS.md` — authority-boundary and counter coverage
  risks.

### Existing Code

- `src/sol_execbench/core/bench/profile_summary.py` — strict
  `profile_summary.sidecar.v1` schema and builder.
- `src/sol_execbench/core/bench/rocm_profiler.py` — registered artifact kinds,
  profile result metadata, and CSV timing parsing helpers.
- `src/sol_execbench/cli/main.py` — CLI wiring for profile-summary sidecar
  generation and citations.
- `src/sol_execbench/core/data/trace.py` — canonical Trace JSONL workload and
  evaluation model.

### Tests And Docs

- `tests/sol_execbench/test_profile_summary.py` — profile-summary schema,
  authority, freshness, and governance tests.
- `tests/sol_execbench/test_cli_environment_snapshot.py` — CLI sidecar
  generation and citation behavior.
- `tests/sol_execbench/fixtures/profile_summary/` — HIP-facing profile summary
  fixtures.
- `docs/user/profile_summary_sidecar.md` — consumer contract.
- `docs/user/agent_feedback_sidecar.md` and `docs/user/EVALUATOR-CONTRACT.md` —
  diagnostic sidecar authority boundaries and capability references.

</canonical_refs>

<code_context>
## Existing Code Insights

- `ProfileSummaryContent.metrics` already exists as a bounded scalar metric
  list, but Phase 190 only populates artifact count and total size.
- `ProfileSummaryAuthority` and `ProfileSummaryGovernanceGuardrail` already
  enforce all authority flags as false except `diagnostic_only`.
- `Rocprofv3ProfileArtifact.kind` distinguishes `trace_csv`, `counter_csv`,
  `agent_info_csv`, `metadata_json`, `perfetto_trace`, `otf2_trace`, `rocpd`,
  and `other`.
- `rocm_profiler.parse_rocprofv3_csv()` and `summarize_rocprofv3_csv()` already
  parse representative CSV timing rows and can be reused or mirrored for
  summary metrics.

</code_context>

<specifics>
## Specific Ideas

- Add structured metric models for workload-level and kernel-level profile
  evidence while keeping scalar `metrics[]` for compatibility.
- Parse registered CSV/JSON artifacts with size and row-count bounds; record
  parse status and warnings rather than failing sidecar generation.
- Derive conservative bottleneck hints from available metrics. If counters are
  absent or ambiguous, emit `insufficient_counters` or `unknown`.
- Keep `.rocpd` citations as evidence pointers only in Phase 191.

</specifics>

<deferred>
## Deferred Ideas

- `.rocpd` database parsing.
- Fine-grained occupancy/VGPR/SGPR/cache/bandwidth bottleneck categories.
- Using profile summary as score, release-gate, cutover, or confirmed benchmark
  authority.

</deferred>

---

*Phase: 191-Structured Profile Summary Evidence*
*Context gathered: 2026-06-21*
