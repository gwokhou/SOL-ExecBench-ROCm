# Phase 191 Discussion Log

**Date:** 2026-06-21
**Phase:** 191 - Structured Profile Summary Evidence

## Questions Asked

### 1. Metric Source Scope

Options:

1. Only derive bounded metrics from registered profiler artifacts and Trace JSONL; do not parse heavy database content.
2. Parse CSV/JSON text artifacts; cite `.rocpd` without parsing.
3. Attempt to parse `.rocpd`/database content.

User selected: **2**

Decision: Phase 191 parses CSV/JSON text artifacts and cites `.rocpd` without parsing.

### 2. Bottleneck Hint Granularity

Options:

1. Conservative categories: `compute_bound`, `memory_l2_bound`, `lds_bound`, `launch_overhead`, `insufficient_counters`, `unknown`.
2. Finer categories including occupancy, VGPR/SGPR pressure, cache, and bandwidth.
3. Record raw metrics only; do not output bottleneck hints.

User selected: **1**

Decision: Phase 191 uses the conservative closed bottleneck taxonomy.

### 3. Schema And Fixture Scope

Options:

1. Allow profile-summary schema fixture changes; update fixtures/docs/tests.
2. Add only backward-compatible fields without changing fixture structure.
3. Do not change schema; only fill internal summary metrics.

User selected: **1**

Decision: Phase 191 may update the profile-summary schema, fixtures, docs, and tests as long as authority boundaries remain diagnostic-only.

## Locked Decisions

- Parse CSV/JSON artifacts, not `.rocpd` databases.
- Emit conservative bounded bottleneck hints.
- Update schema/fixtures/docs/tests explicitly.
- Preserve diagnostic-only authority for profile summary and agent feedback sidecars.
