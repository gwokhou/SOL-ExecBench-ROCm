# Phase 190: Profiler Artifact Registration Closure - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 190 fixes `rocprofv3` profile artifact discovery and registration. When a
requested `rocprofv3` profile run succeeds and produces files, SOL must discover
and register those files as profiler artifacts with concrete citations. This
phase is about artifact discovery, classification, result status semantics, and
citation metadata. It does not parse profiler counters into bottleneck evidence;
that belongs to Phase 191.

</domain>

<decisions>
## Implementation Decisions

### Discovery Scope

- **D-01:** Artifact discovery must recursively scan the requested
  `output_directory` so ROCm-version-specific nested output layouts are covered.
- **D-02:** Recursive scanning must still filter candidates. Register files only
  when they match the requested `output_file` prefix, recognized profiler file
  names, or known `rocprofv3` output directory structures. Do not register
  arbitrary unrelated files just because they are under `output_directory`.

### Artifact Formats

- **D-03:** Phase 190 must explicitly classify common `rocprofv3` output
  formats: `rocpd`, CSV, JSON, Perfetto/PFTrace, and OTF2.
- **D-04:** Unknown profiler files that pass the discovery filter should still be
  registered as `other`. Phase 190 should not parse every format's contents.

### Status Semantics

- **D-05:** If `rocprofv3` exits with return code 0 and at least one artifact is
  discovered, keep the profile result status as `success`.
- **D-06:** Incomplete artifact sets should be expressed through artifact
  coverage status, warnings, and reason codes rather than changing the top-level
  `success` status to `partial` or `success_with_warnings`.
- **D-07:** Command failure, unavailable profiler, no discovered artifacts, and
  partial artifact coverage need stable reason codes that downstream HIP can
  display or gate on.

### Citation Integrity

- **D-08:** Profiler artifact citations should compute SHA256 for registered
  artifacts by default, including `.rocpd`/database artifacts.
- **D-09:** The planner should explicitly account for SHA256 cost risk on large
  profiler databases. If implementation later needs a size limit, that should
  be introduced as a deliberate follow-up rather than silently skipping hashes
  in Phase 190.

### the agent's Discretion

- The agent may choose the exact helper names, reason-code enum/module shape,
  and test fixture layout as long as the decisions above are preserved.
- The agent may keep artifacts as dataclasses if that fits
  `core.bench.rocm_profiler`, but public sidecar payloads must remain stable and
  test-covered.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Scope

- `.planning/ROADMAP.md` — Phase 190 goal and success criteria.
- `.planning/REQUIREMENTS.md` — PROF-01, PROF-02, PROF-03 and traceability.
- `.planning/research/SUMMARY.md` — v1.38 research summary and source links.
- `.planning/research/STACK.md` — AMD/NVIDIA source findings and stack
  additions for confirmed evidence.
- `.planning/research/ARCHITECTURE.md` — data flow and integration points for
  profiler, score, baseline, and HIP evidence.

### Existing Code

- `src/sol_execbench/core/bench/rocm_profiler.py` — current
  `Rocprofv3ProfileArtifact`, `Rocprofv3ProfileResult`,
  `discover_rocprofv3_artifacts`, `collect_rocprofv3_profile`, and artifact
  classification helpers.
- `src/sol_execbench/cli/main.py` — CLI profile sidecar/profile-summary writing
  and artifact citation wiring.
- `src/sol_execbench/core/bench/profile_summary.py` — profile summary artifact
  citation model and diagnostic-only sidecar boundaries.

### Tests And Docs

- `tests/sol_execbench/test_rocm_profiler.py` — current profiler command,
  discovery, collection, and timing evidence tests.
- `tests/sol_execbench/test_cli_environment_snapshot.py` — current profile
  sidecar/profile-summary citation behavior.
- `docs/rocm_timing.md` — existing ROCm timing/profile evidence explanation.
- `docs/profile_summary_sidecar.md` — current diagnostic-only profile summary
  contract.

### External Source References

- `rocprofv3` usage and output formats:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCm Compute Profiler System Speed-of-Light metrics:
  https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/system-speed-of-light.html
- ROCm Systems source repository:
  https://github.com/ROCm/rocm-systems

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `Rocprofv3ProfileArtifact` already records path, kind, and size and is the
  natural place to preserve discovered artifact metadata.
- `Rocprofv3ProfileResult.to_dict()` already serializes diagnostic profile
  metadata and artifacts.
- `profile_summary_artifact_citation_from_path()` already computes SHA256 for
  path citations and can inform profiler artifact citation behavior.

### Established Patterns

- ROCm profile collection is diagnostic and nonfatal; failures must not change
  canonical trace correctness or score semantics.
- Artifact paths exposed to sidecars should be compact/bounded and must avoid
  leaking unstable absolute temp paths.
- Tests use fixture-backed runner functions to simulate profiler output without
  requiring a real GPU.

### Integration Points

- `discover_rocprofv3_artifacts()` is the primary implementation target.
- `collect_rocprofv3_profile()` must preserve success/failure/unavailable
  behavior while adding coverage/warning/reason-code detail.
- CLI profile summary artifact citations should include newly discovered
  nested artifacts.

</code_context>

<specifics>
## Specific Ideas

- Prefer a recursive candidate walk plus explicit filter predicates over
  one-off special cases for the current failure.
- Keep Phase 190 focused on registration/citation. Structured profiling metrics
  and bottleneck hints should wait for Phase 191.
- Preserve top-level `success` for return-code-zero runs with at least one
  discovered artifact to avoid breaking existing `result.succeeded` semantics.

</specifics>

<deferred>
## Deferred Ideas

- Parsing profiler counters into structured bottleneck evidence belongs to
  Phase 191.
- Introducing a SHA256 size limit for large profiler databases may be a future
  optimization if Phase 190's always-hash policy proves too expensive in real
  runs.

</deferred>

---

*Phase: 190-Profiler Artifact Registration Closure*
*Context gathered: 2026-06-21*
