# Phase 85: Compatibility Matrix Schema Export And Semantic Diff - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds diagnostic tooling around the existing ROCm Compatibility
Matrix contract. It must let researchers export JSON Schema for `MatrixEntry`
and `RocmCompatibilityMatrixReport`, and compare two Matrix reports by semantic
changes, while preserving Docker/native-host, score-authority, paper-parity,
and leaderboard claim boundaries.

</domain>

<decisions>
## Implementation Decisions

### Schema Export Boundary
- Export JSON Schema only for `MatrixEntry` and
  `RocmCompatibilityMatrixReport`.
- Include schema identity/version metadata and make strict extra-field behavior
  visible to downstream consumers.
- Treat exported schemas as diagnostic sidecar contracts; do not change Matrix
  model semantics or canonical benchmark schemas.
- Avoid expanding export scope to unrelated sidecars in this phase.

### Semantic Diff Matching And Fields
- Match Matrix entries by Target identity plus validation scope.
- Diff output must classify entries as added, removed, unchanged, or changed.
- Changed entries should include semantic field groups for status, reason code,
  requested Target values, observed host/container/Python dependency/toolchain/
  GPU evidence, dependency policy, Docker image metadata, clock/evidence
  metadata, artifact refs, and claim boundaries.
- Diff logic should be deterministic and should compare normalized JSON
  payloads, not object identity or insertion-order artifacts.

### Output And Severity
- Produce both machine-readable JSON and a human-readable Markdown/summary.
- Severity-ranked transitions must cover validation downgrade, mixed-version
  drift, runtime unavailability, image/dependency drift, GPU architecture
  drift, and claim-boundary escalation.
- All schema and diff outputs remain diagnostic-only: Docker container evidence
  cannot become native-host validation, score authority, paper-parity authority,
  or leaderboard authority.
- Markdown wording must keep authority boundaries visible and avoid implying
  new hardware or native-host validation.

### the agent's Discretion
- The agent may choose exact module names, script names, and export helper
  signatures, with preference for focused helpers near
  `src/sol_execbench/core/compatibility.py` and thin scripts matching existing
  reporting utilities.
- The agent may choose severity enum names and ordering, provided the required
  transition categories are represented and deterministic.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/compatibility.py` defines strict Pydantic v2 models:
  `MatrixEntry`, `RocmCompatibilityMatrixReport`, nested evidence models,
  `MatrixValidationScope`, statuses, reason codes, and claim boundaries.
- Matrix models already use `extra="forbid"`, `frozen=True`, `strict=True`, and
  docstring schema metadata through `BaseModelWithDocstrings`.
- `src/sol_execbench/core/docker_matrix.py` builds/serializes Matrix entries
  and preview payloads for Docker target diagnostics.
- Existing tests under `tests/sol_execbench/test_rocm_compatibility_matrix.py`,
  `test_docker_matrix_targets.py`, `test_matrix_claim_guardrails.py`, and
  Docker matrix tests cover strict model behavior and claim boundaries.

### Established Patterns
- Standalone diagnostic scripts should be thin argparse wrappers over core
  helpers, similar to `scripts/report_parity_gaps.py` and
  `scripts/report_paper_denominator.py`.
- Sidecar/report helpers should produce deterministic sorted JSON and bounded
  Markdown.
- Guardrail tests should prove new Matrix tooling does not mutate primary
  benchmark CLI behavior or public canonical contracts.

### Integration Points
- Add schema export and diff helper tests close to existing Matrix tests.
- If adding script entry points, keep them script-side or module-side
  diagnostics; do not add primary `sol-execbench` CLI options.
- Future docs phase will document the generated schema/diff surfaces, so
  output names and claim-boundary wording should be stable.

</code_context>

<specifics>
## Specific Ideas

Plan around two slices: first schema export helpers and tests; second semantic
diff JSON/Markdown helpers plus a thin script and guardrails.

</specifics>

<deferred>
## Deferred Ideas

- CI policy failure gates for Matrix diff severities remain future work.
- Exporting additional sidecar schemas remains future work unless a downstream
  consumer needs them.
- Native-host Matrix validation, CDNA 3, MI300X, and CDNA 4 validation remain
  out of scope for v1.19.

</deferred>
