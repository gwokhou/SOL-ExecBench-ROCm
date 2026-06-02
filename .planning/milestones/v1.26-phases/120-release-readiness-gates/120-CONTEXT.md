# Phase 120: Release Readiness Gates - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss, infrastructure-style execution

<domain>
## Phase Boundary

This phase delivers a release-readiness gate for the v1.26 prerelease package. The gate validates the Phase 119 artifact bundle, checks required files and checksums, verifies conservative claim-boundary fields and wording, and reports every known gap with an explicit status before publication.

</domain>

<decisions>
## Implementation Decisions

### Gate Contract
- Add a CPU-safe script in `scripts/` rather than wiring a new runtime CLI command.
- Treat missing required bundle files, missing checksums, unknown authority classes, truthy forbidden claim fields, or invalid known-gap statuses as blocking failures.
- Treat deferred/unavailable known gaps as reviewable findings, not hidden success.
- Write machine-readable JSON and Markdown gate reports.

### Claim Boundary
- Keep the corrected MI300X relationship: MI300X is the concrete CDNA3 `gfx942` hardware target, not a separate architecture peer.
- Gate against full paper validation, upstream SOLAR parity, leaderboard readiness, hard-sandbox authority, Docker-to-native-host validation inference, full MI300X validation without evidence, and CDNA4 validation.
- Verify public docs still contain representative non-claim wording.

### Integration
- The gate consumes `prerelease_artifact_bundle.json` and `SHA256SUMS` from Phase 119.
- The gate should be testable with synthetic bundle directories.
- Publication drafting remains Phase 122.

### the agent's Discretion
- The exact report schema and command name are at the agent's discretion as long as GATE-01 through GATE-03 are covered.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/build_prerelease_artifact_bundle.py` emits bundle manifest, Markdown, transcripts, validation summaries, environment evidence, known gaps, claim boundary, and checksums.
- `scripts/release_candidate_validation.py` already classifies validation failures.
- Existing docs and tests carry conservative claim-boundary wording.

### Established Patterns
- Repo-level release workflows live in `scripts/`.
- Report scripts use JSON/Markdown outputs with explicit schema versions and bounded authority language.
- Tests mock filesystem and subprocess boundaries and validate guardrail wording directly.

### Integration Points
- Add gate script under `scripts/`.
- Add focused tests under `tests/sol_execbench/`.
- Add maintainer docs under `docs/`.
- Update prerelease checklist to include the gate command.

</code_context>

<specifics>
## Specific Ideas

- Default gate input should be `out/prerelease_artifact_bundle/<version>` or a caller-provided `--bundle-dir`.
- Gate report should make blocking failures easy to distinguish from deferred/unavailable known gaps.
- Gate should fail if required artifacts referenced in the manifest are missing or have checksum mismatches.

</specifics>

<deferred>
## Deferred Ideas

- Uploading artifacts and generating GitHub release text belongs to Phase 122.
- Research-preview narrative synthesis belongs to Phase 121.

</deferred>
