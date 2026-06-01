# Phase 119: Versioned Prerelease Artifact Bundle - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss, infrastructure-style execution

<domain>
## Phase Boundary

This phase delivers a versioned prerelease artifact bundle workflow for maintainers. The bundle packages release validation output, command transcripts, environment evidence, checksums, and authority classification into reviewable JSON/Markdown artifacts. It does not change benchmark schemas, scoring behavior, runtime semantics, or release claims.

</domain>

<decisions>
## Implementation Decisions

### Bundle Contract
- Treat the bundle as an evidence packaging layer around existing validation and documentation outputs.
- Keep default generation CPU-safe so a maintainer can run it from a clean checkout without ROCm hardware.
- Allow optional evidence to be represented explicitly as `deferred` or `unavailable` rather than silently omitted.
- Make every produced or referenced artifact map to one of the authority classes required by ARTIFACT-03.

### Evidence Boundaries
- Preserve the current prerelease claim boundary: engineering prerelease only, not full 235-problem paper validation, upstream SOLAR parity, leaderboard readiness, hard sandbox evidence, CDNA4 validation, or full MI300X validation on the CDNA3 `gfx942` target.
- Treat command transcripts and environment evidence as diagnostic-only unless a later release gate promotes them.
- Keep release validation summaries diagnostic-only evidence that can still block publication when the validation command fails.
- Record missing optional hardware/dataset evidence as explicit known gaps.

### Usability And Review
- Prefer a script in `scripts/` plus focused docs over adding a new runtime package API.
- Produce stable machine-readable JSON and a concise Markdown review surface.
- Include SHA-256 checksums for copied or referenced files.
- Redact command transcript tails consistently with existing prerelease validation behavior.

### the agent's Discretion
- The exact bundle file names, schema shape, and test helpers are at the agent's discretion as long as the requirements and existing repository conventions are satisfied.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/release_candidate_validation.py` already runs bounded CPU-safe, ROCm smoke, Docker smoke, and dataset-slice checks and writes JSON/Markdown summaries.
- Existing tests import scripts with `importlib.util.spec_from_file_location` and monkeypatch subprocess boundaries.
- `docs/CLAIMS.md`, release notes, support matrix, traceability, and validation docs already carry the public prerelease claim boundary.

### Established Patterns
- Repo-level operational workflows live in `scripts/` and write review artifacts under `out/`.
- JSON artifacts use explicit `schema_version`, status summaries, and claim-boundary blocks.
- Tests prefer small CPU-safe unit coverage and mock subprocess/hardware dependencies.

### Integration Points
- Add the bundle generator under `scripts/`.
- Add tests under `tests/sol_execbench/`.
- Add user-facing instructions under `docs/`.
- Phase completion should update `.planning/ROADMAP.md`, `.planning/STATE.md`, and requirement statuses.

</code_context>

<specifics>
## Specific Ideas

- Default command should generate a bundle without requiring live ROCm, Docker, or downloaded datasets.
- Bundle artifacts should be self-describing enough for a reviewer to determine which evidence is canonical, diagnostic-only, provisional, deferred, or unavailable.
- The implementation should reuse existing release validation rather than duplicating its checks.

</specifics>

<deferred>
## Deferred Ideas

- Signing artifacts, uploading release assets, and GitHub prerelease creation belong to later phases.
- Promoting diagnostic evidence to formal gate decisions belongs to Phase 120.
- Research-preview narrative synthesis belongs to Phase 121.

</deferred>
