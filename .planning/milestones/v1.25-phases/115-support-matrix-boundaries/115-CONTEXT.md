# Phase 115: Support Matrix Boundaries - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase makes prerelease support boundaries easy to interpret across RDNA 4,
Docker/container user-space evidence, MI300X-on-CDNA3, and unavailable CDNA4
validation. It should update documentation and guardrails so users can
distinguish evidence levels without implying new hardware validation.

</domain>

<decisions>
## Implementation Decisions

### Hardware Naming
- Treat MI300X as the concrete CDNA3 hardware target for this project, not as a
  separate architecture category alongside CDNA3.
- Keep `gfx940`, `gfx941`, and `gfx942` as CDNA3 schema/code targets where
  relevant, with MI300X represented by `gfx942`.
- Treat CDNA4 validation as unavailable for the engineering prerelease because
  suitable hardware is not currently accessible.
- Preserve old guardrail phrases where tests depend on them, but add clarifying
  wording so readers do not infer that CDNA3 and MI300X are independent
  validation targets.

### Evidence Levels
- RDNA 4 evidence can be described as prerelease evidence only where archived
  artifacts or prior milestone docs support it.
- Docker/container user-space validation must remain distinct from native-host
  hardware validation.
- MI300X-on-CDNA3 full-suite validation remains deferred unless a complete
  real-hardware evidence chain exists.
- CDNA4 remains unavailable, not merely deferred by project priority.

### Release Interaction
- Reuse the Phase 114 release-candidate validation terminology where possible:
  engineering prerelease, bounded evidence, deferred evidence, and
  diagnostic-only sidecars.
- Do not add paper parity, upstream SOLAR parity, leaderboard readiness, hard
  sandbox, or new hardware-validation claims.

### the agent's Discretion
Choose the most maintainable support-matrix surface, likely documentation plus
focused wording guardrail tests. Avoid adding a new machine-readable support
schema unless existing code already has a natural place for it.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/rocm.md` already has a Hardware Status table covering RDNA 4, CDNA 3,
  and CDNA 4, but its CDNA4 wording predates the user's unavailable-hardware
  clarification.
- `docs/CLAIMS.md` defines allowed claims, Docker Matrix boundaries, and
  forbidden stronger claims.
- `docs/release_candidate_validation.md` now defines engineering prerelease
  validation language and explicit claim boundaries.
- Internal readiness docs under `docs/internal/` cover RDNA4, CDNA3, and MI300X
  readiness context.
- Guardrail tests already assert key phrases in docs and planning metadata.

### Established Patterns
- Support and validation claims are documented through explicit evidence
  levels, sidecar authority boundaries, and tests for forbidden overclaims.
- Docker Matrix entries can validate container ROCm user-space on recorded
  host driver/devices, but they do not prove native-host validation.
- Hardware-sensitive claims require archived environment, clock, trace, and
  pass/skip/fail evidence.

### Integration Points
- Update `docs/rocm.md`, `docs/CLAIMS.md`, and any release/prerelease docs that
  mention MI300X-on-CDNA3 and CDNA4 support status.
- Add or update tests in `tests/sol_execbench/` that read documentation and
  assert support-matrix wording.
- Preserve compatibility with existing public contract guardrail tests.

</code_context>

<specifics>
## Specific Ideas

- Add a prerelease support matrix section or table with rows for RDNA 4,
  Docker/container user-space, MI300X-on-CDNA3, and CDNA4.
- Explicitly state that CDNA4 hardware is not currently accessible, so CDNA4 is
  unavailable for validation in this prerelease.
- Clarify that MI300X is the CDNA3 validation target and full-suite MI300X-on-CDNA3
  validation remains deferred unless complete evidence is archived.

</specifics>

<deferred>
## Deferred Ideas

- Running MI300X-on-CDNA3 validation.
- Running CDNA4 validation while hardware is unavailable.
- Adding native-host Matrix authority, paper parity, upstream SOLAR parity, or
  leaderboard readiness.

</deferred>
