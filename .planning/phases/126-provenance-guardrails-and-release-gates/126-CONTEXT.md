# Phase 126: Provenance Guardrails And Release Gates - Context

**Gathered:** 2026-06-02
**Status:** Ready for planning
**Mode:** Autonomous from completed Phases 123-125

<domain>

## Phase Boundary

Convert copyright/provenance cleanup into repeatable guardrails that prevent
future blanket header drift.

</domain>

<decisions>

## Implementation Decisions

- Use `provenance.toml` as the source of truth for provenance checks.
- Keep `test_rocm_migration_residue_audit.py` as the broad CUDA/NVIDIA residue
  test.
- Add prerelease readiness provenance checks to
  `scripts/check_prerelease_readiness.py`.
- Readiness provenance checks should run with normal readiness checks and not
  mutate benchmark behavior.

</decisions>

<code_context>

## Existing Code Insights

`scripts/check_prerelease_readiness.py` already performs artifact, checksum,
claim-boundary, known-gap, and doc phrase checks. The provenance gate can be a
new finding category using the same `Finding` model.

</code_context>

<deferred>

## Deferred Ideas

Full REUSE conformance and legal review remain future requirements, not this
phase.

</deferred>
