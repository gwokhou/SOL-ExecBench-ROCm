# Phase 123: Provenance Classification Policy - Context

**Gathered:** 2026-06-02
**Status:** Ready for planning
**Mode:** Autonomous from roadmap, research summary, and local upstream diff

<domain>

## Phase Boundary

Establish a defensible provenance classification policy and identify which
active files may retain NVIDIA copyright notices before bulk header cleanup.

This phase does not edit existing file headers. It creates the policy and
classification artifact that Phase 124 will consume.

</domain>

<decisions>

## Implementation Decisions

- Use Apache-2.0 as the unchanged project license.
- Use SPDX file tags as the file-level metadata format.
- Use a project attribution placeholder of `Copyright (c) 2026 contributors to
  SOL ExecBench ROCm Port` until a named legal entity is provided.
- Treat upstream path existence as a provenance signal, not proof of exact
  derivation. Phase 124 may refine individual files if direct code comparison
  finds a stronger or weaker relationship.
- Do not rewrite git history for prior blanket headers.

</decisions>

<code_context>

## Existing Code Insights

- `upstream/main` is available locally.
- Active files that exist at the same path in `upstream/main` are likely
  upstream-retained or derivative candidates.
- Active files absent from `upstream/main`, especially ROCm/AMD scoring,
  evidence, release, Docker dependency, and prerelease scripts, are likely
  independent ROCm work unless they copied upstream expression from another
  path.
- Current NVIDIA header scan found 132 active files with the NVIDIA SPDX line.
  Of those, 52 exist at the same path upstream and 79 do not.

</code_context>

<specifics>

## Specific Ideas

- Add `provenance.toml` as a reviewable manifest with exact
  `nvidia_notice.allowed` and `nvidia_notice.cleanup_candidates` entries.
- Add `docs/provenance.md` to define the classification policy and release
  interpretation.
- Add a focused test proving the manifest accounts for all current NVIDIA SPDX
  headers.

</specifics>

<deferred>

## Deferred Ideas

- Bulk SPDX header changes are deferred to Phase 124.
- Compliance docs and release wording are deferred to Phase 125.
- Provenance-aware readiness gates are deferred to Phase 126.

</deferred>
